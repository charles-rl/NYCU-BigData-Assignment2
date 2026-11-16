"""Model COPY for the ablation suite.

This file is a copy of ``src/models/hetero_model.py``.  The base model is left
untouched.  The copy adds exactly the two switches the ablation suite needs:

1. ``use_residual_fusion`` (default True).  When False, the six prediction
   heads only see the post-GNN user embedding; the direct concatenation of the
   raw ``data['Users'].x`` feature block is removed.  This is the
   ``no-residual-fusion`` ablation.

2. :class:`TemporalHGTConv`, an :class:`HGTConv` that multiplies each edge's
   relation prior by a per-edge scalar weight.  The graph builder copy stores
   ``edge_weight = exp(-dt / tau)`` for event edges and ``1.0`` for the
   ``no-temporal-decay`` ablation.  Passing unit weights is mathematically
   identical to the stock HGT operator, so the two variants differ only in the
   edge weights, not in parameters or code paths.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.nn.conv.hgt_conv import construct_bipartite_edge_index


class NodeMLPEncoder(nn.Module):
    """Encode raw heterogeneous node features to a uniform latent dimension."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.1,
        normalize_latent: bool = True,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln_fc1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln_fc2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln_fc3 = nn.LayerNorm(hidden_dim)

        self.fc4 = nn.Linear(hidden_dim, latent_dim)
        self.ln_fc4 = nn.LayerNorm(latent_dim)

        self.dropout = nn.Dropout(dropout)
        self.normalize_latent = normalize_latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_fc1(F.mish(self.fc1(x)))
        x = self.dropout(x)
        x = self.ln_fc2(F.mish(self.fc2(x)))
        x = self.dropout(x)
        x = self.ln_fc3(F.mish(self.fc3(x)))
        x = self.dropout(x)
        if self.normalize_latent:
            x = self.ln_fc4(F.mish(self.fc4(x)))
        else:
            x = F.mish(self.fc4(x))
        return x


class PredictionHead(nn.Module):
    """Task-specific MLP head returning raw logits for BCEWithLogitsLoss."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln_fc1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln_fc2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_fc1(F.mish(self.fc1(x)))
        x = self.ln_fc2(F.mish(self.fc2(x)))
        return self.fc3(x)


class TemporalHGTConv(HGTConv):
    """HGTConv with an optional per-edge scalar temporal-decay weight.

    The stock operator uses one learned relation prior per edge type, expanded
    to all edges.  This subclass accepts ``edge_weight_dict`` and multiplies
    that prior by the edge's scalar weight before the attention softmax::

        alpha_e = softmax((q_i . k_j) * p_rel[type] * w_e)

    With ``edge_weight_dict=None`` or unit weights, the result is identical to
    :class:`HGTConv`.
    """

    def forward(  # type: ignore[override]
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_weight_dict: dict[tuple[str, str, str], torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | None]:
        out_channels = self.out_channels
        heads = self.heads
        head_dim = out_channels // heads

        k_dict: dict[str, torch.Tensor] = {}
        q_dict: dict[str, torch.Tensor] = {}
        v_dict: dict[str, torch.Tensor] = {}
        out_dict: dict[str, torch.Tensor | None] = {}

        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, heads, head_dim)
            q_dict[key] = q.view(-1, heads, head_dim)
            v_dict[key] = v.view(-1, heads, head_dim)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict
        )

        if edge_weight_dict is None:
            edge_attr_dict: dict[tuple[str, str, str], torch.Tensor] = {
                edge_type: self.p_rel["__".join(edge_type)]
                for edge_type in edge_index_dict
            }
        else:
            weighted: dict[tuple[str, str, str], torch.Tensor] = {}
            for edge_type in edge_index_dict:
                prior = self.p_rel["__".join(edge_type)]
                weight = edge_weight_dict.get(edge_type)
                if weight is None:
                    weighted[edge_type] = prior
                    continue
                edge_index = edge_index_dict[edge_type]
                if not isinstance(edge_index, torch.Tensor):
                    raise TypeError(
                        f"{edge_type}: temporal weights need a dense edge_index"
                    )
                if weight.numel() != edge_index.shape[1]:
                    raise ValueError(
                        f"{edge_type}: {weight.numel()} weights for "
                        f"{edge_index.shape[1]} edges"
                    )
                weighted[edge_type] = prior * weight.to(
                    device=prior.device, dtype=prior.dtype
                ).view(-1, 1)
            edge_attr_dict = weighted

        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict,
            src_offset,
            dst_offset,
            edge_attr_dict=edge_attr_dict,
            num_nodes=k.size(0),
        )

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr)

        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        a_dict = self.out_lin(
            {
                key: F.gelu(value) if value is not None else value
                for key, value in out_dict.items()
            }
        )

        for node_type in out_dict:
            node_out = a_dict[node_type]
            if node_out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                node_out = alpha * node_out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = node_out

        return out_dict


class GitHubHeteroHGT(nn.Module):
    """Relational deep learning model with ablation switches.

    1. Multi-table NodeMLPEncoders -> uniform latent dimension.
    2. Multi-layer HGT message passing across the graph's relation types.
    3. Optional residual fusion of the structural GNN embedding with the raw
       tabular user block (``use_residual_fusion``).
    4. Six task-specific prediction heads.
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_dims: dict[str, int],
        hidden_dim: int,
        latent_dim: int,
        head_hidden_dim: int,
        num_classes: int,
        num_heads: int,
        dropout: float,
        num_gnn_layers: int = 2,
        use_residual_fusion: bool = True,
        use_temporal_decay: bool = True,
    ) -> None:
        super().__init__()

        node_types, _edge_types = metadata
        self.latent_dim = latent_dim
        self.user_raw_dim = in_dims["Users"]
        self.use_residual_fusion = use_residual_fusion
        self.use_temporal_decay = use_temporal_decay

        self.encoders = nn.ModuleDict(
            {
                node_type: NodeMLPEncoder(
                    in_dim=in_dims[node_type],
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    dropout=dropout,
                )
                for node_type in node_types
            }
        )

        self.gnn_layers = nn.ModuleList(
            [
                TemporalHGTConv(
                    in_channels=latent_dim,
                    out_channels=latent_dim,
                    metadata=metadata,
                    heads=num_heads,
                )
                for _ in range(num_gnn_layers)
            ]
        )

        self.layer_norms = nn.ModuleList(
            [
                nn.ModuleDict({node_type: nn.LayerNorm(latent_dim) for node_type in node_types})
                for _ in range(num_gnn_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        # Ablation switch: the raw user block is only concatenated when the
        # residual fusion is enabled.
        combined_dim = latent_dim + self.user_raw_dim if use_residual_fusion else latent_dim

        self.prediction_heads = nn.ModuleList(
            [
                PredictionHead(in_dim=combined_dim, hidden_dim=head_hidden_dim)
                for _ in range(num_classes)
            ]
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Return raw logits of shape ``(num_users, num_classes)``."""
        x_dict: dict[str, torch.Tensor] = {}
        for node_type, x in data.x_dict.items():
            encoder = cast(NodeMLPEncoder, self.encoders[node_type])
            x_dict[node_type] = encoder(x)

        edge_weight_dict = data.edge_weight_dict if self.use_temporal_decay else None

        for conv, ln_dict in zip(self.gnn_layers, self.layer_norms, strict=True):
            conv_typed = cast(TemporalHGTConv, conv)
            out_dict = conv_typed(x_dict, data.edge_index_dict, edge_weight_dict)
            for node_type in x_dict:
                layer_norms = cast(nn.ModuleDict, ln_dict)
                layer_norm = cast(nn.LayerNorm, layer_norms[node_type])
                residual = out_dict[node_type]
                if residual is None:
                    raise RuntimeError(f"HGT produced no embedding for {node_type}")
                x_dict[node_type] = layer_norm(
                    F.mish(residual) + x_dict[node_type]
                )
                x_dict[node_type] = self.dropout(x_dict[node_type])

        user_gnn_emb = x_dict["Users"]

        if self.use_residual_fusion:
            raw_user_features = data["Users"].x
            user_combined = torch.cat([user_gnn_emb, raw_user_features], dim=-1)
        else:
            user_combined = user_gnn_emb

        logits = torch.cat(
            [
                cast(PredictionHead, head)(user_combined)
                for head in self.prediction_heads
            ],
            dim=-1,
        )
        return logits

    @torch.no_grad()
    def predict_proba(self, data: HeteroData) -> torch.Tensor:
        """Return probabilities in [0, 1] for inference and threshold tuning."""
        self.eval()
        logits = self.forward(data)
        return torch.sigmoid(logits)
