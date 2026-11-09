"""Heterogeneous GNN Architecture Skeleton for GitHub User Activity Prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv


class NodeMLPEncoder(nn.Module):
    """Encodes raw heterogeneous node features to a uniform dimension d using Mish and LayerNorm."""

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, dropout: float = 0.1, normalize_latent: bool = True) -> None:
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
    """Task-specific MLP prediction head.
    
    Outputs raw logits for numerically stable training with nn.BCEWithLogitsLoss.
    """

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
    

class GitHubHeteroHGT(nn.Module):
    """Complete Relational Deep Learning architecture:
    1. Multi-table NodeMLPEncoders -> uniform latent dimension d
    2. Multi-layer HGTConv message passing across all 13 relation types
    3. Residual fusion of structural GNN embedding + raw tabular behavioral block
    4. 6 Task-specific prediction heads
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
    ) -> None:
        super().__init__()
        
        node_types, edge_types = metadata
        self.latent_dim = latent_dim
        self.user_raw_dim = in_dims["Users"]

        # 1. Individual MLP encoders for each heterogeneous table
        self.encoders = nn.ModuleDict({
            node_type: NodeMLPEncoder(
                in_dim=in_dims[node_type],
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                dropout=dropout
            )
            for node_type in node_types
        })

        # 2. Stacked Heterogeneous Graph Transformer (HGT) layers
        self.gnn_layers = nn.ModuleList([
            HGTConv(
                in_channels=latent_dim,
                out_channels=latent_dim,
                metadata=metadata,
                heads=num_heads,
            )
            for _ in range(num_gnn_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(latent_dim) for nt in node_types})
            for _ in range(num_gnn_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # 3. Decision 5.5: Fusion dimension (GNN latent embedding + raw user tabular features)
        combined_dim = latent_dim + self.user_raw_dim

        # 4. Decision 5.6: 6 Independent task-specific heads
        self.prediction_heads = nn.ModuleList([
            PredictionHead(in_dim=combined_dim, hidden_dim=head_hidden_dim)
            for _ in range(num_classes)
        ])

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Forward pass returning raw logits of shape (num_users, 6)."""
        x_dict = {}

        # Step A: Encode raw node features into unified latent space
        for node_type, x in data.x_dict.items():
            x_dict[node_type] = self.encoders[node_type](x)

        # Step B: Relational message passing via HGT
        for conv, ln_dict in zip(self.gnn_layers, self.layer_norms):
            out_dict = conv(x_dict, data.edge_index_dict)
            for node_type in x_dict:
                # Residual connection within GNN layer + LayerNorm
                x_dict[node_type] = ln_dict[node_type](F.mish(out_dict[node_type]) + x_dict[node_type])
                x_dict[node_type] = self.dropout(x_dict[node_type])

        # Step C: Extract user latent representations
        user_gnn_emb = x_dict["Users"]

        # Step D: Concatenate GNN representation with raw tabular recency block
        raw_user_features = data["Users"].x
        user_combined = torch.cat([user_gnn_emb, raw_user_features], dim=-1)

        # Step E: Compute 6 activity logits
        logits = torch.cat([head(user_combined) for head in self.prediction_heads], dim=-1)
        return logits

    @torch.no_grad()
    def predict_proba(self, data: HeteroData) -> torch.Tensor:
        """Returns calibrated probabilities in [0, 1] for inference and threshold tuning."""
        self.eval()
        logits = self.forward(data)
        return torch.sigmoid(logits)
