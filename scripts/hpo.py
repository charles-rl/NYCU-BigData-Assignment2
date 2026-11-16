"""Hyperparameter optimisation for the HGT trainer, argparse-only.

One Optuna trial == one subprocess call to ``scripts/run_training.py``. No
existing script is modified: every knob is a flag that already exists. The
objective is the held-out ``best_val_score`` the trainer prints in its final
summary -- the TA metric ``(macro-F1 + micro-F1) / 2`` with CMA-ES thresholds
fitted on training rows only (src/models/train.py:183-219).

Examples
--------
    # 45 TPE trials, 6 at a time, on the primary validation week
    python scripts/hpo.py run --n-trials 45 --n-jobs 6

    # build the command lines and sample configs, but do not train
    python scripts/hpo.py run --n-trials 3 --dry-run

    # confirm the top 3 of that study on the second safe week
    python scripts/hpo.py confirm --top 3 --override val_cutoff=2026-03-14

    # recent-regime collapse check (5-day window; not 7-day comparable)
    python scripts/hpo.py confirm --top 1 --override val_cutoff=2026-10-01 \\
        --override days=5

    # leaderboard of a finished study
    python scripts/hpo.py report --top 10

Search space and all fixed knobs: ``configs/hpo_space.yaml`` (grammar is
documented at the top of that file).

Artifacts land in ``hpo/<study>/``: ``study.db`` (resumable), ``trials.csv``,
``space.snapshot.yaml``, and one directory per trial holding ``model.pt`` and
``train.log``.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "scripts/hpo.py needs Optuna in the active interpreter:\n"
        "  python -m pip install 'optuna>=4'"
    ) from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("scripts/hpo.py needs PyYAML") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "hpo_space.yaml"
TRAINER = REPO_ROOT / "scripts" / "run_training.py"
TRAIN_MODULE = "scripts.run_training"
HPO_ROOT = REPO_ROOT / "hpo"

# "Best validation score: 65.81%" -- scripts/run_training.py:701
# (65.81 = best CLEAN trial, hpo/hgt-hpo/trials.csv. Ignore any 65.5-66.0%
# string from the pre-HPO .wandb-runs/ dirs: those were leaked by CMA-ES
# threshold fitting on validation data and are deleted. See docs/HPO.md.)
BEST_RE = re.compile(r"Best validation score:\s*([0-9]+\.[0-9]+)\s*%")


# Trainer subprocesses live right now, keyed by pid, so SIGINT can stop them
# instead of waiting out a ~100-minute trial. See install_stop_handler().
_LIVE: dict[int, Any] = {}
_LIVE_LOCK = threading.Lock()
_STOP = threading.Event()
# Path of the current study's driver.lock, so an interrupt can remove it.
_LOCK: Path | None = None


def terminate_live(grace: float = 20.0) -> int:
    """Kill every in-flight trainer, giving up on stragglers after ``grace``."""
    with _LIVE_LOCK:
        procs = list(_LIVE.values())
    for proc in procs:
        # Each trainer runs in its own session, so kill the whole group in
        # case torch spawned dataloader workers underneath it.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
    deadline = time.time() + grace
    for proc in procs:
        try:
            proc.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:  # best effort, then SIGKILL
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
    return len(procs)


def install_stop_handler(storage: str) -> None:
    """Make Ctrl-C exit in seconds instead of at the end of a trial.

    ``ThreadPoolExecutor.__exit__`` calls ``shutdown(wait=True)``, so a bare
    KeyboardInterrupt stays blocked until every running trial finishes -- and
    one 310-epoch trial is ~100 minutes. Killing the children first makes
    those futures fail immediately, so the pool drains in seconds.
    """

    def _interrupt(signum: int, frame: Any) -> None:
        del signum, frame
        _STOP.set()
        killed = terminate_live()
        print(f"\ninterrupted; stopped {killed} in-flight trial(s).")
        print(f"  finished trials are already committed to {storage}")
        print("  resume with the SAME command; see docs/HPO.md "
              "('Stopping and resuming')")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _interrupt)


def fail_stale_trials(study: Any) -> int:
    """Mark trials left RUNNING by a killed process as FAIL.

    Optuna's TPE treats RUNNING trials as constant-liar observations, so a
    stale row would keep biasing the surrogate towards a configuration that
    was never actually evaluated -- a permanent distortion rather than a
    cosmetic one.
    """
    stale = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.RUNNING
    ]
    for trial in stale:
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    return len(stale)


def _pid_alive(pid: int) -> bool:
    """True if ``pid`` exists and is not a zombie."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours
    return True


def acquire_driver_lock(home: Path) -> Path | None:
    """Take an exclusive driver lock for a study, or return None if held.

    How this actually gets lost: a driver is killed while two trials are
    running, leaving them marked RUNNING. A second driver then starts, calls
    ``fail_stale_trials``, and fails those rows -- but the first driver's
    children are still alive and will later try to ``tell`` the same trial
    numbers, which Optuna rejects. The second driver also mis-reads its own
    worker set. Two drivers on one study is never correct, so refuse it.

    The lock is a file containing the owner's pid; a lock whose pid is dead
    is stale and gets replaced, so killing a run never wedges the study.
    """
    global _LOCK
    path = home / "driver.lock"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            owner = owner_of_driver_lock(home)
            if owner is not None and _pid_alive(owner):
                return None
            # Stale: the owner died without cleaning up. Replace it.
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(str(os.getpid()))
        _LOCK = path
        return path
    return None


def owner_of_driver_lock(home: Path) -> int | None:
    """The pid recorded in the lock file, or None if unreadable/absent."""
    try:
        text = (home / "driver.lock").read_text().strip()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    if not text.isdigit():
        return None
    return int(text)


def release_driver_lock(path: Path | None) -> None:
    global _LOCK
    if path is not None:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        if path == _LOCK:
            _LOCK = None

warnings.filterwarnings("ignore", message=".*group.*experimental.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ConstraintError(RuntimeError):
    """A sampled configuration cannot satisfy a declared constraint."""


# ---------------------------------------------------------------------------
# search-space grammar
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Spec:
    """One searchable knob, parsed from ``space:`` in the YAML file."""

    key: str
    kind: str  # "categorical" | "float" | "int"
    distribution: str  # "cat" | "log" | "linear" | "int"
    low: float = 0.0
    high: float = 0.0
    choices: tuple[Any, ...] = ()


def _bounds(spec: Spec) -> tuple[float, float]:
    """The (low, high) of a numeric spec, validated at construction time."""
    if spec.low > spec.high:
        raise SystemExit(f"space.{spec.key}: low {spec.low} > high {spec.high}")
    return spec.low, spec.high


_PREFIXES: tuple[tuple[str, str], ...] = (
    ("log:", "log"),
    ("linear:", "linear"),
    ("int:", "int"),
    ("cat:", "cat"),
)


def _unwrap(raw: Any) -> tuple[str | None, Any]:
    """Split an explicit ``"log: [1.0e-5, 1.0e-2]"`` prefix off a value."""
    if isinstance(raw, str):
        text = raw.strip()
        for prefix, name in _PREFIXES:
            if text.startswith(prefix):
                return name, yaml.safe_load(text[len(prefix) :].strip())
    return None, raw


def _num(value: Any, kind: str, where: str) -> float:
    """Parse a config number, with a message that names the offending key."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"space.{where}: {value!r} is not a number ({kind} expected)"
        ) from exc


def parse_spec(key: str, raw: Any) -> Spec | None:
    """Turn one raw ``space:`` entry into a :class:`Spec` (None = skip)."""
    if raw is None:
        return None
    explicit, value = _unwrap(raw)
    if not isinstance(value, list) or not value:
        raise SystemExit(
            f"space.{key}: expected a non-empty list, got {value!r}"
        )

    if explicit == "cat":
        return Spec(key, "categorical", "cat", choices=tuple(value))
    if explicit == "int":
        lo = int(_num(value[0], "int", key))
        hi = int(_num(value[-1], "int", key))
        if lo > hi:
            raise SystemExit(f"space.{key}: int range {lo}..{hi} is inverted")
        return Spec(key, "int", "int", low=float(lo), high=float(hi))
    if explicit in {"log", "linear"}:
        lo = _num(min(_num(v, explicit, key) for v in value), explicit, key)
        hi = _num(max(_num(v, explicit, key) for v in value), explicit, key)
        if explicit == "log" and lo <= 0:
            raise SystemExit(f"space.{key}: log scale needs lo > 0, got {lo}")
        return Spec(key, "float", explicit, low=lo, high=hi)

    # No prefix: infer from the values.
    if any(isinstance(v, str) for v in value):
        return Spec(key, "categorical", "cat", choices=tuple(value))
    numbers = [v for v in value if not isinstance(v, bool)]
    if len(numbers) != len(value):
        raise SystemExit(f"space.{key}: booleans are not a search dimension")
    if all(isinstance(v, int) for v in value):
        return Spec(key, "categorical", "cat", choices=tuple(value))
    lo = _num(min(_num(v, "float", key) for v in value), "float", key)
    hi = _num(max(_num(v, "float", key) for v in value), "float", key)
    if lo > 0 and hi / lo >= 10.0:
        return Spec(key, "float", "log", low=lo, high=hi)
    return Spec(key, "float", "linear", low=lo, high=hi)


def candidates(spec: Spec) -> tuple[Any, ...]:
    """Every value the constraint filter is allowed to keep."""
    if spec.kind == "categorical":
        return spec.choices
    if spec.kind == "int":
        lo, hi = _bounds(spec)
        return tuple(range(int(lo), int(hi) + 1))
    return ()

_NUMERIC_BINOPS = (ast.Mod, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)


class _RuleError(ValueError):
    """A constraint rule used syntax the mini-evaluator does not allow."""


def _eval_node(node: ast.AST, params: dict[str, Any]) -> Any:
    """Evaluate one whitelisted AST node. No ``eval``, no builtins."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, params)
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _RuleError("only numeric constants are allowed")
        return value
    if isinstance(node, ast.Name):
        if node.id not in params:
            raise _RuleError(f"unknown name {node.id!r} in rule")
        return params[node.id]
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _NUMERIC_BINOPS):
            raise _RuleError("that operator is not allowed in a rule")
        left = _eval_node(node.left, params)
        right = _eval_node(node.right, params)
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left // right
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise _RuleError("use one comparison per rule")
        left = _eval_node(node.left, params)
        right = _eval_node(node.comparators[0], params)
        op = node.ops[0]
        checks = {
            ast.Eq: lambda: left == right,
            ast.NotEq: lambda: left != right,
            ast.Lt: lambda: left < right,
            ast.LtE: lambda: left <= right,
            ast.Gt: lambda: left > right,
            ast.GtE: lambda: left >= right,
        }
        for kind, check in checks.items():
            if isinstance(op, kind):
                return check()
        raise _RuleError("that comparison is not allowed in a rule")
    raise _RuleError(f"unsupported rule syntax: {type(node).__name__}")


def rule_holds(rule: str, params: dict[str, Any]) -> bool:
    """Evaluate a constraint rule such as ``latent_dim % num_heads == 0``.

    Deliberately not ``eval``: only names found in ``params``, numeric
    constants, ``+ - * // %`` and single comparisons are accepted, so a YAML
    file cannot execute anything. A rule that cannot be judged yet (a name the
    caller has not sampled) passes, which is what the ordering in
    :func:`sample_params` relies on.
    """
    try:
        return bool(_eval_node(ast.parse(rule, mode="eval"), params))
    except (_RuleError, SyntaxError, TypeError, ZeroDivisionError):
        return True


def suggest_value(trial: optuna.Trial, spec: Spec) -> Any:
    """Sample one knob."""
    if spec.kind == "categorical":
        return trial.suggest_categorical(spec.key, list(spec.choices))
    lo, hi = _bounds(spec)
    if spec.kind == "int":
        return trial.suggest_int(spec.key, int(lo), int(hi))
    if spec.distribution == "log":
        return trial.suggest_float(spec.key, lo, hi, log=True)
    return trial.suggest_float(spec.key, lo, hi)


def sample_params(
    trial: optuna.Trial,
    specs: dict[str, Spec],
    constraints: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Sample every knob, filtering constrained keys against their rules."""
    dependent = {
        str(c["dependent"]): c for c in constraints if c.get("dependent")
    }
    params: dict[str, Any] = {}
    for key, spec in specs.items():
        if key not in dependent:
            params[key] = suggest_value(trial, spec)

    for key, con in dependent.items():
        spec = specs.get(key)
        if spec is None:
            continue
        rule = str(con["rule"])
        pool = [
            v for v in candidates(spec) if rule_holds(rule, {**params, key: v})
        ]
        if not pool:
            raise ConstraintError(f"{con['name']}: no legal value for {key}")
        if spec.kind == "categorical":
            params[key] = trial.suggest_categorical(key, pool)
        else:
            lo, hi = int(min(pool)), int(max(pool))
            if list(range(lo, hi + 1)) != [int(v) for v in pool]:
                params[key] = trial.suggest_categorical(key, pool)
            else:
                params[key] = trial.suggest_int(key, lo, hi)

    for con in constraints:
        if not rule_holds(str(con["rule"]), params):
            raise ConstraintError(f"{con['name']}: violated by {params}")
    return params


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    path: Path
    study: dict[str, Any]
    fixed: dict[str, Any]
    specs: dict[str, Spec]
    constraints: list[dict[str, Any]]

    @property
    def name(self) -> str:
        return str(self.study.get("name", "hgt-hpo"))

    @property
    def storage(self) -> str:
        return str(self.study.get("storage", "sqlite:///hpo/study.db"))


def trainer_flags() -> set[str]:
    """Every ``--flag`` scripts/run_training.py actually accepts."""
    text = TRAINER.read_text()
    return set(re.findall(r'add_argument\(\s*"(--[A-Za-z0-9-]+)"', text))


def load_config(path: Path) -> Config:
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    space = raw.get("space") or {}
    fixed = dict(raw.get("fixed") or {})
    constraints = list(raw.get("constraints") or [])

    specs: dict[str, Spec] = {}
    for key, value in space.items():
        spec = parse_spec(str(key), value)
        if spec is not None:
            specs[key] = spec

    # Fail fast on typos: every key must be a real trainer flag.
    known = trainer_flags()
    unknown = [
        key
        for key in list(fixed) + list(space)
        if "--" + key.replace("_", "-") not in known
    ]
    if unknown:
        raise SystemExit(
            "these keys are not flags of scripts/run_training.py: "
            + ", ".join(sorted(unknown))
        )
    for con in constraints:
        dep = con.get("dependent")
        if dep is not None and dep not in specs:
            raise SystemExit(
                f"constraint {con.get('name')}: {dep} is not searched"
            )
        try:
            ast.parse(str(con.get("rule", "")), mode="eval")
        except SyntaxError as exc:
            raise SystemExit(
                f"constraint {con.get('name')}: bad rule {con.get('rule')!r}"
            ) from exc
    return Config(path, dict(raw.get("study") or {}), fixed, specs, constraints)


def parse_overrides(pairs: Iterable[str]) -> dict[str, Any]:
    """``--override val_cutoff=2026-03-14`` -> ``{"val_cutoff": ...}``."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--override needs key=value, got {pair!r}")
        key, _, text = pair.partition("=")
        out[key.strip()] = yaml.safe_load(text.strip())
    return out


# ---------------------------------------------------------------------------
# command lines
# ---------------------------------------------------------------------------
def _rel(path: Path) -> str:
    """Path relative to the repo when possible, so trials.csv stays portable.

    Falls back to the absolute path when HPO_ROOT was pointed outside the
    repo (tests, or a study kept on another volume). Recording the path is
    the point; crashing the driver over a cosmetic prefix is not.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def as_flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def as_cli(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def build_argv(
    fixed: dict[str, Any],
    params: dict[str, Any],
    trial_dir: Path,
    extra: Sequence[str] = (),
) -> list[str]:
    """The exact ``python -m scripts.run_training ...`` for one trial."""
    argv = [sys.executable, "-m", TRAIN_MODULE]
    for block in (fixed, params):
        for key, value in block.items():
            if value is None:
                continue
            argv += [as_flag(key), as_cli(value)]
    argv += ["--checkpoint", str(trial_dir / "model.pt")]
    # Per-trial predictions: without this, parallel trials all overwrite
    # splits/val_best_pred_<date>.csv and the evaluate.py cross-check in the
    # log reports whichever trial wrote last. The objective itself is computed
    # in memory, so only the cross-check printout is affected -- but a garbled
    # cross-check is worse than none.
    argv += ["--val-pred-path", str(trial_dir / "val_pred.csv")]
    argv += ["--run-name", f"{trial_dir.parent.name}-{trial_dir.name}"]
    argv += list(extra)
    return argv


def worker_env(n_jobs: int, threads_per_job: int) -> dict[str, str]:
    """Thread budget per job, so N trainers do not thrash the whole box."""
    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    env["WANDB_SILENT"] = "true"
    # Without this the child's stdout is block-buffered into train.log, so a
    # 58-minute trial shows an empty file until it exits and nobody can watch
    # progress or spot a NaN loss early.
    env["PYTHONUNBUFFERED"] = "1"
    # Graph batches are large and sizes vary per trial; without this the
    # allocator holds fragmented blocks and OOMs while reporting free memory.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if threads_per_job <= 0:
        threads_per_job = max(1, (os.cpu_count() or 8) // max(1, n_jobs))
    env["OMP_NUM_THREADS"] = str(threads_per_job)
    env["MKL_NUM_THREADS"] = str(threads_per_job)
    return env


@dataclass
class Outcome:
    score: float
    seconds: float
    log: Path


def run_trial(
    argv: Sequence[str], trial_dir: Path, env: dict[str, str]
) -> Outcome:
    """Run one trainer subprocess and read the objective off its summary."""
    log_path = trial_dir / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w") as fh:
        fh.write(" ".join(str(a) for a in argv) + "\n\n")
        fh.flush()
        # start_new_session puts the trainer in its own process group so
        # terminate_live() can take down any workers it spawned.
        proc = subprocess.Popen(
            list(argv),
            cwd=str(REPO_ROOT),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with _LIVE_LOCK:
            _LIVE[proc.pid] = proc
        try:
            returncode = proc.wait()
        finally:
            with _LIVE_LOCK:
                _LIVE.pop(proc.pid, None)
    elapsed = time.time() - start
    text = log_path.read_text(errors="replace")
    match = BEST_RE.search(text)
    if returncode != 0 or match is None:
        tail = "\n".join(text.splitlines()[-15:])
        raise RuntimeError(
            f"trainer failed (rc={proc.returncode})"
            f" -- tail of {log_path}:\n{tail}"
        )
    return Outcome(float(match.group(1)) / 100.0, elapsed, log_path)


# ---------------------------------------------------------------------------
# study helpers
# ---------------------------------------------------------------------------
def storage_url(cfg: Config) -> str:
    """Absolute storage URL, with the SQLite parent directory created."""
    storage = cfg.storage
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return storage
    db = Path(storage[len(prefix) :])
    if not db.is_absolute():
        db = REPO_ROOT / db
    db.parent.mkdir(parents=True, exist_ok=True)
    return prefix + str(db)


def make_study(cfg: Config, n_trials_hint: int) -> optuna.Study:
    storage = storage_url(cfg)

    sampler_name = str(cfg.study.get("sampler", "tpe")).lower()
    if sampler_name == "tpe":
        # Univariate by default: with ~45 trials over 8 dims a joint density is
        # not estimable, and multivariate categorical marginals are exploitative
        # (measured: on a pure-noise objective one of three categorical values
        # got 78% of 200 trials under multivariate=True vs 62% univariate).
        startup = int(cfg.study.get("n_startup_trials", 15))
        sampler: Any = TPESampler(
            seed=int(cfg.study.get("seed", 42)),
            multivariate=bool(cfg.study.get("multivariate", False)),
            group=bool(cfg.study.get("group", False)),
            n_startup_trials=startup,
            constant_liar=True,  # correct for parallel workers
        )
    elif sampler_name == "random":
        from optuna.samplers import RandomSampler

        sampler = RandomSampler(seed=int(cfg.study.get("seed", 42)))
    else:
        raise SystemExit(f"unknown sampler {sampler_name!r}; use tpe or random")

    direction = cast(
        Literal["minimize", "maximize"],
        str(cfg.study.get("direction", "maximize")),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=cfg.name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        load_if_exists=True,
    )
    study.set_user_attr("space", {k: v.__dict__ for k, v in cfg.specs.items()})
    study.set_user_attr("fixed", cfg.fixed)
    study.set_user_attr("n_trials_hint", n_trials_hint)

    # Seed the queue with known-good configurations. Only on a fresh study: on
    # resume they are already recorded, and re-enqueueing would duplicate them.
    if not study.trials:
        for params in cfg.study.get("enqueue") or []:
            if isinstance(params, dict) and params:
                check_enqueued(params, cfg.specs)
                study.enqueue_trial(params)
                print(f"  [seed] enqueued {_short(params)}")
    return study


def check_enqueued(params: dict[str, Any], specs: dict[str, Spec]) -> None:
    """A seeded config must stay inside the search space, else Optuna errors."""
    for key, value in params.items():
        spec = specs.get(key)
        if spec is None:
            raise SystemExit(f"study.enqueue: {key!r} is not in `space:`")
        if spec.kind == "categorical" and value not in spec.choices:
            raise SystemExit(
                f"study.enqueue: {key}={value!r} is not one of {spec.choices}"
            )
        if spec.kind in {"int", "float"}:
            lo, hi = _bounds(spec)
            number = _num(value, spec.kind, f"study.enqueue:{key}")
            if not lo <= number <= hi:
                raise SystemExit(
                    f"study.enqueue: {key}={value} outside {lo}..{hi}"
                )


def study_dir(cfg: Config) -> Path:
    path = HPO_ROOT / cfg.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_csv(path: Path, row: dict[str, Any]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def completed(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]


def summarise(study: optuna.Study, top: int) -> str:
    best = sorted(
        completed(study), key=lambda t: t.value or -math.inf, reverse=True
    )
    if not best:
        return "  (no completed trials yet)"
    width = max(len(k) for k in best[0].params) if best[0].params else 8
    lines = []
    for rank, trial in enumerate(best[:top], start=1):
        score = 100.0 * float(trial.value or 0.0)
        lines.append(f"  #{rank}  trial {trial.number:<4d} {score:6.2f}%")
        for key, value in sorted(trial.params.items()):
            rendered = (
                f"{value:.6g}" if isinstance(value, float) else str(value)
            )
            lines.append(f"      {key:<{width}} {rendered}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
def command_run(cfg: Config, args: argparse.Namespace) -> int:
    overrides = parse_overrides(args.override)
    fixed = {**cfg.fixed, **overrides}
    study = make_study(cfg, args.n_trials)
    home = study_dir(cfg)
    csv_path = home / "trials.csv"
    (home / "space.snapshot.yaml").write_text(cfg.path.read_text())

    target = args.n_trials
    env = worker_env(args.n_jobs, args.threads_per_job)
    if args.dry_run:
        print(f"study {cfg.name!r}: {len(completed(study))} completed trials "
              f"on disk, target {target}, {args.n_jobs} parallel, "
              f"{len(cfg.specs)} search dims")
        print(f"  objective   {study.direction.name} best_val_score on "
              f"val_cutoff={fixed.get('val_cutoff')} "
              f"(docs/VALIDATION_DATA.md)")
        print(f"  epochs      {fixed.get('epochs')}  "
              f"schedule {fixed.get('lr_schedule')}")
        print(f"  artifacts   {home}")
        return _dry_run(cfg, fixed, args)

    lock = acquire_driver_lock(home)
    if lock is None:
        other = owner_of_driver_lock(home)
        print(f"another driver is already running on {home} (pid {other}).")
        print("  resuming would let two drivers submit trials to one study,")
        print("  and would fail the trials that one is still training.")
        print("  stop it first, or watch it with: python -m scripts.hpo "
              "report")
        return 1

    install_stop_handler(storage_url(cfg))
    stale = fail_stale_trials(study)
    already = len(completed(study))
    print(
        f"study {cfg.name!r}: {already} completed trials on disk, "
        f"target {target}, {args.n_jobs} parallel, "
        f"{len(cfg.specs)} search dims"
    )
    if stale:
        print(f"  [note] marked {stale} trial(s) left RUNNING by an earlier "
              f"run as FAIL (their work was lost when it stopped)")
    print(f"  objective   {study.direction.name} best_val_score on "
          f"val_cutoff={fixed.get('val_cutoff')} "
          f"(docs/VALIDATION_DATA.md)")
    print(f"  epochs      {fixed.get('epochs')}  "
          f"schedule {fixed.get('lr_schedule')}")
    print(f"  artifacts   {home}")

    todo = target - already
    if todo <= 0:
        print("\nnothing to do; study already has that many trials.")
        print(summarise(study, args.top))
        release_driver_lock(lock)
        return 0

    failures = 0
    violations = 0
    successes = 0
    max_failures = int(cfg.study.get("max_failures", 3))
    started = time.time()
    pending: dict[
        Future[Outcome], tuple[optuna.Trial, dict[str, Any], float]
    ] = {}

    with ThreadPoolExecutor(max_workers=args.n_jobs) as pool:
        # `pending or todo` (not `todo`): once the last batch is submitted todo
        # hits 0 while jobs are still in flight, and exiting then would throw
        # away finished trials that were never `tell`-ed to the study.
        #
        # The budget counts SUCCESSES, not submissions: an OOM'd trial is
        # replaced rather than eaten, so a shared GPU that evicts a trial does
        # not silently shrink the study from 60 completions to 40.
        while pending or successes < todo:
            if _STOP.is_set():
                break
            while (
                successes + len(pending) < todo
                and len(pending) < args.n_jobs
                and failures <= max_failures
            ):
                trial = study.ask()
                try:
                    params = sample_params(trial, cfg.specs, cfg.constraints)
                except ConstraintError as exc:
                    # A dead constraint (no legal value for the dependent key)
                    # would spin here forever, so cap how often we may retry.
                    violations += 1
                    print(f"  [skip] trial {trial.number}: {exc}")
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    if violations > 50:
                        raise SystemExit(
                            f"constraints rejected {violations} samples; check "
                            "`constraints:` and `space:` in the config"
                        ) from exc
                    continue
                trial_dir = home / f"trial-{trial.number:04d}"
                argv = build_argv(fixed, params, trial_dir, args.extra)
                fut = pool.submit(run_trial, argv, trial_dir, env)
                pending[fut] = (trial, params, time.time())
                print(f"  [run ] trial {trial.number:4d} -> {trial_dir.name}"
                      f"  {_short(params)}")

            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                trial, params, queued = pending.pop(fut)
                del queued  # wall-clock bookkeeping only
                try:
                    outcome = fut.result()
                except Exception as exc:  # noqa: BLE001 - keep the study alive
                    failures += 1
                    print(f"  [fail] trial {trial.number}: {exc}")
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    continue
                study.tell(trial, outcome.score)
                successes += 1
                print(
                    f"  [done] trial {trial.number:4d} "
                    f"{100.0 * outcome.score:6.2f}%  "
                    f"{outcome.seconds / 60.0:5.1f} min  "
                    f"(wall {_hm(time.time() - started)})"
                )
                append_csv(
                    csv_path,
                    {
                        "trial": trial.number,
                        "score": f"{outcome.score:.6f}",
                        "minutes": f"{outcome.seconds / 60.0:.2f}",
                        "params": repr(params),
                        "log": _rel(outcome.log),
                    },
                )

    best = completed(study)
    if best:
        best_trial = max(best, key=lambda t: t.value or -math.inf)
        print(f"\nbest so far: trial {best_trial.number} "
              f"{100.0 * float(best_trial.value or 0.0):.2f}%")
        print(summarise(study, args.top))
    release_driver_lock(lock)
    if failures > max_failures:
        print(f"\nstopped after {failures} failed trials.")
        return 1
    return 0


def _short(params: dict[str, Any]) -> str:
    bits = []
    for key in ("lr", "hidden_dim", "latent_dim", "num_heads", "dropout"):
        if key in params:
            value = params[key]
            bits.append(f"{key}={value:.4g}" if isinstance(value, float) else
                        f"{key}={value}")
    return " ".join(bits)


def _hm(seconds: float) -> str:
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def _dry_run(
    cfg: Config, fixed: dict[str, Any], args: argparse.Namespace
) -> int:
    study = optuna.create_study(
        direction="maximize",
        study_name=cfg.name,
        storage=None,
        sampler=TPESampler(seed=int(cfg.study.get("seed", 42))),
    )
    count = max(1, args.n_trials)
    print(f"\n--dry-run: sampling {count} configs, nothing is trained.\n")
    for index in range(count):
        trial = study.ask()
        params = sample_params(trial, cfg.specs, cfg.constraints)
        trial_dir = study_dir(cfg) / f"dry-{index:02d}"
        argv = build_argv(fixed, params, trial_dir, args.extra)
        print(f"  sample {index}: {_short(params)}")
        print("   ", " ".join(str(a) for a in argv), "\n")
    return 0


def command_confirm(cfg: Config, args: argparse.Namespace) -> int:
    from optuna import load_study

    study = load_study(study_name=cfg.name, storage=storage_url(cfg))

    overrides = parse_overrides(args.override)
    fixed = {**cfg.fixed, **overrides}
    ranked = sorted(completed(study), key=lambda t: t.value or -math.inf,
                    reverse=True)
    if not ranked:
        raise SystemExit("no completed trials to confirm")

    picked, seen = [], set()
    for trial in ranked:
        signature = tuple(sorted(trial.params.items()))
        if signature in seen:
            continue
        seen.add(signature)
        picked.append(trial)
        if len(picked) >= args.top:
            break

    tag = f"{fixed.get('val_cutoff')}-{fixed.get('days')}d"
    home = study_dir(cfg) / f"confirm-{tag}"
    env = worker_env(min(args.n_jobs, len(picked)), args.threads_per_job)
    print(f"confirming {len(picked)} trials on "
          f"val_cutoff={fixed.get('val_cutoff')}"
          f" days={fixed.get('days')} epochs={fixed.get('epochs')}")
    print(f"  artifacts {home}")

    jobs: dict[
        Future[Outcome], tuple[int, optuna.trial.FrozenTrial, float]
    ] = {}
    extra = list(args.extra)
    with ThreadPoolExecutor(max_workers=max(1, args.n_jobs)) as pool:
        for rank, trial in enumerate(picked, start=1):
            trial_dir = home / f"rank{rank}-trial{trial.number:04d}"
            argv = build_argv(fixed, dict(trial.params), trial_dir, extra)
            if args.dry_run:
                print(f"  rank {rank}: " + " ".join(str(a) for a in argv))
                continue
            jobs[pool.submit(run_trial, argv, trial_dir, env)] = (
                rank,
                trial,
                float(trial.value or 0.0),
            )
        rows = []
        for fut, (rank, trial, hpo_score) in list(jobs.items()):
            try:
                outcome = fut.result()
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  [fail] rank {rank}: {exc}")
                continue
            print(f"  rank {rank}: hpo {100.0 * hpo_score:6.2f}% -> "
                  f"confirm {100.0 * outcome.score:6.2f}% "
                  f"({100.0 * (outcome.score - hpo_score):+5.2f} pt)")
            rows.append(
                {
                    "rank": rank,
                    "trial": trial.number,
                    "hpo_score": f"{hpo_score:.6f}",
                    "confirm_score": f"{outcome.score:.6f}",
                    "delta": f"{outcome.score - hpo_score:+.6f}",
                    "val_cutoff": fixed.get("val_cutoff"),
                    "days": fixed.get("days"),
                    "params": repr(dict(trial.params)),
                }
            )
    if args.dry_run:
        return 0
    if rows:
        csv_path = home / "confirm.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {csv_path}")
    return 0


def command_report(cfg: Config, args: argparse.Namespace) -> int:
    from optuna import load_study

    try:
        study = load_study(study_name=cfg.name, storage=storage_url(cfg))
    except KeyError as exc:
        raise SystemExit(
            f"no study named {cfg.name!r} in {storage_url(cfg)} -- run "
            "`python -m scripts.hpo run` first"
        ) from exc
    states: dict[str, int] = {}
    for trial in study.trials:
        name = str(trial.state).split(".")[-1]
        states[name] = states.get(name, 0) + 1
    print(f"study {cfg.name!r}: {len(study.trials)} trials {states}")
    done = completed(study)
    if done:
        print(f"  best {100.0 * float(study.best_value):.2f}%  "
              f"(direction {study.direction.name})")
    print(summarise(study, args.top))
    if done:
        print("\n  params of the top trial:")
        for key, value in sorted(study.best_trial.params.items()):
            print(f"    {as_flag(key)} {as_cli(value)}")
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HPO driver for scripts/run_training.py (argparse only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="search-space YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default=None,
                       help="search-space YAML (default: repo default)")
        p.add_argument("--n-jobs", type=int, default=0,
                       help="parallel trainers (0 = study.n_jobs)")
        p.add_argument("--threads-per-job", type=int, default=0,
                       help="OMP threads per trainer (0 = cores / n_jobs)")
        p.add_argument("--override", action="append", default=[],
                       metavar="KEY=VALUE",
                       help="override a `fixed` entry, e.g. epochs=310")
        p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                       help="extra flags passed through to the trainer")
        p.add_argument("--dry-run", action="store_true",
                       help="print the sampled configs / commands, no training")

    run = sub.add_parser("run", help="sample and train new trials")
    run.add_argument("--n-trials", type=int, default=0,
                     help="total completed trials wanted (0 = study.n_trials)")
    run.add_argument("--top", type=int, default=5, help="summary depth")
    shared(run)

    conf = sub.add_parser("confirm", help="re-score top trials on another week")
    conf.add_argument("--top", type=int, default=3, help="how many to re-run")
    shared(conf)

    rep = sub.add_parser("report", help="leaderboard of a finished study")
    rep.add_argument("--config", default=None,
                     help="search-space YAML (default: the repo default)")
    rep.add_argument("--top", type=int, default=10, help="summary depth")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(Path(args.config or DEFAULT_CONFIG))
    if args.command in {"run", "confirm"} and not getattr(args, "n_jobs", 0):
        args.n_jobs = int(cfg.study.get("n_jobs", 4))
    if args.command == "run" and not getattr(args, "n_trials", 0):
        args.n_trials = int(cfg.study.get("n_trials", 20))
    training_now = args.command in {"run", "confirm"}
    if training_now and not args.dry_run and args.n_jobs < 1:
        raise SystemExit("--n-jobs must be >= 1")

    try:
        if args.command == "run":
            return command_run(cfg, args)
        if args.command == "confirm":
            return command_confirm(cfg, args)
        return command_report(cfg, args)
    except KeyboardInterrupt:
        # install_stop_handler already said what happened; just exit cleanly
        # (130 = 128 + SIGINT) instead of dumping a traceback.
        release_driver_lock(_LOCK)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
