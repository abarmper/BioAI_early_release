"""
Run experiments from experiments/runs.yaml by ID, one fold at a time.

Usage:
    python run_experiment.py --id <id>               # run next unfinished fold
    python run_experiment.py --id <id> --fold <N>    # run specific fold
    python run_experiment.py --id <id> --dry-run     # print command only
    python run_experiment.py --id <id> --validate    # validate all done folds
    python run_experiment.py --id <id> --validate --fold <N>  # validate specific fold
    python run_experiment.py --list [--group <name>] # list experiments
    python run_experiment.py --status                # queue summary

After each fold, the runner automatically reads training_info.json and the
validation summary.json produced by the training pipeline and saves:
  - runs.yaml:                  results.folds[N].best_ema_dice  (best EMA Dice during training)
  - runs.yaml:                  results.folds[N].dice  (foreground mean Dice from full validation)
  - runs.yaml:                  results.mean_dice      (mean across done folds)
  - runs.yaml:                  results.validation_note  (set when --validate re-runs full validation)
  - experiments/results/<id>.json:  full per-fold metrics (dice, iou,
                                    sensitivity, precision, hd95, surface_dice)
"""

import argparse
import ctypes
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import yaml


def _set_pdeathsig():
    """Ask the kernel to SIGKILL this process when its parent dies (Linux only)."""
    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass


_DEFAULT_RUNS_FILE = Path(__file__).parent / "experiments" / "runs.yaml"
# Resolved at import time from BIOAI_RUNS_FILE env var; can be re-bound by main()
# from --runs-file. All other references in this module read this global.
RUNS_FILE = Path(os.environ.get("BIOAI_RUNS_FILE", _DEFAULT_RUNS_FILE)).resolve()
RESULTS_DIR = Path(__file__).parent / "experiments" / "results"


# ── YAML loading ──────────────────────────────────────────────────────────────

def load_runs() -> list[dict]:
    with open(RUNS_FILE) as f:
        groups = yaml.safe_load(f)
    runs = []
    for group in groups:
        for run in group["runs"]:
            run["_group"] = group["group"]
            runs.append(run)
    return runs


def find_run(runs: list[dict], run_id: str) -> dict | None:
    for run in runs:
        if run["id"] == run_id:
            return run
    return None


# ── Fold status helpers ───────────────────────────────────────────────────────

def fold_statuses(run: dict) -> dict[int, str]:
    """Return {fold: status} for all folds_intended."""
    folds = run.get("results", {}).get("folds", {})
    return {f: (folds.get(f, {}) or {}).get("status", "planned")
            for f in run["folds_intended"]}


def experiment_status(run: dict) -> str:
    """Derive experiment-level status from fold statuses."""
    statuses = list(fold_statuses(run).values())
    if all(s == "done" for s in statuses):
        return "done"
    if any(s == "done" for s in statuses):
        return "partial"
    # No folds done yet — pick the most actionable single-word status
    if any(s == "running" for s in statuses):
        return "running"   # interrupted, no completed folds
    if any(s == "failed" for s in statuses):
        return "failed"
    return "planned"


def next_fold(run: dict) -> int | None:
    """Return the lowest-numbered fold that is planned or resumable (running/failed)."""
    for fold, status in sorted(fold_statuses(run).items()):
        if status in ("planned", "running", "failed"):
            return fold
    return None


# ── YAML in-place updates (preserves comments) ───────────────────────────────

def _find_run_bounds(lines: list[str], run_id: str) -> tuple[int, int]:
    """Return (start, end) line indices for the experiment block."""
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^\s+- id: {re.escape(run_id)}\s*$", line):
            start = i
            break
    if start is None:
        raise ValueError(f"Experiment '{run_id}' not found in {RUNS_FILE}")

    indent = len(lines[start]) - len(lines[start].lstrip())
    for i in range(start + 1, len(lines)):
        stripped = lines[i].lstrip()
        if stripped.startswith("- id:") and (len(lines[i]) - len(stripped)) <= indent:
            return start, i
    return start, len(lines)


def update_experiment_status(run_id: str, status: str) -> None:
    text = RUNS_FILE.read_text()
    lines = text.splitlines(keepends=True)
    start, end = _find_run_bounds(lines, run_id)

    for i in range(start, end):
        if re.match(r"^\s+status:\s+\S+\s*$", lines[i]):
            indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
            old = lines[i].strip()
            lines[i] = f"{indent}status: {status}\n"
            print(f"  [{run_id}] experiment status: {old.split(':', 1)[1].strip()} → {status}")
            break

    RUNS_FILE.write_text("".join(lines))


def _fold_line(indent: str, fold: int, inner: dict) -> str:
    return (
        f"{indent}{fold}: {{status: {_yv(inner['status'])}, "
        f"dice: {_yv(inner.get('dice'))}, "
        f"best_ema_dice: {_yv(inner.get('best_ema_dice'))}, "
        f"wandb_run: {_yv(inner.get('wandb_run'))}}}\n"
    )


def update_fold_status(run_id: str, fold: int, status: str) -> None:
    """Update results.folds[fold].status in the YAML file."""
    text = RUNS_FILE.read_text()
    lines = text.splitlines(keepends=True)
    start, end = _find_run_bounds(lines, run_id)

    fold_re = re.compile(rf"^(\s+){fold}: \{{(.*?)\}}\s*$")
    for i in range(start, end):
        m = fold_re.match(lines[i])
        if m:
            indent = m.group(1)
            inner = yaml.safe_load("{" + m.group(2) + "}")
            inner["status"] = status
            lines[i] = _fold_line(indent, fold, inner)
            print(f"  [{run_id}] fold {fold} status → {status}")
            break

    RUNS_FILE.write_text("".join(lines))


def _yv(v) -> str:
    """Format a Python value as a YAML inline scalar."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return v
    return str(v)


# ── Validation results ───────────────────────────────────────────────────────

def get_fold_dir(run: dict, fold: int) -> Path:
    data = run["hydra_groups"]["data"]
    model = run["hydra_groups"]["model"]
    configuration = run["overrides"]["configuration"]
    experiment_name = run["overrides"]["experiment_name"]
    return (
        Path("experiments")
        / data
        / f"{model}_{configuration}"
        / experiment_name
        / f"fold_{fold}"
    )


def get_training_info_path(run: dict, fold: int) -> Path:
    """Derive the training_info.json path written by experiment.py after training."""
    return get_fold_dir(run, fold) / "training_info.json"


def get_summary_path(run: dict, fold: int) -> Path:
    """Derive the validation summary.json path from a run's config."""
    return get_fold_dir(run, fold) / "validation" / "summary.json"


_METRIC_KEYS = [
    ("dice",          "foreground_mean_dice"),
    ("iou",           "foreground_mean_iou"),
    ("sensitivity",   "foreground_mean_sensitivity"),
    ("precision",     "foreground_mean_precision"),
    ("hd95",          "foreground_mean_hd95"),
    ("surface_dice",  "foreground_mean_surface_dice"),
]

_PER_CLASS_KEY_MAP = [
    ("dice",              "Dice"),
    ("dice_std",          "Dice_std"),
    ("iou",               "IoU"),
    ("iou_std",           "IoU_std"),
    ("sensitivity",       "Sensitivity"),
    ("sensitivity_std",   "Sensitivity_std"),
    ("precision",         "Precision"),
    ("precision_std",     "Precision_std"),
    ("hd95",              "HD95"),
    ("surface_dice",      "SurfaceDice"),
    ("surface_dice_std",  "SurfaceDice_std"),
]


def read_fold_metrics(summary_path: Path) -> dict | None:
    """Read foreground and per-class metrics from a validation summary.json."""
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    metrics = {
        short: summary[key]
        for short, key in _METRIC_KEYS
        if key in summary
    }
    if "per_class" in summary:
        metrics["per_class"] = {
            cls: {
                short: cls_data[src_key]
                for short, src_key in _PER_CLASS_KEY_MAP
                if src_key in cls_data
            }
            for cls, cls_data in summary["per_class"].items()
        }
    return metrics


def save_results_json(
    run_id: str,
    fold: int,
    metrics: dict,
    summary_path: Path,
    num_params: int | None = None,
) -> None:
    """Append fold metrics to experiments/results/<run_id>.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"{run_id}.json"

    data: dict = {}
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)

    data.setdefault("id", run_id)
    data.setdefault("folds", {})

    if num_params is not None:
        data["num_params"] = num_params

    fold_data = {k: v for k, v in metrics.items() if k != "per_class"}
    fold_data["summary_path"] = str(summary_path)
    if "per_class" in metrics:
        fold_data["per_class"] = metrics["per_class"]
    data["folds"][str(fold)] = fold_data

    # Recompute mean_dice across all recorded folds
    dices = [v["dice"] for v in data["folds"].values() if "dice" in v and v["dice"] is not None]
    data["mean_dice"] = round(sum(dices) / len(dices), 6) if dices else None

    with open(results_file, "w") as f:
        json.dump(data, f, indent=2)


def update_fold_dice(run_id: str, fold: int, dice: float) -> None:
    """Write dice value into results.folds[fold].dice in the YAML file."""
    text = RUNS_FILE.read_text()
    lines = text.splitlines(keepends=True)
    start, end = _find_run_bounds(lines, run_id)

    fold_re = re.compile(rf"^(\s+){fold}: \{{(.*?)\}}\s*$")
    for i in range(start, end):
        m = fold_re.match(lines[i])
        if m:
            indent = m.group(1)
            inner = yaml.safe_load("{" + m.group(2) + "}")
            inner["dice"] = round(dice, 4)
            lines[i] = _fold_line(indent, fold, inner)
            break

    RUNS_FILE.write_text("".join(lines))


def update_fold_best_ema_dice(run_id: str, fold: int, best_ema_dice: float) -> None:
    """Write best_ema_dice into results.folds[fold].best_ema_dice in the YAML file."""
    text = RUNS_FILE.read_text()
    lines = text.splitlines(keepends=True)
    start, end = _find_run_bounds(lines, run_id)

    fold_re = re.compile(rf"^(\s+){fold}: \{{(.*?)\}}\s*$")
    for i in range(start, end):
        m = fold_re.match(lines[i])
        if m:
            indent = m.group(1)
            inner = yaml.safe_load("{" + m.group(2) + "}")
            inner["best_ema_dice"] = round(best_ema_dice, 4)
            lines[i] = _fold_line(indent, fold, inner)
            print(f"  [{run_id}] fold {fold} best_ema_dice → {round(best_ema_dice, 4)}")
            break

    RUNS_FILE.write_text("".join(lines))


def update_num_params(run_id: str, num_params: int) -> None:
    """Write num_params into results.num_params in the YAML file (no-op if already set)."""
    text = RUNS_FILE.read_text()
    lines = text.splitlines(keepends=True)
    start, end = _find_run_bounds(lines, run_id)

    params_re = re.compile(r"^(\s+num_params:)\s+.*$")
    for i in range(start, end):
        m = params_re.match(lines[i])
        if m:
            lines[i] = f"{m.group(1)} {num_params}\n"
            print(f"  [{run_id}] num_params → {num_params:,}")
            break

    RUNS_FILE.write_text("".join(lines))


def update_mean_dice(run_id: str, mean_dice: float) -> None:
    """Write mean_dice into results.mean_dice in the YAML file."""
    text = RUNS_FILE.read_text()
    lines = text.splitlines(keepends=True)
    start, end = _find_run_bounds(lines, run_id)

    mean_re = re.compile(r"^(\s+mean_dice:)\s+.*$")
    for i in range(start, end):
        m = mean_re.match(lines[i])
        if m:
            lines[i] = f"{m.group(1)} {round(mean_dice, 4)}\n"
            print(f"  [{run_id}] mean_dice → {round(mean_dice, 4)}")
            break

    RUNS_FILE.write_text("".join(lines))


def update_validation_note(run_id: str, note: str) -> None:
    """Write/update results.validation_note in the YAML file (inserted after mean_dice)."""
    text = RUNS_FILE.read_text()
    lines = text.splitlines(keepends=True)
    start, end = _find_run_bounds(lines, run_id)

    note_re = re.compile(r"^(\s+validation_note:)\s*.*$")
    mean_re = re.compile(r"^\s+mean_dice:\s+.*$")

    note_idx = None
    mean_idx = None
    for i in range(start, end):
        if note_re.match(lines[i]):
            note_idx = i
        if mean_re.match(lines[i]):
            mean_idx = i

    if note_idx is not None:
        indent = lines[note_idx][: len(lines[note_idx]) - len(lines[note_idx].lstrip())]
        lines[note_idx] = f"{indent}validation_note: {note}\n"
    elif mean_idx is not None:
        indent = lines[mean_idx][: len(lines[mean_idx]) - len(lines[mean_idx].lstrip())]
        lines.insert(mean_idx + 1, f"{indent}validation_note: {note}\n")
    else:
        print(f"  [{run_id}] WARNING: could not find mean_dice line to insert validation_note")
        return

    RUNS_FILE.write_text("".join(lines))
    print(f"  [{run_id}] validation_note → {note}")


# ── Dependency checking ───────────────────────────────────────────────────────

def check_dependencies(runs: list[dict], run: dict) -> list[str]:
    blocking = []
    for dep_id in run.get("depends_on", []):
        dep = find_run(runs, dep_id)
        if dep is None:
            blocking.append(f"{dep_id} (NOT FOUND)")
        elif experiment_status(dep) != "done":
            blocking.append(f"{dep_id} (status: {experiment_status(dep)})")
    return blocking


# ── Build command for a specific fold ────────────────────────────────────────

def build_cmd(run: dict, fold: int) -> str:
    base = " ".join(run["cmd"].split())          # collapse whitespace/newlines
    base = re.sub(r"\bfold=\d+\b", "", base)     # strip any existing fold=N
    base = re.sub(r"\brun_all_folds=\w+\b", "", base)  # strip run_all_folds
    base = re.sub(r"\s{2,}", " ", base).strip()  # clean up extra spaces
    group_tag = run.get("_group", "")
    return f"{base} fold={fold} \"logging.wandb_logging.tags=[{group_tag}]\""


def build_validate_cmd(run: dict, fold: int) -> str:
    base = " ".join(run["cmd"].split())
    base = re.sub(r"\bcommand=\w+\b", "command=validate", base)
    base = re.sub(r"\bfold=\d+\b", "", base)
    base = re.sub(r"\brun_all_folds=\w+\b", "", base)
    base = re.sub(r"\s{2,}", " ", base).strip()
    return f"{base} fold={fold}"


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_validate(runs: list[dict], run_id: str, fold: int | None, dry_run: bool) -> None:
    run = find_run(runs, run_id)
    if run is None:
        print(f"ERROR: experiment '{run_id}' not found.")
        sys.exit(1)

    statuses = fold_statuses(run)
    done_folds = sorted(f for f, s in statuses.items() if s == "done")

    if fold is not None:
        if fold not in run["folds_intended"]:
            print(f"ERROR: fold {fold} is not in folds_intended {run['folds_intended']} for '{run_id}'.")
            sys.exit(1)
        folds_to_validate = [fold]
        all_done_validation = False
    else:
        if not done_folds:
            print(f"No folds with status 'done' found for '{run_id}'. Nothing to validate.")
            sys.exit(0)
        folds_to_validate = done_folds
        all_done_validation = all(s == "done" for s in statuses.values())

    for f in folds_to_validate:
        cmd = build_validate_cmd(run, f)

        print(f"\nExperiment : {run['id']}")
        print(f"Group      : {run['_group']}")
        print(f"Fold       : {f}")
        print(f"Description: {run.get('description', '').strip()}")
        print(f"\nCommand:\n  {cmd}\n")

        if dry_run:
            print("(dry-run — not executing)")
            continue

        result = subprocess.run(cmd, shell=True, check=False, preexec_fn=_set_pdeathsig)
        if result.returncode == 0:
            summary_path = get_summary_path(run, f)
            metrics = read_fold_metrics(summary_path)
            if metrics:
                training_info_path = get_training_info_path(run, f)
                val_num_params = None
                if training_info_path.exists():
                    with open(training_info_path) as _tf:
                        val_num_params = json.load(_tf).get("num_params")
                update_fold_dice(run_id, f, metrics["dice"])
                save_results_json(run_id, f, metrics, summary_path, num_params=val_num_params)
                print(f"  [{run_id}] fold {f} dice={metrics['dice']:.4f}"
                      + (f"  hd95={metrics['hd95']:.2f}" if "hd95" in metrics else ""))
            else:
                print(f"  [{run_id}] summary.json not found at {summary_path}")
                print(f"  → Fill in results.folds.{f}.dice manually in {RUNS_FILE}")
        else:
            print(f"\n✗ {run_id} fold {f} validation failed (exit code {result.returncode}).")

    if dry_run:
        return

    # Recompute mean_dice from freshly updated dice values
    updated_runs = load_runs()
    updated_run = find_run(updated_runs, run_id)
    done_dices = [
        (updated_run.get("results", {}).get("folds", {}).get(f) or {}).get("dice")
        for f in updated_run["folds_intended"]
        if (updated_run.get("results", {}).get("folds", {}).get(f) or {}).get("status") == "done"
        and (updated_run.get("results", {}).get("folds", {}).get(f) or {}).get("dice") is not None
    ]
    if done_dices:
        update_mean_dice(run_id, sum(done_dices) / len(done_dices))

    if all_done_validation:
        update_validation_note(run_id, "full_validation_rerun")
        print(f"\n✓ Full validation complete for '{run_id}'. mean_dice updated with fresh validation results.")
    else:
        validated_str = ", ".join(str(f) for f in folds_to_validate)
        print(f"\n✓ Validation complete for '{run_id}' fold(s) [{validated_str}]. Dice scores updated.")


def cmd_list(runs: list[dict], group_filter: str | None) -> None:
    current_group = None
    for run in runs:
        if group_filter and run["_group"] != group_filter:
            continue
        if run["_group"] != current_group:
            current_group = run["_group"]
            print(f"\n── {current_group} ──")

        exp_status = experiment_status(run)
        folds = fold_statuses(run)
        intended = run["folds_intended"]

        fold_summary = " ".join(
            f"{f}:{'✓' if folds[f] == 'done' else '✗' if folds[f] == 'failed' else '►' if folds[f] == 'running' else '·'}"
            for f in intended
        )
        mean = run.get("results", {}).get("mean_dice")
        mean_str = f"  mean={mean:.4f}" if mean is not None else ""

        print(f"  [{exp_status:8s}]  {run['id']:<52s}  folds [{fold_summary}]{mean_str}")


def cmd_status(runs: list[dict]) -> None:
    counts: dict[str, int] = {}
    for run in runs:
        s = experiment_status(run)
        counts[s] = counts.get(s, 0) + 1
    total = len(runs)

    print(f"\nTotal experiments: {total}")
    for status, n in sorted(counts.items()):
        bar = "█" * n
        print(f"  {status:10s} {n:3d}  {bar}")

    print("\nBlocked (dependencies not done):")
    any_blocked = False
    for run in runs:
        if experiment_status(run) == "planned":
            blocking = check_dependencies(runs, run)
            if blocking:
                any_blocked = True
                print(f"  {run['id']}")
                for b in blocking:
                    print(f"    ✗ {b}")
    if not any_blocked:
        print("  (none)")

    print("\nResumable / partially done:")
    any_partial = False
    for run in runs:
        es = experiment_status(run)
        if es in ("partial", "running"):
            any_partial = True
            folds = fold_statuses(run)
            fold_str = ", ".join(
                f"fold {f}={s}" for f, s in sorted(folds.items()) if s != "done"
            )
            tag = " [interrupted]" if es == "running" else ""
            print(f"  {run['id']}{tag}  — remaining: {fold_str}")
    if not any_partial:
        print("  (none)")

    print("\nReady to run (planned, no blocking deps):")
    any_ready = False
    for run in runs:
        status = experiment_status(run)
        if status in ("planned", "partial", "running"):
            blocking = check_dependencies(runs, run)
            if not blocking:
                nf = next_fold(run)
                if nf is not None:
                    any_ready = True
                    tag = " [resume]" if status == "running" else ""
                    print(f"  {run['id']}{tag}  →  fold {nf}")
    if not any_ready:
        print("  (none)")


def cmd_run(runs: list[dict], run_id: str, fold: int | None, dry_run: bool, force: bool) -> None:
    run = find_run(runs, run_id)
    if run is None:
        print(f"ERROR: experiment '{run_id}' not found.")
        sys.exit(1)

    exp_status = experiment_status(run)

    # Determine which fold to run
    if fold is None:
        fold = next_fold(run)
        if fold is None:
            if exp_status == "done" and not force:
                print(f"All folds done for '{run_id}'. Use --force to re-run.")
                sys.exit(0)
            fold = run["folds_intended"][0]

    if fold not in run["folds_intended"]:
        print(f"ERROR: fold {fold} is not in folds_intended {run['folds_intended']} for '{run_id}'.")
        sys.exit(1)

    folds = fold_statuses(run)
    fold_status = folds.get(fold, "planned")

    if fold_status == "done" and not force:
        print(f"Fold {fold} already done for '{run_id}'. Use --force to re-run.")
        sys.exit(0)

    if force:
        fold_dir = get_fold_dir(run, fold)
        checkpoints = ["checkpoint_latest.pth", "checkpoint_best.pth", "checkpoint_final.pth"]
        deleted = [ckpt for ckpt in checkpoints if (fold_dir / ckpt).exists()]
        for ckpt in deleted:
            (fold_dir / ckpt).unlink()
        if deleted:
            print(f"  [{run_id}] fold {fold} --force: deleted checkpoints: {', '.join(deleted)}")
        else:
            print(f"  [{run_id}] fold {fold} --force: no checkpoints found, training from scratch")
    elif fold_status == "running":
        print(f"  [{run_id}] fold {fold} was interrupted — resuming from checkpoint_latest.pth")
    elif fold_status == "failed":
        print(f"  [{run_id}] fold {fold} previously failed — retrying")

    blocking = check_dependencies(runs, run)
    if blocking and not force:
        print(f"\nBlocked — dependencies not done:")
        for b in blocking:
            print(f"  ✗ {b}")
        print("\nUse --force to run anyway.")
        sys.exit(1)

    cmd = build_cmd(run, fold)

    print(f"\nExperiment : {run['id']}")
    print(f"Group      : {run['_group']}")
    print(f"Fold       : {fold}  (intended: {run['folds_intended']})")
    print(f"Description: {run.get('description', '').strip()}")
    print(f"\nCommand:\n  {cmd}\n")

    if dry_run:
        print("(dry-run — not executing)")
        return

    update_fold_status(run_id, fold, "running")
    update_experiment_status(run_id, "running")

    try:
        result = subprocess.run(cmd, shell=True, check=False, preexec_fn=_set_pdeathsig)
        if result.returncode == 0:
            update_fold_status(run_id, fold, "done")

            # Auto-read best EMA dice and num_params from training_info.json
            training_info_path = get_training_info_path(run, fold)
            if training_info_path.exists():
                with open(training_info_path) as _f:
                    training_info = json.load(_f)
                best_ema = training_info.get("best_ema_dice")
                if best_ema is not None:
                    update_fold_best_ema_dice(run_id, fold, best_ema)
                num_params = training_info.get("num_params")
                if num_params is not None:
                    update_num_params(run_id, num_params)

            # Auto-read validation results and persist them
            summary_path = get_summary_path(run, fold)
            metrics = read_fold_metrics(summary_path)
            if metrics:
                update_fold_dice(run_id, fold, metrics["dice"])
                save_results_json(run_id, fold, metrics, summary_path, num_params=num_params)
                print(f"  [{run_id}] fold {fold} dice={metrics['dice']:.4f}"
                      + (f"  hd95={metrics['hd95']:.2f}" if "hd95" in metrics else ""))
            else:
                print(f"  [{run_id}] summary.json not found at {summary_path}")
                print(f"  → Fill in results.folds.{fold}.dice manually in {RUNS_FILE}")

            # Recompute mean_dice across all done folds that have a dice value
            updated_runs = load_runs()
            updated_run = find_run(updated_runs, run_id)
            done_dices = [
                (updated_run.get("results", {}).get("folds", {}).get(f) or {}).get("dice")
                for f in updated_run["folds_intended"]
                if (updated_run.get("results", {}).get("folds", {}).get(f) or {}).get("status") == "done"
                and (updated_run.get("results", {}).get("folds", {}).get(f) or {}).get("dice") is not None
            ]
            if done_dices:
                update_mean_dice(run_id, sum(done_dices) / len(done_dices))

            updated_runs = load_runs()
            updated_run = find_run(updated_runs, run_id)
            new_exp_status = experiment_status(updated_run)
            update_experiment_status(run_id, new_exp_status)
            print(f"\n✓ {run_id} fold {fold} done  (experiment: {new_exp_status})")
            print(f"  → Fill in results.folds.{fold}.wandb_run in {RUNS_FILE}")
            nf = next_fold(updated_run)
            if nf is not None:
                print(f"  → Next fold to run: {nf}")
                print(f"     python run_experiment.py --id {run_id} --fold {nf}")
        else:
            update_fold_status(run_id, fold, "failed")
            update_experiment_status(run_id, "failed")
            print(f"\n✗ {run_id} fold {fold} failed (exit code {result.returncode}).")
            sys.exit(result.returncode)
    except KeyboardInterrupt:
        update_fold_status(run_id, fold, "planned")
        update_experiment_status(run_id, experiment_status(find_run(load_runs(), run_id)))
        print(f"\nInterrupted — fold {fold} reset to 'planned'.")
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiments from runs.yaml")
    parser.add_argument("--id", help="Experiment ID to run or validate")
    parser.add_argument("--fold", type=int, help="Fold number (default: next unfinished / all done folds)")
    parser.add_argument("--validate", action="store_true", help="Run validation instead of training")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--status", action="store_true", help="Show queue summary")
    parser.add_argument("--group", help="Filter by group name (use with --list)")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    parser.add_argument("--force", action="store_true", help="Ignore status and dependency checks")
    parser.add_argument(
        "--runs-file",
        help=(
            "Path to the runs YAML to read/write. Defaults to "
            "$BIOAI_RUNS_FILE if set, otherwise experiments/runs.yaml."
        ),
    )
    args = parser.parse_args()

    if args.runs_file is not None:
        global RUNS_FILE
        RUNS_FILE = Path(args.runs_file).resolve()
    if not RUNS_FILE.is_file():
        sys.exit(f"Runs file not found: {RUNS_FILE}")

    runs = load_runs()

    if args.list:
        cmd_list(runs, args.group)
    elif args.status:
        cmd_status(runs)
    elif args.id and args.validate:
        cmd_validate(runs, args.id, fold=args.fold, dry_run=args.dry_run)
    elif args.id:
        cmd_run(runs, args.id, fold=args.fold, dry_run=args.dry_run, force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
