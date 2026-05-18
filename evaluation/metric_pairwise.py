import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, f1_score


LABEL_TO_IDX = {0: 0, 0.5: 1, 1: 2}
FLIP         = {0: 1, 0.5: 0.5, 1: 0}
MODES        = ["raw", "debiased", "p2p"]


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def unswap_predictions(data):
    result = []
    for row in data:
        row = row.copy()
        row["prediction"] = FLIP[row["prediction"]]
        result.append(row)
    return result


def aggregate_prediction(orig: float, flipped: float) -> float:
    LABEL_VALUES = [0, 0.5, 1]
    avg = (orig + flipped) / 2
    return min(LABEL_VALUES, key=lambda v: abs(v - avg))


def build_debiased(pairwise_data, swapped_data, silent=False):
    merge_keys = ("id", "criteria", "sub_id_1", "sub_id_2")

    swapped_flipped = unswap_predictions(swapped_data)

    # Index swapped by key for lookup
    swapped_index = {
        tuple(row[k] for k in merge_keys): row["prediction"]
        for row in swapped_flipped
    }

    result = []
    n_inconsistent = 0

    for row in pairwise_data:
        key = tuple(row[k] for k in merge_keys)
        orig    = row["prediction"]
        flipped = swapped_index.get(key)

        if flipped is None:
            # No swapped counterpart — keep original
            result.append(row.copy())
            continue

        if orig != flipped:
            n_inconsistent += 1

        agg = aggregate_prediction(orig, flipped)
        new_row = row.copy()
        new_row["prediction"] = agg
        result.append(new_row)

    bias_rate = n_inconsistent / len(result) if result else 0.0
    if not silent:
        print(f"  Positional bias rate: {bias_rate:.2%} ({n_inconsistent}/{len(result)} inconsistent)")

    return result


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

LABEL_NAMES = {0: "worse", 0.5: "tie", 1: "better"}

def print_distribution(pairwise_data, swapped_data, p2p_data):
    """
    Print prediction distribution of GT, raw, debiased, and p2p side by side.
    Only computed once per model, not per mode.
    """
    def dist(values):
        total = len(values)
        counts = pd.Series(values).value_counts()
        return {v: counts.get(v, 0) / total for v in [0, 0.5, 1]}

    gt_vals   = [row["label"]      for row in pairwise_data]
    raw_vals  = [row["prediction"] for row in pairwise_data]
    deb_data  = build_debiased(pairwise_data, swapped_data, silent=True) if swapped_data else []
    deb_vals  = [row["prediction"] for row in deb_data] if deb_data else []
    p2p_vals  = [row["prediction"] for row in p2p_data] if p2p_data else []

    gt_d   = dist(gt_vals)
    raw_d  = dist(raw_vals)
    deb_d  = dist(deb_vals)  if deb_vals  else {}
    p2p_d  = dist(p2p_vals)  if p2p_vals  else {}

    print("\n  -- Distribution --")
    header = f"  {'label':<12} {'GT':>8} {'raw':>8} {'debiased':>10} {'p2p':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for v in [0, 0.5, 1]:
        name = LABEL_NAMES[v]
        gt_pct   = f"{gt_d.get(v, 0):.1%}"
        raw_pct  = f"{raw_d.get(v, 0):.1%}"
        deb_pct  = f"{deb_d.get(v, 0):.1%}" if deb_d  else "  n/a"
        p2p_pct  = f"{p2p_d.get(v, 0):.1%}" if p2p_d  else "  n/a"
        print(f"  {name:<12} {gt_pct:>8} {raw_pct:>8} {deb_pct:>10} {p2p_pct:>8}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true_raw, y_pred_raw):
    y_true_idx = np.array([LABEL_TO_IDX[v] for v in y_true_raw])
    y_pred_idx = np.array([LABEL_TO_IDX[v] for v in y_pred_raw])

    qwk = cohen_kappa_score(y_true_idx, y_pred_idx, weights="quadratic")
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    f1_macro = f1_score(y_true_idx, y_pred_idx, average="macro")

    return {
        "n_samples": len(y_true_raw),
        "qwk":       round(qwk, 4),
        "mae":       round(mae, 4),
        "f1_macro":  round(f1_macro, 4),
    }


def evaluate(data):
    """Overall + per-criteria metrics."""
    df = pd.DataFrame(data)
    rows = []

    m = compute_metrics(df["label"].values, df["prediction"].values)
    rows.append({"criteria": "ALL", **m})

    for criteria, group in df.groupby("criteria"):
        m = compute_metrics(group["label"].values, group["prediction"].values)
        rows.append({"criteria": criteria, **m})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-mode runner
# ---------------------------------------------------------------------------

def run_mode(mode, pairwise_path, swapped_path,p2p_path):
    print(f"\n  -- Mode: {mode} --")

    if mode == "raw":
        data = load_jsonl(pairwise_path)

    elif mode == "debiased":
        data = build_debiased(
            load_jsonl(pairwise_path),
            load_jsonl(swapped_path),
        )

    elif mode == "p2p":
        data = load_jsonl(p2p_path)

    results = evaluate(data)
    print(results[["criteria", "n_samples", "qwk", "mae", "f1_macro"]].to_string(index=False))
    return results


def run_model(pairwise_path, swapped_path, p2p_path, modes):
    print("\n" + "=" * 58)
    print(f"Model: {os.path.basename(pairwise_path)}")
    print("=" * 58)

    pairwise_data = load_jsonl(pairwise_path)
    swapped_data  = load_jsonl(swapped_path) if os.path.exists(swapped_path) else []
    p2p_data      = load_jsonl(p2p_path)     if os.path.exists(p2p_path)     else []

    print_distribution(pairwise_data, swapped_data, p2p_data)

    all_results = {}
    for mode in modes:
        result = run_mode(mode, pairwise_path, swapped_path, p2p_path)
        all_results[mode] = result

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate pairwise inference methods.")
    parser.add_argument("--pairwise_dir", required=True,
                        help="Directory with all *_pairwise.jsonl, *_pairwise_swapped.jsonl, "
                             "and *_pairwise_from_pointwise.jsonl files")
    parser.add_argument("--mode",         required=True,
                        choices=MODES + ["all"],
                        help="raw | debiased | p2p | all")
    args = parser.parse_args()

    modes = MODES if args.mode == "all" else [args.mode]

    for f in sorted(os.listdir(args.pairwise_dir)):
        if not f.endswith("_pairwise.jsonl"):
            continue

        pairwise_path = os.path.join(args.pairwise_dir, f)
        swapped_path  = pairwise_path.replace("_pairwise.jsonl", "_pairwise_swapped.jsonl")
        p2p_path      = pairwise_path.replace("_pairwise.jsonl", "_pairwise_from_pointwise.jsonl")

        missing = []
        if "debiased" in modes and not os.path.exists(swapped_path):
            missing.append(swapped_path)
        if "pointwise_to_pairwise" in modes and not os.path.exists(p2p_path):
            missing.append(p2p_path)
        if missing:
            print(f"\n[SKIP] Missing files for {f}:\n  " + "\n  ".join(missing))
            continue

        run_model(pairwise_path, swapped_path, p2p_path, modes)


if __name__ == "__main__":
    main()
