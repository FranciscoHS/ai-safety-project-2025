#!/usr/bin/env python3
# scripts/detect_grokking.py

import os, sys, csv, argparse
import pandas as pd

# Ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.analysis import detect_grokking

CSV_HEADER = [
    "commit","config","seed","d_model","d_mlp",
    "acc_id","acc_ood","acc_pc2_ablated","we_pc12_var","wl_pc12_var",
    "gpu","seconds",
    # new grok fields
    "grok","t_train_99","t_test_95","grok_delay","final_test_acc",
]

def ensure_csv(path: str):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADER).writeheader()

def read_rows(path: str):
    if not os.path.exists(path): return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))

def write_rows(path: str, rows: list[dict]):
    # normalize keys
    for r in rows:
        for k in CSV_HEADER: r.setdefault(k, "")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER); w.writeheader()
        for r in rows: w.writerow(r)

def upsert_row(path: str, key: tuple[str,int], updates: dict):
    ensure_csv(path)
    rows = read_rows(path)
    cfg_key, seed_key = key
    hit = None
    for r in rows:
        if r.get("config","") == cfg_key and str(r.get("seed","")) == str(seed_key):
            hit = r; break
    if hit is None:
        hit = {k:"" for k in CSV_HEADER}
        hit["config"] = cfg_key
        hit["seed"] = str(seed_key)
        rows.append(hit)
    for k,v in updates.items():
        hit[k] = v
    write_rows(path, rows)

def main():
    ap = argparse.ArgumentParser()
    # Identify the run
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--frac_train", type=float, default=0.72)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_mlp", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config_name", default="",
                    help="Row key in experiments.csv. If empty, auto-build from args.")

    # IO
    ap.add_argument("--curves_csv", default="", help="If empty, inferred from config+seed.")
    ap.add_argument("--experiments_csv", default="results/experiments.csv")

    # Detector knobs
    ap.add_argument("--train_thr", type=float, default=0.99)
    ap.add_argument("--test_thr",  type=float, default=0.95)
    ap.add_argument("--min_gap_frac", type=float, default=0.10)
    ap.add_argument("--smooth_window", type=int, default=11)
    args = ap.parse_args()

    config_str = args.config_name or f"N{args.N}_frac{args.frac_train}_d{args.d_model}_h{args.n_heads}_m{args.d_mlp}"
    curves_csv = args.curves_csv or os.path.join("results", "curves", f"{config_str}_seed{args.seed}.csv")

    if not os.path.exists(curves_csv):
        raise FileNotFoundError(f"Curves CSV not found: {curves_csv}. Did you set train_model(..., log_curves=True, curves_csv=...)?")

    df = pd.read_csv(curves_csv)
    info = detect_grokking(
        df,
        train_thr=args.train_thr,
        test_thr=args.test_thr,
        min_gap_frac=args.min_gap_frac,
        smooth_window=args.smooth_window,
        max_steps=float(df["step"].max()),
    )

    updates = {
        "grok": "1" if info["grok"] else "0",
        "t_train_99": "" if info["t_train"] is None else f"{info['t_train']:.0f}",
        "t_test_95":  "" if info["t_test"]  is None else f"{info['t_test']:.0f}",
        "grok_delay": "" if info["delay"]   is None else f"{info['delay']:.0f}",
        "final_test_acc": f"{info['final_test_acc']:.4f}" if info["final_test_acc"] is not None else "",
    }

    upsert_row(args.experiments_csv, key=(config_str, args.seed), updates=updates)
    print(f"[grok] config={config_str} seed={args.seed} "
          f"grok={updates['grok']} delay={updates['grok_delay']} "
          f"final_test_acc={updates['final_test_acc']}")

if __name__ == "__main__":
    main()
