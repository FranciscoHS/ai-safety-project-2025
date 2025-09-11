#!/usr/bin/env python3
# scripts/plot_loss.py
import os, sys, argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Colab/CLI
import matplotlib.pyplot as plt

def infer_curves_path(args) -> str:
    if args.curves_csv:
        return args.curves_csv
    config = args.config_name or f"N{args.N}_frac{args.frac_train}_d{args.d_model}_h{args.n_heads}_m{args.d_mlp}"
    return os.path.join("results", "curves", f"{config}_seed{args.seed}.csv")

def infer_fig_path(args) -> str:
    out_dir = args.out or os.path.join("figures", "curves")
    os.makedirs(out_dir, exist_ok=True)
    base = args.config_name or f"N{args.N}_frac{args.frac_train}_d{args.d_model}_h{args.n_heads}_m{args.d_mlp}"
    return os.path.join(out_dir, f"{base}_seed{args.seed}_loss.png")

def moving_average(x, w):
    if w <= 1: return x
    return pd.Series(x).rolling(window=w, min_periods=1, center=True).mean().to_numpy()

def main():
    ap = argparse.ArgumentParser()
    # Either give curves_csv directly...
    ap.add_argument("--curves_csv", default="", help="Path to results/curves/*.csv. If empty, inferred from args.")
    # ...or let us infer it from the same knobs used in run.py
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--frac_train", type=float, default=0.72)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_mlp", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config_name", default="", help="Row key / filename stem; overrides the inferred one if set.")

    ap.add_argument("--smooth", type=int, default=1, help="Moving-average window (steps) for smoothing curves.")
    ap.add_argument("--out", default="", help="Output directory for the PNG (default: figures/curves)")
    args = ap.parse_args()

    curves_path = infer_curves_path(args)
    if not os.path.exists(curves_path):
        raise FileNotFoundError(f"Curves CSV not found: {curves_path}")

    df = pd.read_csv(curves_path)
    # Expect columns: step, split, loss, acc
    if not {"step", "split", "loss"}.issubset(df.columns):
        raise ValueError(f"CSV must have columns step, split, loss. Found: {df.columns.tolist()}")

    # Separate splits
    tr = df[df["split"] == "train"].sort_values("step")
    te = df[df["split"] == "test"].sort_values("step")

    tr_steps = tr["step"].to_numpy()
    te_steps = te["step"].to_numpy()
    tr_loss = moving_average(tr["loss"].to_numpy(), args.smooth)
    te_loss = moving_average(te["loss"].to_numpy(), args.smooth)

    # Plot
    plt.figure(figsize=(8, 5))
    if len(tr_steps):
        plt.plot(tr_steps, tr_loss, label="train loss")
    if len(te_steps):
        plt.plot(te_steps, te_loss, label="test loss")
    ttl = args.config_name or f"N={args.N}, frac={args.frac_train}, d={args.d_model}, h={args.n_heads}, mlp={args.d_mlp}, seed={args.seed}"
    plt.title(f"Loss vs Step ({ttl})")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    out_path = infer_fig_path(args)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[plot] saved {out_path}")

if __name__ == "__main__":
    main()
