#!/usr/bin/env python3
# scripts/pca_and_ablate.py

import os, sys, csv, argparse
import torch

# Ensure repo root is importable when called as `!python scripts/...`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import make_addition_dataset
from src.model import build_model
from src.train import train_model
from src.analysis import embedding_pca, pc2_ablated_accuracy

CSV_HEADER = [
    "commit","config","seed","d_model","d_mlp",
    "acc_id","acc_ood","acc_pc2_ablated","we_pc12_var","wl_pc12_var",
    "gpu","seconds"
]

def get_commit_hash():
    try:
        import subprocess
        return subprocess.check_output(["git","rev-parse","--short","HEAD"], text=True).strip()
    except Exception:
        return "no-git"

def gpu_name():
    if torch.cuda.is_available():
        try: return torch.cuda.get_device_name(0)
        except Exception: return "cuda"
    return "cpu"

def ensure_csv(path: str):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

def read_rows(path: str):
    if not os.path.exists(path): return []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        return list(r)

def write_rows(path: str, rows: list[dict]):
    # ensure all rows have all columns
    for row in rows:
        for k in CSV_HEADER:
            row.setdefault(k, "")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def upsert_row(path: str, key: tuple[str,int], updates: dict):
    """Update (config, seed) row, else append."""
    ensure_csv(path)
    rows = read_rows(path)
    cfg_key, seed_key = key
    hit = None
    for row in rows:
        if row.get("config","") == cfg_key and str(row.get("seed","")) == str(seed_key):
            hit = row; break
    if hit is None:
        hit = {k:"" for k in CSV_HEADER}
        hit["config"] = cfg_key
        hit["seed"]   = str(seed_key)
        rows.append(hit)
    # apply updates
    for k,v in updates.items():
        hit[k] = v
    write_rows(path, rows)

def main():
    p = argparse.ArgumentParser()
    # Data / model knobs (should match run.py)
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--frac_train", type=float, default=0.72)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_mlp", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)

    # Training (only used if --train_if_missing is true)
    p.add_argument("--steps", type=int, default=0, help="If >0 and training enabled, train for this many steps.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.98)
    p.add_argument("--train_if_missing", action="store_true",
                   help="If set, train a model here (use --steps) instead of relying on a prior run.")

    # I/O
    p.add_argument("--csv", default="results/experiments.csv")
    p.add_argument("--config_name", default="", help="String used as 'config' key in CSV. Default: auto from args.")

    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build test split (ID)
    Xtr, ytr, Xte, yte = make_addition_dataset(args.N, args.frac_train)
    # Build model
    model = build_model(N=args.N, d_model=args.d_model, n_heads=args.n_heads, d_mlp=args.d_mlp).to(device)

    # Optionally (re)train so this script is standalone if needed
    if args.train_if_missing and args.steps > 0:
        _ = train_model(
            model,
            Xtr, ytr,
            test_data=Xte, test_labels=yte,
            lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2),
            num_steps=args.steps, checkpoint_every=0, progress=True, use_checkpoints=False,
        )

    # Analysis: embedding PCA on tokens 0..N-1
    pca = embedding_pca(model, token_ids=range(args.N), n_components=2)
    we_pc12_var = float(pca["explained"][:2].sum().item())

    # PC2 ablation accuracy on ID test split
    accs = pc2_ablated_accuracy(model, Xte, yte, pca)
    acc_pc2 = float(accs["acc_ablated"])

    # Prepare CSV updates
    config_str = args.config_name or f"N{args.N}_frac{args.frac_train}_d{args.d_model}_h{args.n_heads}_m{args.d_mlp}"
    updates = {
        "commit": get_commit_hash(),
        "config": config_str,
        "seed": str(args.seed),
        "d_model": str(args.d_model),
        "d_mlp": str(args.d_mlp),
        "acc_pc2_ablated": f"{acc_pc2:.4f}",
        "we_pc12_var": f"{we_pc12_var:.4f}",
        "gpu": gpu_name(),
        # leave other fields as-is (acc_id/acc_ood/wl_pc12_var/seconds)
    }

    upsert_row(args.csv, key=(config_str, args.seed), updates=updates)
    print(f"[pca_and_ablate] Updated {args.csv} for (config={config_str}, seed={args.seed}) "
          f"with we_pc12_var={we_pc12_var:.4f}, acc_pc2_ablated={acc_pc2:.4f}")

if __name__ == "__main__":
    main()
