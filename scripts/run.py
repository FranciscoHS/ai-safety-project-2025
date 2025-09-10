import argparse, time, random
import numpy as np
import torch

# ensure repository root is on the import path
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Local imports
from src.data import make_addition_dataset
from src.model import build_model
from src.train import train_model

# Try to use your eval helpers if present; otherwise a tiny fallback.
try:
    from src.eval import model_accuracy as _model_accuracy
    def accuracy(model, X, y):
        acc, *_ = _model_accuracy(model, X, y)
        return float(acc)
except Exception:
    @torch.no_grad()
    def accuracy(model, X, y):
        device = next(model.parameters()).device
        X, y = X.to(device), y.to(device)
        logits = model(X)[:, -1, :]
        preds = torch.argmax(logits, dim=-1)
        return float((preds == y).float().mean().item())


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=6)
    p.add_argument("--frac_train", type=float, default=0.72)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_mlp", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.98)
    p.add_argument("--steps", type=int, default=300)  # ~200–500 for a quick check
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}  seed={args.seed}")

    # 1) Data
    Xtr, ytr, Xte, yte = make_addition_dataset(args.N, args.frac_train)
    print(f"[info] train={len(Xtr)}  test={len(Xte)}  N={args.N}")

    # 2) Model
    model = build_model(N=args.N, d_model=args.d_model, n_heads=args.n_heads, d_mlp=args.d_mlp).to(device)

    # 3) Before-training sanity
    acc0_tr = accuracy(model, Xtr, ytr)
    acc0_te = accuracy(model, Xte, yte)
    print(f"[before] acc_train={acc0_tr:.3f}  acc_test={acc0_te:.3f}  (random ≈ {1/(2*args.N-1):.3f})")

    # 4) Train (full-batch)
    t0 = time.time()
    info = train_model(
        model,
        Xtr, ytr,
        test_data=Xte, test_labels=yte,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        num_steps=args.steps,
        checkpoint_every=0,
        progress=True,
        use_checkpoints=False,
    )
    dt = time.time() - t0

    # 5) After-training metrics
    acc1_tr = accuracy(model, Xtr, ytr)
    acc1_te = accuracy(model, Xte, yte)
    tr_loss = info["train_losses"][-1] if info["train_losses"] else float("nan")
    te_loss = info["test_losses"][-1] if info["test_losses"] else float("nan")

    print(f"[after ] acc_train={acc1_tr:.3f}  acc_test={acc1_te:.3f}  "
          f"loss_train={tr_loss:.4f}  loss_test={te_loss:.4f}  time={dt:.1f}s")


if __name__ == "__main__":
    main()