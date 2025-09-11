import os
import csv
import time
import copy
import torch
from typing import Dict, List, Optional, Tuple

try:
    # Your eval.py should expose a loss_fn that accepts (logits, labels)
    from .eval import loss_fn as default_loss_fn
except Exception:
    # Fallback: basic CE that expects logits[:, -1, :] and Long labels
    default_loss_fn = None


def train_model(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: Optional[torch.Tensor] = None,
    test_labels: Optional[torch.Tensor] = None,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    betas: Tuple[float, float] = (0.9, 0.98),
    num_steps: int = 10_000,
    checkpoint_every: int = 100,
    progress: bool = True,
    use_checkpoints: bool = True,
    eval_every: int = 200,
    log_curves: bool = True,
    curves_csv: Optional[str] = None,
    config_name: str = "default",
    seed: Optional[int] = None,
) -> Dict[str, List[float]]:
    """
    Trains a model using full-batch gradient descent.

    Args:
        model (torch.nn.Module): The model to train.
        train_data (torch.Tensor): Training input data.
        train_labels (torch.Tensor): Training labels.
        test_data (Optional[torch.Tensor], optional): Test input data. Defaults to None.
        test_labels (Optional[torch.Tensor], optional): Test labels. Defaults to None.
        lr (float, optional): Learning rate for AdamW. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay for AdamW. Defaults to 1.0.
        betas (Tuple[float, float], optional): AdamW betas. Defaults to (0.9, 0.98).
        num_steps (int, optional): Number of training steps. Defaults to 10_000.
        checkpoint_every (int, optional): Save checkpoint every k steps. Defaults to 100.
        progress (bool, optional): Print progress during training. Defaults to True.
        use_checkpoints (bool, optional): Store model checkpoints in memory. Defaults to True.
        eval_every (int, optional): Evaluate and log metrics every k steps. Defaults to 200.
        log_curves (bool, optional): Log training curves to CSV. Defaults to True.
        curves_csv (Optional[str], optional): Path to save curves CSV. Defaults to None.
        config_name (str, optional): Name for config/logging. Defaults to "default".
        seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.

    Returns:
        Dict[str, List[float]]: Dictionary containing:
            - "train_losses": List of training losses.
            - "test_losses": List of test losses (empty if no test data).
            - "checkpoint_steps": List of steps where checkpoints were saved.
            - "checkpoints": List of model state_dicts (empty if use_checkpoints=False).
            - "seconds": Total training time in seconds.
    """
    device = next(model.parameters()).device
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    if test_data is not None and test_labels is not None:
        test_data = test_data.to(device)
        test_labels = test_labels.to(device)

    # Loss function
    if default_loss_fn is None:
        ce = torch.nn.CrossEntropyLoss()
        def _loss_fn(logits, labels):
            return ce(logits[:, -1, :], labels.to(logits.device))
    else:
        _loss_fn = default_loss_fn

    @torch.no_grad()
    def _accuracy(md: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
        md.eval()
        logits = md(X)
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        preds = logits.argmax(dim=-1)
        return float((preds == y).float().mean().item())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
    )

    train_losses: List[float] = []
    test_losses: List[float] = []
    checkpoints = []
    checkpoint_steps = []

    # Prepare curve logging
    curve_rows: List[dict] = []
    def _maybe_log(step: int):
        # Train split metrics (compute fresh so they're aligned with eval cadence)
        model.eval()
        with torch.no_grad():
            tr_logits = model(train_data)
            tr_loss = float(_loss_fn(tr_logits, train_labels).item())
            tr_acc  = _accuracy(model, train_data, train_labels)
        curve_rows.append({"step": step, "split": "train", "loss": tr_loss, "acc": tr_acc})
        # Test split metrics (if available)
        if test_data is not None:
            with torch.no_grad():
                te_logits = model(test_data)
                te_loss = float(_loss_fn(te_logits, test_labels).item())
                te_acc  = _accuracy(model, test_data, test_labels)
            curve_rows.append({"step": step, "split": "test", "loss": te_loss, "acc": te_acc})
            # Also keep the last in the return lists for convenience
            test_losses.append(te_loss)

        # Progress print
        if progress:
            if test_data is not None:
                print(f"step {step} | train {tr_loss:.4f}/{tr_acc:.3f} | test {te_loss:.4f}/{te_acc:.3f}")
            else:
                print(f"step {step} | train {tr_loss:.4f}/{tr_acc:.3f}")

        # Keep the “classic” lists too (train loss every eval)
        train_losses.append(tr_loss)

    t0 = time.time()
    model.train()
    for step in range(1, num_steps + 1):
        logits = model(train_data)
        loss = _loss_fn(logits, train_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Periodic eval/log
        if eval_every and (step % eval_every == 0 or step == 1 or step == num_steps):
            _maybe_log(step)

        # Optional checkpoints
        if checkpoint_every > 0 and (step % checkpoint_every == 0):
            if use_checkpoints:
                checkpoints.append(copy.deepcopy(model.state_dict()))
            checkpoint_steps.append(step)

    seconds = time.time() - t0

    # Write curves CSV (overwrite for this run)
    if log_curves:
        # Default path if not provided
        if not curves_csv:
            curves_dir = os.path.join("results", "curves")
            os.makedirs(curves_dir, exist_ok=True)
            curves_csv = os.path.join(curves_dir, f"{config_name}_seed{seed if seed is not None else 0}.csv")
        else:
            os.makedirs(os.path.dirname(curves_csv), exist_ok=True)

        with open(curves_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "split", "loss", "acc"])
            writer.writeheader()
            for row in curve_rows:
                writer.writerow(row)

    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "checkpoint_steps": checkpoint_steps,
        "checkpoints": checkpoints,
        "seconds": seconds,
    }
