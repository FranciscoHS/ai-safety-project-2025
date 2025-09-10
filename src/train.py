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
) -> Dict[str, List[float]]:
    """
    Full-batch training loop (mirrors the notebook). Returns losses and (optionally) checkpoints.

    Args:
      model: 1-layer transformer already constructed (device can be cpu/cuda).
      train_data/train_labels: tensors from make_addition_dataset (Long).
      test_data/test_labels: optional eval set.
      lr, weight_decay, betas: AdamW hyperparameters (match notebook defaults).
      num_steps: number of optimization steps (your notebook called these 'epochs').
      checkpoint_every: save a copy of state_dict every k steps (if use_checkpoints=True).
      progress: print step/loss periodically.
      use_checkpoints: toggle storing copies of state_dict in memory.

    Returns:
      {
        "train_losses": [...],
        "test_losses": [...],                 # empty if no test_data
        "checkpoint_steps": [...],
        "checkpoints": [state_dict, ...],     # empty if use_checkpoints=False
        "seconds": float
      }
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
            # Assumes next-token prediction at the final position
            # Ensure labels are 1D and match batch size
            if logits.shape[0] != labels.shape[0]:
                raise ValueError(f"Batch size mismatch: logits {logits.shape}, labels {labels.shape}")
            if labels.ndim != 1:
                raise ValueError(f"Labels should be 1D tensor of shape (batch_size,), got {labels.shape}")
            return ce(logits[:, -1, :], labels.long())
    else:
        _loss_fn = default_loss_fn

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
    )

    train_losses: List[float] = []
    test_losses: List[float] = []
    checkpoints = []
    checkpoint_steps = []

    t0 = time.time()
    model.train()
    for step in range(1, num_steps + 1):
        # Using set_to_none=True for efficiency; if custom layers require gradients to be zeroed, consider removing this argument.
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_data)                     # full-batch forward
        loss = _loss_fn(logits, train_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        train_losses.append(float(loss.item()))

        test_loss = None
        if test_data is not None:
            with torch.inference_mode():
                test_logits = model(test_data)
                test_loss = _loss_fn(test_logits, test_labels).item()
            test_losses.append(float(test_loss))

        if checkpoint_every > 0 and (step % checkpoint_every == 0):
            if use_checkpoints:
                checkpoints.append(copy.deepcopy(model.state_dict()))
            checkpoint_steps.append(step)
            if progress:
                log_msg = {
                    "step": step,
                    "train_loss": float(loss.item()),
                }
                if test_loss is not None:
                    log_msg["test_loss"] = float(test_loss)
                print(log_msg)

    seconds = time.time() - t0
    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "checkpoint_steps": checkpoint_steps,
        "checkpoints": checkpoints,
        "seconds": seconds,
    }
