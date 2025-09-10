import torch
import torch.nn.functional as F

@torch.no_grad()
def model_accuracy(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """
    Computes the accuracy of a model's predictions.

    Args:
        model (torch.nn.Module): The model to evaluate.
        X (torch.Tensor): Input tensor of shape [batch, ...].
        y (torch.Tensor): Target tensor of shape [batch].

    Returns:
        Tuple[float, int, int]: (accuracy, number of correct predictions, total samples).
            - accuracy (float): Proportion of correct predictions.
            - correct (int): Number of correct predictions.
            - total (int): Total number of samples.

    Notes:
        - Uses logits at the last sequence position if input is 3D.
        - Moves X and y to the model's device.
        - Sets model to evaluation mode and disables gradient computation.
    """
    device = next(model.parameters()).device
    model.eval()

    X = X.to(device)
    y = y.to(device, dtype=torch.long)

    logits = model(X)  # [batch, seq_len, d_vocab] or [batch, d_vocab]
    if logits.ndim == 3:
        logits = logits[:, -1, :]  # last position

    preds = logits.argmax(dim=-1)              # [batch]
    correct = (preds == y).sum().item()
    total = y.numel()
    acc = correct / total if total else 0.0
    return acc, correct, total


def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes cross-entropy loss over the final position.

    Args:
        logits (torch.Tensor): Logits tensor of shape [batch, seq_len, d_vocab] or [batch, d_vocab].
        labels (torch.Tensor): Target tensor of shape [batch].

    Returns:
        torch.Tensor: Scalar cross-entropy loss.

    Notes:
        - If logits are 3D, uses the last sequence position.
        - Ensures labels are of type Long and on the same device as logits.
    """
    if logits.ndim == 3:
        logits = logits[:, -1, :]  # [batch, d_vocab]
    labels = labels.to(logits.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)
