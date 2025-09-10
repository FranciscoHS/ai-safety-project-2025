import torch

def model_accuracy(model, X, y):
    """
    Evaluates the accuracy of a model's predictions on a batch of data.

    Args:
        model (torch.nn.Module): The model to evaluate. Should output logits of shape [batch, seq_len, d_vocab].
        X (torch.Tensor): Input tensor to the model.
        y (torch.Tensor): Ground truth labels for the batch. Shape: [batch].

    Returns:
        Tuple[float, int, int]:
            - accuracy (float): The proportion of correct predictions at the last sequence position.
            - correct_predictions (int): Number of correct predictions.
            - total_predictions (int): Total number of predictions evaluated.

    Notes:
        - Only the logits at the last sequence position are used for prediction.
        - Assumes y contains the target token indices for the last position.
    """
    """Calculates accuracy given logits and labels."""
    # Logits shape: [batch, seq_len, d_vocab]
    # We only care about the prediction at the last position (index 2)
    logits = model(X)
    prediction_logits = logits[:, -1, :] # Shape: [batch, d_vocab]
    predicted_tokens = torch.argmax(prediction_logits, dim=-1) # Shape: [batch]
    correct_predictions = (predicted_tokens == y).sum().item()
    total_predictions = y.shape[0]
    accuracy = correct_predictions / total_predictions
    return accuracy, correct_predictions, total_predictions


def loss_fn(logits, labels):
    """
    Computes the negative log-likelihood loss for classification tasks.

    If `logits` has three dimensions, selects the last time step along the second dimension.
    Converts logits to float64, applies log-softmax, and gathers the log-probabilities corresponding to the true labels.
    Returns the mean negative log-probability.

    Args:
        logits (torch.Tensor): The predicted logits of shape (batch_size, num_classes) or
            (batch_size, sequence_length, num_classes).
        labels (torch.Tensor): The true labels of shape (batch_size,).

    Returns:
        torch.Tensor: The mean negative log-likelihood loss (scalar).
    """
    if len(logits.shape)==3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()
