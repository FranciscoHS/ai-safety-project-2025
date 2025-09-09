import torch
import einops
from typing import Tuple, Union

def make_addition_dataset(
    N: int,
    frac_train: float,
    data_seed: int = 598,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build an addition dataset for the sequence [a, b, '='] → target (a+b).

    Inputs:
      - N: integers are 0..N-1; the '=' token id is N
      - frac_train: fraction of all N*N pairs used for training (0<frac<1)
      - data_seed: RNG seed for the train/test split
      - device: 'cpu' or a torch.device (you can keep this 'cpu' and move
                to GPU later in your train/eval code)

    Returns:
      X_train [num_train, 3] (LongTensor)
      y_train [num_train]    (LongTensor)
      X_test  [num_test, 3]  (LongTensor)
      y_test  [num_test]     (LongTensor)

    Notes:
      - a, b ∈ {0..N-1} so (a+b) ∈ {0..2N-2}.
      - Your model’s vocab must cover 0..2N-2 (sums) plus the '=' token id N.
        A common choice is d_vocab = 2*N + 1 (room to spare).
    """
    assert N > 1, "N must be > 1"
    assert 0.0 < frac_train < 1.0, "frac_train must be in (0,1)"

    # All ordered pairs (a,b)
    a = einops.repeat(torch.arange(N, dtype=torch.long), "i -> (i j)", j=N)
    b = einops.repeat(torch.arange(N, dtype=torch.long), "j -> (i j)", i=N)
    eq = torch.full_like(a, fill_value=N)  # '=' token id is N

    X = torch.stack([a, b, eq], dim=1)     # [N*N, 3]
    y = a + b                               # [N*N]

    # Deterministic shuffle & split
    g = torch.Generator()
    g.manual_seed(data_seed)
    idx = torch.randperm(X.size(0), generator=g)
    cutoff = int(idx.numel() * frac_train)
    train_idx, test_idx = idx[:cutoff], idx[cutoff:]

    X_train = X[train_idx].to(device)
    y_train = y[train_idx].to(device)
    X_test  = X[test_idx].to(device)
    y_test  = y[test_idx].to(device)

    return X_train, y_train, X_test, y_test


def make_ood_dataset(
    N: int,
    lo: int,
    hi: int,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build OOD pairs where a,b ∈ [lo..hi] (inclusive), still using '=' token id N.
    Returns X_ood [M,3], y_ood [M].
    """
    a_vals = torch.arange(lo, hi + 1, dtype=torch.long)
    b_vals = torch.arange(lo, hi + 1, dtype=torch.long)
    a = einops.repeat(a_vals, "i -> (i j)", j=b_vals.numel())
    b = einops.repeat(b_vals, "j -> (i j)", i=a_vals.numel())
    eq = torch.full_like(a, fill_value=N)

    X = torch.stack([a, b, eq], dim=1).to(device)
    y = (a + b).to(device)

    return X, y