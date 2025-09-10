from __future__ import annotations
import torch
from typing import Iterable, Dict

@torch.no_grad()
def embedding_pca(
    model: torch.nn.Module,
    token_ids: Iterable[int],
    n_components: int = 2,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, torch.Tensor]:
    """
    PCA on the input token embeddings W_E for the given token ids.

    Args:
      model: TransformerLens model (must have `model.W_E` with shape [d_vocab, d_model]).
      token_ids: which vocab rows to include in PCA (e.g., range(N) for digits 0..N-1).
      n_components: number of principal components to return.
      dtype: computation dtype for numerical stability (float64 recommended).

    Returns (all tensors on CPU):
      {
        "components":  [k, d_model]  # rows are PC1..PCk (right singular vectors)
        "mean":        [d_model]     # mean embedding over the selected tokens
        "scores":      [n_tokens, k] # U * S: coordinates of tokens in PC space
        "singular":    [k]           # top-k singular values
        "explained":   [k]           # variance ratio per component (top-k)
        "token_ids":   [n_tokens]    # the ids used (Long)
      }
    """
    # 1) Grab embeddings and select tokens
    W_E = model.W_E.detach().to("cpu", dtype=dtype)            # [d_vocab, d_model]
    token_ids = torch.as_tensor(list(token_ids), dtype=torch.long)
    W = W_E[token_ids]                                         # [n_tokens, d_model]
    n_tokens, d_model = W.shape

    # 2) Center
    mean = W.mean(dim=0, keepdim=True)                         # [1, d_model]
    Xc = W - mean                                              # [n_tokens, d_model]

    # 3) SVD
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)       # U:[n,k], S:[k], Vh:[k,d]
    k = int(min(n_components, S.numel()))
    comps   = Vh[:k, :]                                        # [k, d_model]
    scores  = U[:, :k] * S[:k]                                 # [n_tokens, k]

    # 4) Explained variance (standard PCA-from-SVD)
    #    eigenvals = S^2 / (n_tokens - 1)
    if n_tokens > 1:
        eigvals = (S**2) / (n_tokens - 1)
        explained = (eigvals / eigvals.sum())[:k]              # [k]
    else:
        explained = torch.ones(k, dtype=dtype) / k

    return {
        "components": comps.contiguous(),                      # [k, d_model]
        "mean": mean.squeeze(0).contiguous(),                  # [d_model]
        "scores": scores.contiguous(),                         # [n_tokens, k]
        "singular": S[:k].contiguous(),                        # [k]
        "explained": explained.contiguous(),                   # [k]
        "token_ids": token_ids,
    }
