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


@torch.no_grad()
def pc2_ablated_accuracy(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    pca_bundle: Dict[str, torch.Tensor],
    *,
    positions: Tuple[int, ...] = (0, 1),
    hook_point: str = "hook_embed",
) -> Dict[str, float]:
    """
    Ablate PC2 in the token-embedding stream (at `hook_embed`) and measure accuracy drop.

    How it works:
      - For positions in `positions` (defaults to the two input tokens), we:
          emb_centered = emb - mean
          emb_ablated  = emb_centered - <emb_centered, pc2> * pc2 + mean
        This *removes the PC2 component only* while preserving PC1 and all other components.

    Args:
      model: HookedTransformer (TransformerLens).
      X, y: input tokens [B, S] and labels [B].
      pca_bundle: output from `embedding_pca(...)` with keys:
          "components" [k, d_model] (rows are PCs), and "mean" [d_model].
          Must have at least 2 PCs.
      positions: which sequence indices to ablate (default: 0 and 1).
      hook_point: usually "hook_embed" (before positional embeddings).

    Returns:
      dict with:
        - acc_orig: accuracy without ablation
        - acc_ablated: accuracy with PC2 removed at `positions`
        - acc_drop: acc_orig - acc_ablated
    """
    device = next(model.parameters()).device
    model.eval()

    # Move inputs to device
    X = X.to(device)
    y = y.to(device, dtype=torch.long)

    # Extract mean and PC2 from the PCA bundle (computed on W_E for your digits)
    comps = pca_bundle["components"]        # [k, d_model]
    mean  = pca_bundle["mean"]              # [d_model]
    if comps.shape[0] < 2:
        raise ValueError("pca_bundle must contain at least 2 principal components.")
    pc2 = comps[1].to(device)               # [d_model]
    mean = mean.to(device)                  # [d_model]

    # Helper: compute accuracy from logits
    def _acc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        preds = logits.argmax(dim=-1)
        return float((preds == labels).float().mean().item())

    # Original accuracy (no hooks)
    logits_orig = model(X)
    acc_orig = _acc_from_logits(logits_orig, y)

    # Build a vectorized hook that removes *only* the PC2 component at specified positions
    pos_idx = torch.as_tensor(positions, device=device)

    def _ablate_pc2(activation: torch.Tensor, hook) -> torch.Tensor:
        # activation: [B, S, d_model] at hook_embed
        out = activation.clone()
        # Select positions (B, P, d)
        sel = out.index_select(dim=1, index=pos_idx)
        # Center
        sel_centered = sel - mean.view(1, 1, -1)
        # Project onto PC2: (B,P)
        coeff = torch.einsum("bpd,d->bp", sel_centered, pc2)
        # Remove PC2 component: (B,P,d)
        sel_no_pc2 = sel_centered - coeff.unsqueeze(-1) * pc2.view(1, 1, -1)
        # Add mean back
        sel_edit = sel_no_pc2 + mean.view(1, 1, -1)
        # Write back into the selected positions
        out[:, pos_idx, :] = sel_edit
        return out

    # Run with hook
    logits_abl = model.run_with_hooks(X, fwd_hooks=[(hook_point, _ablate_pc2)])
    acc_abl = _acc_from_logits(logits_abl, y)

    return {
        "acc_orig": acc_orig,
        "acc_ablated": acc_abl,
        "acc_drop": acc_orig - acc_ablated,
    }