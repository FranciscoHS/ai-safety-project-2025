from __future__ import annotations
import torch
from typing import Iterable, Dict, Tuple, Optional


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

    X = X.to(device)
    y = y.to(device, dtype=torch.long)

    comps = pca_bundle["components"]   # [k, d_model] (float64 likely)
    mean  = pca_bundle["mean"]         # [d_model]    (float64 likely)
    if comps.shape[0] < 2:
        raise ValueError("pca_bundle must contain at least 2 PCs.")
    pc2 = comps[1]                     # [d_model]

    pos_idx = torch.as_tensor(positions, device=device, dtype=torch.long)

    def _ablate_pc2(activation: torch.Tensor, hook) -> torch.Tensor:
        """
        activation: [B, S, d_model] at hook_embed. We:
          - center by mean
          - remove the PC2 component
          - add mean back
          - write only at selected positions
        Critically: cast mean/pc2 to activation.dtype/device to avoid dtype errors.
        """
        out = activation.clone()
        act_dtype = activation.dtype
        mean_local = mean.to(device=activation.device, dtype=act_dtype)    # align dtype/device
        pc2_local  = pc2.to(device=activation.device, dtype=act_dtype)

        sel = out.index_select(dim=1, index=pos_idx)                       # [B, P, d]
        sel_centered = sel - mean_local.view(1, 1, -1)                     # [B, P, d]
        coeff = torch.einsum("bpd,d->bp", sel_centered, pc2_local)         # [B, P]
        sel_no_pc2 = sel_centered - coeff.unsqueeze(-1) * pc2_local.view(1, 1, -1)
        sel_edit = sel_no_pc2 + mean_local.view(1, 1, -1)                  # [B, P, d]
        out[:, pos_idx, :] = sel_edit                                      # write-back: dtypes now match
        return out

    # Original accuracy (no hooks)
    logits_orig = model(X)
    logits_o = logits_orig[:, -1, :] if logits_orig.ndim == 3 else logits_orig
    acc_orig = float((logits_o.argmax(dim=-1) == y).float().mean().item())

    # Ablated accuracy (hooked)
    logits_abl = model.run_with_hooks(X, fwd_hooks=[(hook_point, _ablate_pc2)])
    logits_a = logits_abl[:, -1, :] if logits_abl.ndim == 3 else logits_abl
    acc_abl = float((logits_a.argmax(dim=-1) == y).float().mean().item())

    return {
        "acc_orig": acc_orig,
        "acc_ablated": acc_abl,
        "acc_drop": acc_orig - acc_abl,
    }


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size == 0:
        return x
    w = min(w, x.size // 2 * 2 + 1)  # make odd-ish, avoid huge windows
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(xpad, kernel, mode="valid")

def detect_grokking(
    curves: pd.DataFrame,
    *,
    train_thr: float = 0.99,
    test_thr: float  = 0.95,
    min_gap_frac: float = 0.10,
    smooth_window: int = 11,
    max_steps: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """
    Detect 'grokking' from learning curves:
      - t_train: first step where smoothed train_acc >= train_thr
      - t_test : first step where smoothed test_acc  >= test_thr
      - delay  : t_test - t_train
      - grok   : True if both thresholds hit and delay >= min_gap_frac * max_steps,
                 and final test acc >= test_thr

    Returns dict with keys: grok (bool), t_train, t_test, delay, final_test_acc.
    """
    # Basic checks
    required = {"step", "split", "acc", "loss"}
    if not required.issubset(set(curves.columns)):
        raise ValueError(f"curves must have columns {required}, got {set(curves.columns)}")

    df = curves.sort_values("step")
    if max_steps is None:
        max_steps = float(df["step"].max())

    # Split & smooth
    train_df = df[df["split"] == "train"]
    test_df  = df[df["split"] == "test"]

    if train_df.empty or test_df.empty:
        # Without both splits we can't detect grokking robustly
        return {"grok": False, "t_train": None, "t_test": None, "delay": None, "final_test_acc": None}

    tr_steps = train_df["step"].to_numpy()
    te_steps = test_df["step"].to_numpy()
    tr_acc   = _moving_average(train_df["acc"].to_numpy(), smooth_window)
    te_acc   = _moving_average(test_df["acc"].to_numpy(),   smooth_window)

    # First threshold crossings
    t_train = tr_steps[np.argmax(tr_acc >= train_thr)] if np.any(tr_acc >= train_thr) else None
    t_test  = te_steps[np.argmax(te_acc >= test_thr)]  if np.any(te_acc >= test_thr)  else None

    final_test_acc = float(test_df["acc"].to_numpy()[-1])

    grok = False
    delay = None
    if t_train is not None and t_test is not None:
        delay = float(t_test - t_train)
        grok = (delay >= min_gap_frac * float(max_steps)) and (final_test_acc >= test_thr)

    return {
        "grok": bool(grok),
        "t_train": float(t_train) if t_train is not None else None,
        "t_test": float(t_test) if t_test is not None else None,
        "delay": float(delay) if delay is not None else None,
        "final_test_acc": final_test_acc,
    }
