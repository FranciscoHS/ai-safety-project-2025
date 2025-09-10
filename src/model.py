from transformer_lens import HookedTransformer, HookedTransformerConfig

def build_model(
    N,
    d_model=128,
    n_heads=4,
    d_head=32,
    d_mlp=512,
    act_fn="relu",
    normalization_type=None,
    d_vocab_out=None,
    n_layers=1,
    n_ctx=3,
    init_weights=True,
    device="cpu",
    seed=999
):
    """
    Builds and returns a HookedTransformer model with the specified configuration. The model's biases are disabled.

    Args:
        N (int): Size of the input vocabulary (excluding special tokens).
        d_model (int, optional): Dimension of the model's hidden states. Default is 128.
        n_heads (int, optional): Number of attention heads. Default is 4.
        d_head (int, optional): Dimension of each attention head. Default is 32.
        d_mlp (int, optional): Dimension of the feedforward network (MLP). Default is 512.
        act_fn (str, optional): Activation function to use in the model (e.g., "relu"). Default is "relu".
        normalization_type (str or None, optional): Type of normalization to use (e.g., "layernorm"). Default is None.
        d_vocab_out (int or None, optional): Output vocabulary size. If None, set to 2 * N. Default is None.
        n_layers (int, optional): Number of transformer layers. Default is 1.
        n_ctx (int, optional): Maximum context length (sequence length). Default is 3.
        init_weights (bool, optional): Whether to initialize model weights. Default is True.
        device (str, optional): Device to place the model on (e.g., "cpu" or "cuda"). Default is "cpu".
        seed (int, optional): Random seed for reproducibility. Default is 999.

    Returns:
        HookedTransformer: An instance of the HookedTransformer model configured with the specified parameters.
    """
    if d_vocab_out is None:
        d_vocab_out = 2 * N
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_head,
        d_mlp=d_mlp,
        act_fn=act_fn,
        normalization_type=normalization_type,
        d_vocab=N+1,
        d_vocab_out=d_vocab_out,
        n_ctx=n_ctx,
        init_weights=init_weights,
        device=device,
        seed=seed,
    )
    model = HookedTransformer(cfg)

    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    return model