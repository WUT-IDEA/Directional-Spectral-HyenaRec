"""HyenaRec backbone with Legendre and compact spectral convolution kernels."""

import math


import torch


import torch.nn as nn


import torch.nn.functional as F


from functools import partial


from einops import rearrange


from .spectral import finite_difference_matrix, exposure_spectral_basis


def fftconv(u, k, D):
    """
    We apply a convolution through the fourier domain (from the Convolution Theorem)

    """
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3: k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


def legendre_polynomials(L, K, device=None, dtype=None, x=None):
    """
    生成前 K 阶 Legendre 多项式在 [-1,1] 区间采样的值
    返回 shape: (K, L)
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.get_default_dtype()

    if x is None:
        # Keep the original uniform-grid path byte-for-byte equivalent when the
        # new temporal-grid option is disabled.
        x = torch.linspace(-1.0, 1.0, L, device=device, dtype=dtype)
        P = torch.zeros(K, L, device=device, dtype=dtype)
        P[0, :] = 1.0
        if K > 1:
            P[1, :] = x
        for n in range(1, K - 1):
            P[n + 1, :] = (
                (2.0 * n + 1.0) * x * P[n, :] - n * P[n - 1, :]
            ) / (n + 1.0)
        return P

    if x.ndim != 1 or x.numel() != L:
        raise ValueError(f"x must have shape ({L},), got {tuple(x.shape)}")
    x = x.to(device=device, dtype=dtype)

    # Building a list rather than filling a tensor in-place keeps the recurrence
    # differentiable when x comes from the learnable temporal grid.
    polys = [torch.ones_like(x)]
    if K > 1:
        polys.append(x)
    for n in range(1, K - 1):
        polys.append(
            ((2.0 * n + 1.0) * x * polys[n] - n * polys[n - 1]) / (n + 1.0)
        )
    return torch.stack(polys, dim=0)


class LegendreKernelFilter(nn.Module):
    """Original Legendre kernels, with optional discrete QR for D0 controls."""

    def __init__(self, d_model, seq_len, order=16, dropout=0.0,
                 discrete_orthogonal=False):
        super().__init__()
        if seq_len < 1 or order < 1:
            raise ValueError("Sequence length and polynomial order must be positive")
        self.n_channels = d_model
        self.seq_len = seq_len
        self.K = order
        self.K_eff = min(order, seq_len)
        self.discrete_orthogonal = discrete_orthogonal
        self.dropout = nn.Dropout(dropout)
        self.coeffs = nn.Parameter(torch.randn(d_model, order) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_model))
        if discrete_orthogonal:
            sampled = legendre_polynomials(seq_len, self.K_eff,
                                           device=self.coeffs.device, dtype=self.coeffs.dtype)
            orthogonal, _ = torch.linalg.qr(sampled.T, mode="reduced")
            self.register_buffer("basis", orthogonal.T)
        else:
            self.register_buffer("basis", None, persistent=False)

    def filter(self, length):
        if self.discrete_orthogonal and length == self.seq_len:
            basis = self.basis.to(device=self.coeffs.device, dtype=self.coeffs.dtype)
            coefficients = self.coeffs[:, :self.K_eff]
        else:
            order = min(self.K, length) if self.discrete_orthogonal else self.K
            basis = legendre_polynomials(length, order, device=self.coeffs.device,
                                          dtype=self.coeffs.dtype)
            if self.discrete_orthogonal:
                orthogonal, _ = torch.linalg.qr(basis.T, mode="reduced")
                basis = orthogonal.T
                coefficients = self.coeffs[:, :order]
            else:
                coefficients = self.coeffs
        kernel = torch.einsum("nk,kl->ln", coefficients, basis)
        return (kernel / (kernel.abs().sum(dim=0, keepdim=True) + 1e-6),)

    def forward(self, x, L, k=None, bias=None):
            """
            x: (B, D_x, L)  -- 输入通道 D_x 可以与 self.n_channels 不同（通常 x 的通道数 == k.shape[0]）
            k: 可以是以下形状之一：
               - (d, L)  : d == x.size(1)
               - (L, d)  : will be transposed
               - None    : 那么会使用 self.filter(L) 生成 k_full (L, self.n_channels)，
                           并要求 x.size(1) == self.n_channels 才能直接 conv
            bias: None 或者形状 (d,) 与 k 的第一个维度一致
            返回 y: (B, D_x, L)
            """
            B, D_x, _ = x.shape

            # 准备 kernel (per-channel)
            if k is None:
                k_full = self.filter(L)[0]   # (L, n_channels)
                k_ch = k_full.T              # (n_channels, L)
                if D_x != k_ch.shape[0]:
                    raise RuntimeError(f"forward(): x channels {D_x} != generated kernel channels {k_ch.shape[0]}")
            else:
                # k passed: normalize to (d, L) where d == D_x
                if k.dim() == 2:
                    if k.shape[0] == D_x and k.shape[1] == L:
                        k_ch = k
                    elif k.shape[1] == D_x and k.shape[0] == L:
                        k_ch = k.T
                    else:
                        raise RuntimeError(f"forward(): provided k shape {tuple(k.shape)} not compatible with x channels {D_x} and L {L}")
                else:
                    raise RuntimeError("forward(): provided k must be 2D tensor")

            # bias handling
            if bias is None:
                b = None
            else:
                # bias might be provided as (d,) or (L,d) slice -- expect (d,)
                if bias.dim() == 1 and bias.shape[0] == k_ch.shape[0]:
                    b = bias
                else:
                    # try squeeze if possible
                    b = bias.view(-1) if bias.numel() == k_ch.shape[0] else None
                # if still None -> let conv raise error
            # grouped conv: weight shape must be (out_channels, in_channels_per_group, kernel_size)
            # we want depthwise: out_channels = D_x, in_ch_per_group = 1, groups = D_x
            weight = k_ch.unsqueeze(1)   # (D_x, 1, L)
            # conv1d input shape (B, in_channels, L). out will be (B, out_channels, L + L - 1)
            y = fftconv(x, k_ch, D=torch.zeros(D_x, device=x.device, dtype=x.dtype))
            # y = F.conv1d(x, weight, bias=b, padding=L - 1, groups=D_x)
            y = y[..., :L]  # 取前 L 个时间步（因 causal / truncation）
            return y


class SpectralGeometry(nn.Module):
    """Non-trainable geometry, optionally shared by filters in one backbone."""

    def __init__(self, exposure_profile, rank, difference_order, alpha, epsilon, mode):
        super().__init__()
        basis, values, roughness, weights = exposure_spectral_basis(
            exposure_profile, rank, difference_order, alpha, epsilon
        )
        self.register_buffer("exposure_profile", torch.as_tensor(exposure_profile).float())
        self.register_buffer("spectral_eigenvalues", values.float())
        self.register_buffer("local_smoothness_weights", weights.float())
        self.register_buffer("basis", basis.float() if mode == "hard" else None)
        self.register_buffer("roughness", None)


class SpectralKernelFilter(nn.Module):
    """Direct or exposure-spectral finite-history kernel parameterization."""

    def __init__(self, d_model, seq_len, mode="hard", rank=16,
                 exposure_profile=None, difference_order=2, alpha=1.0,
                 epsilon=1e-3, spectral_geometry_cache=None):
        super().__init__()
        if mode != "hard" or exposure_profile is None:
            raise ValueError("Compact spectral kernels require hard mode and a lag profile")
        if seq_len < 1 or not difference_order < rank <= seq_len:
            raise ValueError("Rank must exceed the difference nullity and fit the sequence")
        self.n_channels = d_model
        self.seq_len = seq_len
        self.mode = mode
        self.rank = rank
        self.difference_order = difference_order
        self.normalization_epsilon = 1e-6
        cache = spectral_geometry_cache
        profile_key = tuple(torch.as_tensor(exposure_profile, dtype=torch.float64)
                            .detach().cpu().flatten().tolist())
        key = (exposure_spectral_basis, seq_len, rank, difference_order, alpha,
               epsilon, profile_key)
        if cache is not None and key in cache:
            self.spectral_geometry = cache[key]
        else:
            self.spectral_geometry = SpectralGeometry(
                exposure_profile, rank, difference_order, alpha, epsilon, mode)
            if cache is not None:
                cache[key] = self.spectral_geometry
        self.coeffs = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_model))

    @property
    def basis(self):
        return self.spectral_geometry.basis if self.spectral_geometry is not None else None

    @property
    def roughness(self):
        return self.spectral_geometry.roughness if self.spectral_geometry is not None else None

    @property
    def exposure_profile(self):
        return self.spectral_geometry.exposure_profile if self.spectral_geometry is not None else None

    @property
    def spectral_eigenvalues(self):
        return self.spectral_geometry.spectral_eigenvalues if self.spectral_geometry is not None else None

    @property
    def local_smoothness_weights(self):
        return self.spectral_geometry.local_smoothness_weights if self.spectral_geometry is not None else None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Legacy checkpoints stored geometry directly on every filter. Keep
        # their saved basis authoritative; only the unused hard operator drops.
        for name in ("basis", "roughness", "exposure_profile",
                     "spectral_eigenvalues", "local_smoothness_weights"):
            old_key = prefix + name
            if old_key in state_dict:
                value = state_dict.pop(old_key)
                if self.mode == "hard" and name == "roughness":
                    continue
                new_key = prefix + "spectral_geometry." + name
                if new_key in state_dict and not torch.equal(state_dict[new_key], value):
                    error_msgs.append("Conflicting legacy and shared geometry: " + old_key)
                else:
                    state_dict[new_key] = value
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _normalized_kernel(self, length):
        if not 1 <= length <= self.seq_len:
            raise ValueError(f"length must be in [1, {self.seq_len}]")
        raw = self.coeffs @ self.basis[:, :length]
        raw = raw.T
        return raw / (raw.abs().sum(dim=0, keepdim=True) + self.normalization_epsilon)

    def filter(self, L):
        return (self._normalized_kernel(L),)

    def forward(self, x, L, k=None, bias=None):
        if k is None:
            k_ch = self.filter(L)[0].T
        elif k.dim() == 2 and k.shape == (x.size(1), L):
            k_ch = k
        elif k.dim() == 2 and k.shape == (L, x.size(1)):
            k_ch = k.T
        else:
            raise RuntimeError(
                f"provided kernel shape is incompatible with {x.size(1)} channels and length {L}"
            )
        return fftconv(
            x,
            k_ch,
            D=torch.zeros(x.size(1), device=x.device, dtype=x.dtype),
        )[..., :L]


class HyenaOperator(nn.Module):
    
    def __init__(self, d_model, l_max, order=2, filter_order=64,
                 dropout=0.0, filter_dropout=0.0, discrete_orthogonal=False,
                 finite_history_mode="none", finite_history_rank=16,
                 finite_history_difference_order=2, finite_history_alpha=1.0,
                 finite_history_epsilon=1e-3, finite_history_profile=None,
                 layer_idx=0, spectral_geometry_cache=None, **layer_metadata):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)
        self.short_filter = nn.Sequential(
            nn.Conv1d(inner_width, 2 * inner_width, 1, groups=inner_width),
            nn.GLU(dim=1),
        )
        self.register_buffer("_folded_kernel", None, persistent=False)
        self._folded_kernel_length = None
        channels = d_model * (order - 1)
        if finite_history_mode == "hard":
            self.filter_fn = SpectralKernelFilter(
                channels, seq_len=l_max, mode="hard", rank=finite_history_rank,
                exposure_profile=finite_history_profile,
                difference_order=finite_history_difference_order,
                alpha=finite_history_alpha, epsilon=finite_history_epsilon,
                spectral_geometry_cache=spectral_geometry_cache,
            )
        elif finite_history_mode == "none":
            self.filter_fn = LegendreKernelFilter(
                channels, seq_len=l_max, order=filter_order,
                dropout=filter_dropout, discrete_orthogonal=discrete_orthogonal,
            )
        else:
            raise ValueError("Supported kernel modes are none and hard")

    @torch.no_grad()
    def fold_filter_for_inference(self, L=None):
        """Precompute the final kernel so serving keeps only the FFT convolution."""
        if L is None:
            L = self.l_max
        if L < 1 or L > self.l_max:
            raise ValueError(f"L must be in [1, {self.l_max}], got {L}")
        self._folded_kernel = self.filter_fn.filter(L)[0].detach()
        self._folded_kernel_length = L
        return self

    def clear_folded_filter(self):
        self._folded_kernel = None
        self._folded_kernel_length = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.clear_folded_filter()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.clear_folded_filter()
        return self

    def forward(self, u, *args, valid_mask=None, **kwargs):
        l = u.size(-2)
        if l > self.l_max:
            raise ValueError('Input sequence exceeds configured maximum length')
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u = rearrange(u, 'b l d -> b d l')

        uc = self.short_filter(u)[...,:l_filter]
        *x, v = uc.split(self.d_model, dim=1)

        if (
            not self.training
            and self._folded_kernel is not None
            and self._folded_kernel_length == l_filter
        ):
            k = self._folded_kernel
        else:
            k = self.filter_fn.filter(l_filter)[0]
        k = rearrange(k, 'l (o d) -> o d l', o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o])

        y = rearrange(v * x[0], 'b d l -> b l d')

        y = self.out_proj(y)
        return y


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, activation=F.gelu,
                 return_residual=False, device=None, dtype=None):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/mlp.py
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class HyenaRecBlock(nn.Module):
    """Pre-normalized Hyena mixing and feed-forward residual block.

    Residual layout follows the FlashAttention block implementation.
    """
    def __init__(self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm,
                 resid_dropout1=0.0, resid_dropout2=0.0, prenorm=True,
                 residual_in_fp32=False):
        super().__init__()
        if not prenorm:
            raise ValueError("HyenaRec uses pre-normalized residual blocks")
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = mixer_cls()
        self.dropout1 = nn.Dropout(resid_dropout1)
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        self.dropout2 = nn.Dropout(resid_dropout2)
        self.norm2 = norm_cls(dim)

    def forward(self, hidden_states, residual=None):
        dropped = self.dropout1(hidden_states)
        residual = dropped + residual if residual is not None else dropped
        hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states)
        residual = self.dropout2(hidden_states) + residual
        hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        return self.mlp(hidden_states), residual


def create_mixer_cls(layer=None, attn_layer_idx=None, attn_cfg=None,
                     layer_idx=None, device=None, dtype=None):
    if attn_layer_idx:
        raise ValueError("HyenaRec uses Hyena blocks at every layer")
    return partial(HyenaOperator, **layer, layer_idx=layer_idx)


def create_mlp_cls(d_model, d_inner=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    inner_dim = d_inner if d_inner is not None else 4 * d_model

    mlp_cls = partial(Mlp, hidden_features=inner_dim,
                          activation=partial(F.gelu, approximate='tanh'), **factory_kwargs)

    return mlp_cls


def create_block(d_model, d_inner=None,
                 layer=None, attn_layer_idx=None,
                 attn_cfg=None, layer_norm_epsilon=1e-5,
                 resid_dropout1=0.0, resid_dropout2=0.0, residual_in_fp32=False,
                 layer_idx=None,
                 device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    mixer_cls = create_mixer_cls(layer=layer,
                                 attn_layer_idx=attn_layer_idx,
                                 attn_cfg=attn_cfg, layer_idx=layer_idx,
                                 **factory_kwargs)
    mlp_cls = create_mlp_cls(d_model, d_inner=d_inner,
                             **factory_kwargs)
    norm_cls = partial(nn.LayerNorm, eps=layer_norm_epsilon, **factory_kwargs)
    
    
    block = HyenaRecBlock(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls,
                  prenorm=True, resid_dropout1=resid_dropout1, resid_dropout2=resid_dropout2,residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True,
                  glu_act=False):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if not glu_act:
                    nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
                else:
                    out_features = p.shape[0]
                    # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                    # on average.
                    nn.init.normal_(p[:out_features // 2], mean=0.0, std=initializer_range / math.sqrt(2 * n_layer) * 2)


class ItemEmbeddings(nn.Module):
    """Item lookup table. Temporal structure is represented by convolution kernels."""
    def __init__(self, embed_dim, vocab_size, max_position_embeddings=0,
                 padding_idx=None, device=None, dtype=None):
        super().__init__()
        if max_position_embeddings > 0:
            raise ValueError("The reported HyenaRec architecture uses kernel-based lag encoding")
        self.word_embeddings = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx, device=device, dtype=dtype)

    def forward(self, input_ids, position_ids=None):
        return self.word_embeddings(input_ids)


class HyenaRecBackbone(nn.Module):

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int,
                 process_group=None, layer=None,
                 attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None,residual_in_fp32=False,
                 device=None, dtype=None, share_spectral_geometry=True, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.process_group = process_group
        self.residual_in_fp32 = residual_in_fp32
        # note max_position_embeddings is 0 for Hyena, and therefore isn't used
        self.embeddings = ItemEmbeddings(d_model, vocab_size, max_position_embeddings,
                                             **factory_kwargs)

        # This cache lives only during construction of this backbone. The
        # shared module remains owned by its filters, including across .to().
        mixer_layer = dict(layer) if layer is not None else layer
        if (share_spectral_geometry and mixer_layer is not None
                and mixer_layer.get("finite_history_mode") == "hard"):
            mixer_layer["spectral_geometry_cache"] = {}

        self.layers = nn.ModuleList([create_block(
            d_model, d_inner=d_inner,
            layer=mixer_layer, attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg, layer_norm_epsilon=layer_norm_epsilon,
            resid_dropout1=embed_dropout if i == 0 else resid_dropout,
            resid_dropout2=resid_dropout, residual_in_fp32=residual_in_fp32,layer_idx=i,
            **factory_kwargs,
        ) for i in range(n_layer)])

        self.drop_f = nn.Dropout(resid_dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon, **factory_kwargs)

        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Shared storage cannot represent distinct per-layer legacy bases.
        # Reject such a checkpoint rather than letting the last copy win.
        seen = {}
        for name, module in self.named_modules():
            if not isinstance(module, SpectralKernelFilter) or module.mode != "hard":
                continue
            for field in ("basis", "exposure_profile", "spectral_eigenvalues",
                          "local_smoothness_weights"):
                field_prefix = prefix + name + "."
                value = state_dict.get(field_prefix + "spectral_geometry." + field)
                if value is None:
                    value = state_dict.get(field_prefix + field)
                if value is None:
                    continue
                key = (id(module.spectral_geometry), field)
                if key in seen and not torch.equal(seen[key], value):
                    error_msgs.append("Inconsistent checkpoint copies of shared spectral "
                                      + field + " at " + field_prefix
                                      + "; load with share_spectral_geometry=False to preserve distinct copies")
                else:
                    seen[key] = value
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids,)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))

        return hidden_states


class HyenaRec(nn.Module):
    """Dataset-aware sequential recommender with a shared Hyena backbone.

    Video, Steam and ML-1M score the full vocabulary using tied item weights.
    XLong scores explicit candidates using its independent output embedding.
    """

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int,
                 layer=None, attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.2, embed_dropout: float = 0.2,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None,residual_in_fp32=False,
                 pad_vocab_size_multiple: int = 1, use_head=False, n_classes: int = 2,
                 device=None, dtype=None, dataset_code=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.dataset_code = dataset_code
        if dataset_code == "xlong":
            self.num_items = n_classes
            self.backbone = HyenaRec(
                d_model=d_model, n_layer=n_layer, d_inner=d_inner, vocab_size=vocab_size,
                layer=layer, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
                max_position_embeddings=max_position_embeddings,
                resid_dropout=resid_dropout, embed_dropout=embed_dropout,
                layer_norm_epsilon=layer_norm_epsilon, initializer_cfg=initializer_cfg,
                residual_in_fp32=residual_in_fp32,
                pad_vocab_size_multiple=pad_vocab_size_multiple, use_head=False,
                n_classes=n_classes, device=device, dtype=dtype, **kwargs,
            )
            self.output = nn.Embedding(n_classes + 2, d_model, **factory_kwargs)
            nn.init.normal_(self.output.weight, std=0.02)
            self.bias = nn.Parameter(torch.zeros(n_classes + 2, **factory_kwargs))
            return
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        self.use_head = use_head
        
        # Match the tied output projection width after vocabulary padding.
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))

        if 'd_model' not in layer:
            layer['d_model'] = d_model

        self.backbone = HyenaRecBackbone(
            d_model=d_model, n_layer=n_layer, d_inner=d_inner, vocab_size=vocab_size,
            layer=layer, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout, embed_dropout=embed_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_cfg=initializer_cfg, residual_in_fp32=residual_in_fp32,
            **factory_kwargs, **kwargs
        )

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))

    @torch.no_grad()
    def fold_filters_for_inference(self):
        """Switch to eval mode and precompute every Hyena layer's final kernel."""
        self.eval()
        for module in self.modules():
            if isinstance(module, HyenaOperator):
                module.fold_filter_for_inference()
        return self

    

    def forward(self, input_ids, position_ids=None, state=None, *, targets=None,
                negative_samples=None, last_only=False):
        if self.dataset_code == "xlong":
            if targets is None or negative_samples is None:
                raise ValueError("XLong scoring requires targets and explicit negative samples")
            hidden = self.backbone(input_ids, position_ids=position_ids)
            if last_only:
                hidden = hidden[:, -1, :]
                candidates = torch.cat((negative_samples, targets[:, None]), dim=1)
            else:
                candidates = torch.cat((negative_samples, targets[:, :, None]), dim=2)
            sampled = self.output(candidates)
            return (hidden.unsqueeze(-2) * sampled).sum(-1) + self.bias[candidates]
        hidden_states = self.backbone(input_ids, position_ids=position_ids)

        if self.use_head:
            return torch.matmul(hidden_states, self.backbone.embeddings.word_embeddings.weight.permute(1, 0)) + self.bias
        
        else:
            return hidden_states
