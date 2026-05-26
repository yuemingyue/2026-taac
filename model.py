"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Union

# ─── Optional FlashAttention (Tri Dao) varlen kernel ──────────────────────────
# We lazily import so that CPU / non-CUDA environments keep working.
try:
    from flash_attn import flash_attn_varlen_func  # type: ignore
    from flash_attn.bert_padding import unpad_input, pad_input  # type: ignore
    _HAS_FLASH_ATTN = True
except Exception:  # pragma: no cover - depends on external package availability
    flash_attn_varlen_func = None
    unpad_input = None
    pad_input = None
    _HAS_FLASH_ATTN = False


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    # 9-column UTC+8 calendar id tensor of shape (B, 9). When time-feature
    # injection is disabled the trainer will pass an empty tensor (B, 0).
    time_feats: torch.Tensor = torch.empty(0, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
        use_flash_varlen: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout
        # Whether we are *allowed* to use FlashAttention varlen kernel for
        # padding-masked self-attention. Runtime gate (CUDA + fp16/bf16 + package
        # available) is applied on every forward.
        self.use_flash_varlen = use_flash_varlen and _HAS_FLASH_ATTN

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 3.5. Fast path: FlashAttention varlen for padding-masked self-attention.
        # Triggered when: flash-attn installed, enabled by user, we are on CUDA
        # with fp16/bf16 weights, there's a key_padding_mask (i.e. actual padding
        # to skip), no explicit additive attn_mask, and this is genuine
        # self-attention (same Q/K length, shared mask). Cross-attention with
        # different Q/K lengths (LongerEncoder cross mode) is intentionally
        # excluded here and continues down the SDPA path below.
        can_flash_varlen = (
            self.use_flash_varlen
            and flash_attn_varlen_func is not None
            and Q.is_cuda
            and Q.dtype in (torch.float16, torch.bfloat16)
            and key_padding_mask is not None
            and attn_mask is None
            and Lq == Lk
            and q_rope_cos is None  # no Q-specific RoPE gathering
        )
        if can_flash_varlen:
            # flash_attn_varlen_func expects tensors in (total_tokens, num_heads,
            # head_dim) layout, so convert back from (B, H, L, D).
            # key_padding_mask: (B, L), True = padding  →  attention_mask True = valid.
            attn_valid = ~key_padding_mask  # (B, L)

            # Unpad Q, K, V using the SAME mask (self-attention assumption).
            # Permute to (B, L, H, D) so unpad_input can flatten on the batch×L axis.
            q_bld = Q.transpose(1, 2).contiguous()  # (B, L, H, D)
            k_bld = K.transpose(1, 2).contiguous()
            v_bld = V.transpose(1, 2).contiguous()

            q_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(q_bld, attn_valid)
            # K/V share indices with Q, but unpad_input recomputes cheaply.
            k_unpad, _, _, _, _ = unpad_input(k_bld, attn_valid)
            v_unpad, _, _, _, _ = unpad_input(v_bld, attn_valid)

            dropout_p = self.dropout if self.training else 0.0
            out_unpad = flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=dropout_p,
                causal=False,
            )  # (total_tokens, H, D)

            # Re-pad back to (B, L, H, D)
            out_bld = pad_input(out_unpad, indices, B, Lq)  # (B, L, H, D)
            out = out_bld.transpose(1, 2)  # (B, H, L, D)

            # NOTE: unpad_input already zeroed padding rows; and fully padded rows
            # (if any) come back as zeros from pad_input, matching the nan_to_num
            # behavior of the SDPA path below.

            # 6. Reshape back and output projection
            out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
            G = self.W_g(query)
            out = out * torch.sigmoid(G)
            out = self.W_o(out)
            return out, None

        # 4. Convert key_padding_mask to SDPA format
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = padding
            # SDPA expects (B, 1, 1, Lk) bool mask, True = attend
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # ─── 方案 A: 加性 float bias 支持 ──────────────────────────────
            # attn_mask 既可能是“2D bool 风格的 -inf 屏蔽”也可能是“真加性 bias”
            # (例如 target_attn_bias 透传过来的 (B, Lk) 或 (B, Lq, Lk))。两种
            # 写法在 PyTorch SDPA 里其实是“float additive mask”同一接口的两端,
            # 这里统一按加性 float mask 处理: 把 padding 位置补一个有限大负数,
            # 其它位置直接相加, 让 softmax 自然吸收 prior. 注意: 用 -1e4 而非
            # -inf, 因为在 bf16 下后者会让 softmax 输出 NaN.
            NEG_LARGE = -1e4
            if attn_mask.dtype == torch.bool:
                # 兼容旧调用: bool mask, True = 允许 / False = 屏蔽
                # 转成 float additive mask: 屏蔽位 -1e4, 允许位 0.
                bool_attn = attn_mask  # (Lq, Lk) or broadcastable
                add_mask = torch.zeros_like(bool_attn, dtype=Q.dtype)
                add_mask = add_mask.masked_fill(~bool_attn, NEG_LARGE)
            else:
                add_mask = attn_mask.to(Q.dtype)
                # float 加性 mask 内部可能含 -inf (例如 nn.Transformer.
                # generate_square_subsequent_mask 的 causal mask), 替换为 -1e4
                # 以保持 bf16 数值稳定.
                add_mask = torch.nan_to_num(
                    add_mask, nan=0.0, posinf=NEG_LARGE * -1.0, neginf=NEG_LARGE)
            # ─── 升维到 (B, num_heads, Lq, Lk) ────────────────────────────
            # 必须显式按 “最后两维=Lq,Lk; 倒数第三维=head 数” 的语义补齐,
            # 不能靠 unsqueeze(0) 循环补 (那样会把 B 推到 head 维, 后续
            # expand 会因为 “非 1 维要求扩到 num_heads” 报 RuntimeError).
            if add_mask.dim() == 2:
                # 形如 (Lq, Lk) 的纯位置 mask (例如 causal mask)
                add_mask = add_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, Lq, Lk)
            elif add_mask.dim() == 3:
                # 可能是 (B, Lq, Lk) 或 (B, 1, Lk)
                # 在 head 维 (dim=1) 处插入单一通道, 让 expand 把它广播到 num_heads
                add_mask = add_mask.unsqueeze(1)               # (B, 1, *, Lk)
            elif add_mask.dim() != 4:
                raise ValueError(
                    f"attn_mask must be 2D / 3D / 4D, got shape {tuple(attn_mask.shape)}"
                )
            add_mask = add_mask.expand(B, self.num_heads, Lq, Lk)
            # 与 padding mask 融合: padding 位补 NEG_LARGE
            if sdpa_attn_mask is not None:
                # sdpa_attn_mask 当前是 bool (B, H, Lq, Lk), True=attend
                add_mask = add_mask.masked_fill(~sdpa_attn_mask, NEG_LARGE)
            sdpa_attn_mask = add_mask  # 替换为 float additive mask

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve original input via residual)
        out = torch.nan_to_num(out, nan=0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre',
        use_flash_varlen: bool = False,
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
            use_flash_varlen=use_flash_varlen,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        attn_score_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.
            attn_score_bias: (B, L) or (B, Nq, L) float, 加在 softmax 之前的
                target-aware additive bias. ``None`` 时退化为标准 cross-attn.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        # 把 (B, L) 的 bias 升维成 (B, 1, L) 以广播到全部 Nq query.
        attn_mask_for_attn = None
        if attn_score_bias is not None:
            if attn_score_bias.dim() == 2:
                attn_mask_for_attn = attn_score_bias.unsqueeze(1)  # (B, 1, L)
            else:
                attn_mask_for_attn = attn_score_bias  # (B, Nq, L)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask_for_attn,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class TargetAttnBiasModule(nn.Module):
    """方案 A: target-aware additive bias 生成器, 替代破坏性的 “seq 加权改写”.

    与 BehaviorImportanceWeighting 的根本区别:
    * BehaviorImportanceWeighting: softmax 归一化后乘到 seq_tokens 上,
      均值 1/L, 整体尺度被压缩, 而且与 DIN 的 “target-aware 残差” 信息重叠;
    * TargetAttnBiasModule: 输出 *未归一化* 的 raw bias, 直接加到 cross-attn
      softmax 之前的 logits 上, 让 attention 自由学习 “是否要听 target prior”.
      与 DIN / cross-attn 信息互补 (DIN: output 残差; cross-attn: 自由学习;
      bias: target prior 引导).

    输入
    ----
    target  : (B, D)        来自 user_ns + item_ns 的 target 摘要
    seq     : (B, L, D)     该 domain 的历史行为序列 (注意是 sequence_encoder
                             *之后* 的表示, 与 cross-attn 看到的 K 同源)
    padding : (B, L) bool   True = padding

    输出
    ----
    bias    : (B, L)        加性 bias, padding 位置已被填 -inf
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = int(d_model)
        # 用 target-conditioned dot-product 算 bias: 简洁、参数少、与 DIN
        # 不冗余 (DIN 内部是另一套 Q/K/V), 这里 q/k 是 *外部* 注入的 prior.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        # 对 seq 做一个轻量 LN, 避免不同 block 的 seq 表示尺度漂移影响 prior.
        self.k_norm = nn.LayerNorm(d_model)
        self.q_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._inv_sqrt_d = float(d_model) ** -0.5

    def forward(
        self,
        target: torch.Tensor,
        seq: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # (B, D) -> (B, 1, D)
        q = self.q_proj(self.q_norm(target)).unsqueeze(1)        # (B, 1, D)
        k = self.k_proj(self.k_norm(seq))                         # (B, L, D)
        bias = (q * k).sum(dim=-1) * self._inv_sqrt_d             # (B, L)
        bias = self.dropout(bias)
        # padding 位置填一个有限大负数 (而非 -inf), 避免 bf16 amp 下 softmax
        # 出现 NaN. -1e4 在 bf16 仍可表示, exp(-1e4)≈0, 效果等价于 -inf.
        # (key_padding_mask 路径会另外把整行填 -inf 处理全 padding 行, 我们
        # 这里只处理本路径的 bias 加性贡献.)
        NEG_LARGE = -1e4
        bias = bias.masked_fill(padding_mask, NEG_LARGE)
        # 防御: 如果某行全 padding, NEG_LARGE 加到 attn_scores 上仍能让
        # softmax 正常工作, 不必额外置零; 这里只是兜底, 万一上游传入异常 mask.
        all_pad = padding_mask.all(dim=-1, keepdim=True)          # (B, 1)
        bias = torch.where(all_pad, torch.zeros_like(bias), bias)
        return bias


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block.

    Performs three steps:
    1. Token Mixing: Parameter-free tensor reshaping.
    2. Per-token FFN: Shared-parameter feedforward network.
    3. Residual connection: Q_boost = Q + Q_e.

    Constraint: d_model must be divisible by n_total in 'full' mode.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full'  # 'full' | 'ffn_only' | 'none'
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == 'none':
            # Pure identity mapping, no submodules created
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        # Per-token FFN (shared parameters) — used by both 'full' and 'ffn_only'
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        # Post-LN after residual to stabilize stacked block outputs
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Steps:
        1. Splits channels into T subspaces: (B, T, D) -> (B, T, T, d_sub).
        2. Swaps token and subspace axes: (B, token, h, d_sub) -> (B, h, token, d_sub).
        3. Flattens back: (B, T, D).

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection.

        Args:
            Q: (B, T, D) where T = Nq + Nns.

        Returns:
            Boosted tensor of shape (B, T, D).
        """
        if self.mode == 'none':
            return Q

        # Token Mixing (parameter-free rewire) or identity
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # Per-token FFN
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        # Residual from original Q
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class ImportanceWeightedQueryGenerator(nn.Module):
    """重要性加权的query生成模块
    
    为每个序列的token计算重要性权重，基于目标物品和序列token的交互
    然后对序列进行重新加权，再输入到query生成器
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4,
        attention_heads: int = 4,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model
        self.attention_heads = attention_heads
        
        # 目标物品与序列token的交互注意力机制
        self.target_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 重要性权重计算网络
        self.importance_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * hidden_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * hidden_mult, 1),
            nn.Sigmoid()
        )
        
        # 标准化层
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 原始query生成器（保持原有逻辑）
        global_info_dim = (num_ns + 1) * d_model
        self.global_info_norm = nn.LayerNorm(global_info_dim)
        
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def _compute_importance_weights(
        self,
        target_tokens: torch.Tensor,
        seq_tokens: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """计算序列中每个token的重要性权重"""
        B, L, D = seq_tokens.shape
        
        # 目标物品表示（平均池化）
        target_repr = target_tokens.mean(dim=1, keepdim=True)  # (B, 1, D)
        
        # 扩展目标表示以匹配序列长度
        target_expanded = target_repr.expand(B, L, D)  # (B, L, D)
        
        # 拼接目标表示和序列token
        interaction_input = torch.cat([target_expanded, seq_tokens], dim=-1)  # (B, L, 2*D)
        
        # 计算重要性权重
        importance_weights = self.importance_mlp(interaction_input).squeeze(-1)  # (B, L)
        
        # 对padding位置应用mask
        importance_weights = importance_weights.masked_fill(padding_mask, 0.0)
        
        # 归一化权重
        weight_sum = importance_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        normalized_weights = importance_weights / weight_sum
        
        return normalized_weights

    def _apply_importance_weighting(
        self,
        seq_tokens: torch.Tensor,
        importance_weights: torch.Tensor
    ) -> torch.Tensor:
        """应用重要性权重到序列token"""
        # 重要性权重扩展以匹配token维度
        weights_expanded = importance_weights.unsqueeze(-1)  # (B, L, 1)
        
        # 加权后的序列表示
        weighted_seq = seq_tokens * weights_expanded  # (B, L, D)
        
        # 聚合加权表示（加权平均）
        seq_pooled = weighted_seq.sum(dim=1)  # (B, D)
        
        return seq_pooled

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list
    ) -> list:
        """生成重要性加权的query tokens"""
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)
        
        q_tokens_list = []
        for i in range(self.num_sequences):
            # 1. 计算重要性权重
            importance_weights = self._compute_importance_weights(
                ns_tokens, seq_tokens_list[i], seq_padding_masks[i]
            )
            
            # 2. 应用重要性加权
            seq_pooled = self._apply_importance_weighting(
                seq_tokens_list[i], importance_weights
            )  # (B, D)
            
            # 3. 生成query tokens（保持原有逻辑）
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)
            
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)
        
        return q_tokens_list


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model

        global_info_dim = (num_ns + 1) * d_model

        # LayerNorm on global_info to prevent gradient explosion from large-dim concat
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # Each sequence has N independent FFNs
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            # MeanPool(Seq_i)
            valid_mask = ~seq_padding_masks[i]  # True = valid
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L_i, 1)
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)  # (B, D)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = Concat(NS_flat, seq_pooled_i)
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)

            # Generate N query tokens
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        use_flash_varlen: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
            use_flash_varlen=use_flash_varlen,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask

class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask) so downstream can update the mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
        use_flash_varlen: bool = False,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # Pre-LN for attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
            use_flash_varlen=use_flash_varlen,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects the latest top_k valid tokens from each sample.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # Start position for each sample: max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # Build gather indices: (B, top_k)
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # For samples with valid_len < top_k, early indices may exceed valid range;
        # clamp to [0, L-1] and handle via mask below
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # New padding mask: first (top_k - actual_k) positions are padding
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # position_indices for Q-side RoPE
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False,
    use_flash_varlen: bool = False,
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', or 'longer'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).
        use_flash_varlen: Enable Flash-Attention varlen kernel for
            padding-masked self-attention (CUDA + fp16/bf16 only, else auto
            fallback to SDPA).

    Returns:
        A sequence encoder module.
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout,
                                  use_flash_varlen=use_flash_varlen)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout,
                             causal, use_flash_varlen=use_flash_varlen)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full',
        use_flash_varlen: bool = False,
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal,
                use_flash_varlen=use_flash_varlen,
            )
            for _ in range(num_sequences)
        ])

        # Independent cross-attention per sequence
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre',
                use_flash_varlen=use_flash_varlen,
            )
            for _ in range(num_sequences)
        ])

        # RankMixer: input token count = Nq * S + Nns
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
        cross_attn_bias_list: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.
            cross_attn_bias_list: 可选, 长度 S 的列表, 每项 (B, L_i) 或
                (B, Nq, L_i), target-aware additive bias 注入到 cross-attn
                的 softmax 之前. ``None`` 时退化为标准 cross-attn.

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            bias_i = (cross_attn_bias_list[i]
                      if cross_attn_bias_list is not None else None)
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs,
                attn_score_bias=bias_i,
            )
            decoded_qs.append(decoded_q_i)

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)

        # 5. Split back into per-sequence Q and NS
        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset:offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# Calendar / Time-feature embedding helper
# ═══════════════════════════════════════════════════════════════════════════════


# Vocab sizes for the 9 calendar id columns produced by the dataset
# (see ``dataset._build_calendar_int_feats``). Order:
#   0: minute_of_day  (0..1439)
#   1: hour_of_day    (0..23)
#   2: day_of_week    (0..6, Mon=0)
#   3: hour_of_week   (0..167)
#   4: day_of_month   (1..31)
#   5: month_of_year  (1..12)
#   6: day_of_year    (1..366)
#   7: is_weekend     (0..1)
#   8: part_of_day    (0..3)
CALENDAR_VOCAB_SIZES: List[int] = [1440, 24, 7, 168, 32, 13, 367, 2, 4]
CALENDAR_NUM_COLUMNS: int = len(CALENDAR_VOCAB_SIZES)


class CalendarTimeEmbedding(nn.Module):
    """Looks up 9 calendar id columns and fuses them into one (B, D) token.

    Each column has its own ``nn.Embedding`` table sized exactly to the
    column vocabulary; the 9 lookups are concatenated and passed through a
    bottleneck projection (Linear → Dropout → LayerNorm → SiLU) to obtain a
    single ``d_model`` representation. The returned tensor is ready to be
    broadcast as a residual onto the user NS tokens.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.tables = nn.ModuleList([
            nn.Embedding(vs, d_model) for vs in CALENDAR_VOCAB_SIZES
        ])
        self.fuse = nn.Sequential(
            nn.Linear(d_model * CALENDAR_NUM_COLUMNS, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

    def forward(self, calendar_ids: torch.Tensor) -> torch.Tensor:
        """calendar_ids: (B, 9) int64 -> (B, d_model)."""
        embs = [self.tables[i](calendar_ids[:, i]) for i in range(CALENDAR_NUM_COLUMNS)]
        return self.fuse(torch.cat(embs, dim=-1))


# ═══════════════════════════════════════════════════════════════════════════════
# User Dense feature projector (typed group split)
# ═══════════════════════════════════════════════════════════════════════════════


class UserDenseGroupedProjector(nn.Module):
    """Splits the 918-d user_dense vector into 3 typed groups and fuses them.

    The TAAC user_dense layout puts two pre-trained semantic embeddings in
    the middle of a stat block:
        [0:256]   = SUM embedding         (fid 61)
        [256:568] = stat slice 1          (fid 62/63/64/65/66, total 5+11+18+49+66=149??)
        [568:888] = LMF4Ads embedding     (fid 87)
        [888:]    = stat slice 2          (fid 89/90/91)

    The two stat slices are concatenated to recover the "plain dense" stream,
    which is independently projected; the two pre-trained embedding blocks
    each get their own projection. The three projected vectors are summed
    and passed through SiLU so the result remains a single (B, 1, D) NS
    token, preserving the existing NS-token count.
    """

    EMB61_DIM = 256
    EMB87_DIM = 320
    # Inclusive end of the first stat slice (also exclusive start of emb87).
    STAT1_END = 568
    # Inclusive start of the second stat slice (exclusive end of emb87).
    STAT2_START = 888

    def __init__(self, total_dim: int, d_model: int) -> None:
        super().__init__()
        self.total_dim = total_dim
        self.d_model = d_model
        # The plain-dense stream covers everything that is not emb61/emb87.
        plain_dim = total_dim - self.EMB61_DIM - self.EMB87_DIM
        if plain_dim <= 0:
            raise ValueError(
                f"UserDenseGroupedProjector: total_dim={total_dim} is too small "
                f"to host both pretrained embedding blocks (256 + 320)."
            )
        self.plain_dim = plain_dim

        self.plain_proj = nn.Sequential(
            nn.Linear(plain_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.emb61_proj = nn.Sequential(
            nn.Linear(self.EMB61_DIM, d_model),
            nn.LayerNorm(d_model),
        )
        self.emb87_proj = nn.Sequential(
            nn.Linear(self.EMB87_DIM, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, dense: torch.Tensor) -> torch.Tensor:
        """dense: (B, total_dim) -> (B, 1, d_model)."""
        emb61_block = dense[:, :self.EMB61_DIM]
        stat_head = dense[:, self.EMB61_DIM:self.STAT1_END]
        emb87_block = dense[:, self.STAT1_END:self.STAT2_START]
        stat_tail = dense[:, self.STAT2_START:]

        plain = torch.cat([stat_head, stat_tail], dim=-1)
        token = (
            self.plain_proj(plain)
            + self.emb61_proj(emb61_block)
            + self.emb87_proj(emb87_block)
        )
        return F.silu(token).unsqueeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# DIN target-aware activation modules
# ═══════════════════════════════════════════════════════════════════════════════


class DINQueryBuilder(nn.Module):
    """Builds a single (B, D) candidate-aware query vector for DIN.

    Pools user-side and item-side static tokens via mean-pooling and feeds
    a 4-way interaction signal through a small MLP:
        cat(item_pool, user_pool, item_pool * user_pool, |item_pool - user_pool|)
    The output acts as the target query vector against which each behavior
    sequence is attended in :class:`DINInterestActivation`.
    """

    def __init__(self, d_model: int, hidden_mult: int = 2) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_model, d_model * hidden_mult),
            nn.SiLU(),
            nn.Linear(d_model * hidden_mult, d_model),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def _mean_pool(tokens: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tokens, dim=1).mean(dim=1)

    def forward(
        self,
        user_tokens: List[torch.Tensor],
        item_tokens: List[torch.Tensor],
    ) -> torch.Tensor:
        u = self._mean_pool(user_tokens)
        v = self._mean_pool(item_tokens)
        feats = torch.cat([v, u, v * u, torch.abs(v - u)], dim=-1)
        return self.mlp(feats)


class DINInterestActivation(nn.Module):
    """Target-aware DIN activation across multiple behavior domains.

    For every domain we project sequence tokens into K/V, score them against
    the DIN query, optionally keep only the top-k highest-scoring positions,
    softmax-normalize, and read out a context vector. A learned soft gate
    over domains (computed from query-context interaction features) merges
    the per-domain contexts into a single matched context, which is finally
    combined with the query through another 4-way interaction MLP to produce
    a residual delta added on top of the backbone embedding.

    The final delta-producing MLP is zero-initialized so the DIN branch
    starts as an identity map and only gradually contributes during training.
    """

    def __init__(
        self,
        d_model: int,
        num_sequences: int,
        hidden_mult: int = 2,
        dropout: float = 0.0,
        top_k: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_sequences = num_sequences
        self.top_k = int(top_k)
        self._inv_sqrt_d = 1.0 / math.sqrt(d_model)

        # Shared Q projection; domain-specific K/V projections + LN.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_projs = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_sequences)])
        self.v_projs = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_sequences)])
        self.ctx_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_sequences)])

        gate_hidden = max(d_model, d_model * hidden_mult)
        self.gate_mlp = nn.Sequential(
            nn.Linear(4 * d_model, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )

        delta_hidden = max(d_model, d_model * hidden_mult)
        self.delta_mlp = nn.Sequential(
            nn.Linear(4 * d_model, delta_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(delta_hidden, d_model),
        )
        # Zero-init the final layer so DIN starts as a no-op residual.
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)

    def _per_domain_context(
        self,
        query: torch.Tensor,
        seq_tokens: torch.Tensor,
        seq_padding_mask: torch.Tensor,
        domain_idx: int,
    ) -> torch.Tensor:
        """Compute the matched context for a single domain.

        Args:
            query: (B, D) DIN query vector.
            seq_tokens: (B, L, D) tokens of the behavior sequence.
            seq_padding_mask: (B, L) bool, True where padding.
            domain_idx: index of the domain (selects K/V projections).
        Returns:
            (B, D) context tensor, normalized by the per-domain LayerNorm.
        """
        q_vec = self.q_proj(query).unsqueeze(1)              # (B, 1, D)
        k_mat = self.k_projs[domain_idx](seq_tokens)         # (B, L, D)
        v_mat = self.v_projs[domain_idx](seq_tokens)         # (B, L, D)

        scores = (q_vec * k_mat).sum(dim=-1) * self._inv_sqrt_d  # (B, L)
        scores = scores.masked_fill(seq_padding_mask, float('-inf'))

        # Optional top-k truncation: positions outside the top-k are forced
        # to -inf before softmax.
        L = scores.shape[-1]
        if self.top_k > 0 and self.top_k < L:
            top_vals, top_idx = torch.topk(scores, k=self.top_k, dim=-1)
            keep = scores.new_full(scores.shape, float('-inf'))
            scores = keep.scatter(-1, top_idx, top_vals)

        attn = F.softmax(scores, dim=-1)
        # All-padding rows would yield NaN softmax → zero them out.
        attn = torch.nan_to_num(attn, nan=0.0)
        ctx = torch.bmm(attn.unsqueeze(1), v_mat).squeeze(1)  # (B, D)
        return self.ctx_norms[domain_idx](ctx)

    def forward(
        self,
        query: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
    ) -> torch.Tensor:
        domain_ctxs = []
        gate_logits = []
        for i in range(self.num_sequences):
            ctx = self._per_domain_context(
                query, seq_tokens_list[i], seq_padding_masks[i], i)
            domain_ctxs.append(ctx)
            interaction = torch.cat([
                query, ctx, query * ctx, torch.abs(query - ctx),
            ], dim=-1)
            gate_logits.append(self.gate_mlp(interaction))

        ctx_stack = torch.stack(domain_ctxs, dim=1)             # (B, S, D)
        gate_logit_mat = torch.cat(gate_logits, dim=-1)         # (B, S)
        gate_w = F.softmax(gate_logit_mat, dim=-1).unsqueeze(-1)  # (B, S, 1)
        merged_ctx = (ctx_stack * gate_w).sum(dim=1)            # (B, D)

        delta_in = torch.cat([
            query, merged_ctx, query * merged_ctx,
            torch.abs(query - merged_ctx),
        ], dim=-1)
        return self.delta_mlp(delta_in)


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    # Filtered high-cardinality feature: output zero vector
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # Single-value feature: direct lookup
                        fid_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
                    else:
                        # Multi-value feature: lookup then mean pooling (ignoring padding=0)
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
                        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Compute total embedding dim: sum of all fids across all groups
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # Per-chunk projection: chunk_dim -> d_model with LayerNorm
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns_tokens)
        ])

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        # 1. Embed all fids in group order → flat cat
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class AuxProjHead(nn.Module):
    """Two-layer MLP + LayerNorm + L2-normalize projection head for the
    auxiliary contrastive losses (InfoNCE / SupCon).

    The output is L2-normalized so cosine similarity reduces to a plain
    dot product downstream. Weights are initialized via the default
    PyTorch scheme; this head is treated as a *dense* parameter (AdamW)
    by ``get_dense_params`` because it contains no ``nn.Embedding``.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        # Cast back to fp32 before normalization for numerical stability
        # under autocast; downstream losses also run in fp32.
        z = F.normalize(z.float(), dim=-1)
        return z


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        # Acceleration
        use_flash_attn_varlen: bool = False,
        # Time-feature injection: when enabled, the model expects ``time_feats``
        # of shape (B, 9) inside ``ModelInput`` and adds a calendar-time
        # residual to the user_ns tokens.
        use_time_feats: bool = False,
        time_feats_dropout: float = 0.1,
        # User dense grouped projection (auto-detected when total dim is
        # large enough to host both pretrained embedding blocks).
        user_dense_grouped: bool = True,
        # DIN target-aware activation branch.
        use_din: bool = False,
        din_top_k: int = 32,
        din_hidden_mult: int = 2,
        # Auxiliary contrastive loss heads (InfoNCE / SupCon). When
        # ``use_aux_loss=False`` no extra parameters are created and
        # ``forward_with_aux`` returns ``aux=None`` so old checkpoints stay
        # bit-exactly compatible.
        use_aux_loss: bool = False,
        aux_proj_dim: int = 64,
        # Behavior importance weighting for query generation.
        use_importance_weighting: bool = False,
        importance_weighting_type: str = 'cross_attention',
        importance_dropout: float = 0.0,
        # ─── 方案 A: Target-aware Cross-Attention Bias ────────────────────
        # 不再原地改写 seq, 而是把 “target × 历史行为” 的相关度做成加性
        # bias, 注入 HyFormer 每层 cross-attn 的 softmax 之前. 与 DIN /
        # cross-attn 信息互补 (DIN 是 output 残差, cross-attn 是自由学习,
        # bias 是 target prior). 初始 ``alpha=0`` 退化为现有结构, 安全.
        use_target_attn_bias: bool = False,
        target_attn_bias_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.use_flash_attn_varlen = use_flash_attn_varlen and _HAS_FLASH_ATTN
        if use_flash_attn_varlen and not _HAS_FLASH_ATTN:
            logging.warning(
                "use_flash_attn_varlen=True but flash-attn is not installed; "
                "falling back to SDPA. Install via: pip install flash-attn --no-build-isolation"
            )
        self.use_time_feats = bool(use_time_feats)
        self.use_din = bool(use_din)
        self.use_aux_loss = bool(use_aux_loss)
        self.aux_proj_dim = int(aux_proj_dim)
        self.use_importance_weighting = bool(use_importance_weighting)
        self.importance_weighting_type = importance_weighting_type
        self.importance_dropout = importance_dropout
        self.use_target_attn_bias = bool(use_target_attn_bias)
        self.target_attn_bias_dropout = float(target_attn_bias_dropout)

        # ================== NS Tokens Construction ==================

        if ns_tokenizer_type == 'group':
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # User dense feature projection (if available)
        self.has_user_dense = user_dense_dim > 0
        # When the dense block is large enough to host the two pretrained
        # embedding banks (256 + 320), we split the vector into typed groups
        # and project them independently. Otherwise we fall back to a single
        # Linear+LN projection (the legacy behavior).
        self.user_dense_grouped = bool(user_dense_grouped) and self.has_user_dense and (
            user_dense_dim >= UserDenseGroupedProjector.EMB61_DIM + UserDenseGroupedProjector.EMB87_DIM + 1
        )
        if self.has_user_dense:
            if self.user_dense_grouped:
                self.user_dense_proj = UserDenseGroupedProjector(
                    total_dim=user_dense_dim, d_model=d_model)
            else:
                self.user_dense_proj = nn.Sequential(
                    nn.Linear(user_dense_dim, d_model),
                    nn.LayerNorm(d_model),
                )

        # Item dense feature projection (if available)
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # Total NS token count
        self.num_ns = (num_user_ns + (1 if self.has_user_dense else 0)
                       + num_item_ns + (1 if self.has_item_dense else 0))

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold or with no vocab info (vs<=0)."""
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # Map from position index to real index in module_list (-1 if skipped)
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        # ================== HyFormer Components ==================
        # MultiSeqQueryGenerator
        from utils import EnhancedMultiSeqQueryGenerator
        self.query_generator = EnhancedMultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
            use_importance_weighting=self.use_importance_weighting,
            weighting_type=self.importance_weighting_type,
            importance_dropout=self.importance_dropout,
        )

        # MultiSeqHyFormerBlock stack
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
                use_flash_varlen=self.use_flash_attn_varlen,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ─── 方案 A: Target-aware Cross-Attention Bias 模块 ───────────────
        # 每层每 domain 一个独立 bias 头, 让浅层与深层可以学到不同的
        # target prior; 每层每 domain 一个可学习 scalar alpha (初值 0),
        # 关闭开关或训练初期等价于现有结构.
        if self.use_target_attn_bias:
            self.target_bias_modules = nn.ModuleList([
                nn.ModuleList([
                    TargetAttnBiasModule(
                        d_model=d_model,
                        dropout=self.target_attn_bias_dropout,
                    )
                    for _ in range(self.num_sequences)
                ])
                for _ in range(num_hyformer_blocks)
            ])
            # Per-block, per-domain learnable scalar; init=0 so block退化
            # 为现有 cross-attn 行为, 模型自适应放大相关 prior.
            self.target_bias_alpha = nn.Parameter(
                torch.zeros(num_hyformer_blocks, self.num_sequences)
            )
        else:
            self.target_bias_modules = None
            self.target_bias_alpha = None

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Classifier
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # ══════ Calendar time-feature module ══════
        # Adds a (B, D) calendar token to user_ns as a residual; does NOT
        # change the NS token count, so existing checkpoints stay shape-
        # compatible when ``use_time_feats=False``.
        if self.use_time_feats:
            self.time_feat_embed = CalendarTimeEmbedding(
                d_model=d_model, dropout=time_feats_dropout)
        else:
            self.time_feat_embed = None

        # ══════ DIN target-aware activation ══════
        if self.use_din:
            self.din_query_builder = DINQueryBuilder(
                d_model=d_model, hidden_mult=din_hidden_mult)
            self.din_activation = DINInterestActivation(
                d_model=d_model,
                num_sequences=self.num_sequences,
                hidden_mult=din_hidden_mult,
                dropout=dropout_rate,
                top_k=din_top_k,
            )
        else:
            self.din_query_builder = None
            self.din_activation = None

        # ══════ Auxiliary contrastive projection heads ══════
        # Three small 2-layer MLP+L2norm heads sharing the same output dim
        # ``aux_proj_dim`` so callers can mix InfoNCE (u vs i) and SupCon
        # (sample-level) losses without re-projecting.
        if self.use_aux_loss:
            self.aux_user_head = AuxProjHead(d_model, aux_proj_dim, dropout_rate)
            self.aux_item_head = AuxProjHead(d_model, aux_proj_dim, dropout_rate)
            self.aux_sample_head = AuxProjHead(d_model, aux_proj_dim, dropout_rate)
            self.aux_rank_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.aux_user_head = None
            self.aux_item_head = None
            self.aux_sample_head = None
            self.aux_rank_scale = None

        # Initialize parameters
        self._init_params()

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # Skipped by emb_skip_threshold, no embedding to reinit
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding is always preserved
        if self.num_time_buckets > 0:
            skip_count += 1

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model."""
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                # Feature skipped by emb_skip_threshold: output zero vector
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True,
        return_seq_repr: bool = False,
        target_for_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs the multi-sequence block stack with dropout and output projection.

        Args:
            return_seq_repr: when True, additionally returns ``curr_seqs``
                (the per-domain token-level representations after the last
                block) for position-level auxiliary losses.
            target_for_bias: 可选 (B, D), 当 ``use_target_attn_bias`` 开启
                时驱动 ``target_bias_modules`` 生成 cross-attn additive bias.
                ``None`` 时退化为标准 cross-attn (与原版本逐位对齐).
        """
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        use_bias = (
            self.use_target_attn_bias
            and self.target_bias_modules is not None
            and target_for_bias is not None
        )

        for block_idx, block in enumerate(self.blocks):
            # Precompute RoPE cos/sin for each sequence
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            # ─── 方案 A: 每层每 domain 计算 cross-attn additive bias ────
            cross_attn_bias_list: Optional[List[Optional[torch.Tensor]]] = None
            if use_bias:
                cross_attn_bias_list = []
                # 注意: bias 的 K 应该来自 sequence_encoder 之后的 seq 表示
                # (与 cross-attn 看到的 K 同源). 这里用 curr_seqs (即上一层
                # 的输出) 作为近似, 让每层 bias 自适应 “seq 演化后的语义”.
                # 第 0 层 curr_seqs == 原始 seq 嵌入, 与 cross-attn 第 0 层
                # 看到的 K 同源. 之后每层都用 “上一层输出” 这个统一约定.
                for i in range(self.num_sequences):
                    bias_module = self.target_bias_modules[block_idx][i]
                    bias_i = bias_module(
                        target_for_bias, curr_seqs[i], curr_masks[i]
                    )  # (B, L_i)
                    # alpha 控制 prior 强度, 初值 0 -> 启动时与原模型等价
                    alpha_i = self.target_bias_alpha[block_idx, i]
                    bias_i = alpha_i * bias_i
                    cross_attn_bias_list.append(bias_i)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
                cross_attn_bias_list=cross_attn_bias_list,
            )

        # Output: concatenate all sequences' Q tokens then project via MLP
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)

        if return_seq_repr:
            # ``curr_seqs`` is a list[(B, L_i, D)] of attention-mixed
            # token-level representations — one per behavior domain. Used
            # by ``forward_with_aux`` to compute position-level next-item
            # InfoNCE.
            return output, curr_seqs
        return output

    def _build_user_dense_token(self, dense: torch.Tensor) -> torch.Tensor:
        """Returns the (B, 1, D) user_dense token, dispatching to the grouped
        or fallback projection."""
        if self.user_dense_grouped:
            # UserDenseGroupedProjector already returns (B, 1, D) with SiLU.
            return self.user_dense_proj(dense)
        return F.silu(self.user_dense_proj(dense)).unsqueeze(1)

    def _encode_inputs(
        self, inputs: ModelInput
    ) -> Tuple[torch.Tensor, list, list, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Shared encoding path used by both ``forward`` and ``predict``.

        Returns:
            ns_tokens: (B, num_ns, D)
            seq_tokens_list: list of (B, L_i, D)
            seq_masks_list: list of (B, L_i)
            user_dense_tok: (B, 1, D) or None — used for the DIN query.
            item_dense_tok: (B, 1, D) or None — used for the DIN query.
        """
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        # Time-feature residual on user_ns. Implemented as residual addition
        # rather than a new NS token, so the existing NS token count and
        # downstream RankMixer T constraint are unchanged.
        if self.use_time_feats and self.time_feat_embed is not None:
            time_tok = self.time_feat_embed(inputs.time_feats.long()).unsqueeze(1)  # (B, 1, D)
            user_ns = user_ns + time_tok

        ns_parts: List[torch.Tensor] = [user_ns]
        user_dense_tok: Optional[torch.Tensor] = None
        item_dense_tok: Optional[torch.Tensor] = None
        if self.has_user_dense:
            user_dense_tok = self._build_user_dense_token(inputs.user_dense_feats)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain])
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(
                inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        # Stash the user_ns/item_ns tokens for DIN; they are the static
        # candidate context used to build the target-aware query.
        self._cached_user_ns = user_ns
        self._cached_item_ns = item_ns

        return (ns_tokens, seq_tokens_list, seq_masks_list,
                user_dense_tok, item_dense_tok)

    def _maybe_din_residual(
        self,
        output: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        user_dense_tok: Optional[torch.Tensor],
        item_dense_tok: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Adds the DIN-derived residual to the backbone output if enabled."""
        if not self.use_din or self.din_activation is None:
            return output
        user_tokens = [self._cached_user_ns]
        if user_dense_tok is not None:
            user_tokens.append(user_dense_tok)
        item_tokens = [self._cached_item_ns]
        if item_dense_tok is not None:
            item_tokens.append(item_dense_tok)
        din_query = self.din_query_builder(user_tokens, item_tokens)
        return output + self.din_activation(din_query, seq_tokens_list, seq_masks_list)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Runs the forward pass of the PCVRHyFormer model."""
        ns_tokens, seq_tokens_list, seq_masks_list, user_dense_tok, item_dense_tok = (
            self._encode_inputs(inputs))

        # 获取目标item特征用于重要性加权 / Target Attn Bias
        target_tokens = None
        if self.use_importance_weighting or self.use_target_attn_bias:
            # 使用item_ns的平均池化作为目标item特征
            item_ns = self._cached_item_ns  # (B, num_item_ns, D)
            target_tokens = item_ns.mean(dim=1)  # (B, D)

        q_tokens_list = self.query_generator(
            ns_tokens, seq_tokens_list, seq_masks_list,
            target_tokens if self.use_importance_weighting else None,
        )
        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=self.training,
            target_for_bias=target_tokens if self.use_target_attn_bias else None,
        )
        output = self._maybe_din_residual(
            output, seq_tokens_list, seq_masks_list,
            user_dense_tok, item_dense_tok)

        logits = self.clsfier(output)  # (B, action_num)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning both logits and embeddings."""
        ns_tokens, seq_tokens_list, seq_masks_list, user_dense_tok, item_dense_tok = (
            self._encode_inputs(inputs))

        # 获取目标item特征用于重要性加权 / Target Attn Bias
        target_tokens = None
        if self.use_importance_weighting or self.use_target_attn_bias:
            item_ns = self._cached_item_ns
            target_tokens = item_ns.mean(dim=1)

        q_tokens_list = self.query_generator(
            ns_tokens, seq_tokens_list, seq_masks_list,
            target_tokens if self.use_importance_weighting else None,
        )
        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=False,
            target_for_bias=target_tokens if self.use_target_attn_bias else None,
        )
        output = self._maybe_din_residual(
            output, seq_tokens_list, seq_masks_list,
            user_dense_tok, item_dense_tok)

        logits = self.clsfier(output)
        return logits, output

    def forward_with_aux(
        self, inputs: ModelInput
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """Training-time forward that ALSO returns auxiliary contrastive
        representations for InfoNCE / SupCon losses.

        Returns:
            logits: (B, action_num)
            aux: tuple of four entries when ``use_aux_loss=True``, else None.
                - u_repr: (B, aux_proj_dim) L2-normed user-side representation
                - i_repr: (B, aux_proj_dim) L2-normed item-side representation
                - s_repr: (B, aux_proj_dim) L2-normed full-sample representation
                - position_aux: dict with ``seq_repr_list`` and ``seq_mask_list``
                  for the optional history-hard InfoNCE branch.
        """
        ns_tokens, seq_tokens_list, seq_masks_list, user_dense_tok, item_dense_tok = (
            self._encode_inputs(inputs))

        # 获取目标item特征用于重要性加权 / Target Attn Bias
        target_tokens = None
        if self.use_importance_weighting or self.use_target_attn_bias:
            item_ns = self._cached_item_ns
            target_tokens = item_ns.mean(dim=1)

        q_tokens_list = self.query_generator(
            ns_tokens, seq_tokens_list, seq_masks_list,
            target_tokens if self.use_importance_weighting else None,
        )
        output, curr_seqs = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=self.training,
            return_seq_repr=True,
            target_for_bias=target_tokens if self.use_target_attn_bias else None,
        )
        output = self._maybe_din_residual(
            output, seq_tokens_list, seq_masks_list,
            user_dense_tok, item_dense_tok)

        logits = self.clsfier(output)

        if not self.use_aux_loss or self.aux_user_head is None:
            return logits, None

        # Pool user-side / item-side static tokens. We reuse the cached
        # NS tokens already computed in ``_encode_inputs`` so this branch
        # adds no extra forward through the tokenizers.
        user_pool_parts = [self._cached_user_ns.mean(dim=1)]
        if user_dense_tok is not None:
            user_pool_parts.append(user_dense_tok.squeeze(1))
        # Mean-pool the masked behavior sequences and feed them into the
        # user side, matching DIN's user-interest semantics.
        for tokens, mask in zip(seq_tokens_list, seq_masks_list):
            keep = (~mask).float().unsqueeze(-1)  # (B, L, 1)
            denom = keep.sum(dim=1).clamp(min=1.0)
            user_pool_parts.append((tokens * keep).sum(dim=1) / denom)
        u_pooled = torch.stack(user_pool_parts, dim=0).mean(dim=0)

        item_pool_parts = [self._cached_item_ns.mean(dim=1)]
        if item_dense_tok is not None:
            item_pool_parts.append(item_dense_tok.squeeze(1))
        i_pooled = torch.stack(item_pool_parts, dim=0).mean(dim=0)

        u_repr = self.aux_user_head(u_pooled)
        i_repr = self.aux_item_head(i_pooled)
        s_repr = self.aux_sample_head(output)
        position_aux = {
            'seq_repr_list': curr_seqs,
            'seq_mask_list': seq_masks_list,
        }
        return logits, (u_repr, i_repr, s_repr, position_aux)

    def score_aux_candidates(
        self,
        u_repr: torch.Tensor,
        i_repr: torch.Tensor,
        option_cols: torch.Tensor,
    ) -> torch.Tensor:
        """Return lightweight candidate rank scores shaped ``[B, K]``.

        The backbone has already produced ``u_repr`` and ``i_repr`` once for
        the mini-batch. This method gathers candidate item representations and
        scores them without adding any raw item-id embedding shortcut.
        """
        if self.aux_item_head is None:
            raise RuntimeError("score_aux_candidates requires use_aux_loss=True")
        if option_cols.dim() != 2:
            raise ValueError(
                f"option_cols must be 2D, got {tuple(option_cols.shape)}")

        device = u_repr.device
        option_cols = option_cols.to(device=device, dtype=torch.long)
        batch_rows, option_count = option_cols.shape
        if u_repr.shape[0] != batch_rows:
            raise ValueError(
                f"u_repr rows {u_repr.shape[0]} != option rows {batch_rows}")

        cand_repr = i_repr.index_select(0, option_cols.reshape(-1)).view(
            batch_rows, option_count, -1)

        scores = torch.einsum('bd,bkd->bk', u_repr, cand_repr)
        if self.aux_rank_scale is not None:
            scores = scores * self.aux_rank_scale.exp().clamp(max=20.0)
        return scores
