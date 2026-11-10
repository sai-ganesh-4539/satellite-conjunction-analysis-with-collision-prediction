"""
model.py  (revised)

RetNet-style retention encoder and a Transformer baseline for CDM-sequence
collision-risk classification.

What changed vs. the previous version
  1. PADDING IS NOW HANDLED.  Padded steps are all-zero rows after scaling.
     Before, input_proj turned them into ordinary tokens (its bias is non-zero),
     so the models read padding as "average measurements".  Now every forward pass
     (a) finds the real steps (rows that are not all zero),
     (b) packs them to the front in their original order (works for left- or
         right-padded input),
     (c) runs the encoder on the packed sequence, and
     (d) pools only over real steps ("last" = latest real CDM).
     Because attention/retention are causal, real steps never see the padding
     that now sits after them, so NO key-padding mask is needed and there is no
     all-masked-row NaN problem.  Output no longer depends on how much padding
     is added.
  2. retention_mode="matrix" is the retention of Sun et al. (2023):
         o_n = sum_{i<=n} gamma^(n-i) (q_n . k_i) v_i          (parallel)
         S_n = gamma S_{n-1} + k_n^T v_n ,  o_n = q_n S_n       (recurrent, S is d x d)
     retention_mode="elementwise" is what the old code actually computed
     (q * sum decay * (k * v), channel by channel).  It is NOT the paper's
     retention; keep it only to reproduce old numbers.
  3. Decay matrices are precomputed buffers (no per-forward Python loop / .item()).
  4. gamma must be in (0, 1).  gamma >= 1 would weight OLD steps more.
  5. pos_embed length is a parameter (max_len), not a hard-coded 25.

All deep-model results must be re-run after this change.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Padding utilities
# --------------------------------------------------------------------------

def pack_real_first(x: torch.Tensor):
    """x: (B, T, F). Real steps = rows that are not all zero.
    Returns x with real steps moved to the front (order preserved) and
    lengths: (B,) number of real steps (>= 1)."""
    real = x.abs().sum(-1) > 0
    lengths = real.sum(1).clamp(min=1)
    order = torch.argsort((~real).to(torch.int8), dim=1, stable=True)
    packed = x.gather(1, order.unsqueeze(-1).expand(-1, -1, x.size(-1)))
    return packed, lengths


def masked_pool(h: torch.Tensor, lengths: torch.Tensor, mode: str) -> torch.Tensor:
    B, T, _ = h.shape
    valid = torch.arange(T, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B,T)
    if mode == "last":
        return h[torch.arange(B, device=h.device), lengths - 1]
    if mode == "mean":
        return (h * valid.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1).to(h.dtype)
    if mode == "max":
        return h.masked_fill(~valid.unsqueeze(-1), float("-inf")).max(1).values
    raise ValueError(f"Unknown pooling: {mode}")


# --------------------------------------------------------------------------
# Multi-scale retention
# --------------------------------------------------------------------------

class MultiScaleRetention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, gammas: Optional[List[float]] = None,
                 dropout: float = 0.1, mode: str = "matrix", max_len: int = 64):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert mode in ("matrix", "elementwise")
        self.d_model, self.n_heads, self.mode = d_model, n_heads, mode
        self.d_head = d_model // n_heads

        gammas = list(gammas) if gammas is not None else [0.90, 0.95, 0.98, 0.99]
        if len(gammas) < n_heads:
            gammas = gammas * (n_heads // len(gammas) + 1)
        gammas = gammas[:n_heads]
        if not all(0.0 < g < 1.0 for g in gammas):
            raise ValueError(f"every gamma must be in (0, 1), got {gammas}")

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        if mode == "matrix":
            self.out_norm = nn.LayerNorm(self.d_head)   # per-head output norm

        g = torch.tensor(gammas, dtype=torch.float32)
        pos = torch.arange(max_len)
        lag = (pos[:, None] - pos[None, :]).float()
        dec = (g.view(-1, 1, 1) ** lag.clamp(min=0)) * (lag >= 0).float()   # (H,L,L)
        self.register_buffer("gammas", g, persistent=False)
        self.register_buffer("decay", dec, persistent=False)
        self.register_buffer("decay_norm", dec / dec.sum(-1, keepdim=True), persistent=False)

    def _split(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        f = lambda t: t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        return f(q), f(k), f(v)                                  # (B,H,T,D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self._split(x)
        if self.mode == "elementwise":
            w = torch.einsum("hts,bhsd->bhtd", self.decay_norm[:, :T, :T], k * v)
            out = q * w
        else:
            scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(self.d_head)
            scores = scores * self.decay[:, :T, :T].unsqueeze(0)
            out = self.out_norm(torch.einsum("bhts,bhsd->bhtd", scores, v))
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.dropout(self.proj(out))

    def forward_recurrent(self, x_t: torch.Tensor, state=None):
        """One streaming step. x_t: (B,1,d_model) or (B,d_model).
        state: (S, A) -- A is only used in elementwise mode."""
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        B, _, C = x_t.shape
        q, k, v = self._split(x_t)                               # (B,H,1,D)
        g = self.gammas.view(1, self.n_heads, 1, 1)
        if self.mode == "elementwise":
            if state is None:
                S = x_t.new_zeros(B, self.n_heads, 1, self.d_head)
                A = x_t.new_zeros(B, self.n_heads, 1, 1)
            else:
                S, A = state
            S = g * S + k * v
            A = g * A + 1.0
            out = q * (S / A.clamp(min=1e-6))
            new_state = (S, A)
        else:
            S = x_t.new_zeros(B, self.n_heads, self.d_head, self.d_head) if state is None else state[0]
            S = g * S + k.transpose(-1, -2) @ v                  # (B,H,D,D)
            out = self.out_norm((q / math.sqrt(self.d_head)) @ S)
            new_state = (S, None)
        out = out.transpose(1, 2).reshape(B, 1, C)
        return self.dropout(self.proj(out)), new_state


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, expansion_ratio: float = 4.0 / 3.0, dropout: float = 0.1):
        super().__init__()
        d_ff = int(d_model * expansion_ratio)
        d_ff = ((d_ff + 1) // 2) * 2
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RetNetBlock(nn.Module):
    def __init__(self, d_model, n_heads, gammas=None, dropout=0.1, pre_norm=True,
                 mode="matrix", max_len=64):
        super().__init__()
        self.retention = MultiScaleRetention(d_model, n_heads, gammas, dropout, mode, max_len)
        self.ffn = SwiGLUFFN(d_model, dropout=dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.pre_norm = pre_norm

    def forward(self, x):
        if self.pre_norm:
            x = x + self.retention(self.norm1(x))
            return x + self.ffn(self.norm2(x))
        x = self.norm1(x + self.retention(x))
        return self.norm2(x + self.ffn(x))

    def forward_recurrent(self, x_t, state=None):
        if self.pre_norm:
            r, st = self.retention.forward_recurrent(self.norm1(x_t), state)
            x_t = x_t + r
            return x_t + self.ffn(self.norm2(x_t)), st
        r, st = self.retention.forward_recurrent(x_t, state)
        x_t = self.norm1(x_t + r)
        return self.norm2(x_t + self.ffn(x_t)), st


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def _init_linear(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class RetNetForCollisionPrediction(nn.Module):
    """Input (B, T, F) with zero-padded steps (left or right) -> logits (B, 1)."""

    def __init__(self, input_dim: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 gammas: Optional[List[float]] = None, dropout: float = 0.1,
                 pooling: str = "last", retention_mode: str = "matrix", max_len: int = 32):
        super().__init__()
        self.max_len, self.pooling = max_len, pooling
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            RetNetBlock(d_model, n_heads, gammas, dropout, mode=retention_mode, max_len=max_len)
            for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        _init_linear(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, lengths = pack_real_first(x)
        T = x.size(1)
        assert T <= self.max_len, f"sequence length {T} > max_len {self.max_len}"
        h = self.input_proj(x) + self.pos_embed[:, :T]
        for blk in self.blocks:
            h = blk(h)
        h = self.final_norm(h)
        return self.classifier(masked_pool(h, lengths, self.pooling))

    def forward_recurrent(self, x_t: torch.Tensor, timestep: int = 0, states=None):
        """Streaming update with the newest CDM only. `timestep` = number of real
        CDMs already consumed (0 for the first). Matches forward() for pooling='last'."""
        if self.pooling != "last":
            raise NotImplementedError("streaming equivalence is defined for pooling='last'")
        assert timestep < self.max_len
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        h = self.input_proj(x_t) + self.pos_embed[:, timestep:timestep + 1]
        states = states if states is not None else [None] * len(self.blocks)
        new_states = []
        for blk, st in zip(self.blocks, states):
            h, st = blk.forward_recurrent(h, st)
            new_states.append(st)
        return self.classifier(self.final_norm(h).squeeze(1)), new_states


class TransformerBaseline(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 d_ff: int = 256, dropout: float = 0.1, pooling: str = "last", max_len: int = 32):
        super().__init__()
        self.max_len, self.pooling = max_len, pooling
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        _init_linear(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, lengths = pack_real_first(x)
        T = x.size(1)
        assert T <= self.max_len, f"sequence length {T} > max_len {self.max_len}"
        h = self.input_proj(x) + self.pos_embed[:, :T]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)      # padding sits AFTER real steps
        h = self.final_norm(h)
        return self.classifier(masked_pool(h, lengths, self.pooling))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FocalLoss(nn.Module):
    """Focal Loss for binary collision triage classification under class imbalance."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(-1)
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - pt) ** self.gamma
        alpha_weight = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_weight * focal_weight * bce).mean()

