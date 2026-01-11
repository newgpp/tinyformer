# model/encoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import AttentionConfig, MultiHeadAttention


@dataclass(frozen=True)
class EncoderConfig:
    d_model: int          # embedding维度
    n_heads: int          # 注意力头数
    d_ff: int             # FFN隐藏层维度（一般是 4*d_model）
    n_layers: int         # encoder层数
    dropout: float = 0.0  # 学习阶段可先设0


class FeedForward(nn.Module):
    """
    标准 FFN：
      Linear(d_model -> d_ff) -> GELU -> Linear(d_ff -> d_model)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        x = self.fc1(x)           # [B, T, d_ff]
        x = F.gelu(x)             # [B, T, d_ff]
        x = self.drop(x)
        x = self.fc2(x)           # [B, T, d_model]
        x = self.drop(x)
        return x


class EncoderLayer(nn.Module):
    """
    教学版 EncoderLayer（Pre-LN）：

    输入 x: [B, T, d_model]

    1) x -> LN -> Self-Attention -> 残差
    2) x -> LN -> FFN            -> 残差
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        attn_cfg = AttentionConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout
        )
        self.self_attn = MultiHeadAttention(attn_cfg)

        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            key_padding_mask: [B, T]，True表示该位置是padding，应被mask掉

        Returns:
            x: [B, T, d_model]
        """

        # ===== 1) Self-Attention 子层（Pre-LN）=====
        # 先做 LN，再做 attention
        a = self.ln1(x)  # [B, T, d_model]

        # encoder self-attention 不需要 causal mask
        attn_out, _ = self.self_attn(
            a,
            memory=None,
            causal=False,
            key_padding_mask=key_padding_mask,
            return_attn=False
        )  # [B, T, d_model]

        # 残差连接：x + attention(a)
        x = x + self.drop(attn_out)

        # ===== 2) FFN 子层（Pre-LN）=====
        b = self.ln2(x)         # [B, T, d_model]
        ffn_out = self.ffn(b)   # [B, T, d_model]
        x = x + self.drop(ffn_out)

        return x


class Encoder(nn.Module):
    """
    Encoder Stack：
      输入 x_embed: [B, T, d_model]
      输出 memory: [B, T, d_model]
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)  # 最后一层 LN（可选但常见）

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]（通常是 embedding + positional 的结果）
            key_padding_mask: [B, T]，True表示padding位置

        Returns:
            memory: [B, T, d_model]
        """
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.ln_f(x)
        return x
