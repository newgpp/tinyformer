# model/decoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import AttentionConfig, MultiHeadAttention


@dataclass(frozen=True)
class DecoderConfig:
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    dropout: float = 0.0


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


class DecoderLayer(nn.Module):
    """
    教学版 DecoderLayer（Pre-LN）：

    输入:
      x:      [B, T, d_model]   (decoder hidden states)
      memory: [B, S, d_model]   (encoder 输出，用于 cross-attention)

    结构（经典 Transformer Decoder）：
      1) LN -> causal self-attention -> residual
      2) LN -> cross-attention       -> residual
      3) LN -> FFN                   -> residual
    """
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        attn_cfg = AttentionConfig(cfg.d_model, cfg.n_heads, cfg.dropout)

        # 1) decoder self-attention（需要 causal mask）
        self.self_attn = MultiHeadAttention(attn_cfg)

        # 2) decoder cross-attention（Q 来自 decoder，K/V 来自 encoder）
        self.cross_attn = MultiHeadAttention(attn_cfg)

        # 三个子层各自一套 LN（Pre-LN）
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln3 = nn.LayerNorm(cfg.d_model)

        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        # encoder padding mask：True 表示该 encoder key 是 padding，应该被屏蔽
        memory_key_padding_mask: Optional[torch.Tensor] = None,  # [B, S]
        # decoder 自身 padding mask（训练 toy task 可先不用）
        self_key_padding_mask: Optional[torch.Tensor] = None,    # [B, T]
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            memory: [B, S, d_model]
            memory_key_padding_mask: [B, S] (True=padding)
            self_key_padding_mask: [B, T] (True=padding)

        Returns:
            x: [B, T, d_model]
        """

        # ===== 1) Causal Self-Attention（decoder 内部看自己，但不能看未来）=====
        # 对应 explainer 的“Masked Self-Attention”：当前位置只看已生成的左侧 token
        a = self.ln1(x)  # [B, T, d_model]

        self_out, _ = self.self_attn(
            a,
            memory=None,                 # self-attn
            causal=True,                 # 关键：开启因果 mask
            key_padding_mask=self_key_padding_mask,
            return_attn=False
        )  # [B, T, d_model]

        x = x + self.drop(self_out)

        # ===== 2) Cross-Attention（decoder 看 encoder 输出）=====
        # 直觉：我现在要生成每个位置的 token，我应该从 encoder 哪些位置取信息？
        # 对应 explainer 的“Encoder-Decoder Attention”：Q 来自 decoder，K/V 来自 encoder
        b = self.ln2(x)  # [B, T, d_model] 作为 Q

        cross_out, _ = self.cross_attn(
            b,
            memory=memory,               # cross-attn：K/V 来自 memory
            causal=False,                # cross-attn 不用 causal
            key_padding_mask=memory_key_padding_mask,  # mask 掉 encoder padding
            return_attn=False
        )  # [B, T, d_model]

        x = x + self.drop(cross_out)

        # ===== 3) FFN =====
        # 对应 explainer 的逐位置前馈网络（不跨 token），再加残差
        c = self.ln3(x)
        ffn_out = self.ffn(c)
        x = x + self.drop(ffn_out)

        return x


class Decoder(nn.Module):
    """
    Decoder Stack：
      输入 x_embed: [B, T, d_model]
      输入 memory:  [B, S, d_model]（来自 encoder）
      输出 h:       [B, T, d_model]
    """
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_key_padding_mask: Optional[torch.Tensor] = None,  # [B, S]
        self_key_padding_mask: Optional[torch.Tensor] = None,    # [B, T]
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
                self_key_padding_mask=self_key_padding_mask
            )
        x = self.ln_f(x)
        return x
