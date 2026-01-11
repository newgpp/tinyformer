# model/attention.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AttentionConfig:
    d_model: int     # embedding 维度，比如 512
    n_heads: int     # 多头数量，比如 8
    dropout: float = 0.0


class MultiHeadAttention(nn.Module):
    """
    教学版 Multi-Head Attention

    支持：
      1) Encoder self-attention
      2) Decoder causal self-attention
      3) Decoder cross-attention（Q 来自 decoder，K/V 来自 encoder）

    统一接口，靠参数控制模式
    """

    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.cfg = cfg
        self.d_head = cfg.d_model // cfg.n_heads  # 每个 head 的维度

        # 线性映射矩阵：把输入映射成 Q / K / V
        self.w_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.w_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.w_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # 多头拼接后再映射回 d_model
        self.w_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,                    # Query 来源 [B, Tq, d_model]
        memory: Optional[torch.Tensor] = None,  # K/V 来源（cross-attn）[B, Tk, d_model]
        *,
        causal: bool = False,              # 是否启用因果 mask（decoder 用）
        key_padding_mask: Optional[torch.Tensor] = None,  # padding mask [B, Tk]
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # 如果没有 memory，就是 self-attention
        if memory is None:
            k_in = x
            v_in = x
        else:
            # cross-attention：K/V 来自 encoder 输出
            k_in = memory
            v_in = memory

        B, Tq, _ = x.shape
        _, Tk, _ = k_in.shape

        # ===== 1. 线性投影 =====
        # 把 token embedding 映射成 Q/K/V
        q = self.w_q(x)       # [B, Tq, d_model]
        k = self.w_k(k_in)   # [B, Tk, d_model]
        v = self.w_v(v_in)   # [B, Tk, d_model]

        # ===== 2. 切分多头 =====
        # [B, T, d_model] → [B, n_heads, T, d_head]
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # ===== 3. 计算注意力分数 QK^T =====
        # [B, nH, Tq, dH] @ [B, nH, dH, Tk] → [B, nH, Tq, Tk]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # ===== 4. padding mask（屏蔽 padding token）=====
        if key_padding_mask is not None:
            # [B, Tk] → [B, 1, 1, Tk]
            mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(mask, float("-inf"))

        # ===== 5. causal mask（屏蔽未来）=====
        if causal:
            # 只能用于 self-attention
            if Tk != Tq:
                raise ValueError("causal=True 时必须是 self-attention")
            causal_mask = self._causal_mask(Tq, device=scores.device)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # ===== 6. softmax 得到注意力权重 =====
        # 每一行表示：当前 token 对所有 key 的注意力分布
        attn = F.softmax(scores, dim=-1)   # [B, nH, Tq, Tk]
        attn = self.drop(attn)

        # ===== 7. 用注意力权重加权 V =====
        # [B, nH, Tq, Tk] @ [B, nH, Tk, dH] → [B, nH, Tq, dH]
        ctx = torch.matmul(attn, v)

        # ===== 8. 合并多头 =====
        # [B, nH, Tq, dH] → [B, Tq, d_model]
        ctx = self._merge_heads(ctx)

        # ===== 9. 输出线性层 =====
        out = self.w_o(ctx)
        out = self.drop(out)

        if return_attn:
            return out, attn
        return out, None

    # -------------------------
    # 工具函数
    # -------------------------

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # [B, T, d_model] → [B, n_heads, T, d_head]
        B, T, D = t.shape
        t = t.view(B, T, self.cfg.n_heads, self.d_head)
        return t.transpose(1, 2)

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        # [B, n_heads, T, d_head] → [B, T, d_model]
        B, nH, T, dH = t.shape
        t = t.transpose(1, 2).contiguous()
        return t.view(B, T, nH * dH)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # 上三角（j > i）为 True，表示“不能看未来”
        m = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return m[None, None, :, :]  # [1, 1, T, T]
