# model/transformer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import Encoder, EncoderConfig
from model.decoder import Decoder, DecoderConfig


@dataclass(frozen=True)
class TransformerConfig:
    # vocab / 序列
    vocab_size: int
    max_len: int

    # 模型宽度
    d_model: int
    n_heads: int
    d_ff: int

    # 层数
    n_encoder_layers: int
    n_decoder_layers: int

    # dropout
    dropout: float = 0.0

    # padding token id（用于 mask）
    pad_id: int = 0


class LearnedPositionalEmbedding(nn.Module):
    """
    最简单的 positional embedding（可学习）
    pos_id = [0..T-1] -> embedding -> [T, d_model]
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        # 返回 [1, T, d_model]，方便广播到 batch
        pos = torch.arange(T, device=device)  # [T]
        return self.pos_emb(pos)[None, :, :]  # [1, T, d_model]


class TransformerSeq2Seq(nn.Module):
    """
    教学版 Encoder-Decoder Transformer（seq2seq）

    输入:
      src_ids: [B, S]
      tgt_ids: [B, T]
    输出:
      logits:  [B, T, vocab_size]
      loss:    (可选) 标准 cross-entropy

    训练时常见做法（teacher forcing）：
      decoder 输入 tgt_in  = [B, T-1] （去掉最后一个token）
      label      tgt_out = [B, T-1] （去掉第一个token）
    """
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        # ===== 1) Embedding（共享也可以，这里先分开写更直观）=====
        self.src_tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.tgt_tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.pos_emb = LearnedPositionalEmbedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # ===== 2) Encoder / Decoder =====
        enc_cfg = EncoderConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            n_layers=cfg.n_encoder_layers,
            dropout=cfg.dropout
        )
        dec_cfg = DecoderConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            n_layers=cfg.n_decoder_layers,
            dropout=cfg.dropout
        )

        self.encoder = Encoder(enc_cfg)
        self.decoder = Decoder(dec_cfg)

        # ===== 3) 输出层（LM head）=====
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # （可选）weight tying：让 tgt embedding 和 lm_head 共享权重（更省参数）
        # 这在语言模型里常见，seq2seq 也能用
        self.lm_head.weight = self.tgt_tok_emb.weight

    def _make_key_padding_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """
        ids: [B, L]
        返回 key_padding_mask: [B, L]
        True 表示 padding，应该被 attention 屏蔽
        """
        return (ids == self.cfg.pad_id)

    def encode(self, src_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        src_ids: [B, S]
        Returns:
          memory: [B, S, d_model]
          src_kpm: [B, S]  (True=padding)
        """
        B, S = src_ids.shape
        device = src_ids.device

        src_kpm = self._make_key_padding_mask(src_ids)           # [B, S]
        src_emb = self.src_tok_emb(src_ids)                      # [B, S, d_model]
        src_emb = src_emb + self.pos_emb(S, device)              # [1,S,d] broadcast
        src_emb = self.drop(src_emb)

        memory = self.encoder(src_emb, key_padding_mask=src_kpm)  # [B, S, d_model]
        return memory, src_kpm

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        src_kpm: torch.Tensor
    ) -> torch.Tensor:
        """
        tgt_ids: [B, T]
        memory:  [B, S, d_model]
        src_kpm: [B, S] (True=padding)

        Returns:
          dec_out: [B, T, d_model]
        """
        B, T = tgt_ids.shape
        device = tgt_ids.device

        tgt_kpm = self._make_key_padding_mask(tgt_ids)  # [B, T]（可选使用）

        tgt_emb = self.tgt_tok_emb(tgt_ids)             # [B, T, d_model]
        tgt_emb = tgt_emb + self.pos_emb(T, device)     # [1,T,d]
        tgt_emb = self.drop(tgt_emb)

        dec_out = self.decoder(
            tgt_emb,
            memory,
            memory_key_padding_mask=src_kpm,
            self_key_padding_mask=tgt_kpm
        )  # [B, T, d_model]

        return dec_out

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            src_ids: [B, S]
            tgt_ids: [B, T]  (decoder 输入，通常是 teacher forcing 的 tgt_in)
            labels:  [B, T]  (要预测的目标 token id，通常是 tgt_out)

        Returns:
            logits: [B, T, vocab_size]
            loss: Optional[Tensor]
        """
        memory, src_kpm = self.encode(src_ids)           # memory: [B,S,d]
        dec_out = self.decode(tgt_ids, memory, src_kpm)  # [B,T,d]

        logits = self.lm_head(dec_out)                   # [B,T,vocab]

        loss = None
        if labels is not None:
            # 典型 seq2seq：忽略 padding 的 loss
            # labels 中 pad_id 的位置不计入loss
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                labels.reshape(-1),
                ignore_index=self.cfg.pad_id
            )

        return logits, loss
