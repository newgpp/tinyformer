import torch
from model.transformer import TransformerConfig, TransformerSeq2Seq

torch.manual_seed(0)

cfg = TransformerConfig(
    vocab_size=100,
    max_len=64,
    d_model=32,
    n_heads=4,
    d_ff=128,
    n_encoder_layers=2,
    n_decoder_layers=2,
    dropout=0.0,
    pad_id=0
)

m = TransformerSeq2Seq(cfg)

B, S, T = 2, 7, 6
src = torch.randint(1, cfg.vocab_size, (B, S))
tgt_in = torch.randint(1, cfg.vocab_size, (B, T))
labels = torch.randint(1, cfg.vocab_size, (B, T))

logits, loss = m(src, tgt_in, labels=labels)
print("logits:", logits.shape)
print("loss:", loss.item())

