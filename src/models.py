"""
Matched Transformer and GRU language models for attractor comparison.
Both models are designed with comparable parameter counts.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLM(nn.Module):
    """Small GPT-style transformer language model."""

    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4,
                 dim_feedforward=512, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, return_hidden=False):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.dropout(self.token_emb(x) + self.pos_emb(positions))

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.ln_f(h)

        logits = self.head(h)
        if return_hidden:
            return logits, h
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GRULM(nn.Module):
    """GRU-based language model matched in parameters to the transformer."""

    def __init__(self, vocab_size, d_model=256, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=d_model, hidden_size=d_model,
            num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, hidden=None, return_hidden=False):
        B, T = x.shape
        h = self.dropout(self.token_emb(x))
        h, hidden_state = self.gru(h, hidden)
        h = self.ln_f(h)

        logits = self.head(h)
        if return_hidden:
            return logits, h
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMLM(nn.Module):
    """LSTM-based language model for additional RNN comparison."""

    def __init__(self, vocab_size, d_model=256, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model,
            num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
                # Forget gate bias = 1
                if 'bias_ih' in name or 'bias_hh' in name:
                    n = p.size(0)
                    p.data[n//4:n//2].fill_(1.0)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, hidden=None, return_hidden=False):
        B, T = x.shape
        h = self.dropout(self.token_emb(x))
        h, hidden_state = self.lstm(h, hidden)
        h = self.ln_f(h)

        logits = self.head(h)
        if return_hidden:
            return logits, h
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_models(vocab_size, scale="small"):
    """Build matched-parameter Transformer and GRU models.

    Returns dict of model_name -> model.
    """
    if scale == "small":
        # ~5M params each
        transformer = TransformerLM(vocab_size, d_model=256, nhead=4,
                                     num_layers=4, dim_feedforward=512)
        gru = GRULM(vocab_size, d_model=256, num_layers=4)
        lstm = LSTMLM(vocab_size, d_model=256, num_layers=4)
    elif scale == "medium":
        # ~15M params each
        transformer = TransformerLM(vocab_size, d_model=384, nhead=6,
                                     num_layers=6, dim_feedforward=768)
        gru = GRULM(vocab_size, d_model=384, num_layers=6)
        lstm = LSTMLM(vocab_size, d_model=384, num_layers=6)
    else:
        raise ValueError(f"Unknown scale: {scale}")

    models = {
        "transformer": transformer,
        "gru": gru,
        "lstm": lstm,
    }

    print(f"Model parameter counts (scale={scale}):")
    for name, model in models.items():
        print(f"  {name}: {model.count_params():,}")

    return models
