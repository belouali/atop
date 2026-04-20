"""Single-stream Transformer for clinical sequence classification."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from atop.config import PAD_IDX


class SingleStreamTransformer(nn.Module):
    """
    Transformer encoder for variable-length token sequences.
    Uses [CLS] token output for binary classification.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128, max_seq_len: int = 512,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor, return_embeddings: bool = False):
        """
        Args:
            input_ids: (B, L) token indices
            return_embeddings: if True, also return per-token embeddings from last layer
        Returns:
            dict with "logits", "y_prob", optionally "embeddings"
        """
        B, L = input_ids.shape
        pad_mask = (input_ids == PAD_IDX)  # True where padded

        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_norm(self.emb_dropout(x))

        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # Use [CLS] token (position 0, or leftmost after truncation)
        # After left-truncation, CLS may be at position 0 if kept, else we use first token
        cls_out = x[:, 0, :]
        logits = self.classifier(cls_out).squeeze(-1)
        y_prob = torch.sigmoid(logits)

        out = {"logits": logits, "y_prob": y_prob}
        if return_embeddings:
            out["embeddings"] = x
        return out

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the input embeddings (token + position) — needed for IG baseline."""
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        return self.token_emb(input_ids) + self.pos_emb(positions)


# ============================================================================
# INTEGRATED GRADIENTS — Custom implementation for single-stream model
# ============================================================================
