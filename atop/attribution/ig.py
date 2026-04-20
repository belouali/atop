"""Integrated Gradients implementation for token-level attribution."""
from __future__ import annotations

import numpy as np
import torch

from atop.config import PAD_IDX


class IntegratedGradientsCustom:
    """
    IG for single-stream Transformer. Interpolates between PAD embedding (baseline)
    and actual embedding, computing gradients of the target class logit w.r.t. embeddings.

    Batched: processes the full batch at once for GPU utilization.
    Falls back to single-sample if OOM.
    """

    def __init__(self, model: SingleStreamTransformer, n_steps: int = 50):
        self.model = model
        self.n_steps = n_steps

    def _attribute_batch(self, input_ids: torch.Tensor, target_class: int = 1) -> torch.Tensor:
        """
        Compute IG for a full batch (B, L).
        Returns: (B, L) attribution scores.
        """
        device = input_ids.device
        B, L = input_ids.shape

        pad_ids = torch.full_like(input_ids, PAD_IDX)
        baseline_emb = self.model.get_input_embeddings(pad_ids).detach()
        actual_emb = self.model.get_input_embeddings(input_ids).detach()
        diff = actual_emb - baseline_emb  # (B, L, D)
        pad_mask = (input_ids == PAD_IDX)

        # Accumulate gradients incrementally
        grad_sum = torch.zeros_like(diff)  # (B, L, D)

        for alpha in np.linspace(0, 1, self.n_steps + 1):
            scaled = baseline_emb + alpha * diff
            scaled = scaled.clone().detach().requires_grad_(True)

            x = self.model.emb_norm(self.model.emb_dropout(scaled))
            x = self.model.transformer(x, src_key_padding_mask=pad_mask)
            cls_out = x[:, 0, :]
            logits = self.model.classifier(cls_out).squeeze(-1)

            target = logits if target_class == 1 else -logits
            grads = torch.autograd.grad(target.sum(), scaled, retain_graph=False)[0]
            grad_sum += grads.detach()

        avg_grads = grad_sum / (self.n_steps + 1)
        ig = diff * avg_grads
        return ig.sum(dim=-1)  # (B, L)

    def _attribute_single(self, input_ids_1: torch.Tensor, target_class: int = 1) -> torch.Tensor:
        """Fallback: compute IG for a SINGLE sample (1, L)."""
        return self._attribute_batch(input_ids_1, target_class)

    def attribute(self, input_ids: torch.Tensor, target_class: int = 1) -> torch.Tensor:
        """
        Compute IG attributions for each token position.
        Tries batched computation first; falls back to per-sample on OOM.
        Args:
            input_ids: (B, L) token indices
            target_class: class index (1 = readmitted)
        Returns:
            attributions: (B, L) — per-token attribution scores
        """
        self.model.eval()
        try:
            return self._attribute_batch(input_ids, target_class)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # Fallback: process one at a time
            B = input_ids.shape[0]
            results = []
            for b in range(B):
                attr_b = self._attribute_single(input_ids[b:b+1], target_class)
                results.append(attr_b)
            return torch.cat(results, dim=0)


# ============================================================================
# TRAINING
# ============================================================================
