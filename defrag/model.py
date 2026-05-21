"""
defrag/model.py

TemporalPredictor: lightweight, causally-correct sequence model
operating over pseudo-timelines produced by VectorDefragRouter.

Architecture: stacked GLU blocks with SSM-inspired gated recurrent state.

Each layer maintains a persistent recurrent state vector that is updated
multiplicatively — inspired by linear recurrences in Mamba/S4 — without
the full state-space machinery. This gives O(sequence_len) compute with
sub-quadratic memory and stable gradient flow.

The combination of:
  - gated linear units (GLU) for position-wise feature mixing
  - gated recurrent carry for temporal information propagation
  - pre-LayerNorm residuals for training stability
  - causal masking on the GLU convolution

ensures the predictor cannot access future pseudo-timeline positions
during both training and inference.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PredictorConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSM-inspired gated recurrent cell
# ---------------------------------------------------------------------------

class GatedRecurrentCell(nn.Module):
    """
    Scalar-gated linear recurrence with input-dependent forget gate.

    State update:
        f_t   = sigmoid(W_f @ x_t + b_f)        # forget gate ∈ (0,1)
        i_t   = tanh(W_i @ x_t + b_i)           # input projection
        h_t   = f_t * h_{t-1} + (1 - f_t) * i_t

    This is deliberately minimal — no key/query attention, no nonlinear
    hidden-to-hidden path — keeping the effective receptive field finite
    and gradient flow clean.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        # Combined projection for forget + input
        self.gate_proj = nn.Linear(dim, dim * 2, bias=True)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        x: torch.Tensor,       # (B, dim)
        h: torch.Tensor,       # (B, dim) — previous state
    ) -> torch.Tensor:
        g = self.gate_proj(x)                          # (B, 2*dim)
        f, i_ = g.chunk(2, dim=-1)
        f  = torch.sigmoid(f)
        i_ = torch.tanh(i_)
        h_new = f * h + (1.0 - f) * i_
        return h_new


# ---------------------------------------------------------------------------
# GLU block with recurrent carry
# ---------------------------------------------------------------------------

class GLURecurrentBlock(nn.Module):
    """
    One layer of the TemporalPredictor.

    Processing per time step t:
        1. Pre-LayerNorm
        2. GLU: split linear projection into value + gate halves
           out = value * sigmoid(gate)
        3. GatedRecurrentCell updates hidden state
        4. Output projection back to model dim
        5. Dropout + residual

    The recurrent state h is carried across time steps sequentially during
    the forward pass (loop over T). This is efficient for the short sequence
    lengths (top_k ~ 16) that characterise pseudo-timelines.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm      = nn.LayerNorm(dim)
        self.glu_proj  = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.recurrent = GatedRecurrentCell(hidden_dim)
        self.out_proj  = nn.Linear(hidden_dim, dim, bias=False)
        self.drop      = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.glu_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,                    # (B, T, dim)
        h0: Optional[torch.Tensor] = None,  # (B, hidden_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        out : (B, T, dim)
        h_T : (B, hidden_dim) — final recurrent state (for stateful inference)
        """
        B, T, D = x.shape

        # Infer hidden_dim from recurrent cell
        hidden_dim = self.recurrent.dim
        h = h0 if h0 is not None else x.new_zeros(B, hidden_dim)

        outputs: List[torch.Tensor] = []

        for t in range(T):
            x_t   = self.norm(x[:, t, :])          # (B, D) pre-norm

            # GLU
            gated = self.glu_proj(x_t)              # (B, 2*hidden_dim)
            val, gate = gated.chunk(2, dim=-1)
            glu_out = val * torch.sigmoid(gate)     # (B, hidden_dim)

            # Recurrent update
            h = self.recurrent(glu_out, h)          # (B, hidden_dim)

            # Project back
            out_t = self.drop(self.out_proj(h))     # (B, D)
            outputs.append(out_t)

        # Stack and add residual
        stacked = torch.stack(outputs, dim=1)       # (B, T, D)
        return x + stacked, h


# ---------------------------------------------------------------------------
# TemporalPredictor
# ---------------------------------------------------------------------------

class TemporalPredictor(nn.Module):
    """
    Lightweight temporal model for alpha projection over pseudo-timelines.

    Input  : (B, S, latent_dim)   — S stitched pseudo-timeline positions
    Output : (B, forecast_dim)    — forward alpha projection

    The model processes the entire pseudo-timeline autoregressively,
    then reads off a prediction from the final sequence position.

    Multi-step projection is supported: forecast_dim > 1 produces a
    trajectory over forecast_dim future steps rather than a scalar.
    """

    def __init__(self, cfg: PredictorConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_norm = nn.LayerNorm(cfg.latent_dim)

        self.blocks = nn.ModuleList([
            GLURecurrentBlock(cfg.latent_dim, cfg.hidden_dim, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

        # Final forecast head: two-layer MLP with GELU
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(cfg.latent_dim),
            nn.Linear(cfg.latent_dim, cfg.hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.forecast_dim, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,                        # (B, S, latent_dim)
        states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Parameters
        ----------
        x      : (B, S, latent_dim) pseudo-timeline latent sequence
        states : optional list of per-layer initial recurrent states (B, hidden_dim)

        Returns
        -------
        alpha  : (B, forecast_dim) — projected alpha signal
        new_states : list of per-layer final recurrent states for stateful inference
        """
        B, S, D = x.shape
        assert D == self.cfg.latent_dim, \
            f"Expected latent_dim={self.cfg.latent_dim}, got D={D}"
        assert S >= 1

        h = self.input_norm(x)    # (B, S, D)

        new_states: List[torch.Tensor] = []

        for layer_idx, block in enumerate(self.blocks):
            h0 = states[layer_idx] if states is not None else None
            h, h_final = block(h, h0=h0)
            new_states.append(h_final)

        # Read from final sequence position
        last = h[:, -1, :]           # (B, D)
        alpha = self.forecast_head(last)   # (B, forecast_dim)

        return alpha, new_states

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def regression_loss(
        pred: torch.Tensor,    # (B, forecast_dim)
        target: torch.Tensor,  # (B, forecast_dim)
    ) -> torch.Tensor:
        """
        Huber loss: less sensitive to outlier returns than MSE,
        avoids gradient explosion on extreme market moves.
        """
        return F.huber_loss(pred, target, delta=1.0)

    # ------------------------------------------------------------------
    # Gradient clipping helper
    # ------------------------------------------------------------------

    def clip_gradients(self) -> float:
        """
        Clip gradient norms in-place. Returns pre-clip global norm.
        Call after loss.backward(), before optimizer.step().
        """
        total_norm = nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=self.cfg.grad_clip,
        )
        return float(total_norm)

    # ------------------------------------------------------------------
    # Stateless single-step inference (live tick path)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(self, context_seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        context_seq : (S, latent_dim) — stitched pseudo-context (no batch dim)

        Returns
        -------
        alpha : (forecast_dim,)
        """
        x = context_seq.unsqueeze(0)    # (1, S, D)
        alpha, _ = self.forward(x)
        return alpha.squeeze(0)         # (forecast_dim,)
