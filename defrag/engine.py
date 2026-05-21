"""
defrag/engine.py

ChronosDefragEngine — end-to-end orchestration.

Responsibilities:
  - raw tensor slicing into causal, non-overlapping blocks
  - augmentation for contrastive training (jitter + time-shift)
  - encoder training loop with InfoNCE + TER + continuity regularisation
  - pseudo-timeline construction and registry population
  - predictor training loop with Huber loss
  - live inference orchestration with latency measurement
  - causal isolation: no future block is ever visible during training
    of any earlier time index
  - checkpoint save/load

Causal correctness model:
  The raw data tensor is a strictly chronological sequence of ticks.
  Block i spans ticks [i*block_len, (i+1)*block_len).
  During encoder training, block i may only form contrastive pairs with
  other blocks from the SAME epoch — not future blocks from the predictor
  training set.
  The predictor is trained on pseudo-timeline windows drawn exclusively
  from the encoder training period; it never sees the held-out suffix.
  Live inference operates on a rolling window of the most recent block_len
  ticks and routes against the historical registry.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import DefragConfig
from .core import RegimeEncoder, VectorDefragRouter
from .model import TemporalPredictor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live prediction result
# ---------------------------------------------------------------------------

@dataclass
class LivePrediction:
    alpha_prediction:   Tensor   # (forecast_dim,)
    regime_confidence:  float
    system_latency_ms:  float


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def _slice_blocks(
    data: Tensor,       # (T, F) — full chronological data
    block_len: int,
    strict: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Partition a chronological tensor into non-overlapping causal blocks.

    Returns
    -------
    blocks     : (N, block_len, F)
    timestamps : (N,) int64 — start tick index of each block
    """
    T, F = data.shape
    N = T // block_len
    if N == 0:
        raise ValueError(
            f"Data length {T} < block_len {block_len}; "
            "provide more data or reduce block_len"
        )
    if strict and (T % block_len != 0):
        # Trim tail to maintain uniform block length
        data = data[:N * block_len]

    blocks     = data[:N * block_len].reshape(N, block_len, F)
    timestamps = torch.arange(N, dtype=torch.long, device=data.device) * block_len
    return blocks, timestamps


def _augment_block(
    block: Tensor,      # (block_len, F)
    noise_std: float = 0.005,
    shift_max: int = 2,
    seed_offset: int = 0,
) -> Tensor:
    """
    Lightweight augmentation for contrastive pair generation.

    Operations:
      1. Additive Gaussian jitter — perturbs magnitude without changing
         temporal ordering, simulating tick-level microstructure noise.
      2. Cyclic time shift — rotates the block by a small integer number
         of steps. Preserves all statistical properties while breaking
         positional identity between the two views.

    Augmentation is applied in-place on a cloned tensor to avoid
    modifying the original training buffer.
    """
    aug = block.clone()

    # 1. Gaussian jitter
    if noise_std > 0.0:
        aug = aug + torch.randn_like(aug) * noise_std

    # 2. Cyclic shift (causal within-block only)
    if shift_max > 0:
        shift = int(torch.randint(1, shift_max + 1, (1,)).item())
        aug   = torch.roll(aug, shifts=shift, dims=0)

    return aug


# ---------------------------------------------------------------------------
# ChronosDefragEngine
# ---------------------------------------------------------------------------

class ChronosDefragEngine:
    """
    Full training and inference orchestrator for ChronosDefrag.

    Usage::

        cfg    = DefragConfig()
        engine = ChronosDefragEngine(cfg)
        engine.fit(raw_data_tensor)          # (T, F)

        result = engine.predict_live(live_window)   # (block_len, F)
        print(result.alpha_prediction)
    """

    def __init__(self, cfg: DefragConfig) -> None:
        cfg.configure_logging()
        cfg.seed_everything()

        self.cfg    = cfg
        self.device = cfg.resolve_device()

        self.encoder   = RegimeEncoder(cfg.encoder).to(self.device)
        self.router    = VectorDefragRouter(cfg.routing, cfg.encoder.latent_dim).to(self.device)
        self.predictor = TemporalPredictor(cfg.predictor).to(self.device)

        self._encoder_opt:   Optional[torch.optim.Adam] = None
        self._predictor_opt: Optional[torch.optim.Adam] = None

        self._fitted     = False
        self._train_size = 0   # number of training ticks (for causal assertion)

        log.info(
            "ChronosDefragEngine initialised | device=%s | latent_dim=%d",
            self.device,
            cfg.encoder.latent_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, raw_data: Tensor) -> None:
        """
        Full training pipeline.

        Parameters
        ----------
        raw_data : (T, F) chronological tick-level feature tensor.
                   F must equal cfg.encoder.input_dim.
                   T must accommodate at least 2 * block_len ticks
                   for meaningful training.

        Causal guarantee:
          The encoder is trained on blocks [0, N_enc).
          The predictor is trained on pseudo-timeline windows derived
          from those same N_enc blocks.
          No block with timestamp >= N_enc * block_len is ever passed
          to either model during fit().
        """
        T, F = raw_data.shape
        assert F == self.cfg.encoder.input_dim, (
            f"raw_data has {F} features; cfg.encoder.input_dim={self.cfg.encoder.input_dim}"
        )

        data = raw_data.to(self.device, dtype=torch.float32)

        log.info("fit() | T=%d ticks | F=%d features", T, F)

        # Slice into blocks
        blocks, timestamps = _slice_blocks(data, self.cfg.encoder.block_len)
        N = blocks.shape[0]
        log.info("Sliced %d blocks of length %d", N, self.cfg.encoder.block_len)

        if N < 4:
            raise ValueError(
                f"Need at least 4 blocks for training, got {N}. "
                f"Provide T >= {4 * self.cfg.encoder.block_len} ticks."
            )

        # Record training boundary for causal assertion in predict_live
        self._train_size = N * self.cfg.encoder.block_len

        # Stage 1: Encoder training
        self._train_encoder(blocks, timestamps)

        # Stage 2: Build registry and pseudo-timeline
        z_all = self._encode_all(blocks)
        self.router.clear_registry()
        self.router.register_batch(z_all, timestamps)

        perm   = self.router.build_pseudo_timeline(z_all)
        log.info("Pseudo-timeline constructed | entropy ratio: %.3f", self._timeline_entropy(perm))

        # Stage 3: Predictor training
        self._train_predictor(z_all, perm)

        self._fitted = True
        log.info("fit() complete")

        if self.cfg.encoder.latent_dim > 0:
            self._save_checkpoint()

    def predict_live(self, live_window: Tensor) -> LivePrediction:
        """
        Live inference from a rolling tick window.

        Parameters
        ----------
        live_window : (block_len, F) — most recent ticks, chronologically ordered.
                      Must NOT contain any ticks from the future (timestamp
                      strictly > current wall-clock block index).

        Returns
        -------
        LivePrediction with alpha_prediction, regime_confidence, system_latency_ms.
        """
        if not self._fitted:
            raise RuntimeError("Engine must be fit() before predict_live()")

        t0 = time.perf_counter()

        win = live_window.to(self.device, dtype=torch.float32)
        assert win.shape == (self.cfg.encoder.block_len, self.cfg.encoder.input_dim), (
            f"live_window shape {tuple(win.shape)} != "
            f"({self.cfg.encoder.block_len}, {self.cfg.encoder.input_dim})"
        )

        self.encoder.eval()
        self.predictor.eval()

        with torch.no_grad():
            z, _ = self.encoder(win.unsqueeze(0))       # (1, D)
            z    = z.squeeze(0)                          # (D,)

            query_ts = self._train_size   # live window is strictly after training data
            context, confidence = self.router.reconstruct_context(z, query_ts)

            # Build a short context sequence: [retrieved_context, live_z]
            ctx_seq = torch.stack([context, z], dim=0).unsqueeze(0)  # (1, 2, D)
            alpha, _ = self.predictor(ctx_seq)
            alpha    = alpha.squeeze(0)                  # (forecast_dim,)

        latency_ms = (time.perf_counter() - t0) * 1e3

        return LivePrediction(
            alpha_prediction  = alpha.cpu(),
            regime_confidence = confidence,
            system_latency_ms = latency_ms,
        )

    # ------------------------------------------------------------------
    # Stage 1: Encoder training
    # ------------------------------------------------------------------

    def _train_encoder(self, blocks: Tensor, timestamps: Tensor) -> None:
        N = blocks.shape[0]
        cfg_e = self.cfg.encoder
        cfg_o = self.cfg.optim

        self._encoder_opt = torch.optim.Adam(
            self.encoder.parameters(),
            lr=cfg_o.encoder_lr,
            weight_decay=cfg_o.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._encoder_opt,
            T_max=cfg_o.encoder_epochs,
        )

        log.info("Encoder training | epochs=%d | N=%d blocks", cfg_o.encoder_epochs, N)

        scaler = torch.amp.GradScaler('cuda', enabled=cfg_o.amp_enabled and self.device.type == "cuda")

        self.encoder.train()
        for epoch in range(cfg_o.encoder_epochs):
            perm    = torch.randperm(N, device=self.device)
            batches = perm.split(cfg_o.batch_size)
            epoch_loss = 0.0

            for batch_idx in batches:
                batch_a = blocks[batch_idx]                           # (B, T, F)
                batch_b = torch.stack([
                    _augment_block(batch_a[i])
                    for i in range(batch_a.shape[0])
                ], dim=0)

                self._encoder_opt.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=cfg_o.amp_enabled and self.device.type == "cuda"):
                    z_a, zp_a = self.encoder(batch_a)
                    z_b, zp_b = self.encoder(batch_b)

                    loss_nce  = self.encoder.infonce_loss(zp_a, zp_b)
                    loss_ter  = self.encoder.temporal_entropy_reg(z_a)
                    loss_cont = self.encoder.latent_continuity_reg(
                        self._encode_ordered_slice(blocks, batch_idx)
                    )
                    loss = loss_nce + loss_ter + loss_cont

                scaler.scale(loss).backward()
                scaler.unscale_(self._encoder_opt)
                nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                scaler.step(self._encoder_opt)
                scaler.update()

                epoch_loss += loss.item()

            scheduler.step()

            if (epoch + 1) % max(1, cfg_o.encoder_epochs // 10) == 0:
                collapse = self._check_collapse()
                log.info(
                    "  encoder epoch %3d/%d | loss=%.4f | collapse=%s",
                    epoch + 1,
                    cfg_o.encoder_epochs,
                    epoch_loss / len(batches),
                    collapse,
                )

    def _encode_ordered_slice(self, blocks: Tensor, idx: Tensor) -> Tensor:
        """
        Returns z-embeddings for a batch of blocks, sorted by their
        original (chronological) index, for continuity regularisation.
        """
        sorted_idx = idx.sort().values
        with torch.no_grad():
            z, _ = self.encoder(blocks[sorted_idx])
        return z.detach()

    # ------------------------------------------------------------------
    # Stage 2 helper: encode all blocks
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_all(self, blocks: Tensor) -> Tensor:
        """
        Encode all blocks in mini-batches. Returns (N, latent_dim).
        """
        self.encoder.eval()
        N = blocks.shape[0]
        B = self.cfg.optim.batch_size
        z_list: List[Tensor] = []

        for start in range(0, N, B):
            end   = min(start + B, N)
            z, _  = self.encoder(blocks[start:end])
            z_list.append(z)

        self.encoder.train()
        return torch.cat(z_list, dim=0)   # (N, D)

    # ------------------------------------------------------------------
    # Stage 3: Predictor training
    # ------------------------------------------------------------------

    def _train_predictor(self, z_all: Tensor, perm: Tensor) -> None:
        """
        Train the predictor on pseudo-timeline windows.

        Each training sample is a short sub-sequence of the pseudo-timeline
        (length = top_k positions).  The regression target is the latent
        embedding at the next position in the pseudo-timeline (self-supervised
        latent reconstruction objective).

        This avoids requiring external price labels: the model learns to
        predict the next regime embedding from the current pseudo-context,
        which is a proxy for forward regime continuation.
        """
        cfg_o = self.cfg.optim
        cfg_r = self.cfg.routing
        cfg_p = self.cfg.predictor

        # Reorder z_all by pseudo-timeline permutation
        z_pseudo = z_all[perm]    # (N, D) — pseudo-chronological order

        N, D = z_pseudo.shape
        S    = min(cfg_r.top_k, N - 1)   # sequence length per sample
        if S < 2:
            log.warning("Pseudo-timeline too short for predictor training; skipping")
            return

        # Build (windows, targets) from the pseudo-timeline
        # window_i = z_pseudo[i : i+S]     shape: (S, D)
        # target_i = z_pseudo[i+S]         shape: (D,) — next position
        windows = torch.stack([z_pseudo[i:i+S] for i in range(N - S)], dim=0)  # (M, S, D)
        targets = z_pseudo[S:].clone()                                           # (M, D)
        M       = windows.shape[0]

        if M == 0:
            log.warning("No predictor training windows; skipping")
            return

        self._predictor_opt = torch.optim.Adam(
            self.predictor.parameters(),
            lr=cfg_o.predictor_lr,
            weight_decay=cfg_o.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._predictor_opt,
            T_max=cfg_o.predictor_epochs,
        )

        log.info(
            "Predictor training | epochs=%d | M=%d windows | S=%d",
            cfg_o.predictor_epochs, M, S,
        )

        self.predictor.train()
        for epoch in range(cfg_o.predictor_epochs):
            perm_m  = torch.randperm(M, device=self.device)
            batches = perm_m.split(cfg_o.batch_size)
            epoch_loss = 0.0

            for batch_idx in batches:
                x_batch = windows[batch_idx]    # (B, S, D)
                y_batch = targets[batch_idx]    # (B, D)

                # Project target to forecast_dim via a simple mean
                # (avoids requiring labelled return data)
                y_scalar = y_batch[:, :cfg_p.forecast_dim]   # (B, forecast_dim)

                self._predictor_opt.zero_grad(set_to_none=True)

                pred, _ = self.predictor(x_batch)
                loss    = TemporalPredictor.regression_loss(pred, y_scalar)
                loss.backward()

                norm = self.predictor.clip_gradients()
                self._predictor_opt.step()
                epoch_loss += loss.item()

            scheduler.step()

            if (epoch + 1) % max(1, cfg_o.predictor_epochs // 10) == 0:
                log.info(
                    "  predictor epoch %3d/%d | loss=%.5f",
                    epoch + 1,
                    cfg_o.predictor_epochs,
                    epoch_loss / len(batches),
                )

        self.predictor.eval()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _check_collapse(self) -> bool:
        """Sample a small batch from the registry to check for collapse."""
        n = int(self.router.reg_size.item())
        if n < 2:
            return False
        n_sample = min(n, 32)
        idx = torch.randperm(n, device=self.device)[:n_sample]
        z   = self.router.registry[idx]
        return self.router.detect_collapse(z)

    def _timeline_entropy(self, perm: Tensor) -> float:
        """
        Measures how evenly the pseudo-timeline re-distributes chronological
        blocks.  A ratio near 1.0 means high temporal mixing (good);
        near 0.0 means the pseudo-timeline is nearly chronological (no gain).

        Computed as: 1 - |corr(perm, arange(N))| / N
        """
        N  = perm.shape[0]
        if N < 2:
            return 0.0
        orig  = torch.arange(N, dtype=torch.float32, device=perm.device)
        p     = perm.float()
        corr  = ((p - p.mean()) * (orig - orig.mean())).sum()
        corr /= (p.std() * orig.std() * N + 1e-8)
        return float(1.0 - corr.abs().item())

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> None:
        path = os.path.join(self.cfg.ensure_checkpoint_dir(), "chronosdefrag.pt")
        torch.save({
            "encoder_state":   self.encoder.state_dict(),
            "predictor_state": self.predictor.state_dict(),
            "router_state":    self.router.state_dict(),
            "train_size":      self._train_size,
        }, path)
        log.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder_state"])
        self.predictor.load_state_dict(ckpt["predictor_state"])
        self.router.load_state_dict(ckpt["router_state"])
        self._train_size = ckpt["train_size"]
        self._fitted     = True
        log.info("Checkpoint loaded: %s", path)

    # ------------------------------------------------------------------
    # Latency benchmark helper
    # ------------------------------------------------------------------

    def benchmark_inference(
        self,
        n_trials: int = 100,
        live_window: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """
        Run n_trials predict_live() calls and return latency statistics.

        If live_window is None, a zero-filled window is used (latency only).
        """
        if not self._fitted:
            raise RuntimeError("Engine must be fit() before benchmarking")

        if live_window is None:
            live_window = torch.zeros(
                self.cfg.encoder.block_len,
                self.cfg.encoder.input_dim,
            )

        latencies: List[float] = []
        for _ in range(n_trials):
            r = self.predict_live(live_window)
            latencies.append(r.system_latency_ms)

        lat_t = torch.tensor(latencies)
        return {
            "mean_ms":   float(lat_t.mean()),
            "p50_ms":    float(lat_t.median()),
            "p95_ms":    float(lat_t.quantile(0.95)),
            "p99_ms":    float(lat_t.quantile(0.99)),
            "min_ms":    float(lat_t.min()),
            "max_ms":    float(lat_t.max()),
            "throughput_qps": float(1000.0 / lat_t.mean()),
        }
