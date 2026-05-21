"""
defrag/config.py

Structured configuration system for ChronosDefrag.
All hyperparameters are typed, validated, and grouped by subsystem.
Deterministic seeding and device resolution are handled here.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deployment Mode
# ---------------------------------------------------------------------------

class DeploymentMode(Enum):
    RESEARCH   = auto()   # full logging, shape assertions, extra diagnostics
    STAGING    = auto()   # reduced logging, assertions enabled
    PRODUCTION = auto()   # minimal overhead, assertions disabled


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    """Conv1D residual encoder hyperparameters."""
    input_dim:       int   = 6       # number of input features per tick
    block_len:       int   = 64      # temporal block length (ticks per window)
    hidden_dim:      int   = 128     # internal channel width
    latent_dim:      int   = 64      # output latent embedding dimension
    projection_dim:  int   = 32      # contrastive projection head output dim
    num_layers:      int   = 4       # number of residual Conv1D layers
    kernel_size:     int   = 3       # Conv1D kernel size (causal padding applied)
    dropout:         float = 0.1     # dropout probability in residual blocks
    temperature:     float = 0.07    # InfoNCE temperature (learnable init value)
    entropy_weight:  float = 0.01    # temporal entropy regularisation coefficient
    continuity_weight: float = 0.005 # latent continuity regularisation coefficient

    def __post_init__(self) -> None:
        assert self.block_len > 0 and (self.block_len & (self.block_len - 1)) == 0, \
            "block_len must be a positive power of 2 for efficient FFT-adjacent ops"
        assert 0.0 < self.temperature < 1.0, "temperature must be in (0, 1)"
        assert self.num_layers >= 2, "minimum 2 residual layers required"
        assert self.kernel_size % 2 == 1, "kernel_size must be odd for symmetric causal padding"


@dataclass
class RoutingConfig:
    """VectorDefragRouter hyperparameters."""
    registry_capacity:  int   = 8192   # maximum historical embeddings stored
    top_k:              int   = 16     # neighbours retrieved per live query
    similarity_decay:   float = 0.92   # exponential decay for similarity persistence
    min_confidence:     float = 0.05   # minimum cosine similarity to accept a match
    stitching_overlap:  int   = 8      # overlap (ticks) between stitched blocks
    collapse_margin:    float = 0.02   # minimum pairwise distance to flag collapse

    def __post_init__(self) -> None:
        assert self.top_k <= self.registry_capacity, \
            "top_k cannot exceed registry_capacity"
        assert 0.0 < self.similarity_decay <= 1.0
        assert self.stitching_overlap >= 0


@dataclass
class PredictorConfig:
    """TemporalPredictor hyperparameters."""
    latent_dim:     int   = 64    # must match EncoderConfig.latent_dim
    hidden_dim:     int   = 128   # GLU / SSM hidden width
    num_layers:     int   = 3     # stacked recurrent/GLU depth
    forecast_dim:   int   = 1     # alpha projection output dimensionality
    dropout:        float = 0.05
    grad_clip:      float = 1.0   # gradient norm clipping threshold

    def __post_init__(self) -> None:
        assert self.forecast_dim >= 1
        assert self.grad_clip > 0.0


@dataclass
class OptimizationConfig:
    """Shared optimisation schedule."""
    encoder_lr:     float = 3e-4
    predictor_lr:   float = 1e-3
    weight_decay:   float = 1e-5
    encoder_epochs: int   = 40
    predictor_epochs: int = 30
    batch_size:     int   = 64
    warmup_steps:   int   = 100
    amp_enabled:    bool  = False   # automatic mixed precision (GPU only)

    def __post_init__(self) -> None:
        assert self.encoder_lr > 0 and self.predictor_lr > 0
        assert self.batch_size >= 8, "batch_size too small for stable InfoNCE"


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class DefragConfig:
    """
    Single entry-point configuration for the ChronosDefrag system.

    Instantiate once; pass by reference through the call stack.
    Never mutate after engine initialisation.
    """
    mode:         DeploymentMode    = DeploymentMode.RESEARCH
    seed:         int               = 42
    device:       str               = "auto"        # "auto" | "cpu" | "cuda" | "mps"
    log_level:    str               = "INFO"
    checkpoint_dir: str             = "./checkpoints"

    encoder:      EncoderConfig     = field(default_factory=EncoderConfig)
    routing:      RoutingConfig     = field(default_factory=RoutingConfig)
    predictor:    PredictorConfig   = field(default_factory=PredictorConfig)
    optim:        OptimizationConfig = field(default_factory=OptimizationConfig)

    def __post_init__(self) -> None:
        # Cross-config consistency
        assert self.encoder.latent_dim == self.predictor.latent_dim, (
            f"encoder.latent_dim ({self.encoder.latent_dim}) must match "
            f"predictor.latent_dim ({self.predictor.latent_dim})"
        )
        self._resolved_device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    def resolve_device(self) -> torch.device:
        if self._resolved_device is not None:
            return self._resolved_device

        if self.device == "auto":
            if torch.cuda.is_available():
                dev = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = torch.device("mps")
            else:
                dev = torch.device("cpu")
        else:
            dev = torch.device(self.device)

        # AMP only meaningful on CUDA
        if self.optim.amp_enabled and dev.type != "cuda":
            log.warning("amp_enabled=True ignored on non-CUDA device (%s)", dev)
            object.__setattr__(self.optim, "amp_enabled", False)

        self._resolved_device = dev
        log.info("Device resolved: %s", dev)
        return dev

    # ------------------------------------------------------------------
    # Deterministic seeding
    # ------------------------------------------------------------------

    def seed_everything(self) -> None:
        """
        Global deterministic seeding.
        Covers Python random, NumPy, PyTorch CPU and CUDA RNGs.
        Does NOT set CUBLAS_WORKSPACE_CONFIG — callers requiring full
        determinism on CUDA must set that env var before importing torch.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        log.debug("Global RNG seeded: %d", self.seed)

    # ------------------------------------------------------------------
    # Logging bootstrap
    # ------------------------------------------------------------------

    def configure_logging(self) -> None:
        numeric = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )

    # ------------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------------

    def ensure_checkpoint_dir(self) -> str:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        return self.checkpoint_dir

    # ------------------------------------------------------------------
    # Assertions guard
    # ------------------------------------------------------------------

    @property
    def assertions_enabled(self) -> bool:
        return self.mode in (DeploymentMode.RESEARCH, DeploymentMode.STAGING)
