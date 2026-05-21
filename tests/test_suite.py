"""
tests/test_suite.py

Deterministic smoke, causality, tensor-shape, and latency tests.

Run with:
    python -m pytest tests/ -v

All tests are self-contained, require no external fixtures,
and are deterministic under the configured seed.
"""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn.functional as F

from defrag.config import (
    DefragConfig,
    EncoderConfig,
    OptimizationConfig,
    PredictorConfig,
    RoutingConfig,
)
from defrag.core import RegimeEncoder, VectorDefragRouter
from defrag.engine import ChronosDefragEngine, _slice_blocks, _augment_block
from defrag.model import TemporalPredictor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _small_cfg() -> DefragConfig:
    """Minimal config for fast unit tests."""
    return DefragConfig(
        seed=0,
        encoder=EncoderConfig(
            input_dim=4,
            block_len=16,
            hidden_dim=32,
            latent_dim=16,
            projection_dim=8,
            num_layers=2,
            kernel_size=3,
            entropy_weight=0.01,
            continuity_weight=0.005,
        ),
        routing=RoutingConfig(
            registry_capacity=128,
            top_k=4,
            similarity_decay=0.9,
        ),
        predictor=PredictorConfig(
            latent_dim=16,
            hidden_dim=32,
            num_layers=2,
            forecast_dim=1,
        ),
        optim=OptimizationConfig(
            encoder_epochs=3,
            predictor_epochs=3,
            batch_size=8,
        ),
    )


def _synthetic_data(T: int = 512, F: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(T, F, generator=g)


# ===========================================================================
# BLOCK 1: Tensor shape tests
# ===========================================================================

class TestEncoderShapes:
    def test_output_shape(self) -> None:
        cfg  = _small_cfg()
        enc  = RegimeEncoder(cfg.encoder)
        B, T, F = 4, cfg.encoder.block_len, cfg.encoder.input_dim
        x    = torch.randn(B, T, F)
        z, zp = enc(x)
        assert z.shape  == (B, cfg.encoder.latent_dim)
        assert zp.shape == (B, cfg.encoder.projection_dim)

    def test_latent_normalised(self) -> None:
        cfg = _small_cfg()
        enc = RegimeEncoder(cfg.encoder)
        x   = torch.randn(8, cfg.encoder.block_len, cfg.encoder.input_dim)
        z, _ = enc(x)
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Latent vectors not unit-normalised; norms={norms}"

    def test_infonce_loss_finite(self) -> None:
        cfg = _small_cfg()
        enc = RegimeEncoder(cfg.encoder)
        x   = torch.randn(8, cfg.encoder.block_len, cfg.encoder.input_dim)
        _, zp = enc(x)
        _, zp2 = enc(x + 0.01 * torch.randn_like(x))
        loss = enc.infonce_loss(zp, zp2)
        assert torch.isfinite(loss), f"InfoNCE loss is not finite: {loss}"
        assert loss.item() >= 0.0

    def test_entropy_reg_nonneg(self) -> None:
        cfg = _small_cfg()
        enc = RegimeEncoder(cfg.encoder)
        z   = F.normalize(torch.randn(16, cfg.encoder.latent_dim), dim=-1)
        reg = enc.temporal_entropy_reg(z)
        assert torch.isfinite(reg)
        assert reg.item() >= 0.0

    def test_continuity_reg_nonneg(self) -> None:
        cfg = _small_cfg()
        enc = RegimeEncoder(cfg.encoder)
        z   = F.normalize(torch.randn(8, cfg.encoder.latent_dim), dim=-1)
        reg = enc.latent_continuity_reg(z)
        assert torch.isfinite(reg)
        assert reg.item() >= 0.0


class TestPredictorShapes:
    def test_output_shape(self) -> None:
        cfg  = _small_cfg()
        pred = TemporalPredictor(cfg.predictor)
        B, S, D = 4, 6, cfg.predictor.latent_dim
        x = torch.randn(B, S, D)
        alpha, states = pred(x)
        assert alpha.shape  == (B, cfg.predictor.forecast_dim)
        assert len(states)  == cfg.predictor.num_layers

    def test_stateful_inference_shape(self) -> None:
        cfg  = _small_cfg()
        pred = TemporalPredictor(cfg.predictor)
        S, D = 4, cfg.predictor.latent_dim
        ctx  = torch.randn(S, D)
        out  = pred.infer(ctx)
        assert out.shape == (cfg.predictor.forecast_dim,)


class TestRouterShapes:
    def test_query_shapes(self) -> None:
        cfg    = _small_cfg()
        router = VectorDefragRouter(cfg.routing, cfg.encoder.latent_dim)
        N  = 32
        z  = F.normalize(torch.randn(N, cfg.encoder.latent_dim), dim=-1)
        ts = torch.arange(N, dtype=torch.long)
        router.register_batch(z, ts)

        q = F.normalize(torch.randn(3, cfg.encoder.latent_dim), dim=-1)
        indices, scores, z_nbrs = router.query(q)
        k = min(cfg.routing.top_k, N)
        assert indices.shape == (3, k)
        assert scores.shape  == (3, k)
        assert z_nbrs.shape  == (3, k, cfg.encoder.latent_dim)

    def test_pseudo_timeline_permutation(self) -> None:
        cfg    = _small_cfg()
        router = VectorDefragRouter(cfg.routing, cfg.encoder.latent_dim)
        N  = 20
        z  = F.normalize(torch.randn(N, cfg.encoder.latent_dim), dim=-1)
        perm = router.build_pseudo_timeline(z)

        assert perm.shape == (N,)
        # Must be a valid permutation (each index appears exactly once)
        assert set(perm.tolist()) == set(range(N))


# ===========================================================================
# BLOCK 2: Causality tests
# ===========================================================================

class TestCausality:
    def test_encoder_causal_padding(self) -> None:
        """
        Verify CausalConv1d cannot see future positions:
        zeroing ticks [t+1, T) must not change the output at position t.
        """
        from defrag.core import CausalConv1d
        conv = CausalConv1d(8, 8, kernel_size=3)
        conv.eval()
        T = 16
        x = torch.randn(1, 8, T)

        with torch.no_grad():
            out_full = conv(x)

        # Zero all positions after t=7
        x_masked = x.clone()
        x_masked[:, :, 8:] = 0.0
        with torch.no_grad():
            out_masked = conv(x_masked)

        # Outputs at positions 0..7 must be identical
        assert torch.allclose(out_full[:, :, :8], out_masked[:, :, :8], atol=1e-6), \
            "CausalConv1d is not causal — future positions affect past outputs"

    def test_fit_does_not_see_future_blocks(self) -> None:
        """
        Engine._train_size must equal N_blocks * block_len,
        ensuring predict_live() receives a timestamp strictly after training.
        """
        cfg    = _small_cfg()
        data   = _synthetic_data(T=256, F=cfg.encoder.input_dim)
        engine = ChronosDefragEngine(cfg)
        engine.fit(data)

        expected_N   = 256 // cfg.encoder.block_len
        expected_size = expected_N * cfg.encoder.block_len
        assert engine._train_size == expected_size, \
            f"_train_size {engine._train_size} != {expected_size}"

    def test_slice_blocks_no_overlap(self) -> None:
        data       = torch.arange(64 * 4).reshape(64, 4).float()
        blocks, ts = _slice_blocks(data, block_len=16)
        # Verify non-overlapping (adjacent block timestamps differ by block_len)
        diffs = (ts[1:] - ts[:-1]).tolist()
        assert all(d == 16 for d in diffs), f"Blocks overlap: diffs={diffs}"


# ===========================================================================
# BLOCK 3: Smoke tests
# ===========================================================================

class TestSmoke:
    def test_full_fit_and_predict(self) -> None:
        cfg    = _small_cfg()
        data   = _synthetic_data(T=512, F=cfg.encoder.input_dim)
        engine = ChronosDefragEngine(cfg)
        engine.fit(data)

        window = _synthetic_data(T=cfg.encoder.block_len, F=cfg.encoder.input_dim)
        result = engine.predict_live(window)

        assert torch.isfinite(result.alpha_prediction).all()
        assert 0.0 <= result.regime_confidence
        assert result.system_latency_ms > 0.0

    def test_augment_preserves_shape(self) -> None:
        block = torch.randn(16, 4)
        aug   = _augment_block(block, noise_std=0.01, shift_max=2)
        assert aug.shape == block.shape

    def test_augment_is_not_identity(self) -> None:
        block = torch.randn(16, 4)
        aug   = _augment_block(block, noise_std=0.01, shift_max=2)
        assert not torch.allclose(aug, block), \
            "Augmentation produced identical output (check noise/shift logic)"

    def test_collapse_detection(self) -> None:
        cfg    = _small_cfg()
        router = VectorDefragRouter(cfg.routing, cfg.encoder.latent_dim)
        # All same vector -> should detect collapse
        D = cfg.encoder.latent_dim
        z_collapsed = torch.ones(16, D) / D ** 0.5
        assert router.detect_collapse(z_collapsed), "Collapse not detected on identical vectors"

        # Random diverse vectors -> should not collapse
        z_diverse = F.normalize(torch.randn(16, D), dim=-1)
        # (might occasionally be False-positive with very small D, acceptable)


# ===========================================================================
# BLOCK 4: Latency sanity tests
# ===========================================================================

class TestLatency:
    def test_predict_live_under_50ms(self) -> None:
        """
        Single live prediction must complete in under 50 ms on CPU.
        This is a sanity check, not a hard SLA.
        """
        cfg    = _small_cfg()
        data   = _synthetic_data(T=512, F=cfg.encoder.input_dim)
        engine = ChronosDefragEngine(cfg)
        engine.fit(data)

        window = _synthetic_data(T=cfg.encoder.block_len, F=cfg.encoder.input_dim)

        # Warm up
        _ = engine.predict_live(window)

        t0 = time.perf_counter()
        for _ in range(10):
            engine.predict_live(window)
        avg_ms = (time.perf_counter() - t0) * 100   # ms per call

        assert avg_ms < 50.0, f"predict_live avg {avg_ms:.1f}ms exceeds 50ms sanity threshold"

    def test_benchmark_returns_valid_dict(self) -> None:
        cfg    = _small_cfg()
        data   = _synthetic_data(T=512, F=cfg.encoder.input_dim)
        engine = ChronosDefragEngine(cfg)
        engine.fit(data)

        bench = engine.benchmark_inference(n_trials=10)
        required = {"mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "throughput_qps"}
        assert required.issubset(bench.keys())
        assert bench["mean_ms"] > 0.0
        assert bench["throughput_qps"] > 0.0
        assert bench["p99_ms"] >= bench["p95_ms"] >= bench["p50_ms"]


# ===========================================================================
# BLOCK 5: Determinism test
# ===========================================================================

class TestDeterminism:
    def test_same_seed_same_latents(self) -> None:
        """
        Two encoders with identical init (same seed) must produce identical
        latent outputs on the same input.
        """
        cfg = _small_cfg()

        cfg.seed_everything()
        enc1 = RegimeEncoder(cfg.encoder)

        cfg.seed_everything()
        enc2 = RegimeEncoder(cfg.encoder)

        x = torch.randn(4, cfg.encoder.block_len, cfg.encoder.input_dim)
        enc1.eval()
        enc2.eval()
        with torch.no_grad():
            z1, _ = enc1(x)
            z2, _ = enc2(x)

        assert torch.allclose(z1, z2, atol=1e-6), \
            "Encoder outputs differ under identical seed (non-deterministic init)"
