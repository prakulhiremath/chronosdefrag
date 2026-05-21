"""
main.py — ChronosDefrag sandbox + verification suite.

Synthetic market generator produces three distinct regime types:

  REGIME_A  low-volatility mean-reversion   AR(1) with |phi| < 0.5
  REGIME_B  high-volatility trending        random-walk with drift
  REGIME_C  stochastic switching            Markov chain between A and B

All data is deterministic given the seed in DefragConfig.
No external datasets are required.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from typing import List

import torch

from defrag.config import DefragConfig, EncoderConfig, OptimizationConfig, RoutingConfig, PredictorConfig
from defrag.engine import ChronosDefragEngine

log = logging.getLogger("chronosdefrag.main")


# ---------------------------------------------------------------------------
# Synthetic market data generator
# ---------------------------------------------------------------------------

def _generate_regime(
    n_ticks: int,
    n_features: int,
    regime: str,
    seed: int,
) -> torch.Tensor:
    """
    Produces a (n_ticks, n_features) float32 tensor for a single regime.

    Feature layout (all normalised to unit scale):
        0  log-return
        1  realised vol (rolling std of log-returns, causal)
        2  order imbalance proxy (signed random walk)
        3  spread proxy (always positive)
        4  momentum (EMA of log-return)
        5  mean-reversion signal (deviation from rolling mean)
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    out = torch.zeros(n_ticks, n_features)

    price = torch.ones(n_ticks + 1)

    if regime == "mean_reversion":
        phi    = 0.35
        sigma  = 0.002
        mu     = 0.0
        noise  = torch.randn(n_ticks, generator=rng) * sigma
        for t in range(n_ticks):
            ret       = phi * (mu - math.log(float(price[t]))) + noise[t]
            price[t+1] = price[t] * math.exp(ret)

    elif regime == "trending":
        drift  = 0.0003
        sigma  = 0.008
        noise  = torch.randn(n_ticks, generator=rng) * sigma
        for t in range(n_ticks):
            ret       = drift + noise[t]
            price[t+1] = price[t] * math.exp(ret)

    elif regime == "switching":
        in_trend = False
        switch_p = 0.04
        sigma_mr = 0.002
        sigma_tr = 0.007
        phi      = 0.3
        drift    = 0.0002
        u        = torch.rand(n_ticks, generator=rng)
        noise    = torch.randn(n_ticks, generator=rng)
        for t in range(n_ticks):
            if u[t] < switch_p:
                in_trend = not in_trend
            if in_trend:
                ret = drift + noise[t] * sigma_tr
            else:
                ret = phi * (-math.log(float(price[t]))) + noise[t] * sigma_mr
            price[t+1] = price[t] * math.exp(ret)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    log_ret = torch.log(price[1:]) - torch.log(price[:-1])

    # Feature 0: log-return
    out[:, 0] = log_ret

    # Feature 1: realised vol (causal rolling std, window=20)
    for t in range(n_ticks):
        start = max(0, t - 19)
        chunk = log_ret[start:t+1]
        out[t, 1] = chunk.std().clamp(min=1e-8) if chunk.numel() > 1 else torch.tensor(1e-8)

    # Feature 2: order imbalance (signed cumulative, normalised)
    oi = torch.randn(n_ticks, generator=rng) * 0.01
    oi = oi.cumsum(0)
    oi = (oi - oi.mean()) / (oi.std() + 1e-8)
    out[:, 2] = oi

    # Feature 3: spread proxy (always positive, regime-dependent scale)
    spread_scale = {"mean_reversion": 0.0005, "trending": 0.0015, "switching": 0.001}[regime]
    out[:, 3] = (torch.rand(n_ticks, generator=rng) * spread_scale).abs()

    # Feature 4: EMA momentum (alpha = 0.1)
    ema = torch.zeros(n_ticks)
    a   = 0.1
    for t in range(n_ticks):
        ema[t] = a * log_ret[t] + (1 - a) * (ema[t-1] if t > 0 else 0.0)
    out[:, 4] = ema

    # Feature 5: mean-reversion signal (deviation from rolling mean)
    for t in range(n_ticks):
        start  = max(0, t - 19)
        mu_t   = log_ret[start:t+1].mean()
        out[t, 5] = log_ret[t] - mu_t

    return out


def generate_synthetic_market(
    n_ticks_per_regime: int = 2048,
    n_features: int = 6,
    seed: int = 42,
) -> torch.Tensor:
    """
    Concatenates three distinct synthetic regime segments into a single
    chronological tensor of shape (3 * n_ticks_per_regime, n_features).
    """
    regimes = [
        _generate_regime(n_ticks_per_regime, n_features, "mean_reversion", seed),
        _generate_regime(n_ticks_per_regime, n_features, "trending",       seed + 1),
        _generate_regime(n_ticks_per_regime, n_features, "switching",      seed + 2),
    ]
    data = torch.cat(regimes, dim=0)   # (3*T, F)

    # Global z-score normalisation (causal: computed over full training set here)
    mu  = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True).clamp(min=1e-8)
    return (data - mu) / std


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

DIVIDER = "─" * 60


def _header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def _tick_report(i: int, result, t_wall: float) -> None:
    alpha_val = float(result.alpha_prediction[0])
    sign      = "+" if alpha_val >= 0 else ""
    conf_pct  = result.regime_confidence * 100.0
    print(
        f"  [TICK {i:04d}]  "
        f"latency={result.system_latency_ms:5.2f}ms  "
        f"confidence={conf_pct:5.1f}%  "
        f"alpha={sign}{alpha_val:.5f}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    cfg = DefragConfig(
        seed=42,
        encoder=EncoderConfig(
            input_dim=6,
            block_len=64,
            hidden_dim=128,
            latent_dim=64,
            projection_dim=32,
            num_layers=4,
            kernel_size=3,
            entropy_weight=0.01,
            continuity_weight=0.005,
        ),
        routing=RoutingConfig(
            registry_capacity=2048,
            top_k=8,
            similarity_decay=0.92,
        ),
        predictor=PredictorConfig(
            latent_dim=64,
            hidden_dim=128,
            num_layers=3,
            forecast_dim=1,
        ),
        optim=OptimizationConfig(
            encoder_epochs=30,
            predictor_epochs=20,
            batch_size=32,
            encoder_lr=3e-4,
            predictor_lr=1e-3,
        ),
    )

    cfg.configure_logging()
    cfg.seed_everything()

    _header("ChronosDefrag — Synthetic Verification Suite")
    print(f"  device      : {cfg.resolve_device()}")
    print(f"  seed        : {cfg.seed}")
    print(f"  block_len   : {cfg.encoder.block_len}")
    print(f"  latent_dim  : {cfg.encoder.latent_dim}")
    print(f"  top_k       : {cfg.routing.top_k}")

    # ------------------------------------------------------------------
    # Synthetic data
    # ------------------------------------------------------------------
    _header("Generating Synthetic Market Data")
    T_per_regime = 2048
    data = generate_synthetic_market(
        n_ticks_per_regime=T_per_regime,
        n_features=cfg.encoder.input_dim,
        seed=cfg.seed,
    )
    T, F = data.shape
    print(f"  shape       : {T} ticks × {F} features")
    print(f"  regimes     : mean-reversion | trending | switching")
    print(f"  feature μ   : {data.mean(0).tolist()}")
    print(f"  feature σ   : {[f'{v:.3f}' for v in data.std(0).tolist()]}")

    # ------------------------------------------------------------------
    # Engine initialisation + training
    # ------------------------------------------------------------------
    _header("Training ChronosDefragEngine")
    t_fit_start = time.perf_counter()

    engine = ChronosDefragEngine(cfg)
    engine.fit(data)

    t_fit_end = time.perf_counter()
    print(f"\n  fit() completed in {t_fit_end - t_fit_start:.2f}s")
    print(f"  registry size : {int(engine.router.reg_size.item())} embeddings")

    # ------------------------------------------------------------------
    # Live streaming inference simulation
    # ------------------------------------------------------------------
    _header("Live Streaming Inference (30 ticks)")

    block_len = cfg.encoder.block_len
    # Use ticks from after the training window (strict future isolation)
    # Here we re-use switching regime ticks with a different seed to simulate
    # unseen live data arriving after training.
    live_data = _generate_regime(
        n_ticks=block_len * 40,
        n_features=cfg.encoder.input_dim,
        regime="switching",
        seed=cfg.seed + 99,
    )
    live_mu  = data.mean(0)
    live_std = data.std(0).clamp(min=1e-8)
    live_data = (live_data - live_mu) / live_std

    latencies: List[float] = []
    for tick_i in range(30):
        start = tick_i * block_len
        end   = start + block_len
        window = live_data[start:end]

        result = engine.predict_live(window)
        latencies.append(result.system_latency_ms)
        _tick_report(tick_i + 1, result, 0.0)

    # ------------------------------------------------------------------
    # Latency benchmark
    # ------------------------------------------------------------------
    _header("Latency Benchmark (100 trials)")
    bench = engine.benchmark_inference(n_trials=100, live_window=live_data[:block_len])

    print(f"  mean    : {bench['mean_ms']:.3f} ms")
    print(f"  p50     : {bench['p50_ms']:.3f} ms")
    print(f"  p95     : {bench['p95_ms']:.3f} ms")
    print(f"  p99     : {bench['p99_ms']:.3f} ms")
    print(f"  min     : {bench['min_ms']:.3f} ms")
    print(f"  max     : {bench['max_ms']:.3f} ms")
    print(f"  QPS     : {bench['throughput_qps']:.1f}")

    # ------------------------------------------------------------------
    # Routing statistics
    # ------------------------------------------------------------------
    _header("Routing Statistics")
    n_reg = int(engine.router.reg_size.item())
    print(f"  registry entries  : {n_reg}")
    print(f"  registry capacity : {cfg.routing.registry_capacity}")
    print(f"  fill ratio        : {n_reg / cfg.routing.registry_capacity:.1%}")
    print(f"  top_k             : {cfg.routing.top_k}")
    print(f"  similarity decay  : {cfg.routing.similarity_decay}")

    lat_t = torch.tensor(latencies)
    print(f"\n  stream latency avg: {lat_t.mean():.2f} ms over 30 live ticks")

    _header("Done")
    print()


if __name__ == "__main__":
    main()
