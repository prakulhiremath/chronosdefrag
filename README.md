# ChronosDefrag

[![CI](https://github.com/prakulhiremath/chronosdefrag/actions/workflows/ci.yml/badge.svg)](https://github.com/prakulhiremath/chronosdefrag/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/chronosdefrag)](https://pypi.org/project/chronosdefrag/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20331774.svg)](https://doi.org/10.5281/zenodo.20331774)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.1-orange)
[![Medium](https://img.shields.io/badge/Medium-Article-black?logo=medium)](https://medium.com/@prakulhiremath/defragmenting-market-time-non-stationary-representation-learning-in-pure-pytorch-15c4a18ccfa3)

**Temporal Defragmentation for Quantitative Finance**

*Rearrange time. Remove noise. Learn regimes.*

</div>

---

Standard sequence models applied to financial data operate under an implicit assumption that is provably wrong: **chronological adjacency implies statistical similarity.**

A trending regime at *t* = 1000 has more in common with a trending regime at *t* = 50 than with the mean-reverting regime at *t* = 999. Training over raw chronological order forces models to waste capacity learning transitions across regime boundaries that carry no predictive signal — and penalises them for the very non-stationarity they are supposed to model.

**ChronosDefrag addresses this at the representation level, not the model level.**

It learns a latent manifold over market regimes via unsupervised contrastive learning, *defragments* the timeline so that statistically similar regimes become locally adjacent, and trains a lightweight temporal predictor over the resulting pseudo-timeline rather than raw chronological order. At inference, a live tick window is routed in sub-millisecond time against a persistent historical registry using pure PyTorch tensor algebra — no external vector databases.

> This is an experimental temporal representation-learning framework, not a trading system. No alpha claims are made.

---

## Contents

- [Why ChronosDefrag](#why-chronosdefrag)
- [Architecture](#architecture)
- [Mathematical Foundations](#mathematical-foundations)
- [Novel Mechanisms](#novel-mechanisms)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Benchmarks](#benchmarks)
- [Configuration](#configuration)
- [Design Philosophy](#design-philosophy)
- [Causal Correctness](#causal-correctness)
- [Failure Modes](#failure-modes)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Why ChronosDefrag

Most time-series ML research treats non-stationarity as a nuisance. ChronosDefrag treats it as a structural property to exploit.

Financial markets cycle through regimes — mean-reversion, momentum, high-volatility dispersion — with no fixed periodicity. A model trained sequentially must learn, forget, and re-learn these regimes repeatedly as they recur. This is computationally wasteful and statistically fragile.

The defragmentation hypothesis: **if we reorder the training timeline so that same-regime windows are locally adjacent, the predictor's effective learning problem simplifies from non-stationary sequence modelling to locally stationary interpolation.**

This does not require labelled regime annotations. The regime structure is discovered from data via contrastive metric learning on a unit-sphere latent manifold.

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                          TRAINING PHASE                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Raw Tick Stream  (T, F)                                             ║
║         │                                                            ║
║         ▼                                                            ║
║  ┌─────────────────┐                                                 ║
║  │  Block Slicer   │  causal · non-overlapping · timestamp-tracked   ║
║  └────────┬────────┘                                                 ║
║           │  (N, block_len, F)                                       ║
║           ▼                                                          ║
║  ┌──────────────────────────────────────────────────────────┐        ║
║  │                    RegimeEncoder                         │        ║
║  │  input projection                                        │        ║
║  │    → N × ResidualBlock (CausalConv1D + pre-norm + GELU)  │        ║
║  │    → temporal mean pooling                               │        ║
║  │    → LayerNorm → latent head → L2-normalise              │        ║
║  │    → projection head  ──►  InfoNCE loss                  │        ║
║  │                                                          │        ║
║  │  Regularisers:  TER (entropy)  ·  LCR (continuity)       │        ║
║  └────────────────────────┬─────────────────────────────────┘        ║
║                           │  (N, latent_dim)  ∈  S^(d-1)             ║
║                           ▼                                          ║
║  ┌──────────────────────────────────────────────────────────┐        ║
║  │                  VectorDefragRouter                      │        ║
║  │  all-pairs cosine  →  greedy NN chain  →  permutation π  │        ║
║  │  circular registry  (capacity, latent_dim)               │        ║
║  └────────────────────────┬─────────────────────────────────┘        ║
║                           │  pseudo-timeline  z[π]                   ║
║                           ▼                                          ║
║  ┌──────────────────────────────────────────────────────────┐        ║
║  │                  TemporalPredictor                       │        ║
║  │  stacked GLURecurrentBlock  (gated linear recurrence)    │        ║
║  │  operates on pseudo-timeline order, not chronological    │        ║
║  │  latent reconstruction objective  (self-supervised)      │        ║
║  └──────────────────────────────────────────────────────────┘        ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                         INFERENCE PHASE                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Live Tick Window  (block_len, F)                                    ║
║         │                                                            ║
║         ▼                                                            ║
║  RegimeEncoder  ──────────────────────►  query latent z              ║
║         │                                                            ║
║         ▼                                                            ║
║  VectorDefragRouter.reconstruct_context()                            ║
║    batched mm()  →  topk()  →  SPD weighting  →  weighted aggregate  ║
║         │                                                            ║
║         ▼                                                            ║
║  TemporalPredictor.infer()                                           ║
║         │                                                            ║
║         ▼                                                            ║
║  {  alpha_prediction  ·  regime_confidence  ·  system_latency_ms  }  ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Module map

| Module | Responsibility |
|---|---|
| `defrag/config.py` | Typed dataclass configs, device resolution, deterministic seeding |
| `defrag/core.py` | `RegimeEncoder`, `VectorDefragRouter`, novel regularisers |
| `defrag/model.py` | `TemporalPredictor` (GLU + gated recurrent cells) |
| `defrag/engine.py` | `ChronosDefragEngine` — end-to-end orchestration |
| `main.py` | Self-contained sandbox with synthetic market generator |
| `tests/` | 19 deterministic tests: causality · shapes · latency · determinism |

---

## Mathematical Foundations

### Latent Manifold

Each market window $x \in \mathbb{R}^{T \times F}$ is mapped to a unit-sphere embedding $z \in \mathcal{S}^{d-1}$ by encoder $\varphi_\theta$. The contrastive objective (InfoNCE / NT-Xent) maximises agreement between two augmented views of the same block while repelling embeddings of different blocks:

```
L_NCE = -E [ log exp(z_i · z_j / τ) / Σ_k exp(z_i · z_k / τ) ]
```

where τ is a learnable log-temperature parameter, clipped to (0, 1) via exponentiation to prevent numerical instability.

Augmentation strategy: (1) additive Gaussian jitter — perturbs magnitude without altering temporal ordering; (2) cyclic time shift — preserves all statistical moments while breaking positional identity. Both are applied in-place on a cloned tensor; the original training buffer is never mutated.

### Pseudo-Timeline Construction

The defragmentation permutation $\pi$ is found via greedy nearest-neighbour chaining on $\mathcal{S}^{d-1}$. Starting from block 0 (preserving the causal anchor for deployment parity), each subsequent position is the unvisited block with the highest cosine similarity to the current position:

```
π* ≈ argmin_{π ∈ Sym(N)}  Σ_{i=1}^{N-1}  ‖z_{π(i+1)} − z_{π(i)}‖²
```

This is the Euclidean TSP on the unit sphere. The greedy solution is $O(N^2)$ but executes inside a single `torch.mm()` call — no Python loops. For the block counts typical of a single training session ($N \leq 5000$), this is fast enough to be negligible relative to encoder training time.

**Timeline entropy diagnostic.** After construction, ChronosDefrag reports a mixing ratio:

```
η = 1 − |corr(π, arange(N))| / N
```

$\eta \approx 1$ indicates high regime mixing (pseudo-timeline is substantially reordered). $\eta \approx 0$ means the latent space has not separated regimes — a signal to increase encoder training depth or adjust regularisation.

### Similarity Persistence Decay

At inference, top-$k$ retrieved neighbour scores are modulated by an exponential temporal-distance penalty before aggregation:

```
ŝ(i) = cos(q, h_i) · δ^(|t_q − t_i| / |H|)
```

where $\delta \in (0,1)$ is the decay coefficient, $t_i$ is the block timestamp of registry entry $i$, $t_q$ is the query timestamp, and $|H|$ is the current registry size. This prevents the router from over-weighting chronologically distant superficially-similar regimes — a common failure mode in non-stationary markets where the same latent region can correspond to structurally different macro environments.

---

## Novel Mechanisms

ChronosDefrag introduces two mechanisms that are not standard components of any existing deep learning or quantitative library:

### 1. Temporal Entropy Regularisation (TER)

**Location:** `RegimeEncoder.temporal_entropy_reg()`

Constructs a soft assignment matrix $P \in \mathbb{R}^{B \times B}$ via row-wise softmax over pairwise cosine similarities within a training batch. Computes the mean Shannon entropy $\bar{H}(P)$ of $P$'s rows and penalises it falling below a target:

```
L_TER = max(0,  H_target − H̄(P))  ·  λ_entropy

where  H_target = ½ log(B)
```

**Why it works:** If the encoder collapses all embeddings to a single point, every row of $P$ becomes a near-uniform or near-degenerate distribution. The penalty forces the encoder to maintain discriminative coverage of the latent sphere without requiring a pre-specified cluster count, discrete assignment, or prototype bank. The threshold adapts to batch size, making it robust across different training configurations.

**Compared to alternatives:** VICReg variance penalty operates on feature variance; SwAV requires prototype assignments; BYOL uses asymmetric networks. TER requires none of these: it operates directly on the geometry of the cosine similarity matrix.

### 2. Similarity Persistence Decay (SPD)

**Location:** `VectorDefragRouter.query()`

A temporal-distance-aware weighting function applied to cosine similarities at inference time. Unlike attention-based re-weighting (which requires a learned query-key interaction), SPD is parameter-free, differentiable with respect to neither the query nor the registry, and adds zero training overhead.

**Key property:** For a fixed cosine similarity value, SPD monotonically down-weights registry entries that are temporally far from the query. This means the router preferentially blends regime contexts from similar market periods — respecting the implicit non-stationarity structure of financial data — without any explicit regime labelling.

---

## Quickstart

```python
import torch
from defrag.config import DefragConfig
from defrag.engine import ChronosDefragEngine

cfg    = DefragConfig()
engine = ChronosDefragEngine(cfg)

# (T, F) float32 — chronological tick-level features
# F must match cfg.encoder.input_dim (default: 6)
data   = torch.randn(4096, 6)

engine.fit(data)

live_window = torch.randn(cfg.encoder.block_len, 6)
result      = engine.predict_live(live_window)

print(f"alpha      : {result.alpha_prediction.item():+.5f}")
print(f"confidence : {result.regime_confidence * 100:.1f}%")
print(f"latency    : {result.system_latency_ms:.2f}ms")
```

Run the full verification suite with synthetic market data:

```bash
python main.py
```

Expected output:

```
  [TICK 0001]  latency= 1.23ms  confidence= 24.3%  alpha=+0.31847
  [TICK 0002]  latency= 1.11ms  confidence= 26.1%  alpha=+0.28902
  [TICK 0003]  latency= 1.08ms  confidence= 27.6%  alpha=+0.19374
  ...
  mean : 1.14ms  p95 : 1.89ms  p99 : 2.31ms  QPS : 877
```

---

## Installation

**From PyPI (recommended):**

```bash
pip install chronosdefrag
```

**From source:**

```bash
git clone https://github.com/prakulhiremath/chronosdefrag
cd chronosdefrag
pip install -e .
```

**Dependencies:** `torch >= 2.1`, `numpy >= 1.24`. No other runtime dependencies.

**Dev dependencies:**

```bash
pip install pytest
pytest tests/ -v
```

---

## Benchmarks

Measured on Apple M2 Pro, CPU-only, PyTorch 2.3, Python 3.11. No GPU required.

### Training

| Stage | N blocks | Time |
|---|---|---|
| Encoder (30 epochs, batch 32) | 96 | ~40s |
| Pseudo-timeline construction | 96 | ~0.4ms |
| Predictor (20 epochs, batch 32) | 88 windows | ~2s |

### Inference (`predict_live`)

| Metric | CPU (M2 Pro) |
|---|---|
| Mean latency | 1.14 ms |
| p50 latency | 1.09 ms |
| p95 latency | 1.89 ms |
| p99 latency | 2.31 ms |
| Throughput | ~877 QPS |
| Peak memory | ~180 MB |

*Includes: encoder forward pass + registry lookup + SPD weighting + predictor forward pass. First 5 calls excluded (warm-up).*

### Test suite

```
19 passed in 4.1s
```

Covers: tensor shapes · causal padding correctness · block non-overlap · pseudo-timeline validity · collapse detection · augmentation integrity · latency budget · determinism under fixed seed.

---

## Configuration

All hyperparameters are typed dataclasses. Override at instantiation:

```python
from defrag.config import (
    DefragConfig, EncoderConfig, RoutingConfig,
    PredictorConfig, OptimizationConfig, DeploymentMode
)

cfg = DefragConfig(
    mode    = DeploymentMode.PRODUCTION,   # disables assertions for speed
    seed    = 42,
    device  = "auto",                      # auto-selects cuda > mps > cpu

    encoder = EncoderConfig(
        input_dim        = 6,
        block_len        = 64,             # must be power of 2
        hidden_dim       = 128,
        latent_dim       = 64,
        num_layers       = 4,
        entropy_weight   = 0.01,           # TER coefficient
        continuity_weight= 0.005,          # LCR coefficient
        temperature      = 0.07,           # InfoNCE init temperature
    ),

    routing = RoutingConfig(
        registry_capacity = 8192,
        top_k             = 16,
        similarity_decay  = 0.92,          # SPD decay coefficient δ
        min_confidence    = 0.05,
    ),

    predictor = PredictorConfig(
        latent_dim   = 64,                 # must match encoder.latent_dim
        hidden_dim   = 128,
        num_layers   = 3,
        forecast_dim = 1,
        grad_clip    = 1.0,
    ),

    optim = OptimizationConfig(
        encoder_epochs   = 40,
        predictor_epochs = 30,
        batch_size       = 64,
        encoder_lr       = 3e-4,
        amp_enabled      = False,          # set True for CUDA
    ),
)
```

**Deployment modes:**

| Mode | Assertions | Logging |
|---|---|---|
| `RESEARCH` | enabled | verbose |
| `STAGING` | enabled | reduced |
| `PRODUCTION` | disabled | minimal |

---

## Design Philosophy

**Causality by construction, not convention.** Every Conv1d in the encoder uses left-only padding (`CausalConv1d`), verified by a dedicated test that zeroes future positions and asserts output invariance at prior positions. The block slicer tracks timestamps explicitly. `predict_live()` asserts the query timestamp exceeds the training boundary before routing — catching any accidental future leakage at the API boundary.

**No external vector databases.** The registry is a fixed-size circular buffer of L2-normalised embeddings stored as a `torch.Tensor`. Lookup is a batched matrix multiply (`mm`) followed by `topk()` — entirely within PyTorch's autograd-free `torch.no_grad()` context. This eliminates operational dependencies on FAISS, Milvus, Hnswlib, or any ANN library, and keeps the entire inference path on-device with no serialisation overhead.

**Self-supervised throughout.** No price labels, forward returns, or external annotations are required at any training stage. The encoder learns via contrastive augmentation pairs. The predictor learns to reconstruct the next pseudo-timeline position's embedding. This makes the system applicable to any multivariate tick-level time series without modification.

**Minimal dependency surface.** `torch`, `numpy`, Python standard library. The full install graph is inspectable in under one minute. No pandas, scikit-learn, Lightning, Hydra, Ray, or vector DB frameworks.

**Deterministic by default.** A single `cfg.seed_everything()` call covers Python `random`, NumPy, and PyTorch CPU/CUDA RNGs. Every test in the suite is deterministic and passes on a cold run with no warm-up state.

---

## Causal Correctness

ChronosDefrag enforces a strict temporal isolation boundary:

```
Training data:  ticks [0,  train_size)
Live inference: ticks [train_size, ∞)
```

This is enforced at three independent levels:

1. **Architecture:** All convolutional operations use left-only padding. No attention mechanism with full-sequence visibility is used at any stage.

2. **Data pipeline:** `_slice_blocks()` produces non-overlapping blocks with monotonically increasing timestamps. Verified by `TestCausality.test_slice_blocks_no_overlap`.

3. **API contract:** `predict_live()` sets `query_timestamp = self._train_size` before calling the router, ensuring SPD correctly penalises all registry entries as being in the historical past relative to the live query.

The causal padding test (`TestCausality.test_encoder_causal_padding`) zeroes positions $[t+1, T)$ and asserts byte-level equality of encoder outputs at positions $[0, t]$ — confirming no future information propagates backward through the residual stack.

---

## Failure Modes

**Latent collapse.** All embeddings converge to a small region of $\mathcal{S}^{d-1}$. Symptoms: `regime_confidence` uniformly near 1.0; all queries route to the same neighbours; timeline entropy $\eta \approx 0$. Root cause: `entropy_weight` too low, `temperature` too high, or batch size too small for stable InfoNCE gradients. Mitigation: increase `entropy_weight`, lower `temperature`, increase `batch_size`.

**Similarity drift.** The encoder's representation geometry shifts during late training epochs, making early registry entries stale. The circular buffer partially mitigates this by evicting old entries, but large representation shifts can cause the router's confidence to degrade over time. Mitigation: reduce `encoder_lr` late in training; re-fit when `regime_confidence` trends downward persistently.

**Non-stationarity mismatch.** The registry is static post-training. A structural market regime change (post-crisis, post-halving, regulatory shock) may not resemble any historical registry entry. Symptom: uniformly low `regime_confidence` across all live ticks. Mitigation: periodic re-fit on a rolling window; monitor `regime_confidence` distribution in production.

**Pseudo-timeline pathology.** The greedy chain produces a near-chronological permutation when the latent space does not cleanly separate regimes. Symptom: timeline entropy $\eta < 0.1$. The predictor in this case trains on an only marginally reordered sequence and provides little advantage over standard chronological training. Mitigation: increase encoder training depth, verify feature quality, check for data normalisation issues.

**O(N²) construction cost.** `build_pseudo_timeline()` computes a full $(N \times N)$ similarity matrix in a single `mm()` call. For $N > 5000$ blocks this allocates $> 200$MB for `latent_dim = 64`. Chunked construction is listed in Future Work.

---

## Limitations

- ChronosDefrag is an **experimental research framework** with no live-trading validation.
- `alpha_prediction` is a **latent-space quantity** — a proxy for regime continuation, not an interpretable return forecast. Downstream calibration to actual returns requires a supervised regression layer trained on labelled data.
- Minimum recommended training size: $N \geq 50$ blocks ($T \geq 50 \times$ `block_len` ticks).
- The greedy pseudo-timeline construction is $O(N^2)$. For $N > 5000$ blocks, a chunked approximate variant is required (not yet implemented).
- No uncertainty quantification is provided over `alpha_prediction`. Point estimates during transitional or ambiguous regimes should be treated with scepticism.

---

## Future Work

- [ ] Chunked $O(N^2 / C)$ pseudo-timeline construction for large $N$
- [ ] Online registry updates with exponential forgetting factor
- [ ] Multi-scale block hierarchies — short / medium / long regime contexts with cross-scale attention
- [ ] Learned augmentation policy replacing fixed jitter + cyclic shift
- [ ] Bayesian uncertainty estimates over `alpha_prediction`
- [ ] Streaming encoder with incremental hidden-state updates
- [ ] Distributed registry across multiple devices for large-scale deployment
- [ ] Formal empirical evaluation on public financial benchmark datasets

---

## Troubleshooting

**`ValueError: Need at least 4 blocks`**
Provide $T \geq 4 \times$ `block_len` ticks. Default `block_len = 64` requires $T \geq 256$.

**`AssertionError: encoder.latent_dim must match predictor.latent_dim`**
Both config fields must be set to the same value. Modify them together.

**`RuntimeError: Registry is empty`**
`engine.fit()` must be called before `engine.predict_live()`.

**Training loss is NaN**
Most common cause: `batch_size < 2` (InfoNCE requires $\geq 2$ samples per batch), or unnormalised input features with extreme values. Apply robust z-score or quantile normalisation before calling `fit()`.

**Uniformly low `regime_confidence`**
Indicates latent collapse or a structural mismatch between training data and live data. Check the fit log for collapse warnings. Increase `entropy_weight`, increase `encoder_epochs`, or re-fit on more representative data.

**`block_len` assertion failure**
`block_len` must be a positive power of 2 (16, 32, 64, 128, ...). This is validated at config instantiation.

---

## References

- Chen, T. et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* ICML. — InfoNCE / NT-Xent loss.
- Grill, J.-B. et al. (2020). *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.* NeurIPS. — Collapse prevention strategies.
- Bardes, A. et al. (2022). *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* ICLR. — Variance-based collapse penalty, contrasted with TER.
- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* — Gated linear recurrence patterns adapted for `GatedRecurrentCell`.
- Gu, A. et al. (2021). *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR. — S4 motivation for sub-quadratic recurrence.
- Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* Econometrica. — Regime-switching model motivation.
- Karpathy, A. (2022). *nanoGPT.* — Minimalist architecture philosophy.
- Kaya, M. & Bilge, H. Ş. (2019). *Deep Metric Learning: A Survey.* Symmetry. — Manifold structure and metric learning context.

---

## Citation

If you use ChronosDefrag in research, please cite:

```bibtex
@software{chronosdefrag2025,
  author  = {Hiremath, Prakul},
  title   = {ChronosDefrag: Temporal Defragmentation for Quantitative Finance},
  year    = {2025},
  url     = {https://github.com/prakulhiremath/chronosdefrag},
  doi     = {10.5281/zenodo.20331773},
  version = {0.1.1}
}
```

---

## License

MIT. See [`LICENSE`](LICENSE).
