# ChronosDefrag

[![CI](https://github.com/prakulhiremath/chronosdefrag/actions/workflows/ci.yml/badge.svg)](https://github.com/prakulhiremath/chronosdefrag/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/chronosdefrag)](https://pypi.org/project/chronosdefrag/)
[![DOI](https://zenodo.org/badge/1246002937.svg)](https://doi.org/10.5281/zenodo.20331773)

**Temporal Defragmentation for Quantitative Finance**

---

Standard sequence models applied to financial data operate under an implicit assumption that is provably wrong: that chronological adjacency implies statistical similarity. A trending regime at *t=1000* has more in common with another trending regime at *t=50* than with the mean-reverting regime at *t=999*. Training over raw time order forces models to waste capacity learning transitions across regime boundaries that offer no predictive signal.

ChronosDefrag addresses this at the data-representation level rather than the model level. It learns a latent manifold over market regimes, *defragments* the timeline so that statistically similar regimes become locally adjacent, and trains a lightweight predictor over the resulting pseudo-timeline rather than raw chronological order.

This is not a trading system. It is an experimental temporal representation-learning framework.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                              │
│                                                                     │
│  Raw Tick Stream (T, F)                                             │
│       │                                                             │
│       ▼                                                             │
│  Block Slicer ──► (N, block_len, F)   causal, non-overlapping       │
│       │                                                             │
│       ▼                                                             │
│  RegimeEncoder                                                      │
│  ├─ CausalConv1D residual stack       no future leakage             │
│  ├─ Temporal mean pooling                                           │
│  ├─ L2-normalised latent head         (N, latent_dim)               │
│  └─ Projection head                   InfoNCE contrastive loss      │
│       │                                                             │
│       │   ┌──────────────────────────────────────────────┐          │
│       │   │  Novel Regularisers                          │          │
│       │   │  • Temporal Entropy Reg. (TER)               │          │ 
│       │   │  • Latent Continuity Reg. (LCR)              │          │
│       │   └──────────────────────────────────────────────┘          │
│       │                                                             │
│       ▼                                                             │
│  VectorDefragRouter                                                 │
│  ├─ All-pairs cosine similarity       (N, N) pure PyTorch           │
│  ├─ Greedy nearest-neighbour chain    pseudo-timeline permutation   │
│  └─ Circular historical registry     O(1) in-memory lookup          │
│       │                                                             │
│       ▼                                                             │
│  TemporalPredictor                                                  │
│  ├─ GLU blocks with gated recurrence                                │
│  ├─ Operates on pseudo-timeline order (not chronological)           │
│  └─ Latent reconstruction objective  (self-supervised)              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PHASE                              │
│                                                                     │
│  Live Tick Window (block_len, F)                                    │
│       │                                                             │
│       ▼                                                             │
│  RegimeEncoder ──► query latent z     (latent_dim,)                 │
│       │                                                             │
│       ▼                                                             │
│  VectorDefragRouter.reconstruct_context()                           │
│  ├─ Vectorised cosine similarity against full registry              │
│  ├─ Similarity Persistence Decay (SPD) weighting                    │
│  └─ Weighted pseudo-neighbourhood embedding                         │
│       │                                                             │
│       ▼                                                             │
│  TemporalPredictor.infer()                                          │
│       │                                                             │
│       ▼                                                             │
│  { alpha_prediction, regime_confidence, system_latency_ms }         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Intuition

### Regime Manifold

Each market window *x* ∈ ℝ^(T×F) is mapped to a unit-sphere latent vector *z* ∈ S^(d-1) by the encoder φ. The contrastive objective (InfoNCE) encourages φ to satisfy:

```
d(φ(x_i), φ(x_j)) small  ↔  x_i, x_j are same-regime augmentations
d(φ(x_i), φ(x_k)) large  ↔  x_i, x_k are different-regime windows
```

The manifold is regularised by two novel penalties:

**Temporal Entropy Regularisation (TER)**

Constructs a soft assignment matrix *P ∈ ℝ^(B×B)* via row-wise softmax of pairwise cosine similarities. Penalises the mean Shannon entropy of *P*'s rows falling below *H_target = ½ log(B)*:

```
L_TER = max(0, H_target − H̄(P)) · λ_entropy
```

This prevents the encoder from collapsing all representations to a single cluster without requiring an explicit cluster-count prior. The threshold is adaptive to batch size.

**Latent Continuity Regularisation (LCR)**

Penalises large L2 jumps between the latent vectors of chronologically adjacent blocks:

```
L_LCR = (1/N−1) Σ ||z_{t+1} − z_t||² · λ_continuity
```

This encourages the manifold to be locally Lipschitz with respect to time — similar to manifold smoothness priors in metric learning — while still permitting sharp regime transitions when the data genuinely supports them.

### Pseudo-Timeline Construction

The defragmentation permutation π is found via greedy nearest-neighbour chaining on the latent manifold. Starting from block 0 (preserving the causal anchor), each subsequent position in the pseudo-timeline is filled by the unvisited block with the highest cosine similarity to the current position. This minimises the total manifold path length:

```
π* ≈ argmin_{π ∈ Sym(N)} Σ_{i=1}^{N−1} ||z_{π(i+1)} − z_{π(i)}||²
```

This is the Euclidean TSP on the unit sphere; the greedy solution is O(N²) but runs entirely within a single PyTorch mm() call.

### Similarity Persistence Decay (SPD)

At inference, retrieved neighbour scores are modulated by:

```
ŝ(i) = cos(q, h_i) · δ^(|t_q − t_i| / |H|)
```

where *δ* ∈ (0,1) is the decay coefficient, *t_i* is the timestamp of the *i*-th registry entry, *t_q* is the query timestamp, and *|H|* is the current registry size. This prevents the router from over-weighting regimes from very different market periods that happen to appear superficially similar in the latent space.

---

## Novel Mechanisms

Two mechanisms are unique to ChronosDefrag and are not standard deep learning components:

| Mechanism | Location | Purpose |
|---|---|---|
| Temporal Entropy Regularisation (TER) | `RegimeEncoder.temporal_entropy_reg()` | Collapse prevention without cluster priors |
| Similarity Persistence Decay (SPD) | `VectorDefragRouter.query()` | Temporal-distance-aware neighbour weighting |

---

## Quickstart

```python
import torch
from defrag.config import DefragConfig
from defrag.engine import ChronosDefragEngine

# 1. Configure
cfg = DefragConfig()   # sane defaults; override as needed

# 2. Prepare chronological data: (T, F) float32
#    F must match cfg.encoder.input_dim (default 6)
data = torch.randn(4096, 6)

# 3. Train
engine = ChronosDefragEngine(cfg)
engine.fit(data)

# 4. Live inference
live_window = torch.randn(cfg.encoder.block_len, 6)
result = engine.predict_live(live_window)

print(f"Alpha:      {result.alpha_prediction.item():.5f}")
print(f"Confidence: {result.regime_confidence * 100:.1f}%")
print(f"Latency:    {result.system_latency_ms:.2f}ms")
```

Run the full sandbox:

```bash
pip install torch numpy
python main.py
```

Run the test suite:

```bash
pip install pytest
pytest tests/ -v
```

---

## Benchmark

Measured on Apple M2 Pro (CPU), PyTorch 2.3, Python 3.11.
No GPU required for the default configuration.

| Metric | Value |
|---|---|
| Encoder training (30 epochs, 48 blocks) | ~4.2s |
| Pseudo-timeline construction (48 blocks) | ~0.3ms |
| predict_live() mean latency | ~1.1ms |
| predict_live() p99 latency | ~2.8ms |
| Inference throughput | ~850 QPS |
| Peak memory (encoder + registry 2k) | ~180MB |

*Latency includes encoder forward pass, registry lookup, SPD weighting, and predictor forward pass. Measured after JIT warm-up (first 5 calls excluded).*

---

## Design Philosophy

**Anti-lookahead by construction.** The encoder uses exclusively causal (left-padded) convolutions. The block slicer is strictly non-overlapping with timestamp tracking. The predictor is trained only on pseudo-timeline windows derived from the encoder training set. `predict_live()` asserts that the query timestamp exceeds the training boundary before routing.

**No external vector databases.** The similarity engine is entirely PyTorch tensor algebra. The registry is a circular buffer of normalised embeddings; lookup is a batched matrix multiply followed by `topk()`. This keeps the dependency graph minimal and avoids the operational complexity of FAISS, Milvus, or Hnswlib.

**Self-supervised throughout.** No price labels, returns, or forward-looking targets are required during training. The encoder objective is contrastive (same-window augmentation pairs). The predictor objective is latent reconstruction (next-pseudo-position prediction). This makes the system applicable to any multivariate tick-level time series.

**Minimal dependency footprint.** `torch`, `numpy`, Python standard library. No pandas, scikit-learn, Lightning, Hydra, or Ray.

---

## Failure Modes

**Latent collapse.** If entropy regularisation weight (`entropy_weight`) is too low relative to the InfoNCE temperature, all embeddings may collapse to a small region of S^(d-1). Symptoms: `regime_confidence` uniformly near 1.0, all live windows routing to the same neighbours. Mitigation: increase `entropy_weight`, decrease `temperature`, increase batch size.

**Similarity drift.** Registry entries from very early in training may become unreliable if the encoder's representations shift significantly late in training. The circular registry evicts the oldest entries, which partially mitigates this, but re-fitting after significant market structure changes is recommended.

**Non-stationarity.** The system makes no stationarity assumptions, but the registry is static post-training. A fundamentally different market microstructure (e.g., post-halving crypto, post-crisis equity) may not match any historical regime in the registry. Symptoms: uniformly low `regime_confidence`. Mitigation: periodic re-training or online registry updates.

**Pseudo-timeline pathologies.** The greedy nearest-neighbour chain can produce poor permutations when the latent space is low-dimensional and regime boundaries are diffuse. In this case the pseudo-timeline may be nearly chronological, offering no advantage over standard sequence training. The `timeline_entropy` metric in the fit log reports permutation mixing quality.

**Memory scaling.** Registry memory scales as `registry_capacity × latent_dim × 4` bytes. At `latent_dim=64` and `registry_capacity=65536`, this is ~16MB — negligible. The all-pairs cosine similarity matrix during `build_pseudo_timeline()` scales as O(N²); for N > 10,000 blocks this requires chunked computation (not yet implemented).

**Regime ambiguity.** Markets frequently exhibit mixed or transitional regimes that do not cleanly separate in the latent space. The SPD-weighted neighbourhood aggregate partially handles this by blending multiple candidate regimes, but `alpha_prediction` values during transitional periods should be treated with higher uncertainty.

---

## Implementation Notes

**Encoder initialisation.** Xavier uniform initialisation is used throughout. The learnable temperature parameter is initialised as `log(cfg.temperature)` and exponentiated during the forward pass, preventing temperature from going negative.

**Augmentation.** Contrastive pairs are generated by (1) additive Gaussian jitter and (2) cyclic time shift within each block. The cyclic shift preserves all statistical moments while breaking positional identity — unlike random masking or dropout, which alter the marginal distribution.

**Predictor training target.** The predictor is trained to predict the first `forecast_dim` coordinates of the next pseudo-timeline position's latent vector. This is a proxy for regime continuation rather than price prediction. Downstream calibration to actual returns requires an additional regression layer trained on labelled data, which is outside the scope of this framework.

**Gradient clipping.** The predictor applies gradient norm clipping (`grad_clip=1.0`) via `clip_grad_norm_` before each optimiser step. The encoder applies the same during contrastive training.

---

## Deployment Flow

```
1. Collect T ticks of (block_len × N)-aligned tick-level features
2. engine.fit(data)                    # trains encoder + predictor, builds registry
3. Checkpoint is saved to ./checkpoints/chronosdefrag.pt
4. On new deployment: engine.load_checkpoint(path)
5. For each incoming block_len-tick window: engine.predict_live(window)
6. Monitor regime_confidence; trigger re-fit when persistently below threshold
```

---

## Limitations

- ChronosDefrag is an **experimental research framework**. It has not been validated in live trading.
- The alpha projection is a **latent-space quantity**, not a directly interpretable return forecast.
- Performance degrades when the number of training blocks N is small (< 50 recommended minimum).
- The greedy pseudo-timeline construction is O(N²) in the number of blocks. For N > 5,000, a chunked approximate version is needed (see Future Work).

---

## Future Work

- [ ] Chunked approximate pseudo-timeline construction for large N
- [ ] Online registry updates with forgetting factor
- [ ] Multi-scale block hierarchies (short/medium/long regime contexts)
- [ ] Learned augmentation policy (replacing fixed jitter + shift)
- [ ] Uncertainty quantification over the alpha projection
- [ ] Streaming encoder with incremental state updates
- [ ] Distributed registry across multiple devices

---

## Troubleshooting

**`ValueError: Need at least 4 blocks`**
Provide at least `4 × block_len` ticks. Default `block_len=64` requires T ≥ 256.

**`AssertionError: encoder.latent_dim must match predictor.latent_dim`**
Both config fields must be identical. Modify both together.

**`RuntimeError: Registry is empty`**
`fit()` must be called before `predict_live()`.

**Training loss is NaN**
Usually caused by `batch_size < 2` (InfoNCE requires ≥ 2 samples) or extreme feature values. Apply robust normalisation (z-score or quantile) before passing data to `fit()`.

**Very low `regime_confidence` on all live ticks**
Likely caused by latent collapse or insufficient training data. Check encoder training log for collapse warnings. Increase `entropy_weight` or `encoder_epochs`.

---

## References and Inspirations

- **SimCLR** — Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (2020). InfoNCE loss design.
- **BYOL** — Grill et al., "Bootstrap Your Own Latent" (2020). Collapse prevention strategies.
- **Mamba / S4** — Gu et al. Gated linear recurrence patterns adapted for the predictor cell.
- **nanoGPT** — Karpathy. Minimalist architecture philosophy.
- **tinygrad** — Hotz. Dependency minimalism and systems-first design.
- **Regime-switching models** — Hamilton (1989). Motivation for regime-conditional temporal modelling.
- **Metric learning** — Kaya & Bilge (2019). Manifold structure intuitions.

---

## License

MIT. See `LICENSE`.

This repository does not constitute financial advice. No representations are made about the fitness of any component for trading or investment purposes.
