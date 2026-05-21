"""
defrag/core.py

Core latent regime learning and pseudo-timeline construction.

Two novel mechanisms unique to ChronosDefrag:

1. Temporal Entropy Regularisation (TER)
   Penalises latent distributions that collapse toward low-entropy (overly
   peaked) regime assignments, maintaining manifold coverage without
   explicit cluster-count priors.

2. Similarity Persistence Decay (SPD)
   During pseudo-timeline stitching, neighbour weights are modulated by an
   exponential decay tied to the temporal distance between the query window
   and each retrieved historical block, preventing the router from
   preferring chronologically distant but superficially similar regimes.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DefragConfig, EncoderConfig, RoutingConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: causal Conv1D (no lookahead)
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """
    Conv1d with left-only padding to guarantee strict causality.
    Output at position t depends only on positions <= t.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.pad = (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Pre-norm residual block: LayerNorm -> CausalConv1d -> GELU -> CausalConv1d.
    Gated skip connection avoids vanishing gradient across depth.
    """

    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.norm   = nn.LayerNorm(channels)
        self.conv1  = CausalConv1d(channels, channels * 2, kernel_size)
        self.conv2  = CausalConv1d(channels * 2, channels, kernel_size)
        self.drop   = nn.Dropout(dropout)
        self.gate   = nn.Parameter(torch.zeros(1))   # learnable residual gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = x
        # LayerNorm operates on channel dim; transpose required
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(self.conv1(h))
        h = self.drop(self.conv2(h))
        return residual + torch.sigmoid(self.gate) * h


# ---------------------------------------------------------------------------
# RegimeEncoder
# ---------------------------------------------------------------------------

class RegimeEncoder(nn.Module):
    """
    Causal Conv1D encoder mapping temporal blocks to a normalised latent space.

    Architecture:
        input projection  -> num_layers x ResidualBlock -> temporal pooling
        -> LayerNorm -> latent head -> L2-normalise
        -> projection head (for InfoNCE)

    The learnable temperature parameter controls InfoNCE sharpness.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Conv1d(cfg.input_dim, cfg.hidden_dim, kernel_size=1, bias=False)

        self.blocks = nn.ModuleList([
            ResidualBlock(cfg.hidden_dim, cfg.kernel_size, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

        self.latent_norm = nn.LayerNorm(cfg.hidden_dim)
        self.latent_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim, bias=False)

        # Projection head for contrastive training (not used at inference)
        self.proj_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim, bias=False),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.projection_dim, bias=False),
        )

        # Learnable log-temperature (more numerically stable than raw temp)
        self.log_temp = nn.Parameter(torch.tensor(math.log(cfg.temperature)))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, F)  — batch, block_len, features

        Returns
        -------
        z  : (B, latent_dim)      — L2-normalised latent embedding
        zp : (B, projection_dim) — projection for InfoNCE loss
        """
        B, T, n_feat = x.shape
        assert n_feat == self.cfg.input_dim, \
            f"Expected {self.cfg.input_dim} features, got {n_feat}"
        assert T == self.cfg.block_len, \
            f"Expected block_len={self.cfg.block_len}, got T={T}"

        h = x.transpose(1, 2).contiguous()   # (B, n_feat, T)
        h = self.input_proj(h)                # (B, H, T)

        for block in self.blocks:
            h = block(h)                      # (B, H, T)

        # Temporal mean pooling (after all causal computation)
        h = h.mean(dim=2)                     # (B, H)
        h = self.latent_norm(h)
        z = self.latent_head(h)               # (B, latent_dim)
        z = F.normalize(z, dim=-1)            # unit sphere

        zp = self.proj_head(z)                # (B, projection_dim)
        zp = F.normalize(zp, dim=-1)

        return z, zp

    # ------------------------------------------------------------------
    # InfoNCE loss
    # ------------------------------------------------------------------

    def infonce_loss(
        self,
        zp_a: torch.Tensor,   # (B, projection_dim)
        zp_b: torch.Tensor,   # (B, projection_dim) — augmented view
    ) -> torch.Tensor:
        """
        Numerically stable InfoNCE (NT-Xent variant).
        Positive pairs: (i, i) across the two views.
        Negative pairs: all cross-batch off-diagonal entries.
        """
        B = zp_a.shape[0]
        assert B >= 2, "InfoNCE requires batch_size >= 2"

        temp = self.log_temp.exp().clamp(min=1e-4, max=1.0)

        # (2B, projection_dim) concatenation
        z_all = torch.cat([zp_a, zp_b], dim=0)

        # Cosine similarity matrix (2B, 2B)
        sim = torch.mm(z_all, z_all.T) / temp

        # Mask self-similarity on diagonal
        mask = torch.eye(2 * B, dtype=torch.bool, device=z_all.device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive indices: for row i in [0, B) -> B+i; for row B+i -> i
        labels = torch.arange(B, device=z_all.device)
        labels = torch.cat([labels + B, labels], dim=0)   # (2B,)

        loss = F.cross_entropy(sim, labels)
        return loss

    # ------------------------------------------------------------------
    # Novel mechanism 1: Temporal Entropy Regularisation (TER)
    # ------------------------------------------------------------------

    def temporal_entropy_reg(self, z: torch.Tensor) -> torch.Tensor:
        """
        Prevents regime collapse by penalising low-entropy assignment
        distributions across a batch of latent vectors.

        Construct a soft assignment via pairwise cosine similarity, interpret
        each row as a categorical distribution, and penalise if the entropy
        of that distribution falls below a target entropy threshold
        (half the maximum entropy log(B)).

        Loss = max(0, H_target - H_mean)
        """
        B = z.shape[0]
        if B < 2:
            return z.new_zeros(1).squeeze()

        sim = torch.mm(z, z.T)                         # (B, B) in [-1, 1]
        # Shift to [0, 1] range for a valid probability simplex
        p = F.softmax(sim, dim=-1)                     # (B, B)

        # Entropy per row: H = -sum(p * log(p + eps))
        H = -(p * (p + 1e-8).log()).sum(dim=-1).mean() # scalar

        H_max   = math.log(B)
        H_target = 0.5 * H_max

        loss = F.relu(torch.tensor(H_target, device=z.device, dtype=z.dtype) - H)
        return loss * self.cfg.entropy_weight

    # ------------------------------------------------------------------
    # Novel mechanism 2 (partial): Latent Continuity Regularisation
    # ------------------------------------------------------------------

    def latent_continuity_reg(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        Encourages smooth manifold traversal across chronologically
        adjacent blocks by penalising large L2 jumps between consecutive
        latent vectors.

        z_seq : (N, latent_dim) — ordered sequence of block embeddings
        """
        if z_seq.shape[0] < 2:
            return z_seq.new_zeros(1).squeeze()

        deltas = z_seq[1:] - z_seq[:-1]                # (N-1, latent_dim)
        loss = (deltas ** 2).sum(dim=-1).mean()
        return loss * self.cfg.continuity_weight


# ---------------------------------------------------------------------------
# VectorDefragRouter
# ---------------------------------------------------------------------------

class VectorDefragRouter(nn.Module):
    """
    Maintains a persistent historical registry of latent embeddings and
    constructs pseudo-timelines by retrieving the k-nearest-neighbour
    regime contexts for any incoming query window.

    All similarity operations are pure PyTorch tensor algebra.
    No external ANN libraries are used.

    Similarity Persistence Decay (SPD) — Novel mechanism 2:
        Retrieved neighbour scores are modulated by an exponential function
        of the temporal lag between the query block and the neighbour block,
        preventing the router from over-weighting chronologically distant
        superficially similar regimes.

        adjusted_score(i) = cosine_sim(q, h_i) * decay^(|t_q - t_i| / T)

    where T is the total registry length at query time.
    """

    def __init__(self, cfg: RoutingConfig, latent_dim: int) -> None:
        super().__init__()
        self.cfg        = cfg
        self.latent_dim = latent_dim

        # Registry buffers — not parameters (not trained by gradient)
        self.register_buffer(
            "registry",
            torch.zeros(cfg.registry_capacity, latent_dim),
        )
        self.register_buffer(
            "reg_timestamps",
            torch.zeros(cfg.registry_capacity, dtype=torch.long),
        )
        self.register_buffer(
            "reg_ptr",
            torch.tensor(0, dtype=torch.long),
        )
        self.register_buffer(
            "reg_size",
            torch.tensor(0, dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    @torch.no_grad()
    def register_batch(
        self,
        z: torch.Tensor,          # (N, latent_dim)
        timestamps: torch.Tensor,  # (N,) int64 block indices
    ) -> None:
        """
        Write-ahead insert into circular registry.
        Oldest entries are evicted when capacity is reached.
        """
        N = z.shape[0]
        cap = self.cfg.registry_capacity

        for i in range(N):
            idx = int(self.reg_ptr.item()) % cap
            self.registry[idx]       = z[i].detach()
            self.reg_timestamps[idx] = timestamps[i]
            self.reg_ptr += 1
            if self.reg_size < cap:
                self.reg_size += 1

    @torch.no_grad()
    def clear_registry(self) -> None:
        self.registry.zero_()
        self.reg_timestamps.zero_()
        self.reg_ptr.zero_()
        self.reg_size.zero_()

    # ------------------------------------------------------------------
    # Core similarity engine — pure PyTorch, no Python loops in hot path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def query(
        self,
        q: torch.Tensor,          # (B, latent_dim) or (latent_dim,)
        query_timestamp: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorised kNN lookup against the historical registry.

        Returns
        -------
        indices  : (B, top_k) — registry row indices of top-k neighbours
        scores   : (B, top_k) — SPD-adjusted cosine similarities
        z_nbrs   : (B, top_k, latent_dim) — neighbour embeddings
        """
        n = int(self.reg_size.item())
        if n == 0:
            raise RuntimeError("Registry is empty; call register_batch first")

        q = q.to(self.registry.device)
        squeeze = q.dim() == 1
        if squeeze:
            q = q.unsqueeze(0)       # (1, D)

        B, D = q.shape
        assert D == self.latent_dim

        # Active slice of registry (contiguous view)
        active = self.registry[:n]                          # (n, D)
        ts     = self.reg_timestamps[:n].float()           # (n,)

        # Cosine similarity: (B, n)  — unit embeddings, so dot = cosine
        sim = torch.mm(q, active.T)                        # (B, n)

        # Similarity Persistence Decay
        if query_timestamp >= 0:
            lag       = (query_timestamp - ts).abs().clamp(min=0)  # (n,)
            lag_norm  = lag / (n + 1e-6)                           # (n,)
            decay     = self.cfg.similarity_decay ** lag_norm       # (n,)
            sim       = sim * decay.unsqueeze(0)                    # (B, n)

        # Top-k retrieval (vectorised)
        k = min(self.cfg.top_k, n)
        scores, indices = sim.topk(k, dim=-1, largest=True, sorted=True)  # (B, k)

        # Confidence gate: mask entries below threshold
        mask   = scores < self.cfg.min_confidence
        scores = scores.masked_fill(mask, 0.0)

        # Gather neighbour embeddings
        flat_idx = indices.reshape(-1)                     # (B*k,)
        z_nbrs   = active[flat_idx].reshape(B, k, D)      # (B, k, D)

        if squeeze:
            indices = indices.squeeze(0)
            scores  = scores.squeeze(0)
            z_nbrs  = z_nbrs.squeeze(0)

        return indices, scores, z_nbrs

    # ------------------------------------------------------------------
    # Pseudo-timeline construction (training time)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_pseudo_timeline(
        self,
        z_all: torch.Tensor,      # (N, latent_dim) — chronological embeddings
    ) -> torch.Tensor:
        """
        Constructs a permutation of block indices such that statistically
        similar regimes become locally adjacent in the pseudo-timeline.

        Algorithm: greedy nearest-neighbour chain with non-repetition.
        Operates on normalised embeddings; all-pairs cosine is computed
        once as a (N, N) matrix.  Diagonal is zeroed to prevent self-loops.

        Returns
        -------
        permutation : (N,) int64 — reordered block indices
        """
        N, D = z_all.shape
        device = z_all.device

        # All-pairs similarity: (N, N)
        sim = torch.mm(z_all, z_all.T)

        # Zero diagonal (no self-loops)
        sim.fill_diagonal_(float("-inf"))

        visited = torch.zeros(N, dtype=torch.bool, device=device)
        order   = torch.zeros(N, dtype=torch.long,  device=device)

        # Start from block 0 (chronologically first — preserves causal anchor)
        current = 0
        visited[0] = True
        order[0]   = 0

        for step in range(1, N):
            row = sim[current].clone()
            row[visited] = float("-inf")           # exclude already-visited
            nxt = int(row.argmax().item())
            if row[nxt] == float("-inf"):
                # All remaining blocks equally distant — fall back to first unvisited
                nxt = int((~visited).nonzero(as_tuple=False)[0].item())
            visited[nxt] = True
            order[step]  = nxt
            current      = nxt

        return order

    # ------------------------------------------------------------------
    # Regime adjacency matrix (training diagnostic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def adjacency_matrix(
        self,
        z: torch.Tensor,   # (N, latent_dim)
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Returns a binary adjacency matrix where entry (i, j) = 1 iff
        cosine_similarity(z_i, z_j) >= threshold.
        """
        sim = torch.mm(z, z.T)
        return (sim >= threshold).float()

    # ------------------------------------------------------------------
    # Collapse detection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def detect_collapse(self, z: torch.Tensor) -> bool:
        """
        Returns True if the batch exhibits signs of representation collapse:
        all pairwise distances are below collapse_margin.
        """
        if z.shape[0] < 2:
            return False
        # Squared L2 between unit vectors = 2(1 - cosine_sim)
        sim  = torch.mm(z, z.T)
        mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        min_dist = (1.0 - sim[mask]).min().item()
        return min_dist < self.cfg.collapse_margin

    # ------------------------------------------------------------------
    # Contextual neighbourhood reconstruction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruct_context(
        self,
        q: torch.Tensor,          # (latent_dim,)
        query_timestamp: int,
    ) -> Tuple[torch.Tensor, float]:
        """
        Reconstruct a local pseudo-neighbourhood embedding for a live query.

        Returns a weighted aggregate of top-k neighbour embeddings,
        normalised by their SPD-adjusted scores.

        Returns
        -------
        context : (latent_dim,) — weighted pseudo-context embedding
        confidence : float      — mean SPD score of retrieved neighbours
        """
        _, scores, z_nbrs = self.query(q, query_timestamp=query_timestamp)
        # scores : (top_k,),  z_nbrs : (top_k, D)

        w = scores.clamp(min=0.0)
        total = w.sum()

        if total < 1e-9:
            # No confident match; return query itself
            return F.normalize(q, dim=0), 0.0

        context    = (w.unsqueeze(-1) * z_nbrs).sum(dim=0) / total
        context    = F.normalize(context, dim=0)
        confidence = float(w.mean().item())
        return context, confidence
