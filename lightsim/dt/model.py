"""Decision Transformer model for traffic signal control.

Reference: Chen et al. (2021), "Decision Transformer: Reinforcement Learning
via Sequence Modeling", NeurIPS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DTConfig:
    """Decision Transformer hyperparameters."""

    obs_dim: int = 12
    act_dim: int = 4
    hidden_dim: int = 64
    n_layers: int = 3
    n_heads: int = 4
    ffn_dim: int = 256
    max_timestep: int = 4096
    context_len: int = 20
    dropout: float = 0.1


class DecisionTransformer(nn.Module):
    """Decision Transformer: predicts actions from (RTG, obs, action) sequences.

    Architecture
    ------------
    - Token embeddings: Linear for RTG and obs, Embedding for actions
    - Timestep embedding added to all tokens at the same timestep
    - Pre-LN Transformer with causal masking (timestep-level: t attends to <= t)
    - Action head predicts from state token positions

    Parameters
    ----------
    config : DTConfig
        Model hyperparameters.
    """

    def __init__(self, config: DTConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim

        # Token embeddings
        self.embed_rtg = nn.Linear(1, d)
        self.embed_obs = nn.Linear(config.obs_dim, d)
        self.embed_act = nn.Embedding(config.act_dim, d)

        # Timestep embedding
        self.embed_timestep = nn.Embedding(config.max_timestep, d)

        # Embedding layer norm + dropout
        self.embed_ln = nn.LayerNorm(d)
        self.embed_drop = nn.Dropout(config.dropout)

        # Transformer encoder (Pre-LN via custom blocks)
        self.blocks = nn.ModuleList([
            _TransformerBlock(d, config.n_heads, config.ffn_dim, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(d)

        # Action prediction head (from state token positions)
        self.action_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, config.act_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        observations: torch.Tensor,   # (B, K, obs_dim)
        actions: torch.Tensor,         # (B, K)
        returns_to_go: torch.Tensor,   # (B, K)
        timesteps: torch.Tensor,       # (B, K)
        mask: torch.Tensor | None = None,  # (B, K), 1=valid, 0=pad
    ) -> torch.Tensor:
        """Forward pass.

        Returns
        -------
        action_logits : torch.Tensor, shape (B, K, act_dim)
            Logits for action prediction at each state position.
        """
        B, K = observations.shape[:2]
        d = self.config.hidden_dim

        # Embed each modality
        rtg_emb = self.embed_rtg(returns_to_go.unsqueeze(-1))  # (B, K, d)
        obs_emb = self.embed_obs(observations)                  # (B, K, d)
        act_emb = self.embed_act(actions)                        # (B, K, d)

        # Add timestep embedding
        time_emb = self.embed_timestep(timesteps)  # (B, K, d)
        rtg_emb = rtg_emb + time_emb
        obs_emb = obs_emb + time_emb
        act_emb = act_emb + time_emb

        # Interleave: [RTG_0, obs_0, act_0, RTG_1, obs_1, act_1, ...]
        # Shape: (B, 3*K, d)
        tokens = torch.stack([rtg_emb, obs_emb, act_emb], dim=2)  # (B, K, 3, d)
        tokens = tokens.reshape(B, 3 * K, d)

        tokens = self.embed_ln(tokens)
        tokens = self.embed_drop(tokens)

        # Build causal mask: (3K, 3K), 2D additive mask
        causal_mask = self._build_causal_mask(K, tokens.device)  # (3K, 3K)

        # Combine with padding mask if provided
        if mask is not None:
            # Expand mask from (B, K) to (B, 3*K)
            pad_mask = mask.unsqueeze(2).expand(B, K, 3).reshape(B, 3 * K)
            # Additive pad mask: (B, 1, 3K) → broadcast to (B, 3K, 3K)
            additive_pad = (1.0 - pad_mask).unsqueeze(1) * (-1e9)
            # Expand causal to (B, 3K, 3K) via broadcast
            n_heads = self.config.n_heads
            # Need (B*n_heads, 3K, 3K) for per-batch masking
            full_mask = causal_mask.unsqueeze(0) + additive_pad  # (B, 3K, 3K)
            full_mask = full_mask.unsqueeze(1).expand(B, n_heads, -1, -1)
            full_mask = full_mask.reshape(B * n_heads, 3 * K, 3 * K)
        else:
            full_mask = causal_mask  # (3K, 3K) — 2D, shared across batch

        # Transformer blocks
        x = tokens
        for block in self.blocks:
            x = block(x, full_mask)
        x = self.ln_f(x)

        # Extract state token positions: indices 1, 4, 7, ... (obs positions)
        state_indices = torch.arange(1, 3 * K, 3, device=x.device)
        state_tokens = x[:, state_indices, :]  # (B, K, d)

        # Predict actions
        action_logits = self.action_head(state_tokens)  # (B, K, act_dim)
        return action_logits

    def _build_causal_mask(self, K: int, device: torch.device) -> torch.Tensor:
        """Build timestep-level causal mask for 3K sequence.

        For mask[i, j]: query position i can attend to key position j
        only if key's timestep (j//3) <= query's timestep (i//3).

        Returns (3K, 3K) additive mask: 0 where allowed, -1e9 where blocked.
        """
        seq_len = 3 * K
        positions = torch.arange(seq_len, device=device)
        query_ts = (positions // 3).unsqueeze(1)  # (3K, 1) — rows
        key_ts = (positions // 3).unsqueeze(0)    # (1, 3K) — columns
        # causal: query at row i can attend to key at col j if key_ts <= query_ts
        causal = (key_ts <= query_ts).float()  # (3K, 3K)
        # Convert to additive mask: 0 where allowed, -1e9 where blocked
        return (1.0 - causal) * (-1e9)  # (3K, 3K)


class _TransformerBlock(nn.Module):
    """Pre-LN Transformer block."""

    def __init__(self, d: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Pre-LN self-attention
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + h
        # Pre-LN FFN
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + h
        return x
