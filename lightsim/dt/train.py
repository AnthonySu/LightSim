"""Training loop and utilities for Decision Transformer."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import Trajectory, TrajectoryDataset
from .model import DTConfig, DecisionTransformer


def train_dt(
    trajectories: list[Trajectory],
    config: DTConfig | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_steps: int = 100,
    context_len: int = 20,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[DecisionTransformer, list[float]]:
    """Train a Decision Transformer on collected trajectories.

    Parameters
    ----------
    trajectories : list[Trajectory]
        Training data from ``collect_trajectories()``.
    config : DTConfig | None
        Model config. If None, inferred from trajectories.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Peak learning rate for AdamW.
    weight_decay : float
        Weight decay for AdamW.
    warmup_steps : int
        Linear warmup steps for cosine schedule.
    context_len : int
        Subsequence length for training.
    device : str
        Device string ("cpu" or "cuda").
    verbose : bool
        Print epoch losses.

    Returns
    -------
    model : DecisionTransformer
        Trained model.
    losses : list[float]
        Per-epoch average loss.
    """
    # Infer dimensions from data
    obs_dim = trajectories[0].observations.shape[1]
    act_dim = int(max(t.actions.max() for t in trajectories)) + 1

    if config is None:
        config = DTConfig(
            obs_dim=obs_dim,
            act_dim=act_dim,
            context_len=context_len,
        )

    dataset = TrajectoryDataset(trajectories, context_len=context_len)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=len(dataset) > batch_size,
    )

    model = DecisionTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine schedule with linear warmup
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_warmup(step, warmup_steps, total_steps),
    )

    epoch_losses: list[float] = []
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            obs = batch["observations"].to(device)
            act = batch["actions"].to(device)
            rtg = batch["returns_to_go"].to(device)
            ts = batch["timesteps"].to(device)
            mask = batch["mask"].to(device)

            logits = model(obs, act, rtg, ts, mask)  # (B, K, act_dim)

            # Cross-entropy loss on valid (non-padded) positions only
            B, K, A = logits.shape
            logits_flat = logits.reshape(B * K, A)
            targets_flat = act.reshape(B * K)
            mask_flat = mask.reshape(B * K)

            loss_per_token = nn.functional.cross_entropy(
                logits_flat, targets_flat, reduction="none",
            )
            # Mask out padded positions
            loss = (loss_per_token * mask_flat).sum() / mask_flat.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

    return model, epoch_losses


def _cosine_warmup(step: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def save_dt_model(model: DecisionTransformer, path: str | Path) -> None:
    """Save model config + state dict to a .pt file."""
    torch.save({
        "config": model.config,
        "state_dict": model.state_dict(),
    }, str(path))


def load_dt_model(path: str | Path, device: str = "cpu") -> DecisionTransformer:
    """Load a saved Decision Transformer."""
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = DecisionTransformer(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    import argparse

    from .dataset import collect_trajectories, save_trajectories

    parser = argparse.ArgumentParser(description="Train Decision Transformer")
    parser.add_argument("--scenario", default="single-intersection-v0")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per controller")
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save", default="dt_model.pt")
    parser.add_argument("--save-data", default=None,
                        help="Save trajectory data to .npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Collecting trajectories ({args.episodes} eps/controller)...")
    trajectories = collect_trajectories(
        scenario=args.scenario,
        episodes_per_controller=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    print(f"  {len(trajectories)} trajectories, "
          f"{sum(t.length for t in trajectories)} total steps")

    if args.save_data:
        save_trajectories(trajectories, args.save_data)
        print(f"  Saved to {args.save_data}")

    print("Training Decision Transformer...")
    model, losses = train_dt(
        trajectories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        context_len=args.context_len,
        lr=args.lr,
    )
    save_dt_model(model, args.save)
    print(f"  Saved model to {args.save}")
    print(f"  Final loss: {losses[-1]:.4f}")
