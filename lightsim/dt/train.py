"""Training loop and utilities for Decision Transformer."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import Trajectory, TrajectoryDataset
from .model import DTConfig, DecisionTransformer


def get_device(device: str = "auto") -> torch.device:
    """Resolve device string to a torch.device.

    Parameters
    ----------
    device : str
        ``"auto"`` picks CUDA if available, else CPU.
        Also accepts ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_dt(
    trajectories: list[Trajectory],
    config: DTConfig | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_steps: int = 100,
    context_len: int = 20,
    device: str = "auto",
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
        ``"auto"`` (default) uses CUDA if available.
        Also accepts ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.
    verbose : bool
        Print epoch losses.

    Returns
    -------
    model : DecisionTransformer
        Trained model (moved to CPU for portability).
    losses : list[float]
        Per-epoch average loss.
    rtg_stats : dict
        RTG normalization stats (``"mean"`` and ``"std"``).
        Pass to ``DTPolicy`` for correct inference conditioning.
    """
    dev = get_device(device)

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
    rtg_stats = {"mean": dataset.rtg_mean, "std": dataset.rtg_std}
    use_cuda = dev.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) > batch_size,
        pin_memory=use_cuda,
        num_workers=0,
    )

    model = DecisionTransformer(config).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine schedule with linear warmup
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_warmup(step, warmup_steps, total_steps),
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Device: {dev} | Params: {n_params:,} | "
              f"Dataset: {len(dataset):,} samples | "
              f"Batches/epoch: {len(loader)}")

    epoch_losses: list[float] = []
    model.train()
    t0 = time.perf_counter()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            obs = batch["observations"].to(dev, non_blocking=use_cuda)
            act = batch["actions"].to(dev, non_blocking=use_cuda)
            rtg = batch["returns_to_go"].to(dev, non_blocking=use_cuda)
            ts = batch["timesteps"].to(dev, non_blocking=use_cuda)
            mask = batch["mask"].to(dev, non_blocking=use_cuda)

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
            elapsed = time.perf_counter() - t0
            print(f"  Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  "
                  f"[{elapsed:.1f}s elapsed]")

    # Move model to CPU for portable saving / inference
    model.cpu()
    model.eval()

    if verbose:
        total_time = time.perf_counter() - t0
        print(f"  Training complete: {total_time:.1f}s total")

    return model, epoch_losses, rtg_stats


def _cosine_warmup(step: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def save_dt_model(
    model: DecisionTransformer,
    path: str | Path,
    rtg_stats: dict | None = None,
) -> None:
    """Save model config + state dict + RTG stats to a .pt file."""
    torch.save({
        "config": model.config,
        "state_dict": model.state_dict(),
        "rtg_stats": rtg_stats,
    }, str(path))


def load_dt_model(
    path: str | Path, device: str = "cpu",
) -> tuple[DecisionTransformer, dict | None]:
    """Load a saved Decision Transformer.

    Parameters
    ----------
    path : str | Path
        Path to .pt checkpoint.
    device : str
        Target device. Defaults to ``"cpu"`` for portable inference.
        Use ``"auto"`` or ``"cuda"`` for GPU inference.

    Returns
    -------
    model : DecisionTransformer
    rtg_stats : dict | None
        RTG normalization stats (``"mean"`` and ``"std"``), or None.
    """
    dev = get_device(device)
    checkpoint = torch.load(str(path), map_location=dev, weights_only=False)
    config = checkpoint["config"]
    model = DecisionTransformer(config).to(dev)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    rtg_stats = checkpoint.get("rtg_stats")
    return model, rtg_stats


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
    parser.add_argument("--device", default="auto",
                        help="Device: 'auto', 'cpu', 'cuda', 'cuda:0'")
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
    model, losses, rtg_stats = train_dt(
        trajectories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        context_len=args.context_len,
        lr=args.lr,
        device=args.device,
    )
    save_dt_model(model, args.save, rtg_stats=rtg_stats)
    print(f"  Saved model to {args.save}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  RTG stats: mean={rtg_stats['mean']:.1f}, std={rtg_stats['std']:.1f}")
