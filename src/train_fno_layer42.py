#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from neuralop.models import FNO
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_channel_norm(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    std = x.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    return mean, std


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FNO on layer42 uniform-time dataset")
    p.add_argument("--dataset", type=Path, default=Path("results/datasets/spe10_layer42_uniform.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("results/models/fno_layer42"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--n-train", type=int, default=8)
    p.add_argument("--hidden-channels", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-modes-h", type=int, default=16)
    p.add_argument("--n-modes-w", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def expand_sample_days(d: np.lib.npyio.NpzFile, n_samples: int) -> np.ndarray:
    if "sample_days" in d.files:
        sd = np.asarray(d["sample_days"])
        if sd.shape[0] == n_samples:
            return sd
    days = np.asarray(d["days"])
    if days.shape[0] == n_samples:
        return days
    if "centers" in d.files:
        centers = np.asarray(d["centers"])
        if days.shape[0] * centers.shape[0] == n_samples:
            return np.repeat(days, centers.shape[0])
    raise ValueError("Could not construct per-sample day labels for this dataset")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.dataset, allow_pickle=True)
    X = torch.from_numpy(d["X"].astype(np.float32))
    Y = torch.from_numpy(d["Y"].astype(np.float32))
    sample_days = expand_sample_days(d, X.shape[0])

    if X.ndim != 4 or Y.ndim != 4:
        raise ValueError("Expected X and Y to be rank-4 tensors (N,C,H,W)")
    if args.n_train <= 0 or args.n_train >= X.shape[0]:
        raise ValueError(f"n-train must be in [1, {X.shape[0]-1}]")

    # Keep chronological split by day to mimic forecasting use.
    x_train, x_val = X[: args.n_train], X[args.n_train :]
    y_train, y_val = Y[: args.n_train], Y[args.n_train :]
    day_train = sample_days[: args.n_train]
    day_val = sample_days[args.n_train :]

    x_mean, x_std = compute_channel_norm(x_train)
    y_mean, y_std = compute_channel_norm(y_train)

    x_train_n = (x_train - x_mean) / x_std
    x_val_n = (x_val - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    train_loader = DataLoader(TensorDataset(x_train_n, y_train_n), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_n, y_val_n), batch_size=args.batch_size, shuffle=False)

    device = select_device()
    model = FNO(
        n_modes=(args.n_modes_h, args.n_modes_w),
        in_channels=X.shape[1],
        out_channels=Y.shape[1],
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.shape[0]
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mae_phys = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.shape[0]

                # MAE in physical units after de-normalization.
                pred_phys = pred * y_std.to(device) + y_mean.to(device)
                yb_phys = yb * y_std.to(device) + y_mean.to(device)
                val_mae_phys += torch.mean(torch.abs(pred_phys - yb_phys)).item() * xb.shape[0]

        val_loss /= len(val_loader.dataset)
        val_mae_phys /= len(val_loader.dataset)
        sched.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": epoch,
                "train_mse_norm": train_loss,
                "val_mse_norm": val_loss,
                "val_mae_phys": val_mae_phys,
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} train_mse={train_loss:.6f} "
                f"val_mse={val_loss:.6f} val_mae_phys={val_mae_phys:.6f}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "fno_layer42_best.pt"
    torch.save(
        {
            "model_state": best_state,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "config": vars(args),
            "days_train": day_train.tolist(),
            "days_val": day_val.tolist(),
            "input_channels": d["input_channels"].tolist(),
            "target_channels": d["target_channels"].tolist(),
        },
        ckpt_path,
    )

    hist_path = args.out_dir / "history.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Saved model:", ckpt_path)
    print("Saved history:", hist_path)
    print("Device:", device)
    print("Train days:", day_train.tolist())
    print("Val days:", day_val.tolist())


if __name__ == "__main__":
    main()
