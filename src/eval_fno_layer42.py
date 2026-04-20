#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from neuralop.models import FNO


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained FNO on layer42 uniform dataset")
    p.add_argument("--dataset", type=Path, default=Path("results/datasets/spe10_layer42_uniform.npz"))
    p.add_argument("--ckpt", type=Path, default=Path("results/models/fno_layer42/fno_layer42_best.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("results/eval/fno_layer42"))
    p.add_argument("--n-train", type=int, default=8)
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


def expand_sample_centers(d: np.lib.npyio.NpzFile, n_samples: int) -> np.ndarray | None:
    if "sample_centers" in d.files:
        sc = np.asarray(d["sample_centers"])
        if sc.shape[0] == n_samples:
            return sc
    if "centers" in d.files and "days" in d.files:
        centers = np.asarray(d["centers"])
        days = np.asarray(d["days"])
        if days.shape[0] * centers.shape[0] == n_samples:
            return np.tile(centers, days.shape[0])
    return None


def build_model_from_ckpt(ckpt: dict, in_channels: int, out_channels: int, device: torch.device) -> FNO:
    cfg = ckpt["config"]
    model = FNO(
        n_modes=(int(cfg["n_modes_h"]), int(cfg["n_modes_w"])),
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=int(cfg["hidden_channels"]),
        n_layers=int(cfg["n_layers"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset, allow_pickle=True)
    X = torch.from_numpy(data["X"].astype(np.float32))
    Y = torch.from_numpy(data["Y"].astype(np.float32))
    sample_days = expand_sample_days(data, X.shape[0])
    sample_centers = expand_sample_centers(data, X.shape[0])
    in_names = [str(v) for v in data["input_channels"]]
    out_names = [str(v) for v in data["target_channels"]]

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    x_mean = ckpt["x_mean"].float()
    x_std = ckpt["x_std"].float()
    y_mean = ckpt["y_mean"].float()
    y_std = ckpt["y_std"].float()

    if args.n_train <= 0 or args.n_train >= X.shape[0]:
        raise ValueError(f"n-train must be in [1, {X.shape[0]-1}]")

    x_val = X[args.n_train :]
    y_val = Y[args.n_train :]
    day_val = sample_days[args.n_train :]
    center_val = sample_centers[args.n_train :] if sample_centers is not None else None

    x_val_n = (x_val - x_mean) / x_std
    device = select_device()
    model = build_model_from_ckpt(ckpt, in_channels=X.shape[1], out_channels=Y.shape[1], device=device)

    with torch.no_grad():
        pred_n = model(x_val_n.to(device)).cpu()
    pred = pred_n * y_std + y_mean

    # Aggregate metrics across all validation samples and cells.
    diff = pred - y_val
    mae = torch.mean(torch.abs(diff), dim=(0, 2, 3))
    rmse = torch.sqrt(torch.mean(diff * diff, dim=(0, 2, 3)))

    metrics = {
        "days_val": day_val.tolist(),
        "mae_per_channel": {out_names[i]: float(mae[i]) for i in range(len(out_names))},
        "rmse_per_channel": {out_names[i]: float(rmse[i]) for i in range(len(out_names))},
    }
    metrics_path = args.out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save side-by-side plots for each validation day and each output channel.
    for i in range(y_val.shape[0]):
        day = int(day_val[i])
        center_suffix = ""
        if center_val is not None:
            center_suffix = f"_z{int(center_val[i])}"
        for c, cname in enumerate(out_names):
            true_map = y_val[i, c].numpy()
            pred_map = pred[i, c].numpy()
            err_map = pred_map - true_map

            vmin = float(min(true_map.min(), pred_map.min()))
            vmax = float(max(true_map.max(), pred_map.max()))
            emax = float(max(abs(err_map.min()), abs(err_map.max())))

            fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

            im0 = axes[0].imshow(true_map, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
            axes[0].set_title(f"True {cname}")
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(pred_map, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
            axes[1].set_title(f"Pred {cname}")
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(err_map, cmap="coolwarm", origin="lower", vmin=-emax, vmax=emax)
            axes[2].set_title(f"Error {cname}")
            axes[2].set_xticks([])
            axes[2].set_yticks([])
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            title = f"Day {day}"
            if center_val is not None:
                title += f" | z={int(center_val[i])}"
            title += f" | Channel {cname}"
            fig.suptitle(title)
            out_img = args.out_dir / f"day_{day}{center_suffix}_{cname}.png"
            fig.savefig(out_img, dpi=140)
            plt.close(fig)

    print("Device:", device)
    print("Validation days:", day_val.tolist())
    print("Input channels:", in_names)
    print("Target channels:", out_names)
    print("MAE:", {out_names[i]: float(mae[i]) for i in range(len(out_names))})
    print("RMSE:", {out_names[i]: float(rmse[i]) for i in range(len(out_names))})
    print("Saved metrics:", metrics_path)
    print("Saved plots dir:", args.out_dir)


if __name__ == "__main__":
    main()
