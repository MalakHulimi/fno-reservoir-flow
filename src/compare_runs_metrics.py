#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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


def build_model(ckpt: dict, in_ch: int, out_ch: int, device: torch.device) -> FNO:
    cfg = ckpt["config"]
    model = FNO(
        n_modes=(int(cfg["n_modes_h"]), int(cfg["n_modes_w"])),
        in_channels=in_ch,
        out_channels=out_ch,
        hidden_channels=int(cfg["hidden_channels"]),
        n_layers=int(cfg["n_layers"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    y_mean_abs = float(np.mean(np.abs(y_true))) + 1e-12
    y_range = float(np.max(y_true) - np.min(y_true)) + 1e-12
    mae_rel_pct = 100.0 * mae / y_mean_abs
    rmse_rel_pct = 100.0 * rmse / y_mean_abs
    nrmse_range_pct = 100.0 * rmse / y_range
    return {
        "mae": mae,
        "rmse": rmse,
        "mae_rel_pct": mae_rel_pct,
        "rmse_rel_pct": rmse_rel_pct,
        "nrmse_range_pct": nrmse_range_pct,
    }


def expand_sample_days(dataset: np.lib.npyio.NpzFile, sample_count: int) -> np.ndarray:
    days = np.asarray(dataset["days"])
    if days.shape[0] == sample_count:
        return days.astype(np.int64)

    centers = np.asarray(dataset.get("centers", []))
    if days.ndim == 1 and centers.ndim == 1 and days.size * max(centers.size, 1) == sample_count:
        if centers.size == 0:
            raise ValueError("Dataset day metadata does not match sample count")
        return np.repeat(days.astype(np.int64), centers.size)

    raise ValueError("Could not expand dataset days to match sample count")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare absolute and relative metrics across multiple runs")
    p.add_argument("--out-dir", type=Path, default=Path("results/eval/summary"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        {
            "name": "baseline_ds2",
            "dataset": Path("results/datasets/spe10_layer42_uniform.npz"),
            "ckpt": Path("results/models/fno_layer42/fno_layer42_best.pt"),
            "n_train": 8,
        },
        {
            "name": "more_samples_ds2",
            "dataset": Path("results/datasets/spe10_layer42_uniform_more.npz"),
            "ckpt": Path("results/models/fno_layer42_more/fno_layer42_best.pt"),
            "n_train": 15,
        },
        {
            "name": "more_samples_fullres",
            "dataset": Path("results/datasets/spe10_layer42_uniform_more_fullres.npz"),
            "ckpt": Path("results/models/fno_layer42_more_fullres/fno_layer42_best.pt"),
            "n_train": 15,
        },
        {
            "name": "finer_fullres",
            "dataset": Path("results/datasets/spe10_layer42_finer_fullres.npz"),
            "ckpt": Path("results/models/fno_layer42_finer_fullres/fno_layer42_best.pt"),
            "n_train": 30,
        },
        {
            "name": "alllayers_window5_fullres",
            "dataset": Path("results/datasets/spe10_alllayers_window5_fullres.npz"),
            "ckpt": Path("results/models/fno_alllayers_window5_fullres/fno_layer42_best.pt"),
            "n_train": 1215,
        },
    ]

    device = select_device()
    rows: list[dict[str, object]] = []

    for run in runs:
        d = np.load(run["dataset"], allow_pickle=True)
        X = torch.from_numpy(d["X"].astype(np.float32))
        Y = torch.from_numpy(d["Y"].astype(np.float32))
        sample_days = expand_sample_days(d, X.shape[0])
        out_names = [str(v) for v in d["target_channels"]]

        ckpt = torch.load(run["ckpt"], map_location="cpu", weights_only=False)
        x_mean = ckpt["x_mean"].float()
        x_std = ckpt["x_std"].float()
        y_mean = ckpt["y_mean"].float()
        y_std = ckpt["y_std"].float()

        n_train = int(run["n_train"])
        x_val = X[n_train:]
        y_val = Y[n_train:]
        day_val = sample_days[n_train:]
        x_val_n = (x_val - x_mean) / x_std

        model = build_model(ckpt, in_ch=X.shape[1], out_ch=Y.shape[1], device=device)
        with torch.no_grad():
            pred_n = model(x_val_n.to(device)).cpu()
        pred = pred_n * y_std + y_mean

        y_true_np = y_val.numpy()
        y_pred_np = pred.numpy()

        run_row: dict[str, object] = {
            "run": run["name"],
            "samples_total": int(X.shape[0]),
            "samples_val": int(y_val.shape[0]),
            "days_val": [int(v) for v in np.unique(day_val).tolist()],
            "grid": f"{X.shape[2]}x{X.shape[3]}",
        }

        for ci, cname in enumerate(out_names):
            m = compute_metrics(y_true_np[:, ci], y_pred_np[:, ci])
            for k, v in m.items():
                run_row[f"{cname}_{k}"] = float(v)

        rows.append(run_row)

    json_path = args.out_dir / "runs_metrics_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    csv_path = args.out_dir / "runs_metrics_summary.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Plot relative errors for quick visual comparison.
    run_names = [r["run"] for r in rows]
    p_mae_rel = [r["pressure_mae_rel_pct"] for r in rows]
    p_rmse_rel = [r["pressure_rmse_rel_pct"] for r in rows]
    sw_mae_rel = [r["swat_mae_rel_pct"] for r in rows]
    sw_rmse_rel = [r["swat_rmse_rel_pct"] for r in rows]

    x = np.arange(len(run_names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    ax.bar(x - 1.5 * width, p_mae_rel, width=width, label="pressure MAE%")
    ax.bar(x - 0.5 * width, p_rmse_rel, width=width, label="pressure RMSE%")
    ax.bar(x + 0.5 * width, sw_mae_rel, width=width, label="swat MAE%")
    ax.bar(x + 1.5 * width, sw_rmse_rel, width=width, label="swat RMSE%")
    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=15)
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Relative Error Comparison Across Runs")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncols=2)

    plot_path = args.out_dir / "runs_relative_error_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("Saved JSON:", json_path)
    print("Saved CSV:", csv_path)
    print("Saved plot:", plot_path)
    print("Rows:")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
