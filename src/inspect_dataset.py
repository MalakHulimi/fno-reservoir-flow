#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def summarize_array(name: str, arr: np.ndarray) -> None:
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    print(
        f"{name}: min={np.nanmin(arr):.6g}, max={np.nanmax(arr):.6g}, "
        f"mean={np.nanmean(arr):.6g}, std={np.nanstd(arr):.6g}"
    )
    print(f"{name}: nan_count={np.isnan(arr).sum()}, inf_count={np.isinf(arr).sum()}")


def plot_sample_maps(dataset_path: Path, out_dir: Path, sample_index: int) -> None:
    d = np.load(dataset_path, allow_pickle=True)
    x = d["X"]
    y = d["Y"]
    days = d["days"] if "days" in d.files else None
    in_names = d["input_channels"] if "input_channels" in d.files else None
    out_names = d["target_channels"] if "target_channels" in d.files else None

    sample_index = max(0, min(sample_index, x.shape[0] - 1))
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1, axes1 = plt.subplots(1, x.shape[1], figsize=(3.6 * x.shape[1], 3.5), constrained_layout=True)
    if x.shape[1] == 1:
        axes1 = [axes1]
    for c in range(x.shape[1]):
        ax = axes1[c]
        im = ax.imshow(x[sample_index, c], cmap="viridis", origin="lower")
        title = f"X channel {c}"
        if in_names is not None:
            title = str(in_names[c])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    day_str = ""
    if days is not None:
        day_str = f" day={int(days[sample_index])}"
    fig1.suptitle(f"Input channels sample={sample_index}{day_str}")
    in_png = out_dir / "dataset_preview_inputs.png"
    fig1.savefig(in_png, dpi=140)
    plt.close(fig1)

    fig2, axes2 = plt.subplots(1, y.shape[1], figsize=(4.2 * y.shape[1], 3.6), constrained_layout=True)
    if y.shape[1] == 1:
        axes2 = [axes2]
    for c in range(y.shape[1]):
        ax = axes2[c]
        im = ax.imshow(y[sample_index, c], cmap="magma", origin="lower")
        title = f"Y channel {c}"
        if out_names is not None:
            title = str(out_names[c])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig2.suptitle(f"Target channels sample={sample_index}{day_str}")
    out_png = out_dir / "dataset_preview_targets.png"
    fig2.savefig(out_png, dpi=140)
    plt.close(fig2)

    print(f"Saved preview images:\n- {in_png}\n- {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect dataset npz content and generate preview plots")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("results/plots"))
    args = p.parse_args()

    d = np.load(args.dataset, allow_pickle=True)
    print("Keys:", list(d.files))

    x = d["X"]
    y = d["Y"]
    summarize_array("X", x)
    summarize_array("Y", y)

    if "days" in d.files:
        print("days:", d["days"])
    if "z_layer" in d.files:
        print("z_layer:", d["z_layer"])
    if "input_channels" in d.files:
        print("input_channels:", d["input_channels"])
    if "target_channels" in d.files:
        print("target_channels:", d["target_channels"])

    plot_sample_maps(args.dataset, args.out_dir, args.sample_index)


if __name__ == "__main__":
    main()
