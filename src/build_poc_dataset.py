#!/usr/bin/env python3
"""Build a lightweight SPE10 proof-of-concept dataset.

This script intentionally avoids heavy binary parsing of UNRST/INIT files.
It uses rock properties (.INC files) to create small 2D samples that are fast
to load and train on Apple Silicon (MPS).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


NX, NY, NZ = 60, 220, 85


def parse_eclipse_values(file_path: Path, keyword: str) -> np.ndarray:
    """Read values for a keyword from an Eclipse-style .INC file.

    Supports repeated-value tokens like "10*0.5".
    """
    values: list[float] = []
    capturing = False

    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("--"):
                continue

            if not capturing:
                if line == keyword:
                    capturing = True
                continue

            end_block = "/" in line
            line = line.replace("/", " ")

            for token in line.split():
                if "*" in token:
                    count_str, value_str = token.split("*", 1)
                    values.extend([float(value_str)] * int(count_str))
                else:
                    values.append(float(token))

            if end_block:
                break

    if not values:
        raise ValueError(f"No values found for keyword {keyword} in {file_path}")

    return np.asarray(values, dtype=np.float32)


def downsample_2d(field: np.ndarray, y_stride: int, x_stride: int) -> np.ndarray:
    """Average-pool a 2D field by integer strides."""
    h, w = field.shape
    h2 = (h // y_stride) * y_stride
    w2 = (w // x_stride) * x_stride
    cropped = field[:h2, :w2]
    pooled = cropped.reshape(h2 // y_stride, y_stride, w2 // x_stride, x_stride).mean(axis=(1, 3))
    return pooled


def build_dataset(
    data_dir: Path,
    out_path: Path,
    z_stride: int,
    y_stride: int,
    x_stride: int,
    max_samples: int,
) -> None:
    perm_file = data_dir / "SPE10MODEL2_PERM.INC"
    phi_file = data_dir / "SPE10MODEL2_PHI.INC"

    permx = parse_eclipse_values(perm_file, "PERMX").reshape(NZ, NY, NX)
    permy = parse_eclipse_values(perm_file, "PERMY").reshape(NZ, NY, NX)
    poro = parse_eclipse_values(phi_file, "PORO").reshape(NZ, NY, NX)

    layer_ids = list(range(0, NZ - 1, z_stride))
    if max_samples > 0:
        layer_ids = layer_ids[: max_samples + 1]

    x_samples: list[np.ndarray] = []
    y_samples: list[np.ndarray] = []
    pair_layers: list[tuple[int, int]] = []

    eps = 1e-8
    for z in layer_ids[:-1]:
        z_next = z + 1

        kx = np.log10(np.maximum(permx[z], eps))
        ky = np.log10(np.maximum(permy[z], eps))
        phi = poro[z]

        kx_ds = downsample_2d(kx, y_stride, x_stride)
        ky_ds = downsample_2d(ky, y_stride, x_stride)
        phi_ds = downsample_2d(phi, y_stride, x_stride)

        # PoC target: next-layer log(Kx). This is a lightweight supervised target
        # that can be built without binary simulator state parsing.
        y_target = np.log10(np.maximum(permx[z_next], eps))
        y_ds = downsample_2d(y_target, y_stride, x_stride)

        x_samples.append(np.stack([kx_ds, ky_ds, phi_ds], axis=0))
        y_samples.append(y_ds[None, ...])
        pair_layers.append((z, z_next))

    x_arr = np.stack(x_samples, axis=0).astype(np.float32)
    y_arr = np.stack(y_samples, axis=0).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=x_arr,
        Y=y_arr,
        pair_layers=np.asarray(pair_layers, dtype=np.int32),
        channels=np.asarray(["log10_permx", "log10_permy", "poro"]),
        target=np.asarray(["next_layer_log10_permx"]),
    )

    print("Saved dataset:", out_path)
    print("X shape:", x_arr.shape, "(N, C, H, W)")
    print("Y shape:", y_arr.shape, "(N, 1, H, W)")
    print("Layer pairs:", pair_layers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lightweight SPE10 PoC dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/opm-data/spe10model2"),
        help="Directory containing SPE10 .INC files",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("results/datasets/spe10_poc.npz"),
        help="Output compressed dataset path",
    )
    parser.add_argument("--z-stride", type=int, default=4, help="Use every Nth layer in Z")
    parser.add_argument("--y-stride", type=int, default=2, help="Downsample factor in Y")
    parser.add_argument("--x-stride", type=int, default=2, help="Downsample factor in X")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=12,
        help="Maximum number of training pairs (<=0 means all available)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        data_dir=args.data_dir,
        out_path=args.out_path,
        z_stride=args.z_stride,
        y_stride=args.y_stride,
        x_stride=args.x_stride,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
