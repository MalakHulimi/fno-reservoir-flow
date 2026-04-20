#!/usr/bin/env python3
"""Build single-layer SPE10 dataset from simulator outputs at uniform report days.

PoC means "proof of concept": a small, fast experiment to verify that the
end-to-end data and model pipeline works before scaling up.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from resdata.grid import Grid
from resdata.resfile import ResdataFile


NX, NY, NZ = 60, 220, 85


def parse_eclipse_values(file_path: Path, keyword: str) -> np.ndarray:
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
                    c, v = token.split("*", 1)
                    values.extend([float(v)] * int(c))
                else:
                    values.append(float(token))
            if end_block:
                break
    return np.asarray(values, dtype=np.float32)


def get_report_days(prt_path: Path) -> list[int]:
    days: list[int] = []
    with prt_path.open("r", encoding="utf-8") as f:
        for line in f:
            if "Report step" not in line or " at day " not in line:
                continue
            part = line.split(" at day ", 1)[1].split("/", 1)[0].strip()
            try:
                days.append(int(round(float(part))))
            except ValueError:
                continue
    # Keep unique values in original order
    seen = set()
    uniq: list[int] = []
    for d in days:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq


def downsample_2d(field: np.ndarray, y_stride: int, x_stride: int) -> np.ndarray:
    h, w = field.shape
    h2 = (h // y_stride) * y_stride
    w2 = (w // x_stride) * x_stride
    cropped = field[:h2, :w2]
    return cropped.reshape(h2 // y_stride, y_stride, w2 // x_stride, x_stride).mean(axis=(1, 3))


def index_from_xyz(ix: int, iy: int, iz: int) -> int:
    # Eclipse ordering: I fastest, then J, then K
    return iz * (NX * NY) + iy * NX + ix


def extract_layer(field_flat: np.ndarray, z_layer: int) -> np.ndarray:
    layer = np.empty((NY, NX), dtype=np.float32)
    for iy in range(NY):
        for ix in range(NX):
            layer[iy, ix] = field_flat[index_from_xyz(ix, iy, z_layer)]
    return layer


def extract_keyword_for_days(rst: ResdataFile, keyword: str, report_indices: list[int]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for rep_idx in report_indices:
        kw = rst.iget_named_kw(keyword, rep_idx)
        out.append(np.asarray(kw, dtype=np.float32))
    return out


def build_dataset(
    data_dir: Path,
    out_path: Path,
    z_layer: int,
    day_start: int,
    day_stop: int,
    day_step: int,
    y_stride: int,
    x_stride: int,
) -> None:
    if not (0 <= z_layer < NZ):
        raise ValueError(f"z_layer must be in [0, {NZ-1}], got {z_layer}")

    base = data_dir / "SPE10_MODEL2"
    prt_path = data_dir / "SPE10_MODEL2.PRT"

    report_days = get_report_days(prt_path)
    desired_days = list(range(day_start, day_stop + 1, day_step))
    used_days = [d for d in desired_days if d in report_days]
    if not used_days:
        raise RuntimeError("No requested uniform days were found in PRT report steps")

    # Map day -> report index based on PRT ordering (report step lines)
    day_to_report_index = {day: idx for idx, day in enumerate(report_days)}
    report_indices = [day_to_report_index[d] for d in used_days]

    perm_file = data_dir / "SPE10MODEL2_PERM.INC"
    phi_file = data_dir / "SPE10MODEL2_PHI.INC"

    permx = parse_eclipse_values(perm_file, "PERMX").reshape(NZ, NY, NX)
    permy = parse_eclipse_values(perm_file, "PERMY").reshape(NZ, NY, NX)
    permz = parse_eclipse_values(perm_file, "PERMZ").reshape(NZ, NY, NX)
    poro = parse_eclipse_values(phi_file, "PORO").reshape(NZ, NY, NX)

    eps = 1e-8
    kx = np.log10(np.maximum(permx[z_layer], eps))
    ky = np.log10(np.maximum(permy[z_layer], eps))
    kz = np.log10(np.maximum(permz[z_layer], eps))
    phi = poro[z_layer]

    static_stack = np.stack(
        [
            downsample_2d(kx, y_stride, x_stride),
            downsample_2d(ky, y_stride, x_stride),
            downsample_2d(kz, y_stride, x_stride),
            downsample_2d(phi, y_stride, x_stride),
        ],
        axis=0,
    )

    rst = ResdataFile(str(base.with_suffix(".UNRST")))
    pressure_list = extract_keyword_for_days(rst, "PRESSURE", report_indices)
    swat_list = extract_keyword_for_days(rst, "SWAT", report_indices)

    x_samples: list[np.ndarray] = []
    y_samples: list[np.ndarray] = []

    for day, p_flat, sw_flat in zip(used_days, pressure_list, swat_list):
        p_layer = downsample_2d(extract_layer(p_flat, z_layer), y_stride, x_stride)
        sw_layer = downsample_2d(extract_layer(sw_flat, z_layer), y_stride, x_stride)

        day_channel = np.full_like(p_layer, float(day), dtype=np.float32)
        x_samples.append(np.concatenate([static_stack, day_channel[None, ...]], axis=0))
        y_samples.append(np.stack([p_layer, sw_layer], axis=0))

    X = np.stack(x_samples, axis=0).astype(np.float32)
    Y = np.stack(y_samples, axis=0).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        Y=Y,
        days=np.asarray(used_days, dtype=np.int32),
        z_layer=np.asarray([z_layer], dtype=np.int32),
        input_channels=np.asarray(["log10_kx", "log10_ky", "log10_kz", "poro", "time_day"]),
        target_channels=np.asarray(["pressure", "swat"]),
    )

    print("PoC = proof of concept (small fast validation dataset)")
    print("Saved:", out_path)
    print("Used days:", used_days)
    print("Layer:", z_layer)
    print("X shape:", X.shape, "(N, C_in, H, W)")
    print("Y shape:", Y.shape, "(N, C_out, H, W)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build layer-specific uniform-time dataset from UNRST")
    p.add_argument("--data-dir", type=Path, default=Path("data/opm-data/spe10model2"))
    p.add_argument("--out-path", type=Path, default=Path("results/datasets/spe10_layer42_uniform.npz"))
    p.add_argument("--z-layer", type=int, default=42)
    p.add_argument("--day-start", type=int, default=1000)
    p.add_argument("--day-stop", type=int, default=2000)
    p.add_argument("--day-step", type=int, default=100)
    p.add_argument("--y-stride", type=int, default=2)
    p.add_argument("--x-stride", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        data_dir=args.data_dir,
        out_path=args.out_path,
        z_layer=args.z_layer,
        day_start=args.day_start,
        day_stop=args.day_stop,
        day_step=args.day_step,
        y_stride=args.y_stride,
        x_stride=args.x_stride,
    )


if __name__ == "__main__":
    main()
