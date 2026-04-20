#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
    seen = set()
    uniq: list[int] = []
    for d in days:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq


def index_from_xyz(ix: int, iy: int, iz: int) -> int:
    return iz * (NX * NY) + iy * NX + ix


def extract_layer(field_flat: np.ndarray, z_layer: int) -> np.ndarray:
    layer = np.empty((NY, NX), dtype=np.float32)
    for iy in range(NY):
        for ix in range(NX):
            layer[iy, ix] = field_flat[index_from_xyz(ix, iy, z_layer)]
    return layer


def downsample_2d(field: np.ndarray, y_stride: int, x_stride: int) -> np.ndarray:
    h, w = field.shape
    h2 = (h // y_stride) * y_stride
    w2 = (w // x_stride) * x_stride
    cropped = field[:h2, :w2]
    return cropped.reshape(h2 // y_stride, y_stride, w2 // x_stride, x_stride).mean(axis=(1, 3))


def extract_keyword_for_days(rst: ResdataFile, keyword: str, report_indices: list[int]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for rep_idx in report_indices:
        kw = rst.iget_named_kw(keyword, rep_idx)
        out.append(np.asarray(kw, dtype=np.float32))
    return out


def build_dataset(
    data_dir: Path,
    out_path: Path,
    center_z: int,
    half_window: int,
    day_start: int,
    day_stop: int,
    day_step: int,
    y_stride: int,
    x_stride: int,
) -> None:
    z_layers = list(range(center_z - half_window, center_z + half_window + 1))
    if min(z_layers) < 0 or max(z_layers) >= NZ:
        raise ValueError(f"Requested z-window {z_layers} goes outside valid range 0..{NZ-1}")

    prt_path = data_dir / "SPE10_MODEL2.PRT"
    report_days = get_report_days(prt_path)
    desired_days = list(range(day_start, day_stop + 1, day_step))
    used_days = [d for d in desired_days if d in report_days]
    if not used_days:
        raise RuntimeError("No requested days found in report steps")
    day_min = float(min(used_days))
    day_max = float(max(used_days))
    day_span = max(day_max - day_min, 1.0)
    day_to_report_index = {day: idx for idx, day in enumerate(report_days)}
    report_indices = [day_to_report_index[d] for d in used_days]

    perm_file = data_dir / "SPE10MODEL2_PERM.INC"
    phi_file = data_dir / "SPE10MODEL2_PHI.INC"
    permx = parse_eclipse_values(perm_file, "PERMX").reshape(NZ, NY, NX)
    permy = parse_eclipse_values(perm_file, "PERMY").reshape(NZ, NY, NX)
    permz = parse_eclipse_values(perm_file, "PERMZ").reshape(NZ, NY, NX)
    poro = parse_eclipse_values(phi_file, "PORO").reshape(NZ, NY, NX)

    eps = 1e-8
    static_channels: list[np.ndarray] = []
    input_channel_names: list[str] = []
    for z in z_layers:
        static_channels.extend(
            [
                downsample_2d(np.log10(np.maximum(permx[z], eps)), y_stride, x_stride),
                downsample_2d(np.log10(np.maximum(permy[z], eps)), y_stride, x_stride),
                downsample_2d(np.log10(np.maximum(permz[z], eps)), y_stride, x_stride),
                downsample_2d(poro[z], y_stride, x_stride),
            ]
        )
        input_channel_names.extend(
            [
                f"log10_kx_z{z}",
                f"log10_ky_z{z}",
                f"log10_kz_z{z}",
                f"poro_z{z}",
            ]
        )

    static_stack = np.stack(static_channels, axis=0)

    rst = ResdataFile(str((data_dir / "SPE10_MODEL2").with_suffix(".UNRST")))
    pressure_list = extract_keyword_for_days(rst, "PRESSURE", report_indices)
    swat_list = extract_keyword_for_days(rst, "SWAT", report_indices)

    x_samples: list[np.ndarray] = []
    y_samples: list[np.ndarray] = []
    for day, p_flat, sw_flat in zip(used_days, pressure_list, swat_list):
        p_layer = downsample_2d(extract_layer(p_flat, center_z), y_stride, x_stride)
        sw_layer = downsample_2d(extract_layer(sw_flat, center_z), y_stride, x_stride)
        day_norm = (float(day) - day_min) / day_span
        day_channel = np.full_like(p_layer, day_norm, dtype=np.float32)
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
        center_z=np.asarray([center_z], dtype=np.int32),
        z_layers=np.asarray(z_layers, dtype=np.int32),
        input_channels=np.asarray(input_channel_names + ["time_norm_0_1"]),
        target_channels=np.asarray(["pressure", "swat"]),
    )

    print("Saved:", out_path)
    print("Used days:", used_days)
    print("Center layer:", center_z)
    print("Input z-layers:", z_layers)
    print("Time normalization: day_norm = (day -", int(day_min), ") /", int(day_span), ")")
    print("X shape:", X.shape, "(N, C_in, H, W)")
    print("Y shape:", Y.shape, "(N, C_out, H, W)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 2.5D layer-window dataset from UNRST")
    p.add_argument("--data-dir", type=Path, default=Path("data/opm-data/spe10model2"))
    p.add_argument("--out-path", type=Path, default=Path("results/datasets/spe10_layer42_window5_fullres.npz"))
    p.add_argument("--center-z", type=int, default=42)
    p.add_argument("--half-window", type=int, default=2)
    p.add_argument("--day-start", type=int, default=200)
    p.add_argument("--day-stop", type=int, default=1900)
    p.add_argument("--day-step", type=int, default=100)
    p.add_argument("--y-stride", type=int, default=1)
    p.add_argument("--x-stride", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        data_dir=args.data_dir,
        out_path=args.out_path,
        center_z=args.center_z,
        half_window=args.half_window,
        day_start=args.day_start,
        day_stop=args.day_stop,
        day_step=args.day_step,
        y_stride=args.y_stride,
        x_stride=args.x_stride,
    )


if __name__ == "__main__":
    main()
