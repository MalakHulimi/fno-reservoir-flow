#!/usr/bin/env python3
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
    uniq: list[int] = []
    seen = set()
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


def build_dataset(
    data_dir: Path,
    out_path: Path,
    half_window: int,
    day_start: int,
    day_stop: int,
    day_step: int,
    y_stride: int,
    x_stride: int,
    center_step: int,
) -> None:
    prt_path = data_dir / "SPE10_MODEL2.PRT"
    report_days = get_report_days(prt_path)
    desired_days = list(range(day_start, day_stop + 1, day_step))
    used_days = [d for d in desired_days if d in report_days]
    if not used_days:
        raise RuntimeError("No requested days found in report steps")

    centers = list(range(half_window, NZ - half_window, center_step))
    if not centers:
        raise RuntimeError("No center layers available with given half-window/center-step")

    day_min = float(min(used_days))
    day_max = float(max(used_days))
    day_span = max(day_max - day_min, 1.0)

    perm_file = data_dir / "SPE10MODEL2_PERM.INC"
    phi_file = data_dir / "SPE10MODEL2_PHI.INC"

    permx = parse_eclipse_values(perm_file, "PERMX").reshape(NZ, NY, NX)
    permy = parse_eclipse_values(perm_file, "PERMY").reshape(NZ, NY, NX)
    permz = parse_eclipse_values(perm_file, "PERMZ").reshape(NZ, NY, NX)
    poro = parse_eclipse_values(phi_file, "PORO").reshape(NZ, NY, NX)

    # Load simulator states for selected report days.
    day_to_report_idx = {d: i for i, d in enumerate(report_days)}
    report_indices = [day_to_report_idx[d] for d in used_days]
    base = data_dir / "SPE10_MODEL2"
    rst = ResdataFile(str(base.with_suffix(".UNRST")))
    grid = Grid(str(base.with_suffix(".EGRID")))
    actnum = np.asarray(grid.export_actnum(), dtype=np.int32).reshape(-1)
    full_size = NX * NY * NZ
    if actnum.size != full_size:
        raise RuntimeError(f"ACTNUM size mismatch: expected {full_size}, got {actnum.size}")
    active_mask = actnum.astype(bool)
    n_active = int(active_mask.sum())

    def to_full(flat_vals: np.ndarray) -> np.ndarray:
        if flat_vals.size == full_size:
            return flat_vals.reshape(NZ, NY, NX)
        if flat_vals.size == n_active:
            full = np.zeros(full_size, dtype=np.float32)
            full[active_mask] = flat_vals
            return full.reshape(NZ, NY, NX)
        raise RuntimeError(
            f"Unsupported restart vector length {flat_vals.size}; expected {full_size} or {n_active}"
        )

    p_days: list[np.ndarray] = []
    sw_days: list[np.ndarray] = []
    for ri in report_indices:
        p_flat = np.asarray(rst.iget_named_kw("PRESSURE", ri), dtype=np.float32)
        sw_flat = np.asarray(rst.iget_named_kw("SWAT", ri), dtype=np.float32)
        p_days.append(to_full(p_flat))
        sw_days.append(to_full(sw_flat))

    eps = 1e-8
    h_out = (NY // y_stride)
    w_out = (NX // x_stride)
    c_in = 4 * (2 * half_window + 1) + 2  # +time_norm +center_norm
    n_samples = len(used_days) * len(centers)

    X = np.empty((n_samples, c_in, h_out, w_out), dtype=np.float32)
    Y = np.empty((n_samples, 2, h_out, w_out), dtype=np.float32)

    input_names: list[str] = []
    for rel in range(-half_window, half_window + 1):
        input_names.extend(
            [
                f"log10_kx_rel{rel}",
                f"log10_ky_rel{rel}",
                f"log10_kz_rel{rel}",
                f"poro_rel{rel}",
            ]
        )
    input_names.extend(["time_norm_0_1", "center_norm_0_1"])

    # Order samples by time first, then center layer to preserve chronological split behavior.
    idx = 0
    for day_i, day in enumerate(used_days):
        p3d = p_days[day_i]
        sw3d = sw_days[day_i]
        day_norm = (float(day) - day_min) / day_span

        for center in centers:
            channels: list[np.ndarray] = []
            for z in range(center - half_window, center + half_window + 1):
                channels.extend(
                    [
                        downsample_2d(np.log10(np.maximum(permx[z], eps)), y_stride, x_stride),
                        downsample_2d(np.log10(np.maximum(permy[z], eps)), y_stride, x_stride),
                        downsample_2d(np.log10(np.maximum(permz[z], eps)), y_stride, x_stride),
                        downsample_2d(poro[z], y_stride, x_stride),
                    ]
                )

            center_norm = float(center) / float(NZ - 1)
            base_shape = channels[0].shape
            channels.append(np.full(base_shape, day_norm, dtype=np.float32))
            channels.append(np.full(base_shape, center_norm, dtype=np.float32))

            X[idx] = np.stack(channels, axis=0)
            Y[idx, 0] = downsample_2d(p3d[center], y_stride, x_stride)
            Y[idx, 1] = downsample_2d(sw3d[center], y_stride, x_stride)
            idx += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        Y=Y,
        days=np.asarray(used_days, dtype=np.int32),
        centers=np.asarray(centers, dtype=np.int32),
        input_channels=np.asarray(input_names),
        target_channels=np.asarray(["pressure", "swat"]),
    )

    print("Saved:", out_path)
    print("Used days:", used_days)
    print("Center layers count:", len(centers), "first/last:", centers[0], centers[-1])
    print("X shape:", X.shape, "(N, C_in, H, W)")
    print("Y shape:", Y.shape, "(N, C_out, H, W)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build all-layer sliding-window dataset from UNRST")
    p.add_argument("--data-dir", type=Path, default=Path("data/opm-data/spe10model2"))
    p.add_argument("--out-path", type=Path, default=Path("results/datasets/spe10_alllayers_window5_fullres.npz"))
    p.add_argument("--half-window", type=int, default=2)
    p.add_argument("--day-start", type=int, default=200)
    p.add_argument("--day-stop", type=int, default=1900)
    p.add_argument("--day-step", type=int, default=100)
    p.add_argument("--y-stride", type=int, default=1)
    p.add_argument("--x-stride", type=int, default=1)
    p.add_argument("--center-step", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        data_dir=args.data_dir,
        out_path=args.out_path,
        half_window=args.half_window,
        day_start=args.day_start,
        day_stop=args.day_stop,
        day_step=args.day_step,
        y_stride=args.y_stride,
        x_stride=args.x_stride,
        center_step=args.center_step,
    )


if __name__ == "__main__":
    main()
