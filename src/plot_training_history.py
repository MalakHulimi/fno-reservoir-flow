#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training history curves")
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.history.open("r", encoding="utf-8") as f:
        hist = json.load(f)

    epochs = [row["epoch"] for row in hist]
    train_mse = [row["train_mse_norm"] for row in hist]
    val_mse = [row["val_mse_norm"] for row in hist]
    val_mae = [row["val_mae_phys"] for row in hist]

    fig, ax1 = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    l1 = ax1.plot(epochs, train_mse, label="train_mse_norm", color="#1f77b4", linewidth=2)
    l2 = ax1.plot(epochs, val_mse, label="val_mse_norm", color="#d62728", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Normalized MSE")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    l3 = ax2.plot(epochs, val_mae, label="val_mae_phys", color="#2ca02c", linestyle="--", linewidth=2)
    ax2.set_ylabel("Validation MAE (physical units)")

    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.set_title("FNO Training Curves")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
