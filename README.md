# FNO Reservoir Flow Prediction

Fourier Neural Operator (FNO) for predicting fluid flow in porous media using the SPE10 Model 2 dataset.

## Project Structure

```
fno-reservoir-flow/
├── data/
│   └── opm-data/
│       └── spe10model2/         ← SPE10 Model 2 dataset files
│           ├── SPE10_MODEL2.DATA
│           ├── SPE10MODEL2_PERM.INC
│           ├── SPE10MODEL2_PHI.INC
│           └── SPE10MODEL2_TOPS.INC
├── notebooks/                   ← Jupyter notebooks for exploration
├── src/                         ← FNO model code
├── results/                     ← Training outputs, plots
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the SPE10 dataset

```bash
cd ~/Desktop/fno-reservoir-flow/data
git clone --depth=1 --filter=blob:none --sparse https://github.com/OPM/opm-data.git
cd opm-data
git sparse-checkout set spe10model2
```

> Uses sparse checkout to download **only** the `spe10model2` folder, not the full repository.

### 2. Install Python dependencies

```bash
cd ~/Desktop/fno-reservoir-flow
pip install -r requirements.txt
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

---

## What We Just Did (Simple)

1. Ran OPM Flow and generated simulator outputs (`.UNRST`, `.PRT`, etc.).
2. Built a learning dataset from real simulator states (not synthetic targets):
     - Fixed one layer: `z = 42`
     - Used uniform report times (every 100 days)
     - Inputs: `log10_kx, log10_ky, log10_kz, poro, time_day`
     - Targets: `pressure, swat`
3. Trained an FNO model locally on Mac (MPS backend).
4. Evaluated predictions on held-out days and saved:
     - loss curve
     - ground truth vs prediction vs error maps

Useful output files:
- Dataset: `results/datasets/spe10_layer42_uniform.npz`
- Trained model: `results/models/fno_layer42/fno_layer42_best.pt`
- Training history: `results/models/fno_layer42/history.json`
- Loss plot: `results/models/fno_layer42/loss_curves.png`
- Evaluation plots: `results/eval/fno_layer42/*.png`

---

## Run With More Samples

To increase sample count, use a wider day range while keeping uniform timestep spacing:

```bash
cd ~/Desktop/fno-reservoir-flow

# 1) Build larger dataset (layer 42, every 100 days)
.venv/bin/python src/build_layer42_uniform_dataset.py \
    --z-layer 42 \
    --day-start 200 \
    --day-stop 2000 \
    --day-step 100 \
    --x-stride 2 \
    --y-stride 2 \
    --out-path results/datasets/spe10_layer42_uniform_more.npz

# 2) Train FNO on the larger dataset
.venv/bin/python src/train_fno_layer42.py \
    --dataset results/datasets/spe10_layer42_uniform_more.npz \
    --out-dir results/models/fno_layer42_more \
    --epochs 60 \
    --batch-size 2 \
    --n-train 15

# 3) Evaluate and generate GT vs Pred vs Error plots
.venv/bin/python src/eval_fno_layer42.py \
    --dataset results/datasets/spe10_layer42_uniform_more.npz \
    --ckpt results/models/fno_layer42_more/fno_layer42_best.pt \
    --out-dir results/eval/fno_layer42_more \
    --n-train 15

# 4) Plot loss curves
.venv/bin/python src/plot_training_history.py \
    --history results/models/fno_layer42_more/history.json \
    --out results/models/fno_layer42_more/loss_curves.png
```

If the available report days differ in your run, the dataset builder automatically keeps only days that exist in `SPE10_MODEL2.PRT`.

---

## Two Most Important Fixes (Pressure Noise)

To keep changes minimal and focused, apply only these two fixes:

1. **Lower Fourier modes** in FNO (less high-frequency capacity):
     - `n_modes_h = 8`
     - `n_modes_w = 6`
2. **Normalize time input** to `[0, 1]` instead of raw day values.

### Reproducible Command (2.5D window, full resolution)

```bash
cd ~/Desktop/fno-reservoir-flow

# 1) Build dataset with time normalization and 2.5D window (z=40..44 -> target z=42)
.venv/bin/python src/build_layer_window_dataset.py \
    --center-z 42 \
    --half-window 2 \
    --day-start 200 \
    --day-stop 1900 \
    --day-step 100 \
    --x-stride 1 \
    --y-stride 1 \
    --out-path results/datasets/spe10_layer42_window5_fullres_timefix.npz

# 2) Train with reduced Fourier modes
.venv/bin/python src/train_fno_layer42.py \
    --dataset results/datasets/spe10_layer42_window5_fullres_timefix.npz \
    --out-dir results/models/fno_layer42_window5_fullres_modes86_timefix \
    --epochs 50 \
    --batch-size 1 \
    --n-train 15 \
    --n-modes-h 8 \
    --n-modes-w 6

# 3) Evaluate and create GT/Pred/Error plots
.venv/bin/python src/eval_fno_layer42.py \
    --dataset results/datasets/spe10_layer42_window5_fullres_timefix.npz \
    --ckpt results/models/fno_layer42_window5_fullres_modes86_timefix/fno_layer42_best.pt \
    --out-dir results/eval/fno_layer42_window5_fullres_modes86_timefix \
    --n-train 15

# 4) Plot training curves
.venv/bin/python src/plot_training_history.py \
    --history results/models/fno_layer42_window5_fullres_modes86_timefix/history.json \
    --out results/models/fno_layer42_window5_fullres_modes86_timefix/loss_curves.png
```

Output locations for this focused run:
- Metrics: `results/eval/fno_layer42_window5_fullres_modes86_timefix/metrics.json`
- Pressure/SWAT maps: `results/eval/fno_layer42_window5_fullres_modes86_timefix/*.png`
- Loss curve: `results/models/fno_layer42_window5_fullres_modes86_timefix/loss_curves.png`

### Results Snapshot (Keep Even If Artifacts Are Deleted)

The following numbers are from completed runs on layer-window setup (center `z=42`, input `z=40..44`, full resolution, days `200..1900` step `100`, train/val split `15/3`).

| Run name | What changed | Pressure MAE | Pressure RMSE | Swat MAE | Swat RMSE |
|---|---|---:|---:|---:|---:|
| `fno_layer42_window5_fullres` | Baseline window5 fullres | 15.5786 | 19.9445 | 0.01187 | 0.02116 |
| `fno_layer42_window5_fullres_modes86_timefix` | Time normalized + lower modes `(8,6)` | 17.8406 | 23.9728 | 0.01474 | 0.02711 |
| `fno_layer42_window5_fullres_timefix_only` | Time normalized only, original modes | 15.5786 | 19.9446 | 0.01187 | 0.02116 |
| `fno_layer20_window9_upper_fullres` | Upper-zone wider window (z=16..24, center z=20) | 13.4001 | 16.8331 | 0.02157 | 0.03209 |

Quick interpretation:
- Lowering modes to `(8,6)` made this case worse.
- Time normalization alone was neutral in this setup.
- Keep `fno_layer42_window5_fullres` as the stronger baseline among these three.
- Weight decay `1e-4` gave a very tiny pressure improvement, not worth changing the default value from `1e-6`.
- Upper-zone wider window improved pressure but degraded swat.

### Next Step to Improve Performance (Single Best Test)

Use the current best setup (`fno_layer42_window5_fullres`) and change only **weight decay** to add mild smoothing regularization while keeping the same Fourier modes.

Recommended trial:

```bash
cd ~/Desktop/fno-reservoir-flow

.venv/bin/python src/train_fno_layer42.py \
    --dataset results/datasets/spe10_layer42_window5_fullres_timefix.npz \
    --out-dir results/models/fno_layer42_window5_fullres_wd1e4 \
    --epochs 50 \
    --batch-size 1 \
    --n-train 15 \
    --weight-decay 1e-4

.venv/bin/python src/eval_fno_layer42.py \
    --dataset results/datasets/spe10_layer42_window5_fullres_timefix.npz \
    --ckpt results/models/fno_layer42_window5_fullres_wd1e4/fno_layer42_best.pt \
    --out-dir results/eval/fno_layer42_window5_fullres_wd1e4 \
    --n-train 15
```

Goal: reduce pressure-map speckle/noise without sacrificing overall error.

---

## Dataset: SPE10 Model 2

SPE10 Model 2 is a standard benchmark dataset for reservoir simulation, originally published by the Society of Petroleum Engineers (SPE) and made publicly available by **Mike Christie** (Heriot-Watt University) and **Martin Blunt** (Imperial College London).

### Grid

| Property | Value |
|---|---|
| Dimensions | 60 × 220 × 85 cells |
| Cell size (x, y) | 20 ft × 10 ft |
| Cell size (z) | 2 ft |
| Total cells | 1,122,000 |
| Depth to top layer | 12,000 ft |

### Geological Formations

The 85 vertical layers represent two distinct geological formations:

| Layers | Formation | Character |
|---|---|---|
| 1 – 35 | Tarbert (fluvial) | Moderate heterogeneity, relatively smooth permeability distribution |
| 36 – 85 | Upper Ness (deltaic) | Extreme heterogeneity, sharp contrasts, channelized structures |

The Upper Ness layers are particularly challenging — permeability can vary by several orders of magnitude within a few cells.

### Data Files

#### `SPE10MODEL2_PERM.INC`
Permeability in the x, y, and z directions, given in millidarcy (mD). Contains three sequential keyword blocks:
- `PERMX` — permeability in the x-direction (60×220×85 = 1,122,000 values)
- `PERMY` — permeability in the y-direction (1,122,000 values)
- `PERMZ` — permeability in the z-direction (1,122,000 values; typically Kz = 0.1 × Kx)

Values range from near zero (tight rock) to over 10,000 mD (highly permeable channels).

#### `SPE10MODEL2_PHI.INC`
Porosity (dimensionless fraction, 0–1) for each grid cell under the keyword `PORO`. Contains 1,122,000 values. Zero porosity cells have been replaced with `1e-7` for numerical stability in simulation. Typical values range from ~0.05 to ~0.35.

#### `SPE10MODEL2_TOPS.INC`
Top depth (in feet) of each grid cell in the first layer, under the keyword `TOPS`. Each of the 85 layers spans 13,200 cells (60×220), with depths increasing by 2 ft per layer from 12,000 ft to 12,168 ft.

#### `SPE10_MODEL2.DATA`
The main Eclipse/OPM simulation input deck. References the three `.INC` files above and defines the full simulation setup including fluid properties, well locations, and schedule.

### Reading the Data in Python

The `.INC` files are plain-text Eclipse keyword format. Each file has a header comment block, followed by a keyword name on its own line, then whitespace-separated float values ending with `/`.

```python
import numpy as np

def read_inc(filepath, keyword):
    """Read an Eclipse .INC file and return values for the given keyword as a numpy array."""
    values = []
    capturing = False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line == keyword:
                capturing = True
                continue
            if not capturing or line.startswith('--'):
                continue
            line = line.rstrip('/')
            values.extend(float(v) for v in line.split() if v)
            if '/' in line:
                break
    return np.array(values)

NX, NY, NZ = 60, 220, 85

permx = read_inc('data/opm-data/spe10model2/SPE10MODEL2_PERM.INC', 'PERMX').reshape(NZ, NY, NX)
permy = read_inc('data/opm-data/spe10model2/SPE10MODEL2_PERM.INC', 'PERMY').reshape(NZ, NY, NX)
poro  = read_inc('data/opm-data/spe10model2/SPE10MODEL2_PHI.INC',  'PORO' ).reshape(NZ, NY, NX)
```

Each reshaped array has shape `(85, 220, 60)` — layers × y-cells × x-cells.

---

## FNO Approach

A **Fourier Neural Operator (FNO)** learns a mapping between function spaces and is well-suited to PDE-governed problems like Darcy flow in porous media.

### Task

Given the static rock properties of a 2D layer (permeability field), predict the resulting pressure or velocity field without running a full numerical simulation.

- **Input:** Log-permeability field `log(Kx)`, optionally with `Ky`, `φ`, and spatial coordinates — shape `(60, 220, C)`
- **Output:** Pressure field `p(x, y)` or Darcy velocity components `(vx, vy)` — shape `(60, 220)`

### Why FNO?

| Method | Cost | Generalizes to new grids |
|---|---|---|
| Full numerical simulator | High (minutes–hours per run) | Yes |
| CNN surrogate | Low (ms) | No (fixed resolution) |
| FNO surrogate | Low (ms) | Yes (resolution-invariant) |

FNO operates in Fourier space, capturing global structure efficiently. It can be trained on one resolution and evaluated on another, which is valuable for multi-scale reservoir problems.

### Training Data Strategy

Each of the 85 layers of SPE10 Model 2 is an independent 60×220 2D domain. The Tarbert and Upper Ness formations provide diverse heterogeneity levels, making this a good benchmark for generalization across geological complexity.

---

## References

- Christie, M.A. & Blunt, M.J. (2001). *Tenth SPE Comparative Solution Project: A Comparison of Upscaling Techniques*. SPE Reservoir Evaluation & Engineering, 4(4), 308–317.
- Li, Z. et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations*. ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
- OPM dataset repository: https://github.com/OPM/opm-data
