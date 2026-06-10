# Trajectory error computation (ATE / RPE)

Computes trajectory errors of algorithm trajectories against a groundtruth:
- Loading of trajectory data (TUM `.txt` or `.csv`).
- Timestamp association and rigid (Umeyama) alignment to groundtruth.
- ATE and RPE computation (vectorized, batched scipy rotations).
- Optional simple 3D plot of the aligned trajectories (pose frame only).

## Modules

| File          | Purpose                                              |
| ------------- | ---------------------------------------------------- |
| `analysis.py` | CLI: align trajectories, compute ATE/RPE, optionally plot |
| `load.py`     | Trajectory loading (`.txt` TUM / `.csv`)             |
| `align.py`    | Timestamp association + Umeyama alignment            |
| `metrics.py`  | ATE / RPE computation                                |
| `plot.py`     | Optional Open3D pose-frame plotting                  |

## Dependencies
```bash
pip install -r requirements.txt
```
`numpy`, `scipy`, `tabulate` are required; `pandas` (CSV input) and
`open3d` (`-p` plotting) are optional.

# Usage
```bash
python metrics/analysis.py <gt_filepath> <alg_filepath1> <alg_filepath2> ... [-d 1 5] [-t N] [-p]
```
## Arguments:
- `<gt_filepath>`: Path to the ground truth trajectory file.
- `<alg_filepathX>`: Paths to algorithm-generated trajectory files.
- `-d`, `--distance`: Sub-trajectory length(s) in meters for RPE (default: 1).
- `-t`, `--start N`: Discard the first `N` seconds before computing metrics,
  measured from the earliest timestamp across all inputs (gt + alg). Useful for
  skipping the initial, unstabilized alignment phase.
- `--save-cropped`: With `-t`, also write each cropped trajectory next to its
  source file as a TUM `.txt` suffixed `_cropped`.
- `-p`, `--plot`: Plot aligned trajectories (requires `open3d`).
- `--gt-quat-order`: Quaternion order in the groundtruth file (`xyzw`/`wxyz`).

Error metrics are always computed (`-c` is kept as a deprecated no-op).

# Output:
## 1. Error Metrics: Console table of ATE and RPE.
```
Trajectory           ATE Pos    ATE Rot    RPE Pos (d=1)    RPE Rot (d=1)
-----------------  ---------  ---------  ---------------  ---------------
trajectory1.txt     0.000000   0.000000         0.000000         0.000000
trajectory2.txt     0.013260   0.008574         0.016014         0.007070
```

## (optional) 2. Plots: Interactive 3D plots of trajectories
![Imgur](https://imgur.com/PwXkyF2.jpg)
