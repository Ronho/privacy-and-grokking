# Evaluation and Visualization

This guide explains how to extract metrics from experiments, evaluate results, and generate plots.

> [!IMPORTANT]
> To successfully run visualization scripts, you must follow the correct pipeline dependency order. Plotting scripts expect specific artifacts and pre-calculated metrics to already exist.

*(This section will be expanded as the visualization scripts are refactored to use the new Tracking & Artifact abstractions).*

## 1. Extracting Metrics
Before you can plot results, you must extract and format the metrics from your training runs. 
This is usually done by running the `evaluate` command or specific metric extraction scripts.

```bash
uv run pag evaluate v2.3.0
```
*(Example only - precise scripts required for extraction will be listed here).*

### Required Artifacts for Extraction
The extraction scripts typically require:
- Model Checkpoints (`model.pth`)
- Training metrics (`metrics.jsonl` or MLflow API access)

### 1.1 PCA Trajectory Extraction
To extract PCA trajectories and evaluate the 2D loss/weight-norm grid surfaces:

```bash
uv run python visualizations/preprocessing/extract_trajectories.py \
    --run-id <run_id> \
    --center-mode final \
    --grid-resolution 40
```

- `--center-mode final` (recommended by Li et al., 2018): Shifts trajectory by the final converged model $\theta_n$ ($M = W - \theta_n$) so that the origin $(0, 0)$ is precisely the final model state where background loss equals the final training loss.
- Output artifacts:
  - `cache/runs/<run_id>/trajectories/projection.json`
  - `cache/runs/<run_id>/trajectories/grid.json`

### 1.2 Filter-Wise Normalized Landscape Extraction (Without Trajectory)
To evaluate the local 2D loss surface and sharpness around a specific checkpoint model $\theta^*$ using scale-invariant filter-wise normalized random directions:

```bash
uv run python visualizations/preprocessing/extract_filter_normalized_landscape.py \
    --run-id <run_id> \
    --resolution 40 \
    --x-min -1.0 --x-max 1.0 \
    --y-min -1.0 --y-max 1.0
```

- Direction normalization rule: $d_{l, j} \leftarrow \frac{d_{l, j}}{\|d_{l, j}\|_F} \|W_{l, j}\|_F$ for each conv filter / linear neuron.
- Output artifacts:
  - `cache/runs/<run_id>/filter_normalized/grid_<step>.json`

### 1.3 Checkpoint Distance Matrix Extraction
To compute pairwise Euclidean distance matrices between parameter vectors across checkpoints:

```bash
# Single pair
uv run python visualizations/preprocessing/compute_checkpoint_distances.py \
    --run-id-1 <run_id_1> \
    --run-id-2 <run_id_2>

# Multi-run batch
uv run python visualizations/preprocessing/compute_checkpoint_distances.py \
    --run-ids <run_id_1>,<run_id_2>,<run_id_3>
```

- Output artifacts:
  - `cache/runs/<run_id_1>/distances/<run_id_2}_distance.json`

## 2. Running Visualizations

### 2.1 2D PCA Loss Landscape & Weight Norm Surface
To plot the 2-panel square visualization with logarithmic loss contours and trajectory overlay:

```bash
uv run python visualizations/images/plot_pca_loss_landscape.py \
    --run-id <run_id> \
    --min-loss 1e-8 \
    --max-loss 10.0 \
    --tag-interval 50000
```

Key features:
- **Logarithmic scale**: LogNorm contouring from $10^1$ down to $10^{-8}$ accurately resolves flat grokking plateaus.
- **Smart milestone tagging**: Automatically groups coincident checkpoints (e.g. `[100k, 150k]` at converged basins) and performs collision-avoidance search to eliminate label overlap.
- **Output files**:
  - `plots/pca_loss_landscape/pca_loss_landscape_<run_id>.pdf`
  - `plots/pca_loss_landscape/pca_loss_landscape_<run_id>.png` (300 DPI)

### 2.2 Filter-Wise Normalized 2D Loss Landscape
To plot the local 2-panel loss & weight norm surface around a model checkpoint:

```bash
uv run python visualizations/images/plot_filter_normalized_landscape.py \
    --run-id <run_id> \
    --min-loss 1e-8 \
    --max-loss 10.0
```

- **Objective Sharpness/Curvature**: Eliminates artificial sharpness caused by weight scale differences.
- **Origin $\theta^*$**: Centered at $(0, 0)$ representing the unperturbed model state.
- **Output files**:
  - `plots/filter_normalized_landscape/filter_normalized_landscape_<run_id>_<step>.pdf`
  - `plots/filter_normalized_landscape/filter_normalized_landscape_<run_id>_<step>.png` (300 DPI)

### 2.3 Checkpoint Distance Heatmaps & Grid
To plot 2D Euclidean distance heatmaps across checkpoint optimization trajectories:

```bash
# Single pair
uv run python visualizations/images/plot_checkpoint_distance_matrix.py \
    --run-id-1 <run_id_1> \
    --run-id-2 <run_id_2>

# Multi-run grid (sorted by model_index then initialization_scale)
uv run python visualizations/images/plot_checkpoint_distance_matrix.py \
    --run-ids <run_id_1>,<run_id_2>,<run_id_3> \
    --log-scale
```

Key features:
- **Logarithmic scale (`--log-scale`)**: Highlights fine-grained weight distances across early steps while preserving global dynamic range.
- **Structured Sorting**: Multi-run grids automatically group by `model_index` and sort ascending by `initialization_scale`.
### 2.3 Joint Multi-Trajectory Loss Landscapes

Visualizes multiple training trajectories simultaneously projected onto a shared 2D PCA subspace with symmetric logarithmic (`symlog`) axes.

```bash
# 1. Preprocessing (Joint PCA & Grid Loss Evaluation)
uv run python visualizations/preprocessing/extract_multi_trajectories.py \
    --run-ids "<run_id_1>,<run_id_2>,..." \
    --group-id joint_12runs \
    --grid-spacing symlog \
    --linthresh 0.05 \
    --grid-resolution 45

# 2. Rendering Figure
uv run python visualizations/images/plot_multi_trajectories_landscape.py \
    --group-id joint_12runs \
    --xscale symlog \
    --yscale symlog \
    --linthresh 0.05 \
    --panels both
```

Key features:
- **Shared Subspace**: Performs SVD on the joint displacement matrix across all runs and checkpoints.
- **Symmetric Logarithmic Scaling (`--xscale symlog --yscale symlog`)**: Resolves trajectories spanning orders of magnitude in initialization scale (e.g. from $1.0$ to $32.0$) on a single coordinate frame.
- **Dual Panels (`--panels both`)**: Displays training loss contours alongside the weight norm surface ($\|\theta\|_2$).
### 2.4 1D Linear Parameter Interpolation

Evaluates and visualizes exact, zero-projection loss, accuracy, and weight norm profiles along parameter line segments:
$$\theta(\alpha) = (1 - \alpha)\,\theta_A + \alpha\,\theta_B \quad \text{for } \alpha \in [\alpha_{\min}, \alpha_{\max}]$$

```bash
# 1. Single pair mode (e.g. from initialization theta_0 to final theta_*)
uv run python visualizations/images/plot_1d_interpolation.py \
    --run-id <run_id> \
    --start-step 0 \
    --end-step 150000 \
    --alpha-min -0.5 \
    --alpha-max 1.5

# 2. Jumps grid mode (all detected slingshot/instability transitions side-by-side)
uv run python visualizations/images/plot_1d_interpolation.py \
    --run-id <run_id> \
    --jumps-grid
```

Key features:
- **Zero Projection Variance**: Evaluates directly in full parameter space, capturing true energy barriers and steep drops (e.g. to $10^{-8}$) without PCA distortion.
- **Dual Axes in Jumps Grid**: Left y-axis displays logarithmic train and test loss ($10^{-8} \dots 10^2$); right y-axis displays weight norm $\|\theta(\alpha)\|_2$.
- **Output files**:
  - `plots/interpolation_1d/interpolation_1d_<run_id>_<start>_<end>.pdf`
  - `plots/interpolation_1d/interpolation_1d_<run_id>_jumps_grid.pdf`
  - `plots/interpolation_1d/interpolation_1d_<run_id>_jumps_grid.png` (300 DPI)

### 2.5 2D Local Subspace Jump Planes (Orthogonal Slingshot Visual Proof)

Evaluates and renders the local 2D subspace plane $(u_1, u_2)$ around each optimization jump/slingshot:
- $u_1$: Unit vector along the incoming drift direction ($v_{\mathrm{in}}$) approaching the minimum.
- $u_2$: Gram-Schmidt orthonormalized jump vector ($v_{\mathrm{jump}}^\perp$) capturing the slingshot kick.

```bash
uv run python visualizations/images/plot_jump_planes.py \
    --run-id <run_id> \
    --window-steps 500 \
    --grid-resolution 35
```

Key features:
- **Visual Proof of $\approx 90^\circ$ Turns**: Proves that the model enters the minimum along $u_1$ and is ejected orthogonally along $u_2$ during Edge-of-Stability events.
- **Logarithmic 2D Landscape**: Evaluates $L(\theta_{\mathrm{pre}} + x u_1 + y u_2)$ on a power-of-10 grid ($10^{-8} \dots 10^2$).
- **Output files**:
  - `plots/jump_planes/jump_planes_2d_<run_id>.pdf`
  - `plots/jump_planes/jump_planes_2d_<run_id>.png` (300 DPI)


