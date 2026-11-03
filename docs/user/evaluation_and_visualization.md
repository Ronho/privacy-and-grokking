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
- **Panel Labels**: Subplots are labeled with lowercase letters `(a)`, `(b)`, ... instead of Roman numerals, matching the usual publication convention.
- **No Post-Jump Trajectory in the Plot**: Checkpoints after $\theta_{\mathrm{post}}$ are intentionally hidden so the orthogonal jump stays visually dominant. Their angle metrics are written to a separate text file.
- **Angle Export**: Per-jump angles (turn angle, outgoing angle, delta theta) and the jump distance are saved as a tab-separated table in `cache/plots/jump_planes/jump_planes_angles_{run_id}.txt`.
- **Extraction Pipeline**: `visualizations/preprocessing/extract_jump_planes.py` detects the jumps, constructs the $(u_1, u_2)$ basis from the incoming drift and the Gram-Schmidt-orthogonalized jump vector, evaluates the 35x35 loss grid, projects the trajectory checkpoints into the plane, and caches the result in `cache/runs/{run_id}/jump_planes/jumps_2d_planes.json`.
### 2.6 Optimizer Sweep Dynamics Overview

Visualizes optimization and training dynamics across optimizers (AdamW, RMSprop, SGD) for the models from the optimizer sweep in a publication-quality multi-row x 4-column layout:
- **Column (a)**: Optimization trajectory in the top 2 PC subspace with explained variance ratio and training step colorbar.
- **Column (b)**: Train & Test Accuracy (left axis) with Train & Test Loss (right axis, logarithmic).
- **Column (c)**: Total Weight Norm (left axis) with Distance from Start and Step Size (right axis, logarithmic).
- **Column (d)**: Optimizer Metrics (First Momentum $m_t$, Second Momentum $\sqrt{v_t}$, Gradient $\|\nabla L\|_2$, Update Size $\|\Delta \theta\|_2$).

```bash
# Generate all publication-ready optimizer dynamics figures
uv run python visualizations/images/plot_os_dynamics.py
```

- **Output files**:
  - `cache/plots/optimizer_sweep_dynamics/os_dynamics_GROK_MNIST_CE_MLP.pdf` (vector) & `.png`
  - `cache/plots/optimizer_sweep_dynamics/os_dynamics_other_models.pdf` (vector) & `.png`

### 2.7 Optimizer Sweep Jump Planes Overview

Generates the 2D local subspace jump planes for the exact same canonical default runs from the optimizer sweep as used in the dynamics plots. For each run, it extracts all slingshot jumps and plots the incoming drift and orthogonal jump over the local 2D loss landscape. Post-jump checkpoints are omitted; the per-jump angles are exported to a separate text file.

```bash
# Generate jump plane plots for all canonical optimizer sweep runs
uv run python visualizations/images/plot_os_jump.py
```

- **Output files**:
  - `cache/plots/jump_planes/jump_planes_2d_{run_id}.pdf` (vector) & `.png` for each canonical run ID


### 2.8 Canary Selection Overview

Generates a publication-quality multi-panel overview figure for canary selection across the 8 canonical model configurations. Rows correspond to the model configurations, and columns correspond to the canary types (e.g., Gaussian Noise, Uniform Noise, Watermark).

```bash
uv run python visualizations/images/plot_canary_selection.py
```

- **Output files**:
  - `cache/plots/canary_accuracy/canary_selection_overview.pdf`
  - `cache/plots/canary_accuracy/canary_selection_overview.png` (300 DPI)

### 2.9 Optimizer Sweep Ablation Overview

Generates one image per canonical model configuration showing optimizer ablations (AdamW Base, AdamW without weight decay, SGD, AdamW b1=b2=0). The visualization has two columns:
- **Column (a)**: Train and Test Accuracy (left axis) with Train and Test Loss (right axis).
- **Column (b)**: Optimizer dynamics (Momentum, Update Step, Weight Decay Term).

```bash
# Generate all optimizer ablation figures
uv run python visualizations/images/plot_os_ablation.py
```

- **Output files**:
  - `cache/plots/os_ablation/{run_name}.pdf`

### 2.10 Adversarial Class-Mean Reconstruction Notebook

The notebook `visualizations/images/plot_adversarial_class_mean.ipynb` demonstrates a model-inversion style attack: it generates one adversarial example per MNIST class that is confidently classified as that class while staying as close as possible to the **training class mean**. The class mean is never used by the optimizer; it is computed separately and only used to score the result.

The attack solves

$$
x_c^* = \arg\min_{x \in [0,1]^{C \times H \times W}}
\mathrm{CE}(f(x), c) + \lambda \lVert x - \mu_{\mathrm{global}} \rVert_2^2,
$$

where $\mu_{\mathrm{global}}$ is the public global pixel mean. The notebook compares the resulting distance to the true class mean with the distance distribution of natural training samples, and reports the optimization trajectory, the generated images, and the class means side by side.

Required inputs:
- `cache/runs/{run_id}/training_config.json` and `cache/runs/{run_id}/checkpoints/{step}/model.pth` for the configured run (the notebook uses the cached MNIST MLP run `088b42f7979f4c769bd6256d29b08ee0` at step `150000`; missing artifacts are fetched through the MLflow artifact cache).
- The MNIST dataset in the local torchvision cache (to compute the evaluation-only class means).

Generated outputs:
- `cache/plots/adversarial_class_mean_grid_<date>.pdf` and `.png`
- `cache/plots/adversarial_class_mean_metrics_<date>.pdf` and `.png`

### 2.11 Reference Quality for Class-Mean and Class-Boundary Attacks

The notebook `visualizations/images/plot_confidence_perturbation_attacks.ipynb` compares four ways of obtaining the class reference used by the Distance-to-Class-Mean (DCM) and Distance-to-Class-Boundary (DCB) membership inference attacks on the canonical `GROK_MNIST_MSE_MLP` model:

1. **Train-lkg:** the attacker knows all training samples and estimates the reference from them.
2. **Test-est:** only test samples are available; the class mean is estimated from the test split and the boundary reference is the KDE mode of the test-split class-0 margins.
3. **Adversarial example:** one synthetic class representative per class, generated from the model with the MSE objective plus a prior towards the global pixel mean.
4. **Perturbed test sample:** the first test image predicted as class $c$, moved towards higher confidence with an $L_\infty$-bounded PGD attack ($\varepsilon=0.05$, 100 steps).

The DCM attack scores penultimate-feature distances to the reference mean. The DCB attack uses the class-0 pairwise margin: $d_{0c}=(f_0-f_c)/\lVert w_0-w_c\rVert$ for $c\ge2$, and the $0$–$1$ margin for actual classes 0 and 1, matching the canary-selection implementation. Boundary references are KDE modes of $|d_{0c}|$.

> [!IMPORTANT]
> The checked-in `configs/GROK_MNIST_MSE_MLP.json` is a template with seed 0. The actual run uses `seed=350876`, `data.seed=604157`, and `mask.seed=604157`. The notebook overrides these seeds so that the train-lkg reference and the membership labels refer to the model's real 1,000-sample training subset. Using the template seeds evaluates on a disjoint subset and collapses the train-lkg attack to chance.

The notebook reports the average feature- and pixel-space distance to the train class mean for train and test samples, plus pooled and per-class AUC and TPR@1%FPR for all reference variants. Train-lkg is the known-training-data scenario; the test-est, adversarial-example, and perturbed-test references do not use training data. The perturbed-test anchor is excluded from the test evaluation so that the reference is never scored against itself.

Required inputs:
- `configs/GROK_MNIST_MSE_MLP.json`
- `cache/runs/2143636bf7d1447e9fa8ef6f107c5d03/checkpoints/150000/model.pth`
- The MNIST dataset in the local torchvision cache.

Generated outputs:
- `cache/plots/reference_quality_attacks_<date>.pdf` and `.png`
- `cache/plots/reference_quality_distance_<date>.pdf` and `.png`
- `cache/plots/reference_quality_perturbations_<date>.pdf` and `.png`

