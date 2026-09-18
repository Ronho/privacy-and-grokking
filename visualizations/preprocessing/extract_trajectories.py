import io
import json
import numpy as np
import torch
import typer
import urllib

from concurrent.futures import ThreadPoolExecutor, as_completed
from mlflow import MlflowClient
from pathlib import Path
from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.utils import Logger
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
from typing import Annotated

DEFAULT_TRACKING_URI = "http://localhost:5051"
_tracking_uri = DEFAULT_TRACKING_URI

BASE_DIR = Path(__file__).parent.parent.parent
CACHE_DIR = f"{BASE_DIR}/cache"
# Runs
RUN_DIR = f"{CACHE_DIR}/runs/{{run_id}}"
CHECKPOINTS_DIR = f"{RUN_DIR}/checkpoints/"
CHECKPOINT_DIR = f"{CHECKPOINTS_DIR}/{{step}}"
MODEL_FILE = f"{CHECKPOINT_DIR}/model.pth"
INFO_FILE = f"{CHECKPOINTS_DIR}/info.json"
CONFIG_FILE = f"{RUN_DIR}/training_config.json"
# Trajectories
TRAJECTORY_DIR = f"{RUN_DIR}/trajectories"
PROJECTION_FILE = f"{TRAJECTORY_DIR}/projection.json"
GRID_FILE = f"{TRAJECTORY_DIR}/grid.json"

app = typer.Typer()

def fetch_run(run_id: str):
    client = MlflowClient(_tracking_uri)
    try:
        run = client.get_run(run_id)
        run_name = run.data.tags.get("mlflow.runName", run.info.run_name or run_id)
        experiment_id = run.info.experiment_id
        print(f"Found Run: '{run_name}' (ID: {run_id}, Status: {run.info.status}, Experiment: {experiment_id})")
        return run_name, experiment_id
    except Exception as e:
        raise Exception(f"Could not fetch run '{run_id}' from MLflow ({e}).") from e

def get_available_checkpoints(run_id: str) -> list[int]:
    info_file = Path(INFO_FILE.format(run_id=run_id))
    info_file.parent.mkdir(parents=True, exist_ok=True)
    if info_file.exists():
        return sorted(json.loads(info_file.read_bytes())["steps"])
    
    url = f"{_tracking_uri}/api/2.0/mlflow/artifacts/list?run_id={run_id}&path=checkpoints"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        files = data.get("files", [])
        steps = []
        for f in files:
            p = f.get("path", "")
            step_str = p.split("/")[-1]
            if step_str.isdigit():
                steps.append(int(step_str))
        presorted = sorted(steps)
        with open(info_file, "w") as f:
            json.dump({"steps": presorted}, f)
        return presorted
    except Exception as e:
        raise Exception(f"Could not fetch checkpoint dir for run '{run_id}' from MLFlow ({e}).") from e

def get_checkpoint_weight(run_id: str, step: int):
    model_file = Path(MODEL_FILE.format(run_id=run_id, step=step))
    model_file.parent.mkdir(parents=True, exist_ok=True)
    if model_file.exists():
        return torch.load(model_file, map_location="cpu", weights_only=True)

    url = f"{_tracking_uri}/get-artifact?path=checkpoints/{step}/model.pth&run_uuid={run_id}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()

        with open(model_file, "wb") as f:
            f.write(data)

        buf = io.BytesIO(data)
        return torch.load(buf, map_location="cpu", weights_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch checkpoint at step {step} for run {run_id} from {url}: {e}"
        ) from e

def get_training_config(run_id: str) -> TrainConfig:
    config_file = Path(CONFIG_FILE.format(run_id=run_id))
    config_file.parent.mkdir(parents=True, exist_ok=True)
    if config_file.exists():
        config = TrainConfig.model_validate(json.loads(config_file.read_bytes()))
        return config

    url = f"{_tracking_uri}/get-artifact?path=training_config.json&run_uuid={run_id}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        with open(config_file, "w") as f:
            json.dump(data, f)
        return TrainConfig.model_validate(data)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch training config for run {run_id} from {url}: {e}") from e

def get_checkpoints(run_id: str, steps: list[int], max_workers: int):
    checkpoints = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_checkpoint_weight, run_id, step): step for step in steps}
        for future in tqdm(as_completed(futures), total=len(steps), desc="Loading checkpoints"):
            step = futures[future]
            checkpoints[step] = future.result()
    return checkpoints

def collect_run(run_id: str, max_workers: int) -> dict[int, torch.Tensor]:
    steps = get_available_checkpoints(run_id)
    return get_checkpoints(run_id=run_id, steps=steps, max_workers=max_workers)

def flatten_weights(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    tensors = []
    for key in sorted(weights.keys()):
        val = weights[key]
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            tensors.append(val.detach().cpu().reshape(-1).float())
    return torch.cat(tensors)

def compute_pca(checkpoints: dict[int, torch.Tensor], n_components: int, identifier: str):
    projection_file = Path(PROJECTION_FILE.format(run_id=identifier))
    projection_file.parent.mkdir(parents=True, exist_ok=True)
    if projection_file.exists():
        data = json.loads(projection_file.read_bytes())
        if data.get("n_components", 0) >= n_components:
            print(f"Loading cached PCA projection from {projection_file}...")
            coords_list = []
            for pt in data.get("trajectories", {}).get("pca_coords", []):
                coords_list.append(pt[:n_components])
            coords = np.array(coords_list)
            evr = np.array(data["explained_variance_ratio"])[:n_components]
            components = np.array(data["components"])[:n_components]
            mean = np.array(data["mean"])
            return coords, evr, components, mean

    steps: list[int] = []
    flat_weights: list[torch.Tensor] = []
    for step in sorted(checkpoints.keys()):
        flat_w = flatten_weights(checkpoints[step])
        flat_weights.append(flat_w)
        steps.append(step)

    weight_matrix = torch.stack(flat_weights)
    print(f"Weight matrix shape: {weight_matrix.shape} (checkpoints x parameters)")

    k = min(n_components, weight_matrix.shape[0], weight_matrix.shape[1])
    pca = PCA(n_components=k)
    coords = pca.fit_transform(weight_matrix.numpy())
    mean = pca.mean_

    trajectories: dict[int, dict] = {"steps": [], "pca_coords": []}
    for step, coord in zip(steps, coords):
        trajectories["steps"].append(step)
        trajectories["pca_coords"].append(coord.tolist() if hasattr(coord, "tolist") else list(coord))

    payload = {
        "trajectory_id": identifier,
        "n_components": int(k),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pca.components_.tolist(),
        "mean": mean.tolist(),
        "trajectories": trajectories,
    }
    projection_file.write_text(json.dumps(payload))
    print(f"Saved PCA projection to {projection_file}")

    return coords, pca.explained_variance_ratio_, pca.components_, mean

def safe_collate(batch):
    if isinstance(batch[0], tuple) and len(batch[0]) == 2:
        xs, ys = zip(*batch)
        if any(isinstance(y, torch.Tensor) for y in ys) and any(not isinstance(y, torch.Tensor) for y in ys):
            ys = [y if isinstance(y, torch.Tensor) else torch.tensor(y) for y in ys]
            batch = list(zip(xs, ys))
    return default_collate(batch)

def unflatten_weights(flat_weights: torch.Tensor | np.ndarray, model: torch.nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    if isinstance(flat_weights, np.ndarray):
        flat_weights = torch.from_numpy(flat_weights)
    flat_weights = flat_weights.to(device=device, dtype=torch.float32)

    state_dict = model.state_dict()
    new_state_dict = {}
    offset = 0
    for key in sorted(state_dict.keys()):
        val = state_dict[key]
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            numel = val.numel()
            param_tensor = flat_weights[offset : offset + numel].reshape(val.shape)
            new_state_dict[key] = param_tensor
            offset += numel
        else:
            new_state_dict[key] = val.to(device)
    return new_state_dict

@torch.no_grad()
def evaluate_loss_tensors(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: torch.nn.Module,
    batch_size: int = 2000,
) -> float:
    model.eval()
    total_loss = 0.0
    n = len(y)
    for i in range(0, n, batch_size):
        bx = x[i : i + batch_size]
        by = y[i : i + batch_size]
        preds = model(bx)
        loss = loss_fn(preds, by)
        total_loss += float(loss.item()) * len(by)
    return total_loss / max(1, n)

def compute_sharpness_power_iteration(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: torch.nn.Module,
    max_iter: int = 6,
    use_sdpa_math: bool = False,
) -> float:
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        ctx = sdpa_kernel([SDPBackend.MATH])
    except ImportError:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        out = model(images)
        loss = loss_fn(out, labels)
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, create_graph=True)
        grad_flat = torch.cat([g.reshape(-1) for g in grads])

        torch.manual_seed(42)
        v = torch.randn_like(grad_flat)
        v_norm = torch.norm(v)
        if v_norm == 0 or not torch.isfinite(v_norm):
            return 0.0
        v = v / v_norm

        top_eigenvalue = 0.0
        for _ in range(max_iter):
            gv = torch.sum(grad_flat * v)
            hv_tuple = torch.autograd.grad(gv, params, retain_graph=True)
            hv = torch.cat([h.reshape(-1) for h in hv_tuple])
            hv_norm = torch.norm(hv)
            if hv_norm == 0 or not torch.isfinite(hv_norm):
                return 0.0
            top_eigenvalue = float(torch.sum(v * hv).item())
            v = hv / hv_norm

        del grads, grad_flat, loss, out

    return abs(top_eigenvalue)

def grid_search(
    coords: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataset_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    identifier: str,
    resolution: int = 25,
    margin: float = 0.1,
    device: torch.device = torch.device("cpu"),
    compute_hessian: bool = False,
    is_vit: bool = False,
):
    grid_file = Path(GRID_FILE.format(run_id=identifier))
    grid_file.parent.mkdir(parents=True, exist_ok=True)
    if grid_file.exists():
        data = json.loads(grid_file.read_bytes())
        if data.get("resolution") == resolution and data.get("margin") == margin:
            if not compute_hessian or (compute_hessian and "sharpness" in data):
                print(f"Loading cached grid with losses from {grid_file}...")
                return data

    x = coords[:, 0]
    y = coords[:, 1]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    x_span = (x_max - x_min) if x_max > x_min else 1.0
    y_span = (y_max - y_min) if y_max > y_min else 1.0

    x_min_grid = x_min - margin * x_span
    x_max_grid = x_max + margin * x_span
    y_min_grid = y_min - margin * y_span
    y_max_grid = y_max + margin * y_span

    x_coords = np.linspace(x_min_grid, x_max_grid, resolution)
    y_coords = np.linspace(y_min_grid, y_max_grid, resolution)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    v0 = components[0]
    v1 = components[1]

    print(f"Evaluating loss and weight norm on {len(grid_points)} grid points...")
    weight_norms: list[float] = []
    sharpness_list: list[float] = []
    losses: dict[str, list[float]] = {name: [] for name in dataset_tensors.keys()}
    
    max_iter_grid = 4 if is_vit else 6

    for pt in tqdm(grid_points, desc="Grid evaluation"):
        gx, gy = pt[0], pt[1]
        w = mean + gx * v0 + gy * v1
        norm = float(np.linalg.norm(w))
        weight_norms.append(norm)

        state = unflatten_weights(w, model, device)
        model.load_state_dict(state)

        for name, (dx, dy) in dataset_tensors.items():
            loss = evaluate_loss_tensors(model, dx, dy, loss_fn, batch_size=2000)
            losses[name].append(loss)
            
        if compute_hessian and "train" in dataset_tensors:
            dx, dy = dataset_tensors["train"]
            # Use smaller batch for hessian if needed or just first 2000 samples
            bx = dx[:2000]
            by = dy[:2000]
            sh_val = compute_sharpness_power_iteration(model, bx, by, loss_fn, max_iter=max_iter_grid, use_sdpa_math=is_vit)
            sharpness_list.append(sh_val)

    grid_payload = {
        "trajectory_id": identifier,
        "resolution": resolution,
        "margin": margin,
        "bounds": {
            "pc1": {"min": x_min, "max": x_max},
            "pc2": {"min": y_min, "max": y_max},
        },
        "grid_bounds": {
            "pc1": {"min": float(x_min_grid), "max": float(x_max_grid)},
            "pc2": {"min": float(y_min_grid), "max": float(y_max_grid)},
        },
        "x_coords": x_coords.tolist(),
        "y_coords": y_coords.tolist(),
        "grid_points": grid_points.tolist(),
        "weight_norms": weight_norms,
        "losses": losses,
    }
    if compute_hessian:
        grid_payload["sharpness"] = sharpness_list
        
    grid_file.write_text(json.dumps(grid_payload))
    print(f"Saved grid to {grid_file} ({resolution}x{resolution} = {len(grid_points)} points)")
    return grid_payload

@app.command()
def main(
    run_id: Annotated[str, typer.Argument()],
    n_components: Annotated[int, typer.Option(help="Number of PCA components to compute")] = 10,
    max_workers: Annotated[int, typer.Option(help="Concurrent workers for loading checkpoints")] = 10,
    grid_resolution: Annotated[int, typer.Option(help="Grid resolution (points per axis)")] = 50,
    grid_margin: Annotated[float, typer.Option(help="Grid margin beyond model bounds")] = 0.1,
    compute_hessian: Annotated[bool, typer.Option(help="Compute sharpness using power iteration")] = False,
    tracking_uri: Annotated[str, typer.Option(help="Tracking URI for MLflow")] = DEFAULT_TRACKING_URI,
):
    global _tracking_uri
    _tracking_uri = tracking_uri
    Logger().setup()
    logger = Logger.get()
    logger.info(f"Run ID: {run_id}")
    training_config = get_training_config(run_id=run_id)
    checkpoints = collect_run(run_id=run_id, max_workers=max_workers)

    logger.info(f"Computing PCA (n_components={n_components})...")
    coords, evr, components, mean = compute_pca(checkpoints, n_components=n_components, identifier=run_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Instantiating dataset and model from config...")
    data_container = training_config.data()
    model = training_config.model(
        input_dim=data_container.input_shape,
        num_classes=data_container.num_classes,
    ).to(device)

    loss_fn = training_config.loss(num_classes=data_container.num_classes)

    norm_mean = (
        torch.tensor(data_container.normalization.mean, device=device, dtype=torch.float32).view(-1, 1, 1)
        if data_container.normalization
        else None
    )
    norm_std = (
        torch.tensor(data_container.normalization.std, device=device, dtype=torch.float32).view(-1, 1, 1)
        if data_container.normalization
        else None
    )

    logger.info("Loading pre-normalized datasets onto device...")
    dataset_tensors = {}
    target_datasets = [("train", data_container.train), ("test", data_container.test)]
    if data_container.train_canary:
        target_datasets.append(("train_canary", data_container.train_canary))
    if data_container.test_canary:
        target_datasets.append(("test_canary", data_container.test_canary))

    for name, ds in target_datasets:
        loader = DataLoader(ds, batch_size=2000, shuffle=False, collate_fn=safe_collate)
        xs, ys = [], []
        for bx, by in loader:
            xs.append(bx)
            ys.append(by)
        x_all = torch.cat(xs).to(device=device, dtype=torch.float32)
        y_all = torch.cat(ys).to(device=device)
        if norm_mean is not None and norm_std is not None:
            x_all = (x_all - norm_mean) / norm_std
        dataset_tensors[name] = (x_all, y_all)

    logger.info("Evaluating exact checkpoint losses on train set...")
    x_train, y_train = dataset_tensors["train"]
    checkpoint_losses = []
    checkpoint_weight_norms = []
    for step in sorted(checkpoints.keys()):
        ckpt = checkpoints[step]
        model.load_state_dict(ckpt)
        loss_val = evaluate_loss_tensors(model, x_train, y_train, loss_fn, batch_size=2000)
        flat_w = flatten_weights(ckpt)
        norm_val = float(torch.norm(flat_w).item())
        checkpoint_losses.append(loss_val)
        checkpoint_weight_norms.append(norm_val)

    proj_path = Path(PROJECTION_FILE.format(run_id=run_id))
    if proj_path.exists():
        proj_data = json.loads(proj_path.read_text())
        proj_data["trajectories"]["losses"] = checkpoint_losses
        proj_data["trajectories"]["weight_norms"] = checkpoint_weight_norms
        proj_path.write_text(json.dumps(proj_data))
        logger.info("Updated projection.json with exact checkpoint losses.")

    is_vit = "vit" in training_config.model.name.lower()

    grid_search(
        coords=coords,
        components=components,
        mean=mean,
        model=model,
        loss_fn=loss_fn,
        dataset_tensors=dataset_tensors,
        identifier=run_id,
        resolution=grid_resolution,
        margin=grid_margin,
        device=device,
        compute_hessian=compute_hessian,
        is_vit=is_vit,
    )
    logger.info(f"Finished grid evaluation for run {run_id}.")


if __name__ == "__main__":
    app()