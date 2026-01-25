import torch
import torch.nn.functional as F
from tqdm import trange

from ..config import TrainConfig
from ..datasets import get_dataset
from ..logger import get_logger
from ..models import create_model
from ..path_keeper import get_path_keeper
from ..utils import get_device


def compute_likelihood(model, x, y, device="cpu"):
    model.eval()
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        true_class_probs = probs.gather(1, y.view(-1, 1))

    return true_class_probs


def rmia_offline(
    target_model: torch.nn.Module,
    reference_models: list[torch.nn.Module],
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    population_data: torch.Tensor,
    population_labels: torch.Tensor,
    gamma: float = 1.0,
    scaling_factor: float = 0.3,
    device="cpu",
) -> torch.Tensor:
    summed_likelihood_target = torch.zeros(
        (target_x.size(0), 1), dtype=torch.float32, device=device
    )

    for ref_model in reference_models:
        likelihood = compute_likelihood(ref_model, target_x, target_y, device=device)
        summed_likelihood_target += likelihood

    p_x_out = summed_likelihood_target / len(reference_models)
    p_x = (1 + scaling_factor) / 2 * p_x_out + (1 - scaling_factor) / 2

    p_x_target = compute_likelihood(target_model, target_x, target_y, device=device)
    ratio_x = p_x_target / p_x

    p_z_out = torch.zeros((population_data.size(0), 1), dtype=torch.float32, device=device)
    for ref_model in reference_models:
        likelihood = compute_likelihood(
            ref_model, population_data, population_labels, device=device
        )
        p_z_out += likelihood
    p_z_out = p_z_out / len(reference_models)
    p_z = (1 + scaling_factor) / 2 * p_z_out + (1 - scaling_factor) / 2

    p_z_target = compute_likelihood(target_model, population_data, population_labels, device=device)
    ratio_z = p_z_target / p_z

    ratios = ratio_x / ratio_z.t()
    scores = (ratios > gamma).float().mean(dim=1)
    return scores


def reduce_data(dataset, max_samples=1000):
    reduced_x = []
    reduced_y = []
    for x, y in dataset:
        if len(reduced_x) >= max_samples:
            break
        reduced_x.append(x)
        reduced_y.append(y)
    return torch.stack(reduced_x), torch.tensor(reduced_y)


def attack(cfg: TrainConfig):
    logger = get_logger()
    logger.info("Starting RMIA attack.", extra={"model": cfg.name})
    device = get_device()
    pk = get_path_keeper()

    train, _, test, input_dim, num_classes, _ = get_dataset(
        name=cfg.dataset.name,
        train_ratio=cfg.dataset.train_ratio,
        train_size=cfg.dataset.train_size,
        canary=None,
    )

    in_target_x, in_target_y = reduce_data(train, max_samples=1000)
    test_x, test_y = reduce_data(test, max_samples=11000)
    out_target_x, out_target_y = test_x[:1000], test_y[:1000]
    population_x, population_y = test_x[1000:], test_y[1000:]

    # TODO: Make somehow adjustable.
    REFERENCE_MODEL_STEP = 250_000
    REFERENCE_MODELS = {
        "MNIST_MLP_": "MNIST_MLP_NOGROK_VAL_NOCAN",
        "CIFAR10_MLP_": "CIFAR10_MLP_NOGROK_VAL_NOCAN",
    }
    model = None
    for model, model_name in REFERENCE_MODELS.items():
        if model in cfg.name:
            model = model_name
            break
    if model is None:
        raise ValueError(f"No reference model defined for dataset {cfg.dataset.name}")

    pk.set_params({"model": model, "step": REFERENCE_MODEL_STEP})
    reference_model = create_model(name=cfg.model, input_dim=input_dim, num_classes=num_classes)
    reference_model.load_state_dict(
        torch.load(pk.MODEL_TORCH, weights_only=True, map_location=device)
    )
    reference_model.to(device)

    train_scores = []
    test_scores = []
    STEP_SIZE = 1_000
    SCALING_FACTOR = 0.3
    GAMMA = 1.0
    steps = list(range(0, cfg.optimization_steps + 1, STEP_SIZE))
    for i in trange(len(steps)):
        pk.set_params({"model": cfg.name, "step": steps[i]})
        target_model = create_model(
            name=cfg.model,
            input_dim=input_dim,
            num_classes=num_classes,
        )
        target_model.load_state_dict(
            torch.load(pk.MODEL_TORCH, weights_only=True, map_location=device)
        )
        target_model.to(device)
        target_model.eval()

        train_scores_per_sample = rmia_offline(
            target_model=target_model,
            reference_models=[reference_model],
            target_x=in_target_x,
            target_y=in_target_y,
            population_data=population_x,
            population_labels=population_y,
            gamma=GAMMA,
            scaling_factor=SCALING_FACTOR,
            device=device,
        )

        test_scores_per_sample = rmia_offline(
            target_model=target_model,
            reference_models=[reference_model],
            target_x=out_target_x,
            target_y=out_target_y,
            population_data=population_x,
            population_labels=population_y,
            gamma=GAMMA,
            scaling_factor=SCALING_FACTOR,
            device=device,
        )

        train_scores.append(train_scores_per_sample.squeeze().cpu())
        test_scores.append(test_scores_per_sample.squeeze().cpu())

    train_scores_over_time = torch.stack(train_scores, dim=0)
    test_scores_over_time = torch.stack(test_scores, dim=0)

    torch.save(
        {
            "train_scores": train_scores_over_time,
            "test_scores": test_scores_over_time,
            "steps": steps,
        },
        pk.ATTACK_FOLDER / "mia_rmia.pt",
    )
