import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.datasets import create_masking, generate_datasets, mask_dataset
from privacy_and_grokking.logger import get_logger
from privacy_and_grokking.models import create_model
from privacy_and_grokking.path_keeper import get_path_keeper
from privacy_and_grokking.utils import get_device

logger = get_logger()


def get_correct_class_probabilities_and_logits(model, dataset):
    dataloader = DataLoader(dataset, batch_size=250)
    device = get_device()

    correct_probs = []
    correct_logits = []
    ce_losses = []
    mse_losses = []
    correctness_list = []

    ce_criterion = nn.CrossEntropyLoss(reduction="none")
    mse_criterion = nn.MSELoss(reduction="none")

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)

            correct_prob = probs.gather(1, y.view(-1, 1))
            correct_logit = logits.gather(1, y.view(-1, 1))
            correct_probs.append(correct_prob)
            correct_logits.append(correct_logit)

            ce_loss = ce_criterion(logits, y)
            ce_losses.append(ce_loss)

            mse_loss = mse_criterion(logits, F.one_hot(y, num_classes=logits.size(1)).float())
            mse_losses.append(mse_loss.gather(1, y.view(-1, 1)))

            correctness = (logits.argmax(dim=1) == y).float()
            correctness_list.append(correctness)

    return (
        torch.cat(correct_probs, dim=0),
        torch.cat(correct_logits, dim=0),
        torch.cat(ce_losses, dim=0),
        torch.cat(mse_losses, dim=0),
        torch.cat(correctness_list, dim=0),
    )


def attack(cfg: TrainConfig):
    pk = get_path_keeper()
    device = get_device()

    train, test = generate_datasets(cfg.dataset)

    masking = create_masking(
        config=cfg.dataset_mask,
        num_samples=len(train),
        num_classes=train.num_classes,
    )
    train_subset = mask_dataset(masking, train, cfg.dataset_mask_idx)

    train_probabilities = []
    test_probabilities = []
    train_logits_list = []
    test_logits_list = []
    train_ce_losses_list = []
    test_ce_losses_list = []
    train_mse_losses_list = []
    test_mse_losses_list = []
    train_correct_list = []
    test_correct_list = []
    steps = pk.get_available_steps(cfg.full_name)
    for i in trange(len(steps), desc="Steps", leave=False):
        pk.set_params({"model": cfg.full_name, "step": steps[i]})
        model = create_model(
            name=cfg.model,
            input_dim=train.input_shape,
            num_classes=train.num_classes,
        )
        model.to(device)
        model.load_state_dict(torch.load(pk.MODEL_TORCH, weights_only=True, map_location=device))
        model.eval()

        train_probs, train_logits, train_ce_losses, train_mse_losses, train_correct = (
            get_correct_class_probabilities_and_logits(model, train_subset)
        )
        test_probs, test_logits, test_ce_losses, test_mse_losses, test_correct = (
            get_correct_class_probabilities_and_logits(model, test)
        )

        train_probabilities.append(train_probs.squeeze())
        test_probabilities.append(test_probs.squeeze())
        train_logits_list.append(train_logits.squeeze())
        test_logits_list.append(test_logits.squeeze())
        train_ce_losses_list.append(train_ce_losses.squeeze())
        test_ce_losses_list.append(test_ce_losses.squeeze())
        train_mse_losses_list.append(train_mse_losses.squeeze())
        test_mse_losses_list.append(test_mse_losses.squeeze())
        train_correct_list.append(train_correct.squeeze())
        test_correct_list.append(test_correct.squeeze())

    train_probs_over_time = torch.stack(train_probabilities, dim=0).detach().cpu()
    test_probs_over_time = torch.stack(test_probabilities, dim=0).detach().cpu()
    train_logits_over_time = torch.stack(train_logits_list, dim=0).detach().cpu()
    test_logits_over_time = torch.stack(test_logits_list, dim=0).detach().cpu()
    train_ce_losses_over_time = torch.stack(train_ce_losses_list, dim=0).detach().cpu()
    test_ce_losses_over_time = torch.stack(test_ce_losses_list, dim=0).detach().cpu()
    train_mse_losses_over_time = torch.stack(train_mse_losses_list, dim=0).detach().cpu()
    test_mse_losses_over_time = torch.stack(test_mse_losses_list, dim=0).detach().cpu()
    train_correct_over_time = torch.stack(train_correct_list, dim=0).detach().cpu()
    test_correct_over_time = torch.stack(test_correct_list, dim=0).detach().cpu()

    torch.save(
        {
            "train_probs": train_probs_over_time,
            "test_probs": test_probs_over_time,
            "train_logits": train_logits_over_time,
            "test_logits": test_logits_over_time,
            "train_ce_losses": train_ce_losses_over_time,
            "test_ce_losses": test_ce_losses_over_time,
            "train_mse_losses": train_mse_losses_over_time,
            "test_mse_losses": test_mse_losses_over_time,
            "train_correct": train_correct_over_time,
            "test_correct": test_correct_over_time,
            "steps": steps,
        },
        pk.ATTACK_FOLDER / "mia_simple.pt",
    )
