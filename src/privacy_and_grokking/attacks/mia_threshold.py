import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from tqdm import trange
from ..config import TrainConfig
from ..datasets import get_dataset
from ..models import create_model
from ..path_keeper import get_path_keeper
from ..utils import get_device


def get_correct_class_probabilities_and_logits(model, dataset):
    dataloader = DataLoader(dataset, batch_size=250)
    
    correct_probs = []
    correct_logits = []
    ce_losses = []
    mse_losses = []
    
    ce_criterion = nn.CrossEntropyLoss(reduction='none')
    mse_criterion = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for x, y in dataloader:
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
    
    return (torch.cat(correct_probs, dim=0), 
            torch.cat(correct_logits, dim=0),
            torch.cat(ce_losses, dim=0),
            torch.cat(mse_losses, dim=0))

def attack(cfg: TrainConfig):
    train, _, test, input_dim, num_classes, _ = get_dataset(
        name=cfg.dataset.name,
        train_ratio=cfg.dataset.train_ratio,
        train_size=cfg.dataset.train_size,
        canary=None,
    )

    train_size = min(1000, len(train))
    test_size = min(1000, len(test))
    train_subset = Subset(train, list(range(train_size)))
    test_subset = Subset(test, list(range(test_size)))

    pk = get_path_keeper()
    train_probabilities = []
    test_probabilities = []
    train_logits_list = []
    test_logits_list = []
    train_ce_losses_list = []
    test_ce_losses_list = []
    train_mse_losses_list = []
    test_mse_losses_list = []
    STEP_SIZE = 1_000
    steps = list(range(0, cfg.optimization_steps+1, STEP_SIZE))
    for i in trange(len(steps), desc="Steps", leave=False):
        pk.set_params({"model": cfg.name, "step": steps[i]})
        model = create_model(
            name=cfg.model,
            input_dim=input_dim,
            num_classes=num_classes,
        )
        model.load_state_dict(torch.load(pk.MODEL_TORCH, weights_only=True, map_location=get_device()))
        model.eval()

        train_probs, train_logits, train_ce_losses, train_mse_losses = get_correct_class_probabilities_and_logits(model, train_subset)
        test_probs, test_logits, test_ce_losses, test_mse_losses = get_correct_class_probabilities_and_logits(model, test_subset)
        train_probabilities.append(train_probs.squeeze())
        test_probabilities.append(test_probs.squeeze())
        train_logits_list.append(train_logits.squeeze())
        test_logits_list.append(test_logits.squeeze())
        train_ce_losses_list.append(train_ce_losses.squeeze())
        test_ce_losses_list.append(test_ce_losses.squeeze())
        train_mse_losses_list.append(train_mse_losses.squeeze())
        test_mse_losses_list.append(test_mse_losses.squeeze())

    train_probs_over_time = torch.stack(train_probabilities, dim=0).detach()
    test_probs_over_time = torch.stack(test_probabilities, dim=0).detach()
    train_logits_over_time = torch.stack(train_logits_list, dim=0).detach()
    test_logits_over_time = torch.stack(test_logits_list, dim=0).detach()
    train_ce_losses_over_time = torch.stack(train_ce_losses_list, dim=0).detach()
    test_ce_losses_over_time = torch.stack(test_ce_losses_list, dim=0).detach()
    train_mse_losses_over_time = torch.stack(train_mse_losses_list, dim=0).detach()
    test_mse_losses_over_time = torch.stack(test_mse_losses_list, dim=0).detach()

    torch.save({
        "train_probs": train_probs_over_time,
        "test_probs": test_probs_over_time,
        "train_logits": train_logits_over_time,
        "test_logits": test_logits_over_time,
        "train_ce_losses": train_ce_losses_over_time,
        "test_ce_losses": test_ce_losses_over_time,
        "train_mse_losses": train_mse_losses_over_time,
        "test_mse_losses": test_mse_losses_over_time,
        "steps": steps,
    }, pk.ATTACK_FOLDER / "mia_threshold.pt")
