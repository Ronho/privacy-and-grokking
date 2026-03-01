import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.utils.data import DataLoader

from privacy_and_grokking.utils import get_device


def compute_mia_signals(
    model: nn.Module,
    dataset,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    dataloader = DataLoader(dataset, batch_size=250)
    device = get_device()

    correct_probs: list[torch.Tensor] = []
    correct_logits: list[torch.Tensor] = []
    ce_losses: list[torch.Tensor] = []
    mse_losses: list[torch.Tensor] = []
    correctness_list: list[torch.Tensor] = []

    ce_criterion = nn.CrossEntropyLoss(reduction="none")
    mse_criterion = nn.MSELoss(reduction="none")

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)

            correct_probs.append(
                probs.gather(1, y.view(-1, 1)),
            )
            correct_logits.append(
                logits.gather(1, y.view(-1, 1)),
            )
            ce_losses.append(ce_criterion(logits, y))
            mse_losses.append(
                mse_criterion(
                    logits,
                    F.one_hot(
                        y,
                        num_classes=logits.size(1),
                    ).float(),
                ).gather(1, y.view(-1, 1))
            )
            correctness_list.append(
                (logits.argmax(dim=1) == y).float(),
            )

    return (
        torch.cat(correct_probs, dim=0),
        torch.cat(correct_logits, dim=0),
        torch.cat(ce_losses, dim=0),
        torch.cat(mse_losses, dim=0),
        torch.cat(correctness_list, dim=0),
    )
