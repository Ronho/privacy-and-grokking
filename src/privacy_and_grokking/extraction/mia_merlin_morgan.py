import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from privacy_and_grokking.utils import get_device

NOISY_SAMPLES = 100
NOISE_SCALE = 0.01


def compute_merlin_morgan_signals(
    model: nn.Module,
    dataset,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = get_device()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    ce_criterion = nn.CrossEntropyLoss(reduction="none")
    mse_criterion = nn.MSELoss(reduction="none")

    ce_votes: list[torch.Tensor] = []
    mse_votes: list[torch.Tensor] = []

    model.eval()

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            output = model(imgs)

            ce_loss = ce_criterion(output, labels)
            labels_one_hot = F.one_hot(
                labels, num_classes=num_classes,
            ).float()
            mse_loss = mse_criterion(output, labels_one_hot).sum(dim=1)

            batch_ce_votes = torch.zeros(
                imgs.size(0), device=device,
            )
            batch_mse_votes = torch.zeros(
                imgs.size(0), device=device,
            )

            for i in range(imgs.size(0)):
                img = imgs[i]
                label = labels[i]
                label_oh = labels_one_hot[i]

                noise = (
                    torch.randn(
                        (NOISY_SAMPLES, *img.shape), device=device,
                    )
                    * NOISE_SCALE
                )
                noisy_imgs = img.unsqueeze(0) + noise
                noisy_output = model(noisy_imgs)

                noisy_ce = ce_criterion(
                    noisy_output, label.repeat(NOISY_SAMPLES),
                )
                batch_ce_votes[i] = (
                    (noisy_ce > ce_loss[i]).float().mean()
                )

                noisy_mse = mse_criterion(
                    noisy_output,
                    label_oh.repeat(NOISY_SAMPLES, 1),
                ).sum(dim=1)
                batch_mse_votes[i] = (
                    (noisy_mse > mse_loss[i]).float().mean()
                )

            ce_votes.append(batch_ce_votes.cpu())
            mse_votes.append(batch_mse_votes.cpu())

    return (
        torch.cat(ce_votes, dim=0),
        torch.cat(mse_votes, dim=0),
    )
