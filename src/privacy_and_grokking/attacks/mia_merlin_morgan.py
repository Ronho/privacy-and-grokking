import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Subset
from tqdm import trange

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.datasets import create_masking, generate_datasets, mask_dataset
from privacy_and_grokking.models import create_model
from privacy_and_grokking.path_keeper import get_path_keeper
from privacy_and_grokking.utils import get_device

NOISY_SAMPLES = 100
NOISE_SCALE = 0.01


def get_merlin_morgan_stats(model, dataset, device, num_classes):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    ce_criterion = nn.CrossEntropyLoss(reduction="none")
    mse_criterion = nn.MSELoss(reduction="none")

    # Store results
    ce_losses = []
    mse_losses = []
    ce_votes = []
    mse_votes = []

    model.eval()

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Original pass
        with torch.no_grad():
            output = model(imgs)

            # Calculate CE Loss
            ce_loss = ce_criterion(output, labels)

            # Calculate MSE Loss
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
            mse_loss = mse_criterion(output, labels_one_hot).sum(dim=1)

            ce_losses.append(ce_loss.cpu())
            mse_losses.append(mse_loss.cpu())

            # Noisy pass
            current_ce_votes = torch.zeros(imgs.size(0), device=device)
            current_mse_votes = torch.zeros(imgs.size(0), device=device)

            # We process sample by sample to match the notebook logic of adding noise
            # Or we can vectorize if memory allows.
            # Notebook:
            # for img, loss, label in zip(imgs, losses, labels):
            #    noise = torch.randn((NOISY_SAMPLES, *img.shape)) ...
            #    noisy_imgs = img.unsqueeze(0) + noise

            for i in range(imgs.size(0)):
                img = imgs[i]
                label = labels[i]
                label_one_hot = labels_one_hot[i]

                base_ce_loss = ce_loss[i]
                base_mse_loss = mse_loss[i]

                # Create noisy samples
                noise = torch.randn((NOISY_SAMPLES, *img.shape), device=device) * NOISE_SCALE
                noisy_imgs = img.unsqueeze(0) + noise

                noisy_output = model(noisy_imgs)

                # CE noisy losses
                noisy_ce_losses = ce_criterion(noisy_output, label.repeat(NOISY_SAMPLES))
                current_ce_votes[i] = (noisy_ce_losses > base_ce_loss).float().mean()

                # MSE noisy losses
                noisy_mse_losses = mse_criterion(
                    noisy_output, label_one_hot.repeat(NOISY_SAMPLES, 1)
                ).sum(dim=1)
                current_mse_votes[i] = (noisy_mse_losses > base_mse_loss).float().mean()

            ce_votes.append(current_ce_votes.cpu())
            mse_votes.append(current_mse_votes.cpu())

    return (
        torch.cat(ce_losses, dim=0),
        torch.cat(mse_losses, dim=0),
        torch.cat(ce_votes, dim=0),
        torch.cat(mse_votes, dim=0),
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
    train_subset_full = mask_dataset(masking, train, cfg.dataset_mask_idx)
    input_dim = train.input_shape
    num_classes = train.num_classes

    # Use a subset of data as per other attacks/notebook (SAMPLE_SIZE = 1000)
    SAMPLE_SIZE = 1000
    train_size = min(SAMPLE_SIZE, len(train_subset_full))
    test_size = min(SAMPLE_SIZE, len(test))

    # Ensure consistent subsets if possible, but random_split is used in notebook.
    # Here we stick to the pattern in mia_threshold which uses Subset(train, list(range(train_size)))
    train_subset = Subset(train_subset_full, list(range(train_size)))
    test_subset = Subset(test, list(range(test_size)))

    train_ce_losses_list = []
    test_ce_losses_list = []
    train_mse_losses_list = []
    test_mse_losses_list = []

    train_ce_votes_list = []
    test_ce_votes_list = []
    train_mse_votes_list = []
    test_mse_votes_list = []

    steps = pk.get_available_steps(cfg.full_name)

    for i in trange(len(steps), desc="Steps", leave=False):
        pk.set_params({"model": cfg.full_name, "step": steps[i]})

        model = create_model(
            name=cfg.model,
            input_dim=input_dim,
            num_classes=num_classes,
        )
        model.load_state_dict(torch.load(pk.MODEL_TORCH, weights_only=True, map_location=device))
        model.to(device)
        model.eval()

        # Train Stats
        tr_ce_loss, tr_mse_loss, tr_ce_vote, tr_mse_vote = get_merlin_morgan_stats(
            model, train_subset, device, num_classes
        )

        # Test Stats
        te_ce_loss, te_mse_loss, te_ce_vote, te_mse_vote = get_merlin_morgan_stats(
            model, test_subset, device, num_classes
        )

        train_ce_losses_list.append(tr_ce_loss)
        train_mse_losses_list.append(tr_mse_loss)
        train_ce_votes_list.append(tr_ce_vote)
        train_mse_votes_list.append(tr_mse_vote)

        test_ce_losses_list.append(te_ce_loss)
        test_mse_losses_list.append(te_mse_loss)
        test_ce_votes_list.append(te_ce_vote)
        test_mse_votes_list.append(te_mse_vote)

    # Stack results
    results = {
        "train_ce_losses": torch.stack(train_ce_losses_list, dim=0),
        "train_mse_losses": torch.stack(train_mse_losses_list, dim=0),
        "train_ce_votes": torch.stack(train_ce_votes_list, dim=0),
        "train_mse_votes": torch.stack(train_mse_votes_list, dim=0),
        "test_ce_losses": torch.stack(test_ce_losses_list, dim=0),
        "test_mse_losses": torch.stack(test_mse_losses_list, dim=0),
        "test_ce_votes": torch.stack(test_ce_votes_list, dim=0),
        "test_mse_votes": torch.stack(test_mse_votes_list, dim=0),
        "steps": steps,
    }

    torch.save(results, pk.ATTACK_FOLDER / "mia_merlin_morgan.pt")
