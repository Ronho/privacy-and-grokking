"""Canary Class Assignment

Contains logic for assigning new labels to canary samples.

(Ideal) Criteria:
- Each canary sample should be assigned a label that is not the same as its original label.
- The distribution of new labels should be equal to the original distribution of labels in the canary set.
- The assignment should be random, without predictable patterns (e.g., not just shifting all labels by one).
- The function should be deterministic given a specific random seed.
- The implementation should be efficient and scalable to larger canary sets.
"""

import torch

from privacy_and_grokking.logger import get_logger

logger = get_logger()


def derange_balanced_indices(canary_lookup: dict[int, list[int]], seed: int) -> torch.Tensor:
    """Returns a deranged list of canary labels matching the original canary indices.

    Why this complicated?
    Criteria:
    - Keep the same amount of entries per class as in canary_lookup.
    - No entry should map to its own class.
    - Randomized assignment. Simple rules like "shift by one" are not acceptable since they lead to predictable patterns.

    Note: This function may fail in some edge cases where derangement is not possible due to previous assignments.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    canary_lbls = torch.Tensor(list(canary_lookup.keys())).to(torch.int64)
    canary_class_amt = torch.Tensor([len(vals) for vals in canary_lookup.values()]).to(torch.int64)
    original_canary_labels = torch.repeat_interleave(canary_lbls, canary_class_amt)
    container_canary_labels = original_canary_labels.clone()
    assigned_canary_labels = torch.zeros(original_canary_labels.size(), dtype=torch.int64)

    for i in range(len(original_canary_labels)):
        available_label_indices = (
            (
                (container_canary_labels != original_canary_labels[i])
                & (container_canary_labels != -1)
            )
            .nonzero()
            .squeeze()
        )
        if available_label_indices.numel() == 0:
            raise ValueError("No available labels to assign for derangement. Try again.")
        elif available_label_indices.numel() == 1:
            choosen_idx = available_label_indices.item()
        else:
            choosen_idx = available_label_indices[
                torch.randint(0, available_label_indices.numel(), (1,), generator=rng).item()
            ]
        assigned_canary_labels[i] = container_canary_labels[choosen_idx]
        container_canary_labels[choosen_idx] = -1  # Mark this label as used
    return assigned_canary_labels


def alternative_derange_balanced_indices(
    canary_lookup: dict[int, list[int]], seed: int, retries: int = 100
) -> torch.Tensor:
    """Uses permutations."""
    canary_lbls = torch.Tensor(list(canary_lookup.keys())).to(torch.int64)
    canary_class_amt = torch.Tensor([len(vals) for vals in canary_lookup.values()]).to(torch.int64)
    original_canary_labels = torch.repeat_interleave(canary_lbls, canary_class_amt)
    rng = torch.Generator()
    rng.manual_seed(seed)
    for i in range(retries):  # Arbitrary number of retries
        assigned_canary_labels = original_canary_labels[
            torch.randperm(original_canary_labels.size(0), generator=rng)
        ]
        if torch.all(assigned_canary_labels != original_canary_labels):
            break
    if i == retries - 1:
        logger.warning(
            "Derangement failed after maximum retries. Returning last attempt, which may contain matches."
        )
    return assigned_canary_labels


def random_derange_indices(canary_lookup: dict[int, list[int]], seed: int) -> torch.Tensor:
    """Uses a random shift."""
    canary_lbls = torch.Tensor(list(canary_lookup.keys())).to(torch.int64)
    canary_class_amt = torch.Tensor([len(vals) for vals in canary_lookup.values()]).to(torch.int64)
    original_canary_labels = torch.repeat_interleave(canary_lbls, canary_class_amt)
    rng = torch.Generator()
    rng.manual_seed(seed)
    shift = torch.randint(1, len(original_canary_labels), (1,), generator=rng).item()
    assigned_canary_labels = (original_canary_labels + shift) % len(canary_lbls)
    return assigned_canary_labels
