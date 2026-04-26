"""
Hyperparameter search config generator for regularizers.

Generates config files in configs/search/ by taking a base config and
producing all combinations of regularizer parameters. The generated
configs can then be picked up by the `command` CLI endpoint.

To adjust what gets searched, edit the SEARCH_SPACE below.
"""

from __future__ import annotations

import itertools
import json
import shutil
from pathlib import Path

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.utils import Logger

CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"
SEARCH_DIR = CONFIG_DIR / "search"

# ---------------------------------------------------------------------------
# Base config to start from (must exist in configs/)
# ---------------------------------------------------------------------------
BASE_CONFIG = "MSE_SGD_DEFAULT.json"

# ---------------------------------------------------------------------------
# Search space definition
#
# Each entry is a regularizer template: a dict that will become the
# "regularizer" field in the config. Use lists for values you want to
# sweep over; scalar values are kept fixed.
#
# The generator takes the cartesian product of all list-valued fields
# within each regularizer template.
# ---------------------------------------------------------------------------
SEARCH_SPACE: list[dict] = [
    # ===================================================================
    # Per-Sample Distance (PSD) — salt & pepper
    # ===================================================================
    {
        "name": "per_sample_distance",
        "weight": [0.1, 1.0, 10.0],
        "metric": ["l1", "l2", "huber"],
        "validation_source": "noisy_self",
        "noise_type": "salt_and_pepper",
        "noise_fraction": [0.05, 0.1, 0.25],
        "num_noisy_samples": 3,
    },
    # Per-Sample Distance (PSD) — gaussian
    {
        "name": "per_sample_distance",
        "weight": [0.1, 1.0, 10.0],
        "metric": ["l1", "l2", "huber"],
        "validation_source": "noisy_self",
        "noise_type": "gaussian",
        "noise_std": [0.1, 0.3, 0.5],
        "num_noisy_samples": 3,
    },
    # ===================================================================
    # MMD — salt & pepper
    # ===================================================================
    {
        "name": "mmd",
        "weight": [0.1, 1.0, 10.0, 20.0],
        "bandwidth": [0.1, 0.5, 1.0],
        "validation_source": "noisy_self",
        "noise_type": "salt_and_pepper",
        "noise_fraction": [0.05, 0.1, 0.25],
        "num_noisy_samples": 3,
    },
    # MMD — gaussian
    {
        "name": "mmd",
        "weight": [0.1, 1.0, 10.0, 20.0],
        "bandwidth": [0.1, 0.5, 1.0],
        "validation_source": "noisy_self",
        "noise_type": "gaussian",
        "noise_std": [0.1, 0.3, 0.5],
        "num_noisy_samples": 3,
    },
    # ===================================================================
    # Overlap (fixed-bin histogram) — salt & pepper
    # ===================================================================
    {
        "name": "overlap",
        "weight": [0.1, 1.0, 2.0, 10.0],
        "n_bins": 50,
        "sigma": 0.05,
        "validation_source": "noisy_self",
        "noise_type": "salt_and_pepper",
        "noise_fraction": [0.05, 0.1, 0.25],
        "num_noisy_samples": 3,
    },
    # Overlap (fixed-bin histogram) — gaussian
    {
        "name": "overlap",
        "weight": [0.1, 1.0, 2.0, 10.0],
        "n_bins": 50,
        "sigma": 0.05,
        "validation_source": "noisy_self",
        "noise_type": "gaussian",
        "noise_std": [0.1, 0.3, 0.5],
        "num_noisy_samples": 3,
    },
    # ===================================================================
    # Overlap KDE — salt & pepper
    # ===================================================================
    {
        "name": "overlap_kde",
        "weight": [0.1, 1.0, 2.0, 10.0],
        "n_points": 200,
        "validation_source": "noisy_self",
        "noise_type": "salt_and_pepper",
        "noise_fraction": [0.05, 0.1, 0.25],
        "num_noisy_samples": 3,
    },
    # Overlap KDE — gaussian
    {
        "name": "overlap_kde",
        "weight": [0.1, 1.0, 2.0, 10.0],
        "n_points": 200,
        "validation_source": "noisy_self",
        "noise_type": "gaussian",
        "noise_std": [0.1, 0.3, 0.5],
        "num_noisy_samples": 3,
    },
]


def _expand_template(template: dict) -> list[dict]:
    """Expand a single regularizer template into all combinations.

    Fields with list values are swept; scalar fields are kept fixed.
    """
    sweep_keys = []
    sweep_values = []
    fixed = {}

    for key, value in template.items():
        if isinstance(value, list):
            sweep_keys.append(key)
            sweep_values.append(value)
        else:
            fixed[key] = value

    if not sweep_keys:
        return [dict(fixed)]

    configs = []
    for combo in itertools.product(*sweep_values):
        cfg = dict(fixed)
        for key, val in zip(sweep_keys, combo):
            cfg[key] = val
        configs.append(cfg)
    return configs


def _regularizer_label(reg: dict) -> str:
    """Create a short descriptive label from a regularizer config."""
    parts = [reg["name"], reg["noise_type"]]
    for key, value in reg.items():
        if key == "name":
            continue
        # Skip fields that are constant / already encoded in the label prefix
        if key in ("validation_source", "noise_type", "num_noisy_samples"):
            continue
        parts.append(f"{key}={value}")
    return "__".join(parts)


def generate_search_configs() -> list[Path]:
    """Generate all search config files. Returns list of written paths."""
    logger = Logger.get()

    base_path = CONFIG_DIR / BASE_CONFIG
    base = TrainConfig.model_validate_json(base_path.read_bytes())
    logger.info("Loaded base config.", base=BASE_CONFIG)

    # Clean and recreate search directory
    if SEARCH_DIR.exists():
        shutil.rmtree(SEARCH_DIR)
    SEARCH_DIR.mkdir(parents=True)

    written: list[Path] = []
    for template in SEARCH_SPACE:
        regularizer_configs = _expand_template(template)
        for reg_cfg in regularizer_configs:
            # Build the full config by overriding the regularizer
            full = base.model_copy(deep=True)
            full.regularizer = None  # clear first, then re-validate with new data
            config_dict = full.model_dump()
            config_dict["regularizer"] = reg_cfg

            # Validate through Pydantic
            validated = TrainConfig.model_validate(config_dict)

            # Write to file
            label = _regularizer_label(reg_cfg)
            filename = f"{base_path.stem}_{label}.json"
            out_path = SEARCH_DIR / filename
            out_path.write_text(
                json.dumps(validated.model_dump(), indent=4) + "\n",
                encoding="utf-8",
            )
            written.append(out_path)
            logger.info("Generated config.", file=filename)

    logger.info("Search config generation complete.", total=len(written), directory=str(SEARCH_DIR))
    return written
