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
# Each entry is a regularizer template using the new nested format:
#   { "weight": ..., "loss_reduction": ..., "regularizer": { "name": ..., "source": { ... } } }
#
# Use lists for values you want to sweep over; scalar values are kept fixed.
# The generator takes the cartesian product of all list-valued fields.
# ---------------------------------------------------------------------------
SEARCH_SPACE: list[dict] = [
    # ===================================================================
    # Per-Sample Distance (PSD) — salt & pepper
    # ===================================================================
    {
        "weight": [0.1, 1.0, 10.0],
        "loss_reduction": [None, "mean", "max"],
        "regularizer": {
            "name": "per_sample_distance",
            "metric": ["l1", "l2", "huber"],
            "source": {
                "name": "salt_and_pepper",
                "fraction": [0.05, 0.1, 0.25],
                "num_noisy_samples": 3,
            },
        },
    },
    # Per-Sample Distance (PSD) — gaussian
    {
        "weight": [0.1, 1.0, 10.0],
        "loss_reduction": [None, "mean", "max"],
        "regularizer": {
            "name": "per_sample_distance",
            "metric": ["l1", "l2", "huber"],
            "source": {
                "name": "gaussian",
                "std": [0.1, 0.3, 0.5],
                "num_noisy_samples": 3,
            },
        },
    },
    # ===================================================================
    # MMD — salt & pepper
    # ===================================================================
    {
        "weight": [0.1, 1.0, 10.0, 20.0],
        "loss_reduction": "mean",
        "regularizer": {
            "name": "mmd",
            "bandwidth": [0.1, 0.5, 1.0],
            "source": {
                "name": "salt_and_pepper",
                "fraction": [0.05, 0.1, 0.25],
                "num_noisy_samples": 3,
            },
        },
    },
    # MMD — gaussian
    {
        "weight": [0.1, 1.0, 10.0, 20.0],
        "loss_reduction": "mean",
        "regularizer": {
            "name": "mmd",
            "bandwidth": [0.1, 0.5, 1.0],
            "source": {
                "name": "gaussian",
                "std": [0.1, 0.3, 0.5],
                "num_noisy_samples": 3,
            },
        },
    },
    # ===================================================================
    # Overlap (fixed-bin histogram) — salt & pepper
    # ===================================================================
    {
        "weight": [0.1, 1.0, 2.0, 10.0],
        "loss_reduction": "mean",
        "regularizer": {
            "name": "overlap",
            "n_bins": 50,
            "sigma": 0.05,
            "source": {
                "name": "salt_and_pepper",
                "fraction": [0.05, 0.1, 0.25],
                "num_noisy_samples": 3,
            },
        },
    },
    # Overlap (fixed-bin histogram) — gaussian
    {
        "weight": [0.1, 1.0, 2.0, 10.0],
        "loss_reduction": "mean",
        "regularizer": {
            "name": "overlap",
            "n_bins": 50,
            "sigma": 0.05,
            "source": {
                "name": "gaussian",
                "std": [0.1, 0.3, 0.5],
                "num_noisy_samples": 3,
            },
        },
    },
    # ===================================================================
    # Overlap KDE — salt & pepper
    # ===================================================================
    {
        "weight": [0.1, 1.0, 2.0, 10.0],
        "loss_reduction": "mean",
        "regularizer": {
            "name": "overlap_kde",
            "n_points": 200,
            "source": {
                "name": "salt_and_pepper",
                "fraction": [0.05, 0.1, 0.25],
                "num_noisy_samples": 3,
            },
        },
    },
    # Overlap KDE — gaussian
    {
        "weight": [0.1, 1.0, 2.0, 10.0],
        "loss_reduction": "mean",
        "regularizer": {
            "name": "overlap_kde",
            "n_points": 200,
            "source": {
                "name": "gaussian",
                "std": [0.1, 0.3, 0.5],
                "num_noisy_samples": 3,
            },
        },
    },
]


def _flatten_sweep_keys(d: dict, prefix: str = "") -> tuple[dict, list[tuple[str, list]]]:
    """Recursively find list-valued fields in a nested dict.

    Returns (fixed_structure, sweep_items) where sweep_items is a list of
    (dotted_key_path, values_list) pairs.
    """
    fixed = {}
    sweeps: list[tuple[str, list]] = []
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            sub_fixed, sub_sweeps = _flatten_sweep_keys(value, full_key)
            fixed[key] = sub_fixed
            sweeps.extend(sub_sweeps)
        elif isinstance(value, list):
            sweeps.append((full_key, value))
        else:
            fixed[key] = value
    return fixed, sweeps


def _set_nested(d: dict, dotted_key: str, value) -> None:
    """Set a value in a nested dict using a dotted key path."""
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        d = d[part]
    d[parts[-1]] = value


def _deep_copy_dict(d: dict) -> dict:
    """Simple deep copy for nested dicts with primitive values."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        else:
            result[k] = v
    return result


def _expand_template(template: dict) -> list[dict]:
    """Expand a regularizer template into all combinations.

    Supports nested dicts — list values at any depth are swept.
    """
    fixed, sweeps = _flatten_sweep_keys(template)

    if not sweeps:
        return [dict(fixed)]

    sweep_keys = [s[0] for s in sweeps]
    sweep_values = [s[1] for s in sweeps]

    configs = []
    for combo in itertools.product(*sweep_values):
        cfg = _deep_copy_dict(fixed)
        for key, val in zip(sweep_keys, combo, strict=True):
            _set_nested(cfg, key, val)
        configs.append(cfg)
    return configs


def _regularizer_label(reg: dict) -> str:
    """Create a short descriptive label from a regularizer config."""
    inner = reg.get("regularizer", {})
    source = inner.get("source", {})
    parts = [inner.get("name", "unknown"), source.get("name", "unknown")]

    # Add weight
    if "weight" in reg:
        parts.append(f"weight={reg['weight']}")

    # Add inner regularizer params (skip name and source)
    for key, value in inner.items():
        if key in ("name", "source"):
            continue
        parts.append(f"{key}={value}")

    # Add source params (skip name and num_noisy_samples)
    for key, value in source.items():
        if key in ("name", "num_noisy_samples"):
            continue
        parts.append(f"{key}={value}")

    # Add loss_reduction if non-default
    if reg.get("loss_reduction") != "mean":
        parts.append(f"loss_reduction={reg.get('loss_reduction')}")

    return "__".join(str(p) for p in parts)


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
