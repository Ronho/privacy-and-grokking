from collections.abc import Callable
from functools import partial

from matplotlib import pyplot as plt

from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.accuracy_over_steps import accuracy_over_steps
from privacy_and_grokking.visualize.visualizations.class_activation import class_activation
from privacy_and_grokking.visualize.visualizations.class_distribution import class_distribution
from privacy_and_grokking.visualize.visualizations.curvature_over_steps import curvature_over_steps
from privacy_and_grokking.visualize.visualizations.loss_components_over_steps import (
    loss_components_over_steps,
)
from privacy_and_grokking.visualize.visualizations.loss_over_steps import loss_over_steps
from privacy_and_grokking.visualize.visualizations.mia_auc_over_steps import mia_auc_over_steps
from privacy_and_grokking.visualize.visualizations.mia_tpr_at_fpr_over_steps import (
    mia_tpr_at_fpr_over_steps,
)
from privacy_and_grokking.visualize.visualizations.norms_over_steps import (
    gradient_norms_over_steps,
    weight_norms_over_steps,
)
from privacy_and_grokking.visualize.visualizations.optimizer_internals import optimizer_internals
from privacy_and_grokking.visualize.visualizations.rdm import rdm
from privacy_and_grokking.visualize.visualizations.training_trajectory import training_trajectory
from privacy_and_grokking.visualize.visualizations.tsne import tsne

SINGLE_AXIS_VISUALIZATIONS: dict[str, Callable[[plt.Axes, DataHandler], None]] = {
    "accuracy_over_steps": accuracy_over_steps,
    "class_distribution": class_distribution,
    "curvature_over_steps": curvature_over_steps,
    "gradient_norms_over_steps": gradient_norms_over_steps,
    "loss_over_steps": loss_over_steps,
    "loss_components_over_steps": loss_components_over_steps,
    "mia_auc_over_steps": mia_auc_over_steps,
    "mia_tpr_at_fpr_over_steps_1": partial(mia_tpr_at_fpr_over_steps, fpr_pct=1),
    "mia_tpr_at_fpr_over_steps_5": partial(mia_tpr_at_fpr_over_steps, fpr_pct=5),
    "mia_tpr_at_fpr_over_steps_10": partial(mia_tpr_at_fpr_over_steps, fpr_pct=10),
    "training_trajectory": training_trajectory,
    "weight_norms_over_steps": weight_norms_over_steps,
}

# Multi-axes visualizations expand into one axes per layer / state-key at runtime.
# Signature: (ax: plt.Axes, dh: DataHandler, layer: str) -> None
#         or (ax: plt.Axes, dh: DataHandler, state_key: str) -> None
MULTI_AXES_VISUALIZATIONS: dict[str, Callable] = {
    "class_layer_activation_grid": class_activation,
    "optimizer_internals_over_steps": optimizer_internals,
    "rdm_per_layer": rdm,
    "tsne_per_layer": tsne,
}

SINGLE_VIZ_NAMES: list[str] = list(SINGLE_AXIS_VISUALIZATIONS) + list(MULTI_AXES_VISUALIZATIONS)
MULTI_VIZ_NAMES: list[str] = SINGLE_VIZ_NAMES
