from pydantic import BaseModel


class MetricsConfig(BaseModel):
    """Configuration for which evaluation metrics to compute during training.

    Controls logging frequency and toggles for individual metric groups.
    Metrics are split into lightweight (computed at ``log_frequency``) and
    heavy (computed at ``heavy_metrics_log_frequency``) categories.
    """

    # Logging frequency
    log_frequency: int = 1000
    heavy_metrics_log_frequency: int = 10000
    log_every_n_epochs: int | None = None
    heavy_log_every_n_epochs: int | None = None

    # Metric groups (lightweight — computed every log_frequency steps)
    loss_stats: bool = True
    accuracy: bool = True
    weight_norms: bool = True
    gradient_norms: bool = True
    optimizer_internals: bool = True
    rnc1: bool = True
    rnc1_train_mean: bool = True
    nhsic: bool = True
    nhsic_max_samples: int = 2048

    # Distribution comparison metrics (between train/test losses)
    distribution_overlap: bool = True
    distribution_overlap_adaptive: bool = True
    distribution_overlap_kde: bool = True
    soft_overlap: bool = True
    kl_divergence: bool = True
    kl_divergence_adaptive: bool = True
    kl_divergence_kde: bool = True
    js_distance: bool = True
    js_distance_adaptive: bool = True
    js_distance_kde: bool = True
    mmd: bool = True

    # Attack / membership inference metrics
    attack_true_class_prob: bool = True
    attack_true_class_logit: bool = True
    attack_ce_loss: bool = True
    attack_mse_loss: bool = True
    attack_correctness: bool = True
    attack_distance_to_class_mean: bool = True
    attack_margin_distance_lf: bool = True

    # Heavy metrics (only computed at heavy_metrics_log_frequency)
    curvature: bool = True
    merlin_morgan: bool = True
    neural_collapse: bool = True
    one_run_audit: bool = True

    @property
    def any_distribution_metric(self) -> bool:
        """Whether any distribution comparison metric is enabled."""
        return any(
            [
                self.distribution_overlap,
                self.distribution_overlap_adaptive,
                self.distribution_overlap_kde,
                self.soft_overlap,
                self.kl_divergence,
                self.kl_divergence_adaptive,
                self.kl_divergence_kde,
                self.js_distance,
                self.js_distance_adaptive,
                self.js_distance_kde,
                self.mmd,
            ]
        )

    @property
    def any_attack_metric(self) -> bool:
        """Whether any attack metric is enabled."""
        return any(
            [
                self.attack_true_class_prob,
                self.attack_true_class_logit,
                self.attack_ce_loss,
                self.attack_mse_loss,
                self.attack_correctness,
                self.attack_distance_to_class_mean,
                self.attack_margin_distance_lf,
                self.merlin_morgan,
            ]
        )
