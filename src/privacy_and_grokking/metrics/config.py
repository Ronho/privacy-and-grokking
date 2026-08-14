from pydantic import BaseModel


class MetricsConfig(BaseModel):
    """Configuration for which evaluation metrics to compute during training.

    Controls logging frequency and toggles for individual metric groups.
    Metrics are split into lightweight (computed at ``log_frequency``) and
    heavy (computed at ``heavy_metrics_log_frequency``) categories.
    """

    # Logging frequency
    log_frequency: int = 1000
    heavy_metrics_log_frequency: int = 1000
    log_every_n_epochs: int | None = 10
    heavy_log_every_n_epochs: int | None = 10

    # Metric groups (lightweight — computed every log_frequency steps)
    loss_stats: bool = True
    accuracy: bool = True
    weight_norms: bool = True
    gradient_norms: bool = True

    # Distribution comparison metrics (between train/test losses)
    distribution_overlap: bool = True
    mmd: bool = False

    # Attack / membership inference metrics
    attack_true_class_prob: bool = True
    attack_true_class_logit: bool = True
    attack_ce_loss: bool = True
    attack_mse_loss: bool = True
    attack_correctness: bool = True

    # Heavy metrics (only computed at heavy_metrics_log_frequency)
    neural_collapse: bool = True
    attack_distance_to_class_mean: bool = False
    attack_margin_distance_lf: bool = False
    curvature: bool = False

    @property
    def any_distribution_metric(self) -> bool:
        """Whether any distribution comparison metric is enabled."""
        return any(
            [
                self.distribution_overlap,
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
            ]
        )
