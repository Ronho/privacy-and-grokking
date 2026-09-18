from typing import Literal

import torch

from privacy_and_grokking.optimizer.base import OptimizerConfig


class RMSpropConfig(OptimizerConfig):
    name: Literal["RMSprop"] = "RMSprop"

    lr: float = 0.01
    alpha: float = 0.99
    eps: float = 1e-08
    weight_decay: float = 0
    momentum: float = 0
    centered: bool = False

    def __call__(self, params) -> torch.optim.Optimizer:
        return torch.optim.RMSprop(
            params,
            **self.model_dump(exclude={"name"}),
        )

    def get_tracking_metrics(
        self,
        optimizer: torch.optim.Optimizer,
        pre_step_params: list[torch.Tensor],
        model_parameters: list[torch.nn.Parameter],
    ) -> dict[str, float]:
        lr = optimizer.param_groups[0]['lr']
        weight_decay = optimizer.param_groups[0].get('weight_decay', 0.0)

        total_update_sq = 0.0
        weight_reg_norm = 0.0
        effective_grad_sq = 0.0
        grad_sq = 0.0
        weight_sq = 0.0
        square_avg_sq = 0.0
        momentum_sq = 0.0

        for p_old, p_new in zip(pre_step_params, model_parameters):
            if p_new.grad is None:
                continue

            grad_sq += p_new.grad.pow(2).sum().item()
            weight_sq += p_new.pow(2).sum().item()

            weight_reg = weight_decay * p_old
            weight_reg_norm += weight_reg.pow(2).sum().item()

            step_diff = p_new.detach() - p_old
            total_update_sq += step_diff.pow(2).sum().item()

            effective_grad = step_diff / (-lr)
            effective_grad_sq += effective_grad.pow(2).sum().item()

            state = optimizer.state[p_new]
            if 'square_avg' in state:
                square_avg_sq += state['square_avg'].sum().item()
            if 'momentum_buffer' in state and state['momentum_buffer'] is not None:
                momentum_sq += state['momentum_buffer'].pow(2).sum().item()

        metrics = {
            "optim/global_weight_norm": (weight_sq ** 0.5),
            "optim/global_grad_norm": (grad_sq ** 0.5),
            "optim/global_weight_reg_norm": (weight_reg_norm ** 0.5),
            "optim/global_sqrt_square_avg_norm": (square_avg_sq ** 0.5),
            "optim/global_step_norm": (effective_grad_sq ** 0.5), # without learning rate
            "optim/global_update_norm": (total_update_sq ** 0.5), # with learning rate
        }

        if self.momentum > 0:
            metrics["optim/global_momentum_buffer_norm"] = momentum_sq ** 0.5
            
        return metrics
