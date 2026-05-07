"""Power decay one cycle LR scheduler class."""

import math
from typing import Any

import torch
from torch.optim import Optimizer


class ExamplePowerDecayOneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    """LR scheduler that extends OneCycleLR with power decay.

    The number of steps/epochs and parameters other than power should be
    chosen to match the desired OneCycleLR behaviour.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float | list[float],
        power: float,
        one_cycle_total_steps: int | None = None,
        one_cycle_epochs: int | None = None,
        one_cycle_steps_per_epoch: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the power decay one cycle LR scheduler.

        Args:
            optimizer: Optimizer instance.
            power: Power of the exponential decay.
            max_lr: Upper learning rate boundaries in the cycle
                for each parameter group in one cycle LR.
            one_cycle_total_steps: Total number of steps in the
                one cycle LR portion. Defaults to None.
            one_cycle_epochs: Number of epochs in the one cycle LR
                portion. Defaults to None.
            one_cycle_steps_per_epoch: Steps per epoch in the one cycle
                LR portion. Defaults to None.
            **kwargs: Any other keyword-arguments to be passed to the OneCycleLR class.
        """
        self._power = power
        super().__init__(
            optimizer,
            max_lr=max_lr,
            total_steps=one_cycle_total_steps,
            epochs=one_cycle_epochs,
            steps_per_epoch=one_cycle_steps_per_epoch,
            **kwargs,
        )

    def _annealing_exp(self, start: float, end: float, pct: float) -> float:
        """Exponential anneal from `start` to `end` as pct goes from 0.0 to 1.0.

        Args:
            start: Learning rate at the start.
            end: Learning rate at the end.
            pct: Current step as the proportion between `start` and `end`. It is
                between 0 and 1.

        Returns:
            Learning rate.
        """
        beta = -math.pi * (end - start) / (end + start) * math.pow(2, self._power - 1) / self._power
        alpha = beta / math.pow(2, self._power) + math.log((end + start) / 2)
        return math.exp(alpha - beta * math.pow(pct, self._power))

    def get_lr(self) -> list[float]:
        """Compute learning rate.

        Raises:
            NotImplementedError: If there are three schedule phases.

        Returns:
            Learning rate.
        """
        if len(self._schedule_phases) != 2:
            raise NotImplementedError("Three phase is not implemented")

        phase = self._schedule_phases[1]
        start_step = self._schedule_phases[0]["end_step"]
        end_step = self._schedule_phases[1]["end_step"]

        exponential_start = (start_step + end_step) // 2

        lrs: list[float] = []
        step_num = self.last_epoch
        if step_num > exponential_start:
            for group in self.optimizer.param_groups:
                pct = (step_num - start_step) / (end_step - start_step)
                computed_lr = self._annealing_exp(
                    group[phase["start_lr"]], group[phase["end_lr"]], pct
                )
                lrs.append(computed_lr)

            return lrs
        return super().get_lr()
