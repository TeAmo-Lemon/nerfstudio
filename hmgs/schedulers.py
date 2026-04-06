from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import numpy as np
from torch.optim import Optimizer, lr_scheduler

from hmgs.config_base import InstantiateConfig

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


@dataclass
class SchedulerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Scheduler)


class Scheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        raise NotImplementedError


@dataclass
class ExponentialDecaySchedulerConfig(SchedulerConfig):
    _target: Type = field(default_factory=lambda: ExponentialDecayScheduler)
    lr_pre_warmup: float = 1e-8
    lr_final: Optional[float] = None
    warmup_steps: int = 0
    max_steps: int = 100000
    ramp: Literal["linear", "cosine"] = "cosine"


class ExponentialDecayScheduler(Scheduler):
    config: ExponentialDecaySchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        lr_final = lr_init if self.config.lr_final is None else self.config.lr_final

        def func(step: int) -> float:
            if step < self.config.warmup_steps:
                if self.config.ramp == "cosine":
                    lr = self.config.lr_pre_warmup + (lr_init - self.config.lr_pre_warmup) * np.sin(
                        0.5 * np.pi * np.clip(step / self.config.warmup_steps, 0, 1)
                    )
                else:
                    lr = self.config.lr_pre_warmup + (lr_init - self.config.lr_pre_warmup) * step / self.config.warmup_steps
            else:
                t = np.clip(
                    (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps), 0, 1
                )
                lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return float(lr / lr_init)

        return lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
