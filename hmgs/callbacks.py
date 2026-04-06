from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from inspect import signature
from typing import Callable, Dict, List, Optional, Tuple

from torch.cuda.amp.grad_scaler import GradScaler


@dataclass
class TrainingCallbackAttributes:
    optimizers: Optional[object]
    grad_scaler: Optional[GradScaler]
    pipeline: Optional[object]
    trainer: Optional[object]


class TrainingCallbackLocation(Enum):
    BEFORE_TRAIN_ITERATION = auto()
    AFTER_TRAIN_ITERATION = auto()
    AFTER_TRAIN = auto()


class TrainingCallback:
    def __init__(
        self,
        where_to_run: List[TrainingCallbackLocation],
        func: Callable,
        update_every_num_iters: Optional[int] = None,
        iters: Optional[Tuple[int, ...]] = None,
        args: Optional[List] = None,
        kwargs: Optional[Dict] = None,
    ) -> None:
        assert "step" in signature(func).parameters
        self.where_to_run = where_to_run
        self.func = func
        self.update_every_num_iters = update_every_num_iters
        self.iters = iters
        self.args = args or []
        self.kwargs = kwargs or {}

    def run_callback(self, step: int) -> None:
        if self.update_every_num_iters is not None:
            if step % self.update_every_num_iters == 0:
                self.func(*self.args, **self.kwargs, step=step)
        elif self.iters is not None:
            if step in self.iters:
                self.func(*self.args, **self.kwargs, step=step)
        else:
            self.func(*self.args, **self.kwargs, step=step)

    def run_callback_at_location(self, step: int, location: TrainingCallbackLocation) -> None:
        if location in self.where_to_run:
            self.run_callback(step)
