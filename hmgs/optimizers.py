from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parameter import Parameter

from hmgs.config_base import PrintableConfig


@dataclass
class OptimizerConfig(PrintableConfig):
    _target: Type = torch.optim.Adam
    lr: float = 0.0005
    eps: float = 1e-8
    max_norm: Optional[float] = None

    def setup(self, params) -> torch.optim.Optimizer:
        kwargs = vars(self).copy()
        kwargs.pop("_target")
        kwargs.pop("max_norm")
        return self._target(params, **kwargs)


@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    _target: Type = torch.optim.Adam
    weight_decay: float = 0.0


class Optimizers:
    def __init__(self, config: Dict[str, Any], param_groups: Dict[str, List[Parameter]]) -> None:
        self.config = config
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.schedulers = {}
        self.parameters = {}
        for name, params in param_groups.items():
            if name not in config:
                raise RuntimeError(f"Missing optimizer config for param group: {name}")
            lr_init = config[name]["optimizer"].lr
            self.optimizers[name] = config[name]["optimizer"].setup(params=params)
            self.parameters[name] = params
            if config[name]["scheduler"] is not None:
                self.schedulers[name] = config[name]["scheduler"].setup().get_scheduler(
                    optimizer=self.optimizers[name], lr_init=lr_init
                )

    def zero_grad_some(self, param_groups: List[str]) -> None:
        for group in param_groups:
            self.optimizers[group].zero_grad()

    def optimizer_scaler_step_some(self, grad_scaler: GradScaler, param_groups: List[str]) -> None:
        for group in param_groups:
            optimizer = self.optimizers[group]
            max_norm = self.config[group]["optimizer"].max_norm
            if max_norm is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters[group], max_norm)
            if any(any(p.grad is not None for p in g["params"]) for g in optimizer.param_groups):
                grad_scaler.step(optimizer)

    def scheduler_step_all(self) -> None:
        for scheduler in self.schedulers.values():
            scheduler.step()

    def load_optimizers(self, loaded_state: Dict[str, Any]) -> None:
        for key, value in loaded_state.items():
            self.optimizers[key].load_state_dict(value)

    def load_schedulers(self, loaded_state: Dict[str, Any]) -> None:
        for key, value in loaded_state.items():
            self.schedulers[key].load_state_dict(value)
