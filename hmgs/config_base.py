from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type


class PrintableConfig:
    def __str__(self) -> str:
        parts = [f"{self.__class__.__name__}:"]
        for key, value in vars(self).items():
            parts.append(f"  {key}: {value}")
        return "\n".join(parts)


@dataclass
class InstantiateConfig(PrintableConfig):
    _target: Type

    def setup(self, **kwargs) -> Any:
        return self._target(self, **kwargs)
