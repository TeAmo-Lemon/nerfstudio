from __future__ import annotations

from contextlib import contextmanager


class SimpleConsole:
    def print(self, *args, **kwargs) -> None:
        del kwargs
        print(*args)

    def log(self, *args, **kwargs) -> None:
        del kwargs
        print(*args)

    @contextmanager
    def status(self, *args, **kwargs):
        del args, kwargs
        yield


CONSOLE = SimpleConsole()
