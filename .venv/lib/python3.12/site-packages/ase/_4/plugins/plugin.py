from __future__ import annotations

import importlib
from abc import ABCMeta
from collections.abc import Callable
from functools import cached_property, singledispatchmethod
from importlib.metadata import metadata
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from ase.calculators.calculator import BaseCalculator

if TYPE_CHECKING:
    from importlib.metadata import PackageMetadata


_I = TypeVar('_I')


class MissingFunction(ValueError):
    """Reader/writer function not found in relevant ase io module"""


class Plugin(metaclass=ABCMeta):
    def __init__(
        self, name: str, *, citation: str, module: ModuleType | None = None
    ) -> None:
        self.name = name
        self.citation = citation
        self.module = module

    def display(self) -> str:
        module_name = (
            self.module.__name__
            if isinstance(self.module, ModuleType)
            else self.module
        )
        return f'{self.name:<20} ({module_name} {self.version})'

    @cached_property
    def package(self) -> str:
        if self.module:
            return self.module.__name__.split('.')[0]
        return ''

    @cached_property
    def package_data(self) -> PackageMetadata | None:
        if not self.package:
            return None
        return metadata(self.package)

    @property
    def version(self) -> str:
        if self.package_data is None:
            return ''
        return self.package_data.get('Version', '')

    @singledispatchmethod
    @staticmethod
    def _resolve(implementation: str) -> Any:
        modulename, classname = implementation.rsplit('.', maxsplit=1)
        module = importlib.import_module(modulename)
        return getattr(module, classname)

    @_resolve.register(Callable)  # type: ignore[call-overload,misc]
    @staticmethod
    def _(implementation: Callable[[], Any]) -> Any:
        return implementation()


class CalculatorPlugin(Plugin):
    """Container for a Calculator with metadata about where it came from"""

    def __init__(
        self,
        name: str,
        *,
        citation: str = '',
        module: ModuleType | None = None,
        implementation: str | Callable[[], type[BaseCalculator]],
    ) -> None:
        super().__init__(name=name, module=module, citation=citation)
        self._implementation = implementation

    @cached_property
    def implementation(self) -> type[BaseCalculator]:
        implementation = self._resolve(self._implementation)
        assert isinstance(implementation, type)
        assert issubclass(implementation, BaseCalculator)
        return implementation

    def __call__(self) -> type[BaseCalculator]:
        return self.implementation
