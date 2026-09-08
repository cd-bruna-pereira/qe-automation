from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from importlib import import_module
from types import ModuleType
from typing import (
    TYPE_CHECKING,
)

from typing_extensions import Self

from ase._4.plugins.plugin import Plugin
from ase.utils.plugins import ExternalViewer
from ase.visualize.viewers import AbstractViewer

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint


class ViewerPlugin(Plugin):
    """Container for a Viewer with access to citation and package info"""

    def __init__(
        self,
        name: str,
        *,
        citation: str,
        module: ModuleType | None = None,
        implementation: str | Callable[[], AbstractViewer],
    ) -> None:
        super().__init__(name=name, module=module, citation=citation)
        self._implementation = implementation

    @cached_property
    def implementation(self) -> AbstractViewer:
        implementation = self._resolve(self._implementation)
        assert isinstance(implementation, AbstractViewer)
        return implementation

    def __call__(self, atoms, *args, **kwargs):
        return self.implementation.view(atoms, *args, **kwargs)


class LegacyViewerPlugin(ViewerPlugin):
    """Wrapper for external viewer plugin using legacy format/entrypoint"""

    def __init__(
        self,
        name: str,
        *,
        citation: str,
        module: ModuleType | None = None,
        implementation: str | Callable[[], AbstractViewer],
    ) -> None:
        super().__init__(
            name=name,
            citation=citation,
            module=module,
            implementation=implementation,
        )

    @classmethod
    def from_externalviewer(
        cls,
        name: str,
        ext_viewer: ExternalViewer,
        *,
        citation: str = 'Imported from legacy ExternalViewer',
        module: ModuleType | None = None,
    ) -> Self:
        from ase.visualize.viewers import CLIViewer, PyViewer

        implementation: AbstractViewer
        if ext_viewer.cli:
            implementation = CLIViewer(name, ext_viewer.fmt, ext_viewer.argv)
        else:
            implementation = PyViewer(
                name, ext_viewer.desc, module_name=ext_viewer.module
            )

        return cls(
            name=name,
            citation=citation,
            implementation=(lambda: implementation),
        )

    @classmethod
    def from_legacy_entry_point(
        cls,
        entry_point: EntryPoint,
    ) -> Self:
        external_viewer = entry_point.load()
        if not isinstance(external_viewer, ExternalViewer):
            raise TypeError(
                'Wrong type for registering external IO formats '
                f'in format {entry_point.name}, expected '
                'ExternalViewer'
            )

        module = import_module(external_viewer.__module__)

        return cls.from_externalviewer(
            entry_point.name, external_viewer, module=module
        )
