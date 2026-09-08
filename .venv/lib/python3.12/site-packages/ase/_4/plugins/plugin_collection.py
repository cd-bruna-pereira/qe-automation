from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterator
from functools import cached_property
from importlib.metadata import entry_points
from types import ModuleType
from typing import Any, Callable, TypeVar, cast

from typing_extensions import Self

from .io import IOFormatPlugin
from .plugin import CalculatorPlugin, Plugin
from .viewer import ViewerPlugin

P = TypeVar('P', bound=Plugin, covariant=True)
_P = TypeVar('_P', bound=Plugin)


class PluginCollection(Collection[P]):
    """Collection of ASE plugins

    The quasi-global set of installed ASE plugins will be available as an
    instance of Plugins at ase.plugins.plugins

    Other instances of Plugins might be created as a useful subset (e.g. just
    instances of CalculatorPlugin) or for testing purposes.

    Plugins.plugin_set is a frozenset of Plugin; it cannot be mutated but can
    be replaced if you have a really good reason to do so.
    """

    _cached_properties = 'by_package', 'by_type'

    def __init__(self, plugins: Collection[P]) -> None:
        self.plugins = frozenset(plugins)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

        """Invalidate cached properties"""
        if name == 'plugins':
            for attr in self._cached_properties:
                self.__dict__.pop(attr, None)

    def __contains__(self, other: object) -> bool:
        return other in self.plugins

    def __iter__(self) -> Iterator[P]:
        return iter(self.plugins)

    def __len__(self) -> int:
        return len(self.plugins)

    def __hash__(self) -> int:
        return hash((type(self), self.plugins))

    def __eq__(self, other: Any) -> bool:
        return (type(other) is type(self)) and other.plugins == self.plugins

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.plugins)})'

    @staticmethod
    def _modulesetter(module: ModuleType) -> Callable[[_P], _P]:
        def set_module(plugin: _P) -> _P:
            if plugin.module is None:
                plugin.module = module
            return plugin

        return set_module

    @classmethod
    def from_entry_points(cls, group: str = 'ase.plugins') -> Self:
        """Instantiate collection from Plugins registered as entry points

        Note that if __ase_plugins__ is missing from a registered entry point,
        it will be skipped quietly.
        """
        group_entry_points = entry_points(group=group)

        modules = (e_p.load() for e_p in group_entry_points)
        plugin_set: set[P] = set()

        for module in modules:
            plugins: Collection[P] = getattr(module, '__ase_plugins__', set())
            plugin_set = plugin_set | set(
                map(cls._modulesetter(module), plugins)
            )

        return cls(plugin_set)

    @cached_property
    def by_package(self) -> defaultdict[str, PluginCollection[Plugin]]:
        plugins_by_package = defaultdict(set)
        for plugin in self:
            plugins_by_package[plugin.package].add(plugin)

        plugincollections_by_package: defaultdict[
            str, PluginCollection[Plugin]
        ] = defaultdict(lambda: type(self)(set()))
        plugincollections_by_package.update(
            {
                key: type(self)(value)
                for key, value in plugins_by_package.items()
            }
        )
        return plugincollections_by_package

    @cached_property
    def by_type(self) -> dict[type[Plugin], PluginCollection[Plugin]]:
        """Get a dict of PluginCollections grouped by Plugin type

        The CalculatorPlugin and IOFormatPlugin groups are treated specially,
        and will include plugins that are subclasses. Any remaining plugin
        types will be listed separately.
        """
        plugins_by_type: defaultdict[type[Plugin], set[Plugin]] = defaultdict(
            set
        )

        for plugin in self.plugins:
            if isinstance(plugin, CalculatorPlugin):
                plugins_by_type[CalculatorPlugin].add(plugin)
            elif isinstance(plugin, IOFormatPlugin):
                plugins_by_type[IOFormatPlugin].add(plugin)
            else:
                plugins_by_type[type(plugin)].add(plugin)

        plugincollections_by_type: dict[
            type[Plugin], PluginCollection[Plugin]
        ] = {
            # From Python 3.13 should be able to use type(self)[Plugin]
            # (So far mypy doesn't like it...)
            key: PluginCollection(plugin_set)
            for key, plugin_set in plugins_by_type.items()
        }

        # CalculatorPlugin and IOFormatPlugin groups should always be present
        # but may be empty collections.
        plugincollections_by_type.setdefault(
            CalculatorPlugin, type(self)(set())
        )
        plugincollections_by_type.setdefault(IOFormatPlugin, type(self)(set()))

        return plugincollections_by_type

    @property
    def calculators(self) -> PluginCollection[CalculatorPlugin]:
        """Get a PluginCollection containing only CalculatorPlugins"""
        return cast(
            PluginCollection[CalculatorPlugin], self.by_type[CalculatorPlugin]
        )

    @property
    def io_formats(self) -> PluginCollection[IOFormatPlugin]:
        """Get a PluginCollection containing only IOFormatPlugins"""
        return cast(
            PluginCollection[IOFormatPlugin], self.by_type[IOFormatPlugin]
        )

    @property
    def viewers(self) -> PluginCollection[ViewerPlugin]:
        """Get a PluginCollection containing only ViewerPlugins"""
        return cast(PluginCollection[ViewerPlugin], self.by_type[ViewerPlugin])

    def display(self) -> str:
        return '\n'.join(sorted(plugin.display() for plugin in self))
