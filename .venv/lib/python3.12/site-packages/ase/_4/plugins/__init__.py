"""Experimental plugin system for ASE v4"""

from .plugin import Plugin
from .plugin_collection import PluginCollection

plugins: PluginCollection[Plugin] = PluginCollection.from_entry_points()

__all__ = ['Plugin', 'PluginCollection', 'plugins']
