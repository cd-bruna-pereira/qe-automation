"""Legacy ExternalIOFormatPlugin are imported to this namespace

From there they are exposed to the new plugin system
"""

from importlib.metadata import entry_points

from .io import LegacyIOFormatPlugin

legacy_entry_points = entry_points(group='ase.ioformats')

__ase_plugins__ = {
    LegacyIOFormatPlugin.from_legacy_entry_point(entry_point)
    for entry_point in legacy_entry_points
}
