from __future__ import annotations

import sys

from ase._4.plugins.io import LegacyIOFormatPlugin
from ase._4.plugins.viewer import ViewerPlugin
from ase.io.formats import ioformats
from ase.visualize.viewers import CLI_VIEWERS, PY_VIEWERS

_this_module = sys.modules[__name__]

wrapped_ioformats = {
    # Wrap all _internal_ formats from ase.io.formats;
    # skip formats from legacy entry-point, they are handled in legacy_plugins
    LegacyIOFormatPlugin.from_legacy_io_format(ioformat, module=_this_module)
    for ioformat in ioformats.values()
    if ioformat.module_name[:7] == 'ase.io.'
}

wrapped_cli_viewers = {
    ViewerPlugin(
        name=key,
        module=_this_module,
        citation='imported from ase.visualize.viewers.CLI_VIEWERS',
        implementation=(lambda: value),
    )
    for key, value in CLI_VIEWERS.items()
}

wrapped_py_viewers = {
    ViewerPlugin(
        name=key,
        module=_this_module,
        citation='imported from ase.visualize.viewers.PY_VIEWERS',
        implementation=(lambda: value),
    )
    for key, value in PY_VIEWERS.items()
}

__ase_plugins__ = wrapped_ioformats | wrapped_cli_viewers | wrapped_py_viewers
