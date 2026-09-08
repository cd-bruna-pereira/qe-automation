from __future__ import annotations

import sys
from functools import cached_property
from typing import TYPE_CHECKING

from ase._4.plugins.io import (
    IOFormatPlugin,
    LegacyIOFormatPlugin,
)
from ase._4.plugins.plugin import CalculatorPlugin
from ase._4.plugins.viewer import ViewerPlugin
from ase.io.formats import IOFormat
from ase.visualize.viewers import VIEWERS

if TYPE_CHECKING:
    from ase.calculators.lj import LennardJones
    from ase.calculators.vasp import Vasp


# Plugins should generally set their module to the place they are defined;
# this is optional and will be done automatically for entrypoint imports
THIS_MODULE = sys.modules[__name__]

# Calculator plugin using string to defer import
emt_plugin = CalculatorPlugin(
    name='emt',
    citation='Big Prof',
    module=THIS_MODULE,
    implementation='ase.calculators.emt.EMT',
)


# Calculator plugin using function to defer import
def get_lj_calc() -> type[LennardJones]:
    from ase.calculators.lj import LennardJones

    return LennardJones


lj_plugin = CalculatorPlugin(
    name='lj',
    citation='Other Big Prof',
    module=THIS_MODULE,
    implementation=get_lj_calc,
)


def get_ase_io_format(name: str) -> IOFormat:
    from ase.io.formats import ioformats

    return ioformats[name]


# Calculator plugin using a custom class (Not generally needed!)
class VaspCalculatorPlugin(CalculatorPlugin):
    def __init__(self, citation: str):
        super().__init__(
            name='vasp',
            implementation='',  # To be overriden
            citation=citation,
            module=THIS_MODULE,
        )

    @cached_property
    def implementation(self) -> type[Vasp]:
        from ase.calculators.vasp import Vasp

        # This is some pedantry to inform mypy, we don't expect real-world
        # plugins to do this nonsense. Just return the thing!
        calc_class: type[Vasp] = Vasp

        return calc_class


vasp_plugin = VaspCalculatorPlugin(
    citation=(
        'ASE developer community, based on modules by '
        'Jussi Enkovaara and John Kitchin'
    )
)

# IOFormat plugin created using __init__
poscar_plugin = IOFormatPlugin(
    name='vasp',
    citation=vasp_plugin.citation,
    # no module, lets pretend this is an interactive shell
    description='VASP POSCAR/CONTCAR',
    code='1F',
    extensions=['poscar'],
    globs=['*POSCAR*', '*CONTCAR*', '*CENTCAR*'],
    reader='ase.io.vasp.read_vasp',
    writer='ase.io.vasp.write_vasp',
)

# IOFormat plugin converted from existing ASE setup
extxyz_plugin = LegacyIOFormatPlugin.from_legacy_io_format(
    get_ase_io_format('extxyz'),
    citation='James Kermode',
    module=THIS_MODULE,
)

# A read-only format from existing ASE setup
castep_castep_plugin = LegacyIOFormatPlugin.from_legacy_io_format(
    get_ase_io_format('castep-castep'),
    citation='',
    module=THIS_MODULE,
)


# Plugin creation wrapped by a function
def create_dftb_input_plugin() -> LegacyIOFormatPlugin:
    return LegacyIOFormatPlugin.from_legacy_io_format(
        io_format=get_ase_io_format('dftb'),
        citation='',
        module=THIS_MODULE,
    )


# Viewer plugins; wrap viewers from ase.visualize.viewers
avogadro_plugin = ViewerPlugin(
    name='Cloned avogadro viewer',
    citation='N/A: Mock Viewer plugin',
    module=THIS_MODULE,
    implementation=(lambda: VIEWERS['avogadro']),
)

ase_gui_plugin = ViewerPlugin(
    name='ASE GUI',
    citation='ASE developers',
    # no module, lets pretend this is an interactive shell
    implementation=(lambda: VIEWERS['ase']),
)


__ase_plugins__ = {
    ase_gui_plugin,
    avogadro_plugin,
    castep_castep_plugin,
    create_dftb_input_plugin(),
    emt_plugin,
    extxyz_plugin,
    lj_plugin,
    poscar_plugin,
    vasp_plugin,
}
