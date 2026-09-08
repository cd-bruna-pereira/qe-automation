#!/usr/bin/env python

"""Tests for experimental ASE v4 plugin system."""

import os
import re
import textwrap
from pathlib import Path

import pytest

import ase.test.ase4.sample_plugins as sample_plugins
from ase import Atoms
from ase._4.plugins import PluginCollection
from ase._4.plugins import plugins as auto_plugins
from ase._4.plugins.plugin import CalculatorPlugin
from ase.visualize.viewers import AbstractViewer


@pytest.fixture
def sample_calc_plugins():
    """Collection of two calculator plugins"""

    return [sample_plugins.emt_plugin, sample_plugins.lj_plugin]


@pytest.fixture
def sample_mixed_plugins():
    """Collection of two calculators, one ioformat and two visualizers"""
    return [
        sample_plugins.emt_plugin,
        sample_plugins.lj_plugin,
        sample_plugins.extxyz_plugin,
        sample_plugins.avogadro_plugin,
        sample_plugins.ase_gui_plugin,
    ]


@pytest.fixture
def two_instances_plugins():
    another_emt_plugin = CalculatorPlugin(
        name='emt',
        citation='',
        implementation='ase.calculators.emt.EMT',
    )

    return [
        sample_plugins.emt_plugin,
        sample_plugins.lj_plugin,
        another_emt_plugin,
    ]


@pytest.fixture
def atoms() -> Atoms:
    import ase.build

    return ase.build.bulk('Au', cubic=True)


@pytest.fixture(
    scope='module',
    params=[('POSCAR', 'vasp'), ('dftb_in.hsd', 'dftb'), ('tmp.xyz', 'extxyz')],
)
def sample_file_and_format(request) -> tuple[Path, str]:
    filename, format_name = request.param
    path = Path(__file__).parent / f'data/{filename}'
    return path, format_name


@pytest.fixture(scope='module')
def cat_plugin():
    from ase_entry_demo.plugin import Plugin

    class CatPlugin(Plugin): ...

    return CatPlugin(name='Felix', citation={'Fur et al. (1996)'})


def test_build_from_entry_points():
    """Check that plugins were automatically loaded"""
    assert len(auto_plugins) > 0
    assert auto_plugins == auto_plugins
    assert PluginCollection.from_entry_points() == auto_plugins


def test_display(sample_calc_plugins):
    """Check that plugin listing looks correct"""
    assert re.match(
        textwrap.dedent(
            """\
        emt \\s+ \\(ase.test.ase4.sample_plugins \\d\\.\\d+\\.\\w+\\)
        lj  \\s+ \\(ase.test.ase4.sample_plugins \\d\\.\\d+\\.\\w+\\)"""
        ),
        PluginCollection(sample_calc_plugins).display(),
    )


def test_cache_invalidation(sample_calc_plugins):
    plugins = PluginCollection(sample_calc_plugins)

    initial_calculators = plugins.calculators
    initial_by_package = plugins.by_package

    # Change the underlying data
    plugins.plugins = frozenset(sample_calc_plugins[:-1])
    new_calculators = plugins.calculators
    new_by_package = plugins.by_package

    assert len(initial_calculators) == 2
    assert len(new_calculators) == 1

    assert len(initial_by_package['ase']) != len(new_by_package['ase'])


def test_missing_writer():
    from .sample_plugins import castep_castep_plugin

    assert castep_castep_plugin.can_read
    assert not castep_castep_plugin.can_write


def test_viewer_plugin(sample_mixed_plugins, atoms):
    plugins = PluginCollection(sample_mixed_plugins)

    assert len(plugins) > 2
    assert len(plugins.viewers) == 2
    for viewer in plugins.viewers:
        assert isinstance(viewer.implementation, AbstractViewer)

        # If factoring this out to another test, consider formally skipping
        # with pytest.skip when DISPLAY unavailable.
        if viewer.name == 'ASE GUI' and os.environ.get('DISPLAY'):
            # Shortcut to the .view() method of implementation
            ase_gui_view = viewer(atoms, repeat=(1, 1, 1))
            ase_gui_view.terminate()
