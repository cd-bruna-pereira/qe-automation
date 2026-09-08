"""Conditional execution of sphinx-gallery scripts via companion probe files.

Overview
--------
The purpose of this module is for sphinx gallery to optionally generate certain
tutorials. This means the docs can be built without all the DFT codes installed.
If sphinx-gallery one day makes this possible via configuration (e.g. "optional"
tutorials or better code injection around examples), then this module can be
deleted.

Probe file convention
---------------------
For every gallery script ``plot_something.py``, an optional companion file
``plot_something_prerequisites.py`` may exist in the same directory.  This file
contains code that raises the ``SphinxPrerequisitesError`` defined here if the
requirements for the execution of the actual gallery script are not met::

    from ase.utils.sphinx_assert impoty SphinxPrerequisitesError
    try:
        import gpaw
    except ImportError as e:
        raise SphinxPrerequisitesError from e

How it works
------------
The module monkey-patches ``gen_rst.execute_script`` from sphinx-gallery.
Before each script runs, the patch looks for a companion probe file next to
the script and tries to import it.  A failed import is caught and reported as
a descriptive warning, and ``script_vars['execute_script']`` is set to False
so sphinx-gallery renders the code cells without executing them.
"""

import importlib
import warnings
from pathlib import Path

PROBE_SUFFIX = '_prerequisites'


class SphinxPrerequisitesError(Exception):
    """Raised by a probe file to signal that its gallery script should be
    skipped.

    Raise this when a required dependency is missing or incompatible.  Any other
    exception raised by a probe file is treated as an unexpected error and
    propagated normally.
    """


def probe_path(script_path):
    """Return the Path of the companion probe file for *script_path*.

    E.g. ``examples/**/plot_something.py`` ->
    ``examples/**/plot_something_prerequisites.py``.
    """
    p = Path(script_path)
    return p.with_stem(p.stem + PROBE_SUFFIX)


def run_probe(script_path):
    """Run the probe file for *script_path*, if one exists.

    The probe is imported in a fresh empty namespace so its side effects do
    not leak into the gallery environment.  If the import fails with
    SphinxPrerequisitesError the gallery script should be skipped.

    Returns True if the probe imported successfully or no probe file exists.
    Returns False on SphinxPrerequisitesError in import.
    """
    probe = probe_path(script_path)
    if not probe.is_file():
        return True

    spec = importlib.util.spec_from_file_location('probe', probe)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SphinxPrerequisitesError:
        warnings.warn(
            f'Probe {str(probe)!r} failed — '
            f'{str(script_path)!r} will be shown but NOT executed.'
        )
        return False

    return True


def setup(app):
    import sphinx_gallery.gen_rst as gen_rst

    original_execute_script = gen_rst.execute_script

    def patched_execute_script(script_blocks, script_vars, *args, **kwargs):
        """Wrap ``execute_script`` to skip execution when the probe file
        fails."""

        if not run_probe(script_vars['src_file']):
            script_vars['execute_script'] = False
        return original_execute_script(
            script_blocks, script_vars, *args, **kwargs
        )

    gen_rst.execute_script = patched_execute_script
