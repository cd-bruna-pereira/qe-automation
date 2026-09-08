from functools import cache

import ase
from ase.utils import (
    get_python_package_path_description,
    search_current_git_hash,
)


def format_dependency(modname: str) -> tuple[str, str]:
    """Return (name, info) for given module.

    If possible, info is the path to the module's package."""
    import importlib.metadata

    try:
        module = importlib.import_module(modname)
    except ImportError:
        return modname, 'not installed'

    if modname == 'flask':
        version = importlib.metadata.version('flask')
    else:
        version = getattr(module, '__version__', '?')

    if modname == 'ase':
        name = ase_version_info()
    else:
        name = f'{modname}-{version}'

    # (only packages have __path__, but we are importing packages.)
    info = get_python_package_path_description(module)
    return name, info


def all_dependencies() -> list[tuple[str, str]]:
    names = [
        'ase',
        'numpy',
        'scipy',
        'matplotlib',
        'spglib',
        'ase_ext',
        'flask',
        'psycopg2',
        'pyamg',
    ]
    return [format_dependency(name) for name in names]


@cache
def ase_version_info() -> str:
    """Return "ase-<version>".

    If ASE is installed from source, version includes shortened git hash."""
    version = f'ase-{ase.__version__}'
    githash = search_current_git_hash(ase)
    if githash:
        version += f'-{githash:.10}'
    return version
