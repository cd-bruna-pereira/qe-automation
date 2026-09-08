"""RuNNer input.data support

Write files in RuNNer's input.data file format.

Contains
--------
* `write_runnerdata`: Write structures to a RuNNer input.data file.

Reference
---------
* [The online documentation of RuNNer 2.0](https://runner-suite.gitlab.io/runner2)

Contributors
------------
* Maintainer and Author: [Alexander Knoll](mailto:alexander.knoll@rub.de)
* Author: [Redouan El Haouari](mailto:redouan.elhaouari@rub.de)
"""

from typing import TextIO

from ase.atoms import Atoms
from ase.utils import writer

from .runneratoms import RuNNerAtoms, Units


def _format_atom_section(runneratoms: RuNNerAtoms, fmt: str) -> str:
    """Format the per-structure block of lines containing atom information."""
    lines = []

    for i, position in enumerate(runneratoms.positions):
        fields = [f'{j:{fmt}}' for j in position]
        fields.append(f'{runneratoms.symbols[i]:2s}')

        for name, length in runneratoms.atom_layout:
            atom_arr = runneratoms.atom_arrays[name][i]

            if length == 1:
                fields.append(f'{atom_arr:{fmt}}')
            else:
                assert isinstance(atom_arr, list)
                fields.extend(f'{v:{fmt}}' for v in atom_arr)

        if runneratoms.forces is not None:
            fields.extend(f'{f:{fmt}}' for f in runneratoms.forces[i])

        lines.append('atom ' + ' '.join(fields))

    return '\n'.join(lines) + '\n'


def _compose_begin_line(atoms: RuNNerAtoms) -> str:
    """Compose the field names in the begin line."""
    begin_fields = ['begin', 'position(3)', 'element']

    if atoms.atom_arrays is not None:
        for name, length in atoms.atom_layout:
            if length == 1:
                begin_fields.append(name)
            else:
                begin_fields.append(f'{name}({length})')

    if atoms.forces is not None:
        begin_fields.append('forces(3)')

    return ' '.join(begin_fields) + '\n'


@writer
def write_runnerdata(
    outfile: TextIO,
    images: list[Atoms],
    comment: str = '',
    fmt: str = '20.10f',
    input_units: Units = Units.ASE,
) -> None:
    """Write series of ASE Atoms to a RuNNer input.data file.

    For further details see the `read_runnerdata` routine.

    Parameters
    ----------
    outfile: TextIOWrapper
        Python fileobj with the target input.data file.
    images: array-like
        List of `Atoms` objects.
    comment: str
        A comment message to be added to each structure.
    fmt: str
        A format specifier for float values.
    input_units: str
        The given input units. Can be 'Units.ASE' or 'Units.ATOMIC'.

    Raises
    ------
    ValueError: exception
        Raised if the comment line contains newline characters.
    """
    # Preprocess the comment.
    comment = comment.rstrip()
    if '\n' in comment:
        raise ValueError('Comment line cannot contain line breaks.')

    for atoms in images:
        runneratoms = RuNNerAtoms.from_ase_atoms(atoms, input_units=input_units)
        runneratoms.convert(Units.ATOMIC)

        outfile.write(_compose_begin_line(runneratoms))

        if comment:
            outfile.write(f'comment {comment}\n')

        for vec in runneratoms.cell:
            outfile.write(
                f'lattice {vec[0]:{fmt}} {vec[1]:{fmt}} {vec[2]:{fmt}}\n'
            )

        outfile.write(_format_atom_section(runneratoms, fmt))

        if runneratoms.energy is not None:
            outfile.write(f'energy {runneratoms.energy:{fmt}}\n')

        if runneratoms.total_charge is not None:
            outfile.write(f'charge {runneratoms.total_charge:{fmt}}\n')

        outfile.write('end\n')
