"""RuNNer input.data support

Read files in RuNNer's input.data file format.

Contains
--------
* `read_runnerdata`: Read structures from a RuNNer input.data file.

Reference
---------
* [The online documentation of RuNNer 2.0](https://runner-suite.gitlab.io/runner2)

Contributors
------------
* Maintainer and Author: [Alexander Knoll](mailto:alexander.knoll@rub.de)
* Author: [Redouan El Haouari](mailto:redouan.elhaouari@rub.de)
"""

import re
from collections.abc import Iterator
from typing import TextIO

from ase.atoms import Atoms
from ase.utils import reader

from .runneratoms import (
    DEFAULT_ATOM_LAYOUT,
    PROP_NAME_DICT,
    RuNNerAtoms,
    Units,
)

# Regex pattern for vector properties like "forces(3)"
_vector_pattern = re.compile(r'([\w\-]+)\((\d+)\)')


def _parse_begin(begin_line: str) -> list[tuple[str, int]]:
    """Parse the begin line to determine per-atom property layout.

    Returns a list of (name, length) for all properties (vector or scalar).
    Scalars have length 1.

    The two default fields "positions" and "element" are skipped.
    """
    fields = begin_line.strip().split()[1:]
    layout: list[tuple[str, int]] = []

    if len(fields) == 0:
        return DEFAULT_ATOM_LAYOUT

    # skip the first two fields (position, element), since they are fixed.
    for field in fields[2:]:
        match = _vector_pattern.fullmatch(field)
        if match:
            name, length = match.group(1), int(match.group(2))
        else:
            name = field
            length = 1

        # Normalize property names by replacing them with the defaults defined
        # in the dictionary PROP_NAME_DICT.
        name = PROP_NAME_DICT.get(name, name)

        layout.append((name, length))

    return layout


def _parse_chunk(
    chunk: str, begin_line: str, input_units: Units
) -> RuNNerAtoms:
    """Parse a single chunk into a RuNNerAtoms object."""
    atom_layout = _parse_begin(begin_line)

    runneratoms = RuNNerAtoms(
        atom_layout=atom_layout,
        input_units=input_units,
        num_atom_columns=sum([i for _, i in atom_layout]) + 4,
    )

    for line in chunk.splitlines():
        runneratoms.parse_line(line)

    return runneratoms


@reader
def read_runnerdata(
    infile: TextIO,
    index: int | slice = -1,
    input_units: Units = Units.ATOMIC,
    output_units: Units = Units.ASE,
) -> Iterator[Atoms]:
    """Parse all structures within a RuNNer input.data file.

    input.data files contain all structural information needed to train a
    Behler-Parrinello-type neural network potential, e.g. Cart. coordinates,
    atomic forces, and energies. This function reads the file object `infile`
    and returns the slice of structures given by `index`. All structures will
    be converted to ASE units by default.

    Parameters
    ----------
    infile:
        Python fileobj with the target input.data file.
    index:
        The slice of structures which should be returned. Returns only the last
        structure by default.
    input_units:
        The given input units. Can be 'Units.ASE' or 'Units.ATOMIC'.
    output_units:
        The desired output units. Can be 'Units.ASE' or 'Units.ATOMIC'.

    Yields
    ------
    images:
        All information about the structures within `index` of `infile`,
        including symbols, positions, atomic charges, and cell lattice. Every
        `Atoms` object has a `RunnerSinglePointCalculator` attached with
        additional information on the total energy, atomic forces, and total
        charge.

    """
    # First, split input.data into separate structure "chunks".
    read_infile = infile.read()
    # First chunk is discarded because it is the data before the
    # first "begin".
    begin_pattern = re.compile(r'begin.*\n')
    chunks = begin_pattern.split(read_infile)[1:]
    begin_lines = begin_pattern.findall(read_infile)

    # Second, only parse the chunks which the user asked for.
    for begin_line, chunk in zip(begin_lines[index], chunks[index]):
        runneratoms = _parse_chunk(chunk, begin_line, input_units)

        runneratoms.convert(output_units)
        yield runneratoms.to_ase_atoms()
