"""Helper functions to parse different blocks."""

import re

import numpy as np


def tensor33(x):
    return np.squeeze(np.reshape(x, (3, 3))).tolist()


def tensor31(x):
    return np.squeeze(np.reshape(x, (3, 1))).tolist()


def get_version(file_contents):
    """
    Look for and parse the magres file format version line
    """

    lines = file_contents.split('\n')
    match = re.match(r'\#\$magres-abinitio-v([0-9]+).([0-9]+)', lines[0])

    if match:
        version = match.groups()
        version = tuple(vnum for vnum in version)
    else:
        version = None

    return version


def parse_blocks(file_contents):
    """
    Parse series of XML-like deliminated blocks into a list of
    (block_name, contents) tuples
    """
    blocks_re = re.compile(
        r'[\[<](?P<block_name>.*?)[>\]](.*?)[<\[]/' + r'(?P=block_name)[\]>]',
        re.M | re.S,
    )

    blocks = blocks_re.findall(file_contents)

    return blocks


def parse_block(block):
    """
    Parse block contents into a series of (tag, data) records
    """

    def clean_line(line):
        # Remove comments and whitespace at start and ends of line
        line = re.sub('#(.*?)\n', '', line)
        line = line.strip()

        return line

    name, data = block

    lines = [clean_line(line) for line in data.split('\n')]

    records = []

    for line in lines:
        xs = line.split()

        if len(xs) > 0:
            tag = xs[0]
            data = xs[1:]

            records.append((tag, data))

    return (name, records)


def check_units(d):
    """
    Verify that given units for a particular tag are correct.
    """

    allowed_units = {
        'lattice': 'Angstrom',
        'atom': 'Angstrom',
        'ms': 'ppm',
        'efg': 'au',
        'efg_local': 'au',
        'efg_nonlocal': 'au',
        'isc': '10^19.T^2.J^-1',
        'isc_fc': '10^19.T^2.J^-1',
        'isc_orbital_p': '10^19.T^2.J^-1',
        'isc_orbital_d': '10^19.T^2.J^-1',
        'isc_spin': '10^19.T^2.J^-1',
        'sus': '10^-6.cm^3.mol^-1',
        'calc_cutoffenergy': 'Hartree',
    }

    if d[0] in d and d[1] == allowed_units[d[0]]:
        pass
    else:
        raise RuntimeError(f'Unrecognized units: {d[0]} {d[1]}')

    return d


def parse_magres_block(block):
    """
    Parse magres block into data dictionary given list of record
    tuples.
    """

    _name, records = block

    # 3x3 tensor
    def ntensor33(name):
        return lambda d: {name: tensor33([float(x) for x in data])}

    # Atom label, atom index and 3x3 tensor
    def sitensor33(name):
        return lambda d: _parse_sitensor33(name, data)

    # 2x(Atom label, atom index) and 3x3 tensor
    def sisitensor33(name):
        return lambda d: {
            'atom1': {'label': data[0], 'index': int(data[1])},
            'atom2': {'label': data[2], 'index': int(data[3])},
            name: tensor33([float(x) for x in data[4:]]),
        }

    tags = {
        'ms': sitensor33('sigma'),
        'sus': ntensor33('S'),
        'efg': sitensor33('V'),
        'efg_local': sitensor33('V'),
        'efg_nonlocal': sitensor33('V'),
        'isc': sisitensor33('K'),
        'isc_fc': sisitensor33('K'),
        'isc_spin': sisitensor33('K'),
        'isc_orbital_p': sisitensor33('K'),
        'isc_orbital_d': sisitensor33('K'),
        'units': check_units,
    }

    data_dict = {}

    for record in records:
        tag, data = record

        if tag not in data_dict:
            data_dict[tag] = []

        data_dict[tag].append(tags[tag](data))

    return data_dict


def _unmunge_label_index(label_index: str) -> tuple[str, str]:
    """Splits a label_index string into a label and an index,
    where the index is always the final 3 digits.

    This function handles cases where the site label and index are combined
    in CASTEP magres files (versions < 23),
    e.g., 'H1222' instead of 'H1' and '222'.

    Since site labels can contain numbers (e.g., H1, H2, H1a),
    we extract the index as the final 3 digits.
    The remaining characters form the label.

    Note: Only call this function when label and index are confirmed
    to be combined (detected by the line having 10 fields instead of 11).

    Parameters
    ----------
    label_index : str
        The input string containing the combined label and index
        (e.g., 'H1222')

    Returns
    -------
    tuple[str, str]
        A tuple of (label, index) strings (e.g., ('H1', '222'))

    Raises
    ------
    RuntimeError
        If the index is >999 (not supported by this solution))
        If invalid data format or regex match failure

    Examples
    --------
    >>> _unmunge_label_index('H1222')
    ('H1', '222')
    >>> _unmunge_label_index('C201')
    ('C', '201')
    >>> _unmunge_label_index('H23104')
    ('H23', '104')
    >>> _unmunge_label_index('H1a100')
    ('H1a', '100')
    """
    match = re.match(r'(.+?)(\d{3})$', label_index)
    if match:
        label, index = match.groups()
        if not isinstance(label, str) or not isinstance(index, str):
            raise RuntimeError('Regex match produced non-string values')
        if index == '000':
            raise RuntimeError(
                'Index greater than 999 detected. This is not supported in '
                'magres files with munged label and indices. '
                'Try manually unmunging the label and index.'
            )
        return (label, index)
    raise RuntimeError(
        'Invalid data in magres block. Check the site labels and indices.'
    )


def _parse_sitensor33(name, data):
    # We expect label, index, and then the 3x3 tensor
    if len(data) == 10:
        label, index = _unmunge_label_index(data[0])
        data = [label, index] + data[1:]
    if len(data) != 11:
        raise ValueError(
            f'Expected 11 values for {name} tensor data, got {len(data)}'
        )

    return {
        'atom': {'label': data[0], 'index': int(data[1])},
        name: tensor33([float(x) for x in data[2:]]),
    }


def parse_atoms_block(block):
    """
    Parse atoms block into data dictionary given list of record tuples.
    """

    _name, records = block

    # Lattice record: a1, a2 a3, b1, b2, b3, c1, c2 c3
    def lattice(d):
        return tensor33([float(x) for x in data])

    # Atom record: label, index, x, y, z
    def atom(d):
        return {
            'species': data[0],
            'label': data[1],
            'index': int(data[2]),
            'position': tensor31([float(x) for x in data[3:]]),
        }

    def symmetry(d):
        return ' '.join(data)

    tags = {
        'lattice': lattice,
        'atom': atom,
        'units': check_units,
        'symmetry': symmetry,
    }

    data_dict = {}

    for record in records:
        tag, data = record
        if tag not in data_dict:
            data_dict[tag] = []
        data_dict[tag].append(tags[tag](data))

    return data_dict


def parse_generic_block(block):
    """
    Parse any other block into data dictionary given list of record
    tuples.
    """

    _name, records = block

    data_dict = {}

    for record in records:
        tag, data = record

        if tag not in data_dict:
            data_dict[tag] = []

        data_dict[tag].append(data)

    return data_dict
