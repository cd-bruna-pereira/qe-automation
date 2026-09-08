# fmt: off

"""This module provides I/O functions for the MAGRES file format, introduced
by CASTEP as an output format to store structural data and ab-initio
calculated NMR parameters.
Authors: Simone Sturniolo (ase implementation), Tim Green (original magres
    parser code)
"""

from collections import OrderedDict

import numpy as np

import ase.io._magres as _magres
import ase.units
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError

_mprops = {
    'ms': ('sigma', 1),
    'sus': ('S', 0),
    'efg': ('V', 1),
    'isc': ('K', 2)}
# (matrix name, number of atoms in interaction) for various magres quantities


def read_magres(fd, include_unrecognised=False):
    """Read magres file."""

    block_parsers = {'magres': _magres.parse_magres_block,
                     'atoms': _magres.parse_atoms_block,
                     'calculation': _magres.parse_generic_block, }

    file_contents = fd.read()

    # This works as a validity check
    version = _magres.get_version(file_contents)
    if version is None:
        # This isn't even a .magres file!
        raise RuntimeError('File is not in standard Magres format')
    blocks = _magres.parse_blocks(file_contents)

    data_dict = {}

    for block_data in blocks:
        block = _magres.parse_block(block_data)

        if block[0] in block_parsers:
            block_dict = block_parsers[block[0]](block)
            data_dict[block[0]] = block_dict
        else:
            # Throw in the text content of blocks we don't recognise
            if include_unrecognised:
                data_dict[block[0]] = block_data[1]

    # Now the loaded data must be turned into an ASE Atoms object

    # First check if the file is even viable
    if 'atoms' not in data_dict:
        raise RuntimeError('Magres file does not contain structure data')

    # Allowed units handling. This is redundant for now but
    # could turn out useful in the future

    magres_units = {'Angstrom': ase.units.Ang}

    # Lattice parameters?
    if 'lattice' in data_dict['atoms']:
        try:
            u = dict(data_dict['atoms']['units'])['lattice']
        except KeyError:
            raise RuntimeError('No units detected in file for lattice')
        u = magres_units[u]
        cell = np.array(data_dict['atoms']['lattice'][0]) * u
        pbc = True
    else:
        cell = None
        pbc = False

    # Now the atoms
    symbols = []
    positions = []
    indices = []
    labels = []

    if 'atom' in data_dict['atoms']:
        try:
            u = dict(data_dict['atoms']['units'])['atom']
        except KeyError:
            raise RuntimeError('No units detected in file for atom positions')
        u = magres_units[u]
        # Now we have to account for the possibility of there being CASTEP
        # 'custom' species amongst the symbols
        custom_species = None
        for a in data_dict['atoms']['atom']:
            spec_custom = a['species'].split(':', 1)
            if len(spec_custom) > 1 and custom_species is None:
                # Add it to the custom info!
                custom_species = list(symbols)
            symbols.append(spec_custom[0])
            positions.append(a['position'])
            indices.append(a['index'])
            labels.append(a['label'])
            if custom_species is not None:
                custom_species.append(a['species'])

    atoms = Atoms(cell=cell,
                  pbc=pbc,
                  symbols=symbols,
                  positions=positions)

    # Add custom species if present
    if custom_species is not None:
        atoms.new_array('castep_custom_species', np.array(custom_species))

    # Add the spacegroup, if present and recognizable
    if 'symmetry' in data_dict['atoms']:
        try:
            spg = Spacegroup(data_dict['atoms']['symmetry'][0])
        except SpacegroupNotFoundError:
            # Not found
            spg = Spacegroup(1)  # Most generic one
        atoms.info['spacegroup'] = spg

    # Set up the rest of the properties as arrays
    atoms.new_array('indices', np.array(indices))
    atoms.new_array('labels', np.array(labels))

    # Now for the magres specific stuff
    li_list = list(zip(labels, indices))

    def create_magres_array(name, order, block):

        if order == 1:
            u_arr = [None] * len(li_list)
        elif order == 2:
            u_arr = [[None] * (i + 1) for i in range(len(li_list))]
        else:
            raise ValueError(
                'Invalid order value passed to create_magres_array')

        for s in block:
            # Find the atom index/indices
            if order == 1:
                # First find out which atom this is
                at = (s['atom']['label'], s['atom']['index'])
                try:
                    ai = li_list.index(at)
                except ValueError:
                    raise RuntimeError('Invalid data in magres block')
                # Then add the relevant quantity
                u_arr[ai] = s[mn]
            else:
                at1 = (s['atom1']['label'], s['atom1']['index'])
                at2 = (s['atom2']['label'], s['atom2']['index'])
                ai1 = li_list.index(at1)
                ai2 = li_list.index(at2)
                # Sort them
                ai1, ai2 = sorted((ai1, ai2), reverse=True)
                u_arr[ai1][ai2] = s[mn]

        if order == 1:
            return np.array(u_arr)
        else:
            return np.array(u_arr, dtype=object)

    if 'magres' in data_dict:
        if 'units' in data_dict['magres']:
            atoms.info['magres_units'] = dict(data_dict['magres']['units'])
            for u in atoms.info['magres_units']:
                # This bit to keep track of tags
                u0 = u.split('_')[0]

                if u0 not in _mprops:
                    raise RuntimeError('Invalid data in magres block')

                mn, order = _mprops[u0]

                if order > 0:
                    u_arr = create_magres_array(mn, order,
                                                data_dict['magres'][u])
                    atoms.new_array(u, u_arr)
                else:
                    # atoms.info['magres_data'] = atoms.info.get('magres_data',
                    #                                            {})
                    # # We only take element 0 because for this sort of data
                    # # there should be only that
                    # atoms.info['magres_data'][u] = \
                    #     data_dict['magres'][u][0][mn]
                    if atoms.calc is None:
                        calc = SinglePointDFTCalculator(atoms)
                        atoms.calc = calc
                        atoms.calc.results[u] = data_dict['magres'][u][0][mn]

    if 'calculation' in data_dict:
        atoms.info['magresblock_calculation'] = data_dict['calculation']

    if include_unrecognised:
        for b in data_dict:
            if b not in block_parsers:
                atoms.info['magresblock_' + b] = data_dict[b]

    return atoms


def tensor_string(tensor):
    return ' '.join(' '.join(str(x) for x in xs) for xs in tensor)


def write_magres(fd, image):
    """
    A writing function for magres files. Two steps: first data are arranged
    into structures, then dumped to the actual file
    """

    image_data = {}
    image_data['atoms'] = {'units': []}
    # Contains units, lattice and each individual atom
    if np.all(image.get_pbc()):
        # Has lattice!
        image_data['atoms']['units'].append(['lattice', 'Angstrom'])
        image_data['atoms']['lattice'] = [image.get_cell()]

    # Now for the atoms
    if image.has('labels'):
        labels = image.get_array('labels')
    else:
        labels = image.get_chemical_symbols()

    if image.has('indices'):
        indices = image.get_array('indices')
    else:
        indices = [labels[:i + 1].count(labels[i]) for i in range(len(labels))]

    # Iterate over atoms
    symbols = (image.get_array('castep_custom_species')
               if image.has('castep_custom_species')
               else image.get_chemical_symbols())

    atom_info = list(zip(symbols,
                         image.get_positions()))
    if len(atom_info) > 0:
        image_data['atoms']['units'].append(['atom', 'Angstrom'])
        image_data['atoms']['atom'] = []

    for i, a in enumerate(atom_info):
        image_data['atoms']['atom'].append({
            'index': indices[i],
            'position': a[1],
            'species': a[0],
            'label': labels[i]})

    # Spacegroup, if present
    if 'spacegroup' in image.info:
        image_data['atoms']['symmetry'] = [image.info['spacegroup']
                                           .symbol.replace(' ', '')]

    # Now go on to do the same for magres information
    if 'magres_units' in image.info:

        image_data['magres'] = {'units': []}

        for u in image.info['magres_units']:
            # Get the type
            p = u.split('_')[0]
            if p in _mprops:
                image_data['magres']['units'].append(
                    [u, image.info['magres_units'][u]])
                image_data['magres'][u] = []
                mn, order = _mprops[p]

                if order == 0:
                    # The case of susceptibility
                    tens = {
                        mn: image.calc.results[u]
                    }
                    image_data['magres'][u] = tens
                else:
                    arr = image.get_array(u)
                    li_tab = zip(labels, indices)
                    for i, (lab, ind) in enumerate(li_tab):
                        if order == 2:
                            for j, (lab2, ind2) in enumerate(li_tab[:i + 1]):
                                if arr[i][j] is not None:
                                    tens = {mn: arr[i][j],
                                            'atom1': {'label': lab,
                                                      'index': ind},
                                            'atom2': {'label': lab2,
                                                      'index': ind2}}
                                    image_data['magres'][u].append(tens)
                        elif order == 1:
                            if arr[i] is not None:
                                tens = {mn: arr[i],
                                        'atom': {'label': lab,
                                                 'index': ind}}
                                image_data['magres'][u].append(tens)

    # Calculation block, if present
    if 'magresblock_calculation' in image.info:
        image_data['calculation'] = image.info['magresblock_calculation']

    def write_units(data, out):
        if 'units' in data:
            for tag, units in data['units']:
                out.append(f'  units {tag} {units}')

    def write_magres_block(data):
        """
            Write out a <magres> block from its dictionary representation
        """

        out = []

        def nout(tag, tensor_name):
            if tag in data:
                out.append(' '.join([' ', tag,
                                     tensor_string(data[tag][tensor_name])]))

        def siout(tag, tensor_name):
            if tag in data:
                for atom_si in data[tag]:
                    out.append(('  %s %s %d '
                                '%s') % (tag,
                                         atom_si['atom']['label'],
                                         atom_si['atom']['index'],
                                         tensor_string(atom_si[tensor_name])))

        write_units(data, out)

        nout('sus', 'S')
        siout('ms', 'sigma')

        siout('efg_local', 'V')
        siout('efg_nonlocal', 'V')
        siout('efg', 'V')

        def sisiout(tag, tensor_name):
            if tag in data:
                for isc in data[tag]:
                    out.append(('  %s %s %d %s %d '
                                '%s') % (tag,
                                         isc['atom1']['label'],
                                         isc['atom1']['index'],
                                         isc['atom2']['label'],
                                         isc['atom2']['index'],
                                         tensor_string(isc[tensor_name])))

        sisiout('isc_fc', 'K')
        sisiout('isc_orbital_p', 'K')
        sisiout('isc_orbital_d', 'K')
        sisiout('isc_spin', 'K')
        sisiout('isc', 'K')

        return '\n'.join(out)

    def write_atoms_block(data):
        out = []

        write_units(data, out)

        if 'lattice' in data:
            for lat in data['lattice']:
                out.append(f"  lattice {tensor_string(lat)}")

        if 'symmetry' in data:
            for sym in data['symmetry']:
                out.append(f'  symmetry {sym}')

        if 'atom' in data:
            for a in data['atom']:
                out.append(('  atom %s %s %s '
                            '%s') % (a['species'],
                                     a['label'],
                                     a['index'],
                                     ' '.join(str(x) for x in a['position'])))

        return '\n'.join(out)

    def write_generic_block(data):
        out = []

        for tag, data in data.items():
            for value in data:
                out.append('{} {}'.format(tag, ' '.join(str(x) for x in value)))

        return '\n'.join(out)

    # Using this to preserve order
    block_writers = OrderedDict([('calculation', write_generic_block),
                                 ('atoms', write_atoms_block),
                                 ('magres', write_magres_block)])

    # First, write the header
    fd.write('#$magres-abinitio-v1.0\n')
    fd.write('# Generated by the Atomic Simulation Environment library\n')

    for b in block_writers:
        if b in image_data:
            fd.write(f'[{b}]\n')
            fd.write(block_writers[b](image_data[b]))
            fd.write(f'\n[/{b}]\n')

    # Now on to check for any non-standard blocks...
    for i in image.info:
        if '_' in i:
            ismag, b = i.split('_', 1)
            if ismag == 'magresblock' and b not in block_writers:
                fd.write(f'[{b}]\n')
                fd.write(image.info[i])
                fd.write(f'[/{b}]\n')
