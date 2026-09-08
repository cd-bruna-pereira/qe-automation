import numpy as np
import pytest

from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.units import Bohr, Hartree

# Periodic System

sample_inputfile = """
begin position(3) elements charges magmoms forces(3) prop1 prop2(2)
lattice  1.7097166001e+01  0.0000000000e+00  0.0000000000e+00
lattice  0.0000000000e+00  1.7097166001e+01  0.0000000000e+00
lattice  0.0000000000e+00  0.0000000000e+00  5.0000000001e+01
atom  1.4275939497e+01  1.4231407235e+01  3.6076973257e+01 O  -2.8918410600e-01\
 0.0 -2.8119721838e-03  5.1917246104e-04 -6.1571751814e-03 2.0 3.0 4.0
atom  1.1412436308e+01  1.1399368513e+01  4.0122450030e+01 O  -3.6358139600e-01\
 0.0 -8.7882158509e-04 -9.1956096436e-04 -2.2866358560e-02 2.0 3.0 4.0
atom  1.4206953179e+01  1.4201432515e+01  4.0151366429e+01 Mg  3.7335335400e-01\
 0.0  3.7450445387e-03  5.1525057893e-03  1.4302349011e-02 2.0 3.0 4.0
atom  2.7674441650e+00  3.2676547854e+00  1.5266538963e+01 Au -1.8129486000e-02\
 0.0  4.7752095368e-04 -2.2142671411e-03  5.1007575752e-02 2.0 3.0 4.0
atom  2.7884963173e+00  3.1844017731e+00  1.0716139442e+01 Au -2.1088749600e-01\
 0.0 -3.8108279331e-05  1.9467681277e-04 -4.1132711694e-02 2.0 3.0 4.0
energy -5.4395981012e+04
charge -0.5084291
end
begin position(3) elements charges magmoms forces(3) prop1 prop2(2)
lattice  1.7097166001e+01  0.0000000000e+00  0.0000000000e+00
lattice  0.0000000000e+00  1.7097166001e+01  0.0000000000e+00
lattice  0.0000000000e+00  0.0000000000e+00  5.0000000001e+01
atom  1.4272572875e+01  1.4211055189e+01  3.6029266916e+01 O  -2.9285677027e-01\
 0.0 -3.5309877319e-03  5.0460293558e-03  4.2060129135e-03 2.0 3.0 4.0
atom  1.1354692329e+01  1.1460175814e+01  4.0165924689e+01 O  -3.6393253027e-01\
 0.0  6.7307290176e-03 -5.8410162857e-03 -2.6114879133e-02 2.0 3.0 4.0
atom  1.4282415362e+01  1.4216391775e+01  4.0116261083e+01 Mg  3.7507213973e-01\
 0.0 -4.5330918345e-03  2.1468339206e-03  1.3403211731e-02 2.0 3.0 4.0
atom  1.1072914323e+01  1.1349504011e+01  1.4330323695e+01 Au -9.5324370273e-02\
 0.0  2.0451169014e-02 -2.6611347314e-04  1.2209050962e-03 2.0 3.0 4.0
atom  6.3571436157e+00  1.1482303636e+01  1.4536148776e+01 Au -7.9902230273e-02\
 0.0 -2.0898800524e-02  7.5900764900e-04  6.1314623489e-04 2.0 3.0 4.0
energy -5.4395988758e+04
charge -0.4569438
end
"""

ref_outputfile = """\
begin position(3) element charge magmom prop1 prop2(2) forces(3)
lattice           17.0971660            0.0000000            0.0000000
lattice            0.0000000           17.0971660            0.0000000
lattice            0.0000000            0.0000000           50.0000000
atom           14.2759395           14.2314072           36.0769733 O          \
  -0.2891841            0.0000000            2.0000000            3.0000000    \
        4.0000000           -0.0028120            0.0005192           -0.0061572
atom           11.4124363           11.3993685           40.1224500 O          \
  -0.3635814            0.0000000            2.0000000            3.0000000    \
        4.0000000           -0.0008788           -0.0009196           -0.0228664
atom           14.2069532           14.2014325           40.1513664 Mg         \
   0.3733534            0.0000000            2.0000000            3.0000000    \
        4.0000000            0.0037450            0.0051525            0.0143023
atom            2.7674442            3.2676548           15.2665390 Au         \
  -0.0181295            0.0000000            2.0000000            3.0000000    \
        4.0000000            0.0004775           -0.0022143            0.0510076
atom            2.7884963            3.1844018           10.7161394 Au         \
  -0.2108875            0.0000000            2.0000000            3.0000000    \
        4.0000000           -0.0000381            0.0001947           -0.0411327
energy       -54395.9810120
charge            0.0000000
end
begin position(3) element charge magmom prop1 prop2(2) forces(3)
lattice           17.0971660            0.0000000            0.0000000
lattice            0.0000000           17.0971660            0.0000000
lattice            0.0000000            0.0000000           50.0000000
atom           14.2725729           14.2110552           36.0292669 O          \
  -0.2928568            0.0000000            2.0000000            3.0000000    \
        4.0000000           -0.0035310            0.0050460            0.0042060
atom           11.3546923           11.4601758           40.1659247 O          \
  -0.3639325            0.0000000            2.0000000            3.0000000    \
        4.0000000            0.0067307           -0.0058410           -0.0261149
atom           14.2824154           14.2163918           40.1162611 Mg         \
   0.3750721            0.0000000            2.0000000            3.0000000    \
        4.0000000           -0.0045331            0.0021468            0.0134032
atom           11.0729143           11.3495040           14.3303237 Au         \
  -0.0953244            0.0000000            2.0000000            3.0000000    \
        4.0000000            0.0204512           -0.0002661            0.0012209
atom            6.3571436           11.4823036           14.5361488 Au         \
  -0.0799022            0.0000000            2.0000000            3.0000000    \
        4.0000000           -0.0208988            0.0007590            0.0006131
energy       -54395.9887580
charge            0.0000000
end
"""

# Non-Periodic System Data (New)

sample_np_inputfile = """\
begin
atom         0.0000000000         0.0000000000        -4.0364550000 Ag        \
-0.4356515741         0.0000000000         0.0000000000         0.0000000000  \
      -0.1339292767
atom         0.0000000000         0.0000000000         0.0000000000 Ag        \
-0.1234211190         0.0000000000         0.0000000000         0.0000000000  \
       0.0151742830
atom         0.0000000000         0.0000000000         4.0568410000 Ag        \
-0.4409273069         0.0000000000         0.0000000000         0.0000000000  \
       0.1187549937
energy    -16138.7661406807
charge        -1.0000000000
end
begin
atom         0.0000000000         0.0000000000        -4.0568410000 Ag        \
 0.4709505873         0.0000000000         0.0000000000         0.0000000000  \
      -0.1593636539
atom         0.0000000000         0.0000000000         0.0000000000 Ag        \
 0.0523806844         0.0000000000         0.0000000000         0.0000000000  \
       0.1595173562
atom         0.0000000000         0.0000000000         5.0965340000 Ag        \
 0.4766687283         0.0000000000         0.0000000000         0.0000000000  \
      -0.0001537023
energy    -16138.4957065618
charge         1.0000000000
end
begin
atom         0.0000000000         0.0000000000        -5.8712070000 Ag        \
 0.0189245590         0.0000000000         0.0000000000         0.0000000000  \
       0.0166777468
atom         0.0000000000         0.0000000000         0.0000000000 Ag        \
-0.0373113812         0.0000000000         0.0000000000         0.0000000000  \
      -0.0006403173
atom         0.0000000000         0.0000000000         5.9935240000 Ag        \
 0.0183868222         0.0000000000         0.0000000000         0.0000000000  \
      -0.0160364294
energy    -16138.7578612655
charge         0.0000000000
end
"""

ref_np_outputfile = """\
begin position(3) element charge magmom forces(3)
atom            0.0000000            0.0000000           -4.0364550 Ag         \
  -0.4356516            0.0000000            0.0000000            0.0000000    \
       -0.1339293
atom            0.0000000            0.0000000            0.0000000 Ag         \
  -0.1234211            0.0000000            0.0000000            0.0000000    \
        0.0151743
atom            0.0000000            0.0000000            4.0568410 Ag         \
  -0.4409273            0.0000000            0.0000000            0.0000000    \
        0.1187550
energy       -16138.7661407
charge           -1.0000000
end
begin position(3) element charge magmom forces(3)
atom            0.0000000            0.0000000           -4.0568410 Ag         \
   0.4709506            0.0000000            0.0000000            0.0000000    \
       -0.1593637
atom            0.0000000            0.0000000            0.0000000 Ag         \
   0.0523807            0.0000000            0.0000000            0.0000000    \
        0.1595174
atom            0.0000000            0.0000000            5.0965340 Ag         \
   0.4766687            0.0000000            0.0000000            0.0000000    \
       -0.0001537
energy       -16138.4957066
charge            1.0000000
end
begin position(3) element charge magmom forces(3)
atom            0.0000000            0.0000000           -5.8712070 Ag         \
   0.0189246            0.0000000            0.0000000            0.0000000    \
        0.0166777
atom            0.0000000            0.0000000            0.0000000 Ag         \
  -0.0373114            0.0000000            0.0000000            0.0000000    \
       -0.0006403
atom            0.0000000            0.0000000            5.9935240 Ag         \
   0.0183868            0.0000000            0.0000000            0.0000000    \
       -0.0160364
energy       -16138.7578613
charge            0.0000000
end
"""

float_cmp_prec = 1e-7

# References for periodic system tests
ref_pos = [
    np.array(
        [
            [1.4275939497e01, 1.4231407235e01, 3.6076973257e01],
            [1.1412436308e01, 1.1399368513e01, 4.0122450030e01],
            [1.4206953179e01, 1.4201432515e01, 4.0151366429e01],
            [2.7674441650e00, 3.2676547854e00, 1.5266538963e01],
            [2.7884963173e00, 3.1844017731e00, 1.0716139442e01],
        ]
    ),
    np.array(
        [
            [1.4272572875e01, 1.4211055189e01, 3.6029266916e01],
            [1.1354692329e01, 1.1460175814e01, 4.0165924689e01],
            [1.4282415362e01, 1.4216391775e01, 4.0116261083e01],
            [1.1072914323e01, 1.1349504011e01, 1.4330323695e01],
            [6.3571436157e00, 1.1482303636e01, 1.4536148776e01],
        ]
    ),
]
ref_symbols = [['O', 'O', 'Mg', 'Au', 'Au'], ['O', 'O', 'Mg', 'Au', 'Au']]
ref_charges = [
    np.array(
        [
            -2.89184106e-01,
            -3.63581396e-01,
            3.73353354e-01,
            -1.81294860e-02,
            -2.10887496e-01,
        ]
    ),
    np.array(
        [
            -2.92856770e-01,
            -3.63932530e-01,
            3.75072140e-01,
            -9.53243703e-02,
            -7.99022303e-02,
        ]
    ),
]
ref_magmoms = [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
]
ref_prop1 = [
    np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
    np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
]
ref_prop2 = [
    np.array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]),
    np.array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]),
]
ref_forces = [
    np.array(
        [
            [-2.8119721838e-03, 5.1917246104e-04, -6.1571751814e-03],
            [-8.7882158509e-04, -9.1956096436e-04, -2.2866358560e-02],
            [3.7450445387e-03, 5.1525057893e-03, 1.4302349011e-02],
            [4.7752095368e-04, -2.2142671411e-03, 5.1007575752e-02],
            [-3.8108279331e-05, 1.9467681277e-04, -4.1132711694e-02],
        ]
    ),
    np.array(
        [
            [-3.5309877319e-03, 5.0460293558e-03, 4.2060129135e-03],
            [6.7307290176e-03, -5.8410162857e-03, -2.6114879133e-02],
            [-4.5330918345e-03, 2.1468339206e-03, 1.3403211731e-02],
            [2.0451169014e-02, -2.6611347314e-04, 1.2209050962e-03],
            [-2.0898800524e-02, 7.5900764900e-04, 6.1314623489e-04],
        ]
    ),
]
ref_cell = [
    np.array(
        [
            [1.7097166001e01, 0.0000000000e00, 0.0000000000e00],
            [0.0000000000e00, 1.7097166001e01, 0.0000000000e00],
            [0.0000000000e00, 0.0000000000e00, 5.0000000001e01],
        ]
    ),
    np.array(
        [
            [1.7097166001e01, 0.0000000000e00, 0.0000000000e00],
            [0.0000000000e00, 1.7097166001e01, 0.0000000000e00],
            [0.0000000000e00, 0.0000000000e00, 5.0000000001e01],
        ]
    ),
]
ref_pbc = [np.array([True, True, True]), np.array([True, True, True])]
ref_energy = [-5.4395981012e04, -5.4395988758e04]
ref_total_charge = [-0.5084291, -0.4569438]


# References for non-periodic system tests
ref_np_pos = [
    np.array([[0.0, 0.0, -4.036455], [0.0, 0.0, 0.0], [0.0, 0.0, 4.056841]]),
    np.array([[0.0, 0.0, -4.056841], [0.0, 0.0, 0.0], [0.0, 0.0, 5.096534]]),
    np.array([[0.0, 0.0, -5.871207], [0.0, 0.0, 0.0], [0.0, 0.0, 5.993524]]),
]
ref_np_symbols = [['Ag', 'Ag', 'Ag']] * 3
ref_np_charges = [
    np.array([-0.4356515741, -0.1234211190, -0.4409273069]),
    np.array([0.4709505873, 0.0523806844, 0.4766687283]),
    np.array([0.0189245590, -0.0373113812, 0.0183868222]),
]
ref_np_magmoms = [np.array([0.0, 0.0, 0.0])] * 3
ref_np_forces = [
    np.array(
        [
            [0.0, 0.0, -0.1339292767],
            [0.0, 0.0, 0.0151742830],
            [0.0, 0.0, 0.1187549937],
        ]
    ),
    np.array(
        [
            [0.0, 0.0, -0.1593636539],
            [0.0, 0.0, 0.1595173562],
            [0.0, 0.0, -0.0001537023],
        ]
    ),
    np.array(
        [
            [0.0, 0.0, 0.0166777468],
            [0.0, 0.0, -0.0006403173],
            [0.0, 0.0, -0.0160364294],
        ]
    ),
]
ref_np_cell = [np.zeros((3, 3))] * 3
ref_np_pbc = [np.array([False, False, False])] * 3
ref_np_energy = [-16138.7661406807, -16138.4957065618, -16138.7578612655]
ref_np_total_charge = [-1.0, 1.0, 0.0]


# Grouping parameters for test parametrization
periodic_data = {
    'sample_input': sample_inputfile,
    'ref_output': ref_outputfile,
    'frames': 2,
    'pos': ref_pos,
    'symbols': ref_symbols,
    'charges': ref_charges,
    'magmoms': ref_magmoms,
    'prop1': ref_prop1,
    'prop2': ref_prop2,
    'forces': ref_forces,
    'cell': ref_cell,
    'pbc': ref_pbc,
    'energy': ref_energy,
    'total_charge': ref_total_charge,
    'is_periodic': True,
}

np_data = {
    'sample_input': sample_np_inputfile,
    'ref_output': ref_np_outputfile,
    'frames': 3,
    'pos': ref_np_pos,
    'symbols': ref_np_symbols,
    'charges': ref_np_charges,
    'magmoms': ref_np_magmoms,
    'prop1': None,
    'prop2': None,
    'forces': ref_np_forces,
    'cell': ref_np_cell,
    'pbc': ref_np_pbc,
    'energy': ref_np_energy,
    'total_charge': ref_np_total_charge,
    'is_periodic': False,
}


@pytest.mark.parametrize('data', [periodic_data, np_data])
def test_read_input_data(data) -> None:
    """Test reading functionality of input.data."""
    testfile_name = 'input.data_test'

    with open(testfile_name, 'wt', encoding='utf-8') as outfile:
        outfile.write(data['sample_input'])

    input_data = read(testfile_name, format='runnerdata', index=':')

    for iframe, frame in enumerate(input_data):
        np.testing.assert_allclose(
            frame.get_cell() / Bohr, data['cell'][iframe]
        )
        np.testing.assert_array_equal(frame.get_pbc(), data['pbc'][iframe])
        np.testing.assert_allclose(frame.positions / Bohr, data['pos'][iframe])

        np.testing.assert_allclose(
            frame.get_initial_charges(), data['charges'][iframe]
        )
        np.testing.assert_allclose(
            frame.get_initial_magnetic_moments(), data['magmoms'][iframe]
        )
        np.testing.assert_allclose(
            frame.get_forces() * Bohr / Hartree, data['forces'][iframe]
        )

        if data['prop1'] is not None:
            np.testing.assert_allclose(
                frame.get_array('prop1'), data['prop1'][iframe]
            )
        if data['prop2'] is not None:
            np.testing.assert_allclose(
                frame.get_array('prop2'), data['prop2'][iframe]
            )

        assert frame.get_chemical_symbols() == data['symbols'][iframe]
        assert (
            abs(frame.get_potential_energy() - data['energy'][iframe] * Hartree)
            < float_cmp_prec
        )
        assert (
            abs(frame.info['total_charge'] - data['total_charge'][iframe])
            < float_cmp_prec
        )
        assert (
            abs(sum(frame.get_initial_charges()) - data['total_charge'][iframe])
            < float_cmp_prec
        )


@pytest.mark.parametrize('data', [periodic_data, np_data])
def test_write_input_data(data) -> None:
    """Test writing functionality of input.data."""
    input_data: list[Atoms] = []
    for iframe in range(data['frames']):
        frame = Atoms(
            positions=data['pos'][iframe] * Bohr,
            symbols=data['symbols'][iframe],
            cell=data['cell'][iframe] * Bohr,
            pbc=data['pbc'][iframe],
        )

        frame.set_initial_charges(data['charges'][iframe])
        frame.set_initial_magnetic_moments(data['magmoms'][iframe])

        if not data['is_periodic']:
            frame.info['total_charge'] = data['total_charge'][iframe]

        if data['prop1'] is not None:
            frame.set_array('prop1', data['prop1'][iframe], dtype=np.float64)
        if data['prop2'] is not None:
            frame.set_array('prop2', data['prop2'][iframe], dtype=np.float64)

        calc = SinglePointCalculator(
            frame,
            energy=data['energy'][iframe] * Hartree,
            forces=data['forces'][iframe] * Hartree / Bohr,
            charges=data['charges'][iframe],
        )
        frame.calc = calc
        input_data.append(frame)

    testfile_name = 'input.data_test'
    write(testfile_name, input_data, format='runnerdata', fmt='20.7f')

    with open(testfile_name, 'rt', encoding='utf-8') as infile:
        assert infile.read() == data['ref_output']
