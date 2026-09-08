import pytest

from ase.io import read
from ase.io.formats import UnknownFileTypeError


@pytest.mark.parametrize(
    'name, file_data',
    [
        (
            'minimal_valid',
            b'begin\natom 1.0 2.0 3.0 H 0.0 0.0 0.0 0.0 0.0\nend',
        ),
        (
            'extended_keys_and_lattice',
            b'begin position(3) element velocity(3) mass\n'
            b'lattice 10.0 0.0 0.0\n'
            b'lattice 0.0 10.0 0.0\n'
            b'lattice 0.0 0.0 10.0\n'
            b'atom 1.0 2.0 3.0 Fe 0.0 0.0 0.0 1.0\n',
        ),
        (
            'float_varieties',
            b'begin position(3) element\n'
            # Matches: trailing dot (12.), leading dot (.5), scientific (1.5e-3)
            b'atom 12. .5 1.5e-3 Pr\n',
        ),
        (
            'all_comment_types_interspersed',
            b'# Start of file comment\n'
            b'! Another comment format\n'
            b'begin position(3) element\n'
            b'comment This is a word comment\n'
            b'lattice 10.0 0.0 0.0\n'
            b'lattice 0.0 10.0 0.0\n'
            b'lattice 0.0 0.0 10.0\n'
            b'# Interspersed comment\n'
            b'atom -1.0 +2.0 3 Fe\n',
        ),
        ('windows_newlines', b'begin\r\natom 0 0 0 C 0 0 0 0 0\r\n'),
        (
            'mixed_tabs_and_spaces',
            b'begin position(3) element\n  atom\t 1.0   2.0\t3.0  He\n',
        ),
    ],
)
def test_valid_formats(tmp_path, name, file_data):
    # Use a neutral file extension so ASE cannot guess by filename
    test_file = tmp_path / 'unknown_file.txt'
    test_file.write_bytes(file_data)

    try:
        # If magic_regex works, this is identified as 'runnerdata' and calls the
        # parser
        atoms = read(test_file)
        assert atoms is not None
    except UnknownFileTypeError:
        pytest.fail(
            f'ASE failed to recognize the format using magic_regex for: {name}'
        )


@pytest.mark.parametrize(
    'name, file_data',
    [
        (
            'invalid_element_lowercase',
            b'begin position(3) element\natom 1.0 2.0 3.0 fe\n',
        ),
        (
            'invalid_element_too_many_caps',
            b'begin position(3) element\natom 1.0 2.0 3.0 HEllo\n',
        ),
        (
            'invalid_element_too_long',
            b'begin position(3) element\natom 1.0 2.0 3.0 Uss\n',
        ),
        (
            'missing_begin_keyword',
            b'lattice 10.0 0.0 0.0\natom 1.0 2.0 3.0 H\n',
        ),
        (
            'bad_float_in_atom',
            b'begin position(3) element\natom 1.0 two 3.0 H\n',
        ),
        (
            'wrong_begin_keys',
            b'begin positions(wrong) element\natom 1.0 2.0 3.0 H\n',
        ),
        ('missing_atom_line', b'begin position(3) element\n'),
    ],
)
def test_invalid_formats(tmp_path, name, file_data):
    test_file = tmp_path / 'unknown_file.txt'
    test_file.write_bytes(file_data)

    # Because the magic_regex should fail, ASE will exhaust all other formats
    # and eventually throw an UnknownFileTypeError.
    with pytest.raises(UnknownFileTypeError):
        read(test_file)
