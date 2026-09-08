from dataclasses import asdict, dataclass

import numpy as np
import scipy

from ase import Atoms
from ase._4.optimize._symopt.relax_print import (
    pprint_atoms,
    pretty,
    pretty_atomic_dofs,
    pretty_dofs,
    pretty_header,
    pretty_subheader,
)
from ase.cell import Cell
from ase.utils import spglib_new_errorhandling
from ase.utils.abc import Optimizable


def green(text: str) -> str:
    return f'\x1b[32m{text}\x1b[0m'


def argsort_rotations(rotations):
    return sorted(
        range(len(rotations)),
        key=lambda i: tuple(rotations[i].reshape(9).tolist()),
    )


def snap_translations_to_zero(translations, tol=1e-13):
    # spglib may produce translations that are +/- 1e-15, we want those to be 0
    # It may also produce 1 (+/- 1e-15), and we want those to be zero as well.

    # If a translation is 1, it means a whole lattice vector and that we
    # can also use 0 instead (no matter the corresponding rotation).
    cleaned_translations = translations - np.rint(translations)
    cleaned_translations[abs(cleaned_translations) < tol] = 0.0
    assert (abs(cleaned_translations) <= 0.5).all()
    return cleaned_translations


def chol_derivative(A, dA):
    """
    Compute the derivative of the Cholesky factorization.

        A = L L^T

    where L is lower-triangular. For a small symmetric perturbation `dA`,
    this function returns an approximation of the differential `dL` of
    the Cholesky factor:

        L + dL ≈ chol(A + dA).

    Either L or A must be given as an input.
    """
    A = (A + A.T) / 2
    dA = (dA + dA.T) / 2

    L = np.linalg.cholesky(A)
    Linv = np.linalg.inv(L)
    S = Linv @ dA @ Linv.T
    X = np.tril(S)
    X[np.diag_indices_from(X)] *= 0.5
    return L @ X


def symmetrize_atoms(
    S_ac: np.ndarray, U_scc: np.ndarray, f_sc: np.ndarray, atommap_sa, tol=1e-12
):
    """
    Symmetrize fractional atomic coordinates under a space-group.

    Given atomic scaled positions `S_ac` and a set of space-group operations
    (U_scc, f_sc), this function projects the positions onto the symmetry-
    invariant subspace by averaging over all symmetry-related images.

    Parameters
    ----------
    S_ac : ndarray, shape (na, 3)
        Scaled atomic coordinates.

    U_scc : ndarray, shape (ns, 3, 3)
        Rotation matrices.

    f_sc : ndarray, shape (ns, 3)
        Translation vectors.

    atommap_sa : ndarray, shape (ns, na)
        Mapping such that atommap_sa[s, a] gives the index of the atom
        to which atom `a` is mapped by symmetry operation `s`.

    tol : float, optional
        Tolerance for snapping values close to 0 or 1 back to 0.

    Returns
    -------
    Ssym_ac : ndarray, shape (na, 3)
        Symmetrized fractional atomic coordinates in [0, 1).

    Notes
    -----
    The symmetrization is done in the complex phase representation
    exp(2πi x) to correctly average periodic fractional coordinates.
    """
    ns, na = atommap_sa.shape
    complex_Ssym_ac = np.zeros_like(S_ac, dtype=np.complex128)
    for a in range(na):
        for s in range(ns):
            new = U_scc[s].T @ S_ac[a] - f_sc[s]
            complex_Ssym_ac[atommap_sa[s, a]] += np.exp(2j * np.pi * new)
    Ssym_ac = (np.angle(complex_Ssym_ac) / (2 * np.pi)) % 1.0 % 1.0
    Ssym_ac[np.abs(Ssym_ac) < tol] = 0.0
    Ssym_ac[np.abs(Ssym_ac - 1.0) < tol] = 0.0
    return Ssym_ac


def apply_pbc_to_symops(rotations, translations, pbc):
    """spglib thinks we are a 3d system, but actually we may be 2D/1D.

    spglib will produce symops that involve e.g. a z-flip, but that
    flips the positions to negative z-values, and we need to apply the
    full z axis as translation vector in order to map the position back
    into the cell.

    This function equips such operations with the necessary translations.

    We also do some sanity checking.
    """
    translations = translations.copy()
    assert len(rotations) == len(translations)

    for i in range(3):
        if pbc[i]:
            continue

        for s in range(len(rotations)):
            nonpbc_rot = rotations[s, i, i]
            assert nonpbc_rot in (-1, 1)
            nonpbc_op_row = np.zeros(3, int)
            nonpbc_op_row[i] = nonpbc_rot
            assert (rotations[s, :, i] == nonpbc_op_row).all()
            assert (rotations[s, i, :] == nonpbc_op_row).all()
            if nonpbc_rot == -1:
                assert abs(translations[s, i]) < 1e-15, translations[s, i]
                translations[s, i] = 1.0

    return translations


@dataclass
class AtomsSymmetries:
    """
    Dataclass to contain symmetry information from atoms.

    This is to set up an interface for spglib/GPAW or whatever
    source of symmetry operations.
    """

    rotation_scc: np.ndarray
    atommap_sa: np.ndarray
    translation_sc: np.ndarray

    def __post_init__(self):
        ns = len(self.rotation_scc)
        assert self.rotation_scc.shape == (ns, 3, 3)
        assert len(self.atommap_sa) == ns
        assert self.translation_sc.shape == (ns, 3)

    @classmethod
    def from_spglib_auto(cls, atoms, *, symprec=1e-4):
        from spglib import get_symmetry_dataset

        dataset = spglib_new_errorhandling(get_symmetry_dataset)(
            (atoms.cell, atoms.get_scaled_positions(), atoms.numbers),
            symprec=symprec,
        )
        return cls.from_spglib(atoms, dataset)

    @classmethod
    def from_spglib(cls, atoms, dataset):
        from ase._4.optimize._symopt.atommap import build_atommaps

        args = argsort_rotations(dataset.rotations.transpose(0, 2, 1))

        rotations = dataset.rotations[args]
        translations = apply_pbc_to_symops(
            dataset.rotations, dataset.translations, pbc=atoms.pbc
        )[args]
        translations = snap_translations_to_zero(translations)

        return cls(
            rotation_scc=rotations.transpose(0, 2, 1).copy(),
            translation_sc=-translations,
            atommap_sa=build_atommaps(
                atoms.copy(),
                rotations,
                translations.copy(),
            ),
        )

    @classmethod
    def from_GPAW(cls, atoms, *, log, tolerance, symmorphic):
        from gpaw.new.symmetry import create_symmetries_object

        gpaw_symmetries = create_symmetries_object(
            atoms, tolerance=tolerance, symmorphic=symmorphic
        )
        args = argsort_rotations(gpaw_symmetries.rotation_scc)
        log(gpaw_symmetries)
        return cls(
            gpaw_symmetries.rotation_scc[args],
            gpaw_symmetries.atommap_sa[args],
            gpaw_symmetries.translation_sc[args],
        )


# The symmetry-reduced coordinates for positions and cell
# have an arbitrary scale.
#
# It is nice that both sets of coordinates produce human-friendly
# numbers of approximately the same magnitude in typical cases.  To
# this effect, we scale the cell coordinates by this number.  This
# actually affects the run if there's a specific initial Hessian, step
# limitation using "maxstep", etc., but any value is in principle
# valid.
CELL_COORDINATE_SCALING_FACTOR = 40


@dataclass
class SymmetryAdaptedCellCoordinates:
    """Class for defining symmetry adapted cell coordinates

    Note: This is not symmetry adapted cell, it just provides the set of
    generalized coordinates for the symmetry adapted cell.
    To get the cell, call C_cv = get_cell(cell_z).

    sacc = SymmetryAdaptedCellCoordinates(...)
    sacc.get_cell(cell_z), where cell_z is 1D array of the cell coordinates.

    Thus, the M_cc and C_cv here is just the origin of the coordinate system.
    """

    # Symmetrized cell
    M_cc: np.ndarray  # Rename to M0_cc
    C_cv: np.ndarray  # Rename to C0_cv
    dM_zcc: np.ndarray
    dM_zvv: np.ndarray
    rot_vv: np.ndarray

    def get_cell(self, cell_z):
        """
        Construct the real-space unit cell from symmetry-adapted coordinates.

        Given generalized cell coordinates `cell_z`, this method reconstructs
        the metric tensor (see get_M_cc) and then computes a corresponding
        cell matrix C_cv via a Cholesky factorization of M_cc,
        followed by a fixed rotation `rot_vv`:

            C_cv = chol(M_cc) @ rot_vv.T

        Parameters
        ----------
        cell_z : ndarray, shape (nz,)
            Symmetry-adapted cell coordinates.

        Returns
        -------
        cell : ase.geometry.Cell
            The reconstructed unit cell.
        """

        M_cc = self.get_M_cc(cell_z)
        try:
            C_cv = np.linalg.cholesky(M_cc) @ self.rot_vv.T
        except np.linalg.LinAlgError:
            raise RuntimeError('Failed to create cell from metric', M_cc)

        return Cell(C_cv)

    def get_M_cc(self, cell_z):
        """
        Reconstruct the metric tensor from symmetry-adapted coordinates.

        Computes the metric tensor as a linear expansion around a reference
        metric M0_cc in the symmetry-allowed tangent directions:

            M_cc = M0_cc + sum_z cell_z[z] * dM_zcc[z]

        Parameters
        ----------
        cell_z : ndarray, shape (nz,)
            Symmetry-adapted cell coordinates.

        Returns
        -------
        M_cc : ndarray, shape (3, 3)
            Symmetrized metric tensor corresponding to `cell_z`.
        """
        return self.M_cc + np.einsum('z,zcd->cd', cell_z, self.dM_zcc)

    @classmethod
    def build(cls, cell, pbc_c, rotation_scc: np.ndarray, *, log):
        return cls(*cls.unit_cell_symmetry(cell, rotation_scc, pbc_c, log=log))

    @classmethod
    def symmetrize_cell(cls, C_cv, rotation_scc):
        """Symmetrize the cell

        Calculates the cell metric, and applies the rotation operations to it.
        New cell lower diagonal cell is calculated via Cholesky decomposition.
        By doing polar decomposition to the deformation gradient, the rotation
        back to the original cell rotation is obtained.

        Returns osymC_cV, symC_cv, M_cc, rot_vv

        Where osymC_cv is the symmetrized original like cell
        symC_cv is the lower diagonal symmetrized cell
        M_cc is the symmetrized cell metric
        rot_vv is the rotation matrix between osymC_Cv and symC_cv
            such that osymC_cv = symC_cv @ rot_vv.T

        """
        # Calculate the cell metric
        M_cc = C_cv @ C_cv.T
        # Symmetrize the cell metric
        M_cc = np.einsum(
            'scd,de,sfe->cf', rotation_scc, M_cc, rotation_scc, optimize=True
        ) / len(rotation_scc)

        symC_cv = np.linalg.cholesky(M_cc)

        # Deformation gradient
        F_vv = np.linalg.inv(C_cv) @ symC_cv

        # Sanity check
        assert np.allclose(C_cv @ F_vv, symC_cv)

        # Do a polar decomposition to rotate the symmetrized cell back
        rot_vv, P_vv = scipy.linalg.polar(F_vv)
        osymC_cv = symC_cv @ rot_vv.T

        return osymC_cv, symC_cv, M_cc, rot_vv

    @classmethod
    def unit_cell_symmetry(cls, C_cv, rotation_scc, pbc_c, units='Å^2', *, log):
        pretty(
            C_cv @ C_cv.T, "Cell metric (M_cc' = C_cv C_c'v)", units, log=log
        )
        osymC_cv, symC_cv, M_cc, rot_vv = cls.symmetrize_cell(
            C_cv, rotation_scc
        )
        pretty(
            M_cc, "Symmetrized cell metric (M_cc' = C_cv C_c'v)", units, log=log
        )

        # Now we can construct exact Cartesian rotation matrices
        iosymC_cv = np.linalg.inv(osymC_cv)
        U_svv = [osymC_cv.T @ U_cc.T @ iosymC_cv.T for U_cc in rotation_scc]
        U_svv = np.array(U_svv)

        # Build unit vector in symmetric matrix space
        def e(i, j):
            eps_ij = np.zeros((3, 3))
            eps_ij[i, j] = 1.0
            eps_ij[j, i] = 1.0
            return eps_ij

        eps_ijk = np.zeros((3, 3, 6))
        k = 0
        for i in range(3):
            for j in range(i, 3):
                if i == j:
                    s = 1.0
                else:
                    s = 2 ** (-0.5)
                eps_ijk[i, j, k] = s
                eps_ijk[j, i, k] = s
                k += 1

        A_blocks = []
        for U_vv in U_svv:
            rows = []
            for k in range(6):
                row = U_vv @ eps_ijk[:, :, k] @ U_vv.T - eps_ijk[:, :, k]
                rows.append(row.reshape(9))
            A_blocks.append(np.vstack(rows))
        for c in range(3):
            if not pbc_c[c]:
                for cprime in range(3):
                    A_blocks.append(e(c, cprime).reshape(9))
        A = np.vstack(A_blocks)
        A = A @ eps_ijk.reshape((9, 6))
        # Compute null space via SVD
        U, S, Vh = np.linalg.svd(A)
        tol = 1e-6
        null_mask = S < tol
        nullspace = Vh[null_mask]
        dM_zcc = []
        dM_zvv = []
        for B in nullspace:
            B = B @ eps_ijk.reshape((9, 6)).T
            dM_vv = B.reshape((3, 3))
            dof = osymC_cv @ dM_vv @ osymC_cv.T
            dM_zcc.append(dof)
            dM_zvv.append(rot_vv @ dM_vv @ rot_vv.T)
        dM_zcc = np.array(dM_zcc).reshape((-1, 3, 3))
        dM_zvv = np.array(dM_zvv).reshape((-1, 3, 3))

        # Do a QR decomposition, try to get more zeros to coordinates
        basis = np.array(dM_zcc).reshape((-1, 9))
        Q, R = np.linalg.qr(basis)
        dM_zcc = (Q.T @ basis).reshape((-1, 3, 3))

        # Normalize tangent space vectors
        Cinv = np.linalg.inv(C_cv)
        for z in range(len(dM_zcc)):
            dC = chol_derivative(M_cc, dM_zcc[z]) @ rot_vv.T
            eps = 0.5 * (Cinv @ dC + dC.T @ Cinv.T)

            dM_zcc[z] /= np.sum(np.abs(eps)) * np.linalg.det(C_cv)
            dM_zcc[z] *= CELL_COORDINATE_SCALING_FACTOR
            # Define the direction of the tangent vector such that
            # it increases the volume. Sign cannot be used because of shear
            if np.trace(np.linalg.inv(C_cv) @ dC) < 0:
                dM_zcc[z] *= -1

        pretty_dofs(dM_zcc, M_cc, rot_vv, osymC_cv, log=log)

        # TODO: Move U_svv
        return M_cc, osymC_cv, dM_zcc, dM_zvv, rot_vv


@dataclass
class SymmetryAdaptedScaledCoordinates:
    dof_zac: np.ndarray
    s0_ac: np.ndarray

    def get_scaled_coordinates(self, atoms_z: np.ndarray):
        return self.s0_ac + np.einsum('zac,z->ac', self.dof_zac, atoms_z)

    @classmethod
    def build(
        cls,
        s_ac,
        rotation_scc,
        translation_sc,
        atommap_sa,
        C_cv,
        *,
        log,
    ):
        ns, na = atommap_sa.shape
        B_ascac = np.zeros((na, ns, 3, na, 3), int)
        for s, U_cc in enumerate(rotation_scc):
            for a in range(na):
                a2 = atommap_sa[s, a]
                B_ascac[a, s, :, a] = U_cc.T
                B_ascac[a, s, :, a2] -= np.eye(3, dtype=int)
        B_EA = B_ascac.reshape((na * ns * 3, na * 3))
        # Extra translational gauge degrees of freedom
        B_A = np.zeros((na * 3, 3))
        for a in range(na):
            B_A[(a * 3) : (a * 3 + 3), :] = np.eye(3)
        B_EA = np.vstack([B_EA, B_A.T])

        # Make sure the old svd code reproduces the same result
        U, S, Vh = np.linalg.svd(B_EA, False)
        tol = 1e-6
        null_mask = S < tol
        nullspace = Vh[null_mask]

        # def same_rowspace(N, M, tol=1e-10):
        #    A = np.vstack([N, M])
        #    rA = np.linalg.matrix_rank(A, tol)
        #    rN = np.linalg.matrix_rank(N, tol)
        #    rM = np.linalg.matrix_rank(M, tol)
        #    return rA == rN == rM
        # assert same_rowspace(nullspace, nullspace2)

        # Just make the printing prettyer for now
        nullspace = np.where(np.abs(nullspace) < 1e-10, 0, nullspace)

        s0_ac = symmetrize_atoms(
            s_ac,
            rotation_scc,
            translation_sc,
            atommap_sa,
        )

        log(f'Atomic degrees of freedom: {len(nullspace)}')

        dof_zac = nullspace.reshape((-1, na, 3))

        if len(dof_zac):
            dof_zav = np.einsum('zac,cv->zav', dof_zac, C_cv)
            # Normalize such that the distance in Cartesian real space
            # is reflected on the generalized coordinate
            dof_zac /= np.max(np.linalg.norm(dof_zav, axis=2), axis=1)[
                :, None, None
            ]

        return SymmetryAdaptedScaledCoordinates(dof_zac, s0_ac)


class SymmetryAdaptedAtoms(Optimizable):
    """Implementation of symmetry adapted atoms

    Symmetry adapted atoms WILL symmetrize the actual_atoms given to init.

    SymmetryAdaptedAtoms does not behave like Atoms object, but will expose the
    __ase_optimizable__ protocol, so it can be optimized with ASE.
    """

    def __init__(
        self,
        actual_atoms: Atoms,
        symmetries: AtomsSymmetries,
        log=print,
        cell_coordinates: SymmetryAdaptedCellCoordinates | None = None,
        atom_coordinates=None,
        value_z=None,
    ):
        self.actual_atoms = actual_atoms
        self.symmetries = symmetries

        # XXX remove these attributes:
        self.symmetry_force_violation = np.inf
        self.fmax = 0.01

        if cell_coordinates is None:
            pretty_subheader('Symmetry adapted cell coordinates', log)
            cell_coordinates = SymmetryAdaptedCellCoordinates.build(
                self.actual_atoms.cell,
                self.actual_atoms.pbc,
                self.symmetries.rotation_scc,
                log=log,
            )

        self.cell_coordinates = cell_coordinates

        if atom_coordinates is None:
            pretty_subheader('Symmetry adapted atomic coordinates', log)
            atom_coordinates = SymmetryAdaptedScaledCoordinates.build(
                self.actual_atoms.get_scaled_positions(),
                self.symmetries.rotation_scc,
                self.symmetries.translation_sc,
                self.symmetries.atommap_sa,
                self.cell_coordinates.C_cv,
                log=log,
            )
            pretty_atomic_dofs(actual_atoms, atom_coordinates.dof_zac, log=log)

        assert isinstance(atom_coordinates, SymmetryAdaptedScaledCoordinates)
        self.atom_coordinates = atom_coordinates

        # s_ac = dof_zac s_z -> ds_ac/d_sz = dof_zac
        # dR_av / dsz = dR_av / d_sac ds_ac / ds_z
        # R_av = s_ac C_cv
        #
        # self.actual_atoms.set_cell(self.cell_coordinates.C_cv,
        #                           scale_atoms=True)
        #
        # self.actual_atoms.wrap()
        # self.actual_atoms.set_scaled_positions(self.S_ac)
        # if 1:
        #     log('Skipping sanity checks for now')
        # else:
        #     pass
        #    # new_positions = atoms.get_positions()
        #    # dR_av = new_positions - old_positions
        #    # s_ac = np.linalg.solve(self.C_cv, dR_av.T)
        #    # assert (
        #    #     np.max(np.abs(new_positions.flatten() -
        #    #                   old_positions.flatten()))
        #    #     < symprec
        #    # )

        self._ndofs_cell = len(self.cell_coordinates.dM_zcc)
        self._ndofs_atoms = len(self.atom_coordinates.dof_zac)
        self._ndofs = self._ndofs_cell + self._ndofs_atoms

        if value_z is None:
            value_z = np.zeros((self._ndofs))

        self.value_z = value_z
        # !!! This actually symmetrizes actual atoms
        self.set_x(self.value_z)
        self.actual_atoms.wrap()

    def datafy(self):
        return {
            'actual_atoms': self.actual_atoms,
            'symmetries': asdict(self.symmetries),
            'value_z': self.value_z,
            'cell_coordinates': asdict(self.cell_coordinates),
            'atom_coordinates': asdict(self.atom_coordinates),
        }

    @classmethod
    def undatafy(cls, dct, calc):
        dct = dct.copy()
        dct['symmetries'] = AtomsSymmetries(**dct['symmetries'])
        dct['cell_coordinates'] = SymmetryAdaptedCellCoordinates(
            **dct['cell_coordinates']
        )
        dct['atom_coordinates'] = SymmetryAdaptedScaledCoordinates(
            **dct['atom_coordinates']
        )
        dct['actual_atoms'].calc = calc
        return cls(**dct, log=lambda *args, **kwargs: None)

    @classmethod
    def from_atoms_spglib(cls, atoms, log=print, *, symprec):
        symmetries = AtomsSymmetries.from_spglib_auto(atoms, symprec=symprec)
        return cls(atoms, symmetries, log=log)

    @classmethod
    def from_spglib_dataset(cls, atoms, dataset, log=print):
        # XXX This will fail in atommaps if atoms are not exactly symmetrized.
        # Better to not fail now, but rather wait until after the
        # symmetrization and then check the atommaps against the
        # symmetrized atoms with a fine tolerance.
        #
        # Here we can only promise ~symprec tolerance.
        symmetries = AtomsSymmetries.from_spglib(atoms, dataset)
        return cls(atoms, symmetries, log=log)

    @classmethod
    def from_atoms(cls, atoms, log=print, *, symprec, symmorphic):
        symmetries = AtomsSymmetries.from_GPAW(
            atoms,
            tolerance=symprec,
            symmorphic=symmorphic,
            log=log,
        )
        return cls(atoms, symmetries, log=log)

    @property
    def stress_conv(self):
        S_vv = self.actual_atoms.get_stress(voigt=False)
        C_cv = self.actual_atoms.cell
        S_cc = C_cv @ S_vv @ np.linalg.inv(C_cv)
        for c, periodic in enumerate(self.actual_atoms.pbc):
            if periodic:
                continue
            S_cc[c, :] = 0.0
            S_cc[:, c] = 0.0
        S_vv = np.linalg.inv(C_cv) @ S_cc @ C_cv
        return np.linalg.norm(S_vv)

    # Properties for internal degrees of freedom
    @property
    def cell_z(self):
        return self.get_x()[: self._ndofs_cell]

    # From here on out, these are the __ase_optimizable__ interface
    def ndofs(self):
        return self._ndofs

    def get_x(self):
        return self.value_z.copy()

    def set_x(self, x):
        self.value_z[:] = x
        self.actual_atoms.set_cell(self.cell_coordinates.get_cell(self.cell_z))

        self.actual_atoms.set_scaled_positions(
            self.atom_coordinates.get_scaled_coordinates(self.atoms_z)
        )

    def get_Fback_av(self, F_av, C_cv, atoms_grad_z):
        natomz = len(self.atom_coordinates.dof_zac)
        # For sanity check, we want to project the atomic gradient back
        # minimizing the Cartesian metrix.
        if natomz > 0:
            dof_zX = np.einsum(
                'cv,zac->zav', C_cv, self.atom_coordinates.dof_zac
            ).reshape((natomz, -1))
            back_Fav = -(
                dof_zX.T @ np.linalg.inv(dof_zX @ dof_zX.T) @ atoms_grad_z
            ).reshape(F_av.shape)
        else:
            # Even if there is degrees of freedom, it is possible to
            # get symmetry violation
            back_Fav = np.zeros_like(F_av)
        return back_Fav

    def get_Sback_vv(self, S_vv, A_zV, grad_z):
        ncellz = len(A_zV)
        if ncellz > 0:
            return (A_zV.T @ np.linalg.inv(A_zV @ A_zV.T) @ grad_z).reshape(
                (3, 3)
            )

        # Even if there are no cell degrees of freedom, the stress can
        # still violate the imposed symmetry in Cartesian space.
        return np.zeros((3, 3))

    def _get_gradient(self, fmax, smax):
        grad_z = np.zeros(self._ndofs_cell)
        S_vv = self.actual_atoms.get_stress(voigt=False)
        C_cv = self.cell_coordinates.get_cell(self.cell_z)
        V = np.linalg.det(C_cv)
        Cinv = np.linalg.inv(C_cv)

        M_cc = self.cell_coordinates.get_M_cc(self.cell_z)

        # TODO: Move to SymmetryAdaptedCellCoordinates
        # dE/deps_vv deps_vv/dC_cv dC_cv/dz

        ncellz = len(self.cell_coordinates.dM_zcc)
        A_zV = np.zeros((ncellz, 9))
        for z in range(ncellz):
            dC_cv = (
                chol_derivative(M_cc, self.cell_coordinates.dM_zcc[z])
                @ self.cell_coordinates.rot_vv.T
            )
            A_zV[z] = (V * (Cinv @ dC_cv + dC_cv.T @ Cinv.T) / 2).reshape(9)

        grad_z[:] = np.dot(A_zV, S_vv.reshape(9))
        back_Svv = self.get_Sback_vv(S_vv, A_zV, grad_z)

        def get_mask_3x3(pbc):
            from ase._4.optimize.frechet import default_mask
            from ase.stress import voigt_6_to_full_3x3_stress

            mask_6 = default_mask(pbc)
            return voigt_6_to_full_3x3_stress(mask_6)

        mask_3x3 = get_mask_3x3(self.actual_atoms.pbc)

        dS_vv = S_vv * mask_3x3 - back_Svv
        symmetry_stress_violation = np.linalg.norm(dS_vv)

        if symmetry_stress_violation > smax:
            import warnings

            warnings.warn(
                'Warning!!! Back projection of symmetry adapted '
                'stresses to Cartesian space failed by '
                f'{symmetry_stress_violation:7.13f}\n'
                f'Cartesian stress:\n{S_vv * mask_3x3}\n'
                f'Back projected stress:\n{back_Svv}'
            )

        F_av = self.actual_atoms.get_forces()
        # dE/ds_z = dE/dR_av dR_av/ds_ac ds_ac/ds_z
        # R_av = ds_ac C_cv
        # ds_ac = self.dof_zac S_z

        atoms_grad_z = -np.einsum(
            'av,cv,zac->z', F_av, C_cv, self.atom_coordinates.dof_zac
        )

        back_Fav = self.get_Fback_av(F_av, C_cv, atoms_grad_z)

        dF_av = F_av - back_Fav
        symmetry_force_violation = np.max(np.linalg.norm(dF_av, axis=1))

        if symmetry_force_violation > fmax / 20:
            # Should probably be logged somehow instead of being a warning
            # as such.  This may happen if the code's forces are noisy.
            import warnings

            warning_chunks = [
                'Warning!!! Back projection of symmetry adapted'
                ' forces to Cartesian space failed by '
                f'{symmetry_force_violation:7.13f}\n'
                'atom Obtained force           Back projected force'
            ]

            for a, (F_v, F2_v) in enumerate(zip(F_av, back_Fav)):
                warning_chunks.append(
                    f'{a:5d} {F_v[0]:7.4f} {F_v[1]:7.4f} {F_v[2]:7.4f}'
                    f' {F2_v[0]:7.4f} {F2_v[1]:7.4f} {F2_v[2]:7.4f}'
                )

            warnings.warn('\n'.join(warning_chunks))

        converged = self._converged(back_Fav, fmax, back_Svv, smax)

        from ase._4.optimize.symopt import SymGradient

        return SymGradient(
            gradient=np.hstack([grad_z, atoms_grad_z]),
            backF_av=back_Fav,
            backS_vv=back_Svv,
            symmetry_force_violation=symmetry_force_violation,
            symmetry_stress_violation=symmetry_stress_violation,
            converged=converged,
            fnorm=self._fnorm(back_Fav),
            snorm=self._snorm(back_Svv),
            volume=self.actual_atoms.cell.volume,
        )

    def get_gradient(self):
        grad_obj = self._get_gradient(fmax=self.fmax, smax=self.smax)
        self.back_Fav = grad_obj.backF_av
        self.back_Svv = grad_obj.backS_vv
        self.symmetry_force_violation = grad_obj.symmetry_force_violation
        self.symmetry_stress_violation = grad_obj.symmetry_stress_violation
        return grad_obj.gradient

    def gradient_norm(self, grad_z):
        # Go actually to cell metric
        return np.max(np.abs(grad_z))

    def get_value(self):
        return self.actual_atoms.get_potential_energy()

    def iterimages(self):
        return [self.actual_atoms]

    def converged(self, gradient, fmax):
        # Convergence needs to be from the back projected forces.
        # The symmetry violating forces will never converge.
        return self._converged(
            backF_av=self.back_Fav,
            fmax=self.fmax,
            smax=self.smax,
            backS_av=self.stress_conv,
        )  # XXX use backS_av

    def _fnorm(self, backF_av):
        return np.max(np.linalg.norm(backF_av, axis=1))

    def _snorm(self, backS_av):
        return abs(backS_av).max()

    def _converged(self, backF_av, fmax, backS_av, smax):
        Fconv = self._fnorm(backF_av)
        return Fconv < fmax and self._snorm(backS_av) < smax

    @property
    def atoms_z(self):
        return self.get_x()[self._ndofs_cell :]

    def logheaders(self):
        dtitles = '    '.join([f'q{i:02d}' for i in range(len(self.atoms_z))])
        return (
            f'iter  time     E           maxF   maxS     maxG   a1    a2'
            f'    a3    L1      L2       L3     {dtitles}'
            '   log_10 viol. (F / S)'
        )

    def step_to_string(self, i, fmax, smax, gradient_obj):
        import time

        tstr = time.strftime('%H:%M:%S')
        E = self.actual_atoms.get_potential_energy()
        g = gradient_obj.gradient  # XXX clean up
        Fmax = gradient_obj.fnorm
        sFmax = f'{Fmax:7.3f}'

        if Fmax < fmax:
            sFmax = green(sFmax)

        Smax = gradient_obj.snorm
        sSmax = f'{Smax:7.4f}'
        if Smax < smax:
            sSmax = green(sSmax)

        gmax = np.max(np.abs(g))

        cell = self.actual_atoms.cell
        a = cell.angles()
        l = cell.lengths()
        cell = f'{a[0]:5.1f} {a[1]:5.1f} {a[2]:5.1f} '
        cell += f'{l[0]:7.3f} {l[1]:7.3f} {l[2]:7.3f}'

        dofs = ''
        for Z in self.atoms_z:
            dofs += f' {Z:6.3f}'
        if gradient_obj.symmetry_force_violation:
            syviol = np.log10(gradient_obj.symmetry_force_violation)
        else:
            syviol = -np.inf
        symviol = f'{syviol:4.1f}'
        if syviol < fmax:
            symviol = green(symviol)
        if gradient_obj.symmetry_stress_violation:
            syviol2 = np.log10(gradient_obj.symmetry_stress_violation)
        else:
            syviol2 = -np.inf
        symviol2 = f'{syviol2:4.1f}'
        if syviol2 < smax:
            symviol2 = green(symviol2)
        return (
            f'{i:5d} {tstr} {E:9.5f} {sFmax} {sSmax} {gmax:7.3f}'
            f' {cell}{dofs} {symviol} {symviol2}'
        )


def print_headers(atoms, log):
    pretty_header('Symmetry adapted Cell and Atomic Relaxation', log)
    pretty_subheader('Original atoms', log)
    pprint_atoms(atoms, log)


def print_symmetrized_atoms(atoms, log):
    pretty_subheader('Symmetrized atoms', log)
    pprint_atoms(atoms, log)


class Relax:
    """General utility class to log and perform symmetry adapted relax"""

    def __init__(
        self,
        symmorphic=False,
        logfile=None,
        teelog=True,
        *,
        atoms: Atoms,
        calc,
        optimizer_factory,
        symprec,
        comm,
        atoms_symmetries=None,
    ):
        self.comm = comm
        self.logfile = logfile
        self.logf = None
        if self.logfile:
            if self.comm.rank == 0:
                self.logf = open(self.logfile, 'w')
        self.teelog = teelog

        if atoms.calc is not None:
            raise ValueError('Do not attach a calculator to Atoms yet.')

        self.symprec = symprec

        self.original_atoms = atoms.copy()
        print_headers(atoms, self.log)

        self.atoms = atoms
        if atoms_symmetries is None:
            atoms_symmetries1 = AtomsSymmetries.from_spglib_auto(
                self.atoms, symprec=symprec
            )
            atoms_symmetries2 = AtomsSymmetries.from_GPAW(
                self.atoms,
                tolerance=symprec,
                symmorphic=symmorphic,
                log=self.log,
            )

            assert len(atoms_symmetries1.rotation_scc) == len(
                atoms_symmetries2.rotation_scc
            )

            for rot1, rot2 in zip(
                atoms_symmetries1.rotation_scc, atoms_symmetries2.rotation_scc
            ):
                assert (rot1.ravel() == rot2.ravel()).all()

            # for T1, T2 in zip(
            #    atoms_symmetries1.translation_sc,
            #    atoms_symmetries2.translation_sc,
            # ):
            #    err = abs(T2 - T1).max()
            #    assert err < 1e-13, err

            atoms_symmetries = atoms_symmetries1

        self.symmetry_adapted_atoms = SymmetryAdaptedAtoms(
            self.atoms, atoms_symmetries
        )

        # Now, with cell and atoms symmetrized,
        # it is safe to assign the calculator
        # TODO: Implement Setter or something
        self.calc = calc
        self.symmetry_adapted_atoms.actual_atoms.calc = calc()

        print_symmetrized_atoms(
            self.symmetry_adapted_atoms.actual_atoms, self.log
        )

        self.optimizer_factory = optimizer_factory
        self.optimizer = self.optimizer_factory(self.symmetry_adapted_atoms)

    def log(self, *args, **kwargs):
        if self.comm.rank == 0:
            if self.logf:
                print(*args, **kwargs, flush=True, file=self.logf)
            if self.teelog:
                print(*args, **kwargs)

    def run(self, *, fmax=0.01, smax=0.0001, steps=20):
        # Why would symmetry adapted atoms care about smax and fmax
        # But it needs to be ase optimizable, so it needs to do that
        self.symmetry_adapted_atoms.smax = smax
        self.symmetry_adapted_atoms.fmax = fmax

        self.smax = smax
        self.fmax = fmax
        self.maxiter = steps

        self.log(self.symmetry_adapted_atoms.logheaders())

        for i, _ in enumerate(self.optimizer.irun(fmax=fmax)):
            gradient_obj = self.symmetry_adapted_atoms._get_gradient(
                fmax=fmax, smax=smax
            )
            line = self.symmetry_adapted_atoms.step_to_string(
                i, self.fmax, self.smax, gradient_obj
            )
            self.log(line)

            if i > self.maxiter or i > 40:
                self.log(f'Not converged in {self.maxiter} or 40 steps.')
                return False

        return True

    def visualize_modes(self):
        from ase.io.trajectory import Trajectory

        with Trajectory('modes.traj', 'w') as traj:
            for z in range(self.ndofs()):
                x = np.zeros((self.ndofs(),))
                for i in np.arange(0, 6 * np.pi, 0.1):
                    x[z] = np.sin(i) * 0.004
                    self.set_x(x)
                    traj.write(self.atoms.copy())
