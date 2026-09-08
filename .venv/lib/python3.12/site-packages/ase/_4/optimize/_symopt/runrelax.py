"""Test on structures from C2DB."""

from sys import argv

from ase._4.optimize._symopt.relax import Relax
from ase.parallel import world


def main():
    # Tests:
    # Wurtzite, distorted structure, nice logging, quick convergence

    if world.rank == 0:
        import requests  # type: ignore[import-untyped]

        url = f'https://c2db.fysik.dtu.dk/material/{argv[1]}/download/xyz'
        print(url)
        request = requests.get(url)
        with open('atoms.xyz', 'wb') as f:
            f.write(request.content)
        print('Written to atoms.xyz')
    world.barrier()
    # atoms = bulk("NaCl", "rocksalt", a=5.2)
    # atoms = bulk("ZnO", crystalstructure="wurtzite", a=3.24, c=5.20)
    # atoms = bulk("ZnO", crystalstructure="wurtzite", a=3.14, c=5.30)
    # Avoid rotating the cell (making it symmetric)
    # eps = np.array([[0,1,0], [1,0,0], [0,0, 0]]) * 0.02
    # atoms.set_cell(atoms.get_cell() @ (np.eye(3) + eps + eps.T))
    # atoms.rattle(0.1)
    from ase.io import read

    # atoms = read('2AlCl3-1.xyz').copy()
    atoms = read('atoms.xyz').copy()
    atoms.center()

    def calc():
        from gpaw.new.ase_interface import GPAW

        return GPAW(
            mode={'name': 'pw', 'ecut': 800},
            kpts={'density': 4, 'gamma': True},
            symmetry={'symmorphic': False},
            txt='ZnO.txt',
            xc='PBE',
            convergence={'density': 1e-7},
        )

    from ase.optimize.bfgs import BFGS

    relax = Relax(
        atoms=atoms,
        calc=calc,
        optimizer_factory=lambda atoms: BFGS(
            atoms, maxstep=0.5, logfile='bfgs.log', trajectory='a.traj'
        ),
        symprec=0.003,
        logfile='relax.log',
        teelog=True,
        comm=world,
    )

    relax.run(fmax=0.01, smax=0.0005)


if __name__ == '__main__':
    main()
