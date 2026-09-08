# fmt: off

from dataclasses import dataclass

# Note: There could be more than one "calculator" for any given code;
# for example Espresso can work both as GenericFileIOCalculator and
# SocketIOCalculator, or as part of some DFTD3 combination.
#
# Also, DFTD3 is one external code but can be invoked alone (as PureDFTD3)
# as well as together with a DFT code (the main DFTD3 calculator).
#
# The current CodeMetadata object only specifies a single calculator class.
# We should be wary of these invisible "one-to-one" restrictions.


@dataclass
class CodeMetadata:
    name: str
    longname: str
    modulename: str
    classname: str
    configurable: bool = True
    is_wrapper: bool = False

    def calculator_class(self):
        from importlib import import_module
        module = import_module(self.modulename)
        cls = getattr(module, self.classname)
        return cls

    @classmethod
    def define_code(
        cls,
        name: str,
        longname: str,
        importpath: str,
        configurable: bool = True,
        is_wrapper: bool = False,
    ):
        modulename, classname = importpath.rsplit('.', 1)
        return cls(
            name,
            longname,
            modulename,
            classname,
            configurable=configurable,
            is_wrapper=is_wrapper,
        )

    def _description(self):
        yield f'Name:     {self.longname}'
        yield f'Import:   {self.modulename}.{self.classname}'
        yield f'Type:     {self.calculator_type()}'
        yield from self._wrapper_description()
        yield from self._config_description()

    def description(self, indent=''):
        return '\n'.join(indent + line for line in self._description())

    def is_legacy_fileio(self):
        from ase.calculators.calculator import FileIOCalculator
        return issubclass(self.calculator_class(), FileIOCalculator)

    def is_generic_fileio(self):
        from ase.calculators.genericfileio import CalculatorTemplate

        # It is nicer to check for the template class, since it has the name,
        # but then calculator_class() should be renamed.
        return issubclass(self.calculator_class(), CalculatorTemplate)

    def is_calculator_oldbase(self):
        from ase.calculators.calculator import Calculator
        return issubclass(self.calculator_class(), Calculator)

    def is_base_calculator(self):
        from ase.calculators.calculator import BaseCalculator
        return issubclass(self.calculator_class(), BaseCalculator)

    def calculator_type(self):
        cls = self.calculator_class()

        if self.is_generic_fileio():
            return 'GenericFileIOCalculator'

        if self.is_legacy_fileio():
            return 'FileIOCalculator (legacy)'

        if self.is_calculator_oldbase():
            return 'Calculator (legacy base class)'

        if self.is_base_calculator():
            return 'Base calculator'

        return f'BAD: Not a proper calculator (superclasses: {cls.__mro__})'

    def profile(self):
        from ase.calculators.calculator import FileIOCalculator
        from ase.calculators.genericfileio import CalculatorTemplate
        from ase.config import cfg
        cls = self.calculator_class()
        if issubclass(cls, CalculatorTemplate):
            return cls().load_profile(cfg)
        elif hasattr(cls, 'fileio_rules'):
            assert issubclass(cls, FileIOCalculator)
            return cls.load_argv_profile(cfg, self.name)
        else:
            raise NotImplementedError('profile() not implemented')

    def _wrapper_description(self):
        if self.is_wrapper:
            yield ('')
            yield ('Wraps other Calculator(s)')

    def _config_description(self):
        from ase.calculators.genericfileio import BadConfiguration
        from ase.config import cfg

        if not self.configurable:
            return

        yield ''

        parser = cfg.parser
        if self.name not in parser:
            yield f'Not configured: No [{self.name}] section in configuration'
            return

        try:
            profile = self.profile()
        except BadConfiguration as ex:
            yield f'Error in configuration section [{self.name}]'
            yield 'Missing or bad parameters:'
            yield f'  {ex}'
            return
        except NotImplementedError as ex:
            yield f'N/A: {ex}'
            return

        yield f'Configured by section [{self.name}]:'
        configvars = vars(profile)
        for name in sorted(configvars):
            yield f'  {name} = {configvars[name]}'

        return


def register_codes():

    codes = {}

    def reg(name, *args, configurable=False, is_wrapper=False):
        code = CodeMetadata.define_code(
            name, *args, configurable=configurable, is_wrapper=is_wrapper
        )
        codes[name] = code

    reg('abinit', 'Abinit',
        'ase.calculators.abinit.AbinitTemplate', configurable=True)
    reg('ace', 'ACE molecule', 'ase.calculators.acemolecule.ACE',
        configurable=True)
    reg('acn', 'ACN force field', 'ase.calculators.acn.ACN')
    reg('aims', 'FHI-Aims', 'ase.calculators.aims.AimsTemplate',
        configurable=True)
    reg('amber', 'Amber', 'ase.calculators.amber.Amber', configurable=True)
    reg('average', 'AverageCalculator',
        'ase.calculators.mixing.AverageCalculator', is_wrapper=True)
    reg('castep', 'Castep', 'ase.calculators.castep.Castep', configurable=True)
    reg('checkpoint', 'CheckpointCalculator',
        'ase.calculators.checkpoint.CheckpointCalculator', is_wrapper=True)
    reg('combine_mm', 'CombineMM',
        'ase.calculators.combine_mm.CombineMM', is_wrapper=True)
    reg('counterion', 'AtomicCounterIon',
        'ase.calculators.counterions.AtomicCounterIon')
    reg('cp2k', 'CP2K', 'ase.calculators.cp2k.CP2K', configurable=True)
    reg('crystal', 'CRYSTAL', 'ase.calculators.crystal.CRYSTAL',
        configurable=True)
    reg('demon', 'deMon', 'ase.calculators.demon.Demon', configurable=True)
    reg('demonnano', 'deMon-nano',
        'ase.calculators.demonnano.DemonNano', configurable=True)
    reg('dftb', 'DFTB+', 'ase.calculators.dftb.Dftb', configurable=True)
    reg('dftd3', 'DFT-D3', 'ase.calculators.dftd3.DFTD3', configurable=True)
    reg('dmol', 'DMol3', 'ase.calculators.dmol.DMol3', configurable=True)
    reg('eam', 'EAM', 'ase.calculators.eam.EAM')
    reg('eiqmmmm', 'Explicit interaction QMMM',
        'ase.calculators.qmmm.EIQMMM', is_wrapper=True)
    reg('elk', 'ELK', 'ase.calculators.elk.ELK', configurable=True)
    reg('emt', 'EMT potential', 'ase.calculators.emt.EMT')
    reg('espresso', 'Quantum Espresso',
        'ase.calculators.espresso.EspressoTemplate', configurable=True)
    reg('exciting', 'Exciting',
        'ase.calculators.exciting.exciting.ExcitingGroundStateTemplate',
        configurable=True)
    reg('ff', 'FF', 'ase.calculators.ff.ForceField')
    reg('fd', 'Finite Difference',
        'ase.calculators.fd.FiniteDifferenceCalculator', is_wrapper=True)
    reg('forceconstant', 'ForceConstantCalculator',
        'ase.calculators.qmmm.ForceConstantCalculator')
    reg('force-qmmmm', 'Force-based QM/MM Calculator',
        'ase.calculators.qmmm.ForceQMMM', is_wrapper=True)
    reg('gamess_us', 'GAMESS-US',
        'ase.calculators.gamess_us.GAMESSUS', configurable=True)
    reg('gaussian', 'Gaussian',
        'ase.calculators.gaussian.Gaussian', configurable=True)
    reg('gromacs', 'Gromacs', 'ase.calculators.gromacs.Gromacs',
        configurable=True)
    reg('gulp', 'GULP', 'ase.calculators.gulp.GULP', configurable=True)
    reg('harmonic', 'Harmonic potential',
        'ase.calculators.harmonic.HarmonicCalculator')
    reg('idealgas', 'Ideal gas', 'ase.calculators.idealgas.IdealGas')
    reg('lammpslib', 'Lammps (python library)',
        'ase.calculators.lammpslib.LAMMPSlib', configurable=True)
    reg('lammpsrun', 'Lammps (external)',
        'ase.calculators.lammpsrun.LAMMPS', configurable=True)
    reg('linearcombination', 'LinearCombinationCalculator',
        'ase.calculators.mixing.LinearCombinationCalculator', is_wrapper=True)
    reg('lj', 'Lennard–Jones potential',
        'ase.calculators.lj.LennardJones', is_wrapper=True)
    reg('logging', 'LoggingCalculator',
        'ase.calculators.loggingcalc.LoggingCalculator', is_wrapper=True)
    reg('mixed', 'MixedCalculator',
        'ase.calculators.mixing.MixedCalculator', is_wrapper=True)
    reg('mopac', 'MOPAC', 'ase.calculators.mopac.MOPAC', configurable=True)
    reg('morse', 'Morse potential', 'ase.calculators.morse.MorsePotential')
    reg('nwchem', 'NWChem', 'ase.calculators.nwchem.NWChem', configurable=True)
    reg('octopus', 'Octopus',
        'ase.calculators.octopus.OctopusTemplate', configurable=True)
    reg('onetep', 'Onetep',
        'ase.calculators.onetep.OnetepTemplate', configurable=True)
    reg('openmx', 'OpenMX', 'ase.calculators.openmx.OpenMX', configurable=True)
    reg('orca', 'ORCA', 'ase.calculators.orca.OrcaTemplate', configurable=True)
    reg('plumed', 'Plumed', 'ase.calculators.plumed.Plumed', configurable=True)
    reg('psi4', 'Psi4', 'ase.calculators.psi4.Psi4', configurable=True)
    reg('python-subprocess', 'PythonSubProcessCalculator',
        'ase.calculators.subprocesscalculator.PythonSubProcessCalculator',
        is_wrapper=True)
    reg('qchem', 'QChem', 'ase.calculators.qchem.QChem', configurable=True)
    reg('rescaled', 'Rescaled Calculator',
        'ase.calculators.qmmm.RescaledCalculator', is_wrapper=True)
    reg('siesta', 'SIESTA', 'ase.calculators.siesta.Siesta', configurable=True)
    reg('simple-qmmmm', 'Simple QMMM',
        'ase.calculators.qmmm.SimpleQMMM', is_wrapper=True)
    reg('singlepoint', 'SinglePoint',
        'ase.calculators.singlepoint.SinglePointCalculator')
    reg('singlepoint-dft', 'SinglePointDFTCalculator',
        'ase.calculators.singlepoint.SinglePointDFTCalculator')
    reg('socketio', 'SocketIOCalculator',
        'ase.calculators.socketio.SocketIOCalculator', is_wrapper=True)
    reg('spring', 'Spring Calculator',
        'ase.calculators.harmonic.SpringCalculator')
    reg('sum', 'SumCalculator', 'ase.calculators.mixing.SumCalculator',
        is_wrapper=True)
    reg('tersoff', 'Tersoff potential', 'ase.calculators.tersoff.Tersoff')
    reg('tip3p', 'TIP3P', 'ase.calculators.tip3p.TIP3P')
    reg('tip4p', 'TIP4P', 'ase.calculators.tip4p.TIP4P')
    reg('turbomole', 'Turbomole',
        'ase.calculators.turbomole.Turbomole', configurable=True)
    reg('vasp', 'VASP', 'ase.calculators.vasp.Vasp', configurable=True)
    # internal: vdwcorrection  # This only really works with GPAW, could move?

    return codes


codes = register_codes()


def list_codes(names: list[str]):
    from ase.config import cfg
    cfg.print_header()
    print()

    for name in names:
        code = codes[name]
        print(code.name)
        try:
            print(code.description(indent='  '))
        except Exception as ex:
            print(f'Bad configuration of {name}: {ex!r}')
        print()


if __name__ == '__main__':
    import sys
    names = sys.argv[1:]
    if not names:
        names = [*codes]
    list_codes(names)
