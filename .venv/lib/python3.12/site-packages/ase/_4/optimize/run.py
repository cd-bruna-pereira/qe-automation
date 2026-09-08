from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ase._4.optimize.lbfgs import OptimizerMethod
from ase._4.optimize.optimizable import Optimizable4
from ase.dependencies import ase_version_info
from ase.io.jsonio import default, object_hook
from ase.io.trajectory import Trajectory
from ase.parallel import world


def toplevel_header(target, method):
    import datetime

    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

    # Log initialization
    # Log fmax/smax if known?
    # Allow each object to write description of itself

    return (
        f'{ase_version_info()}\n'
        f"Optimizing '{target.__class__.__name__}' "
        f'using method {method.methodname!r} '
        f'starting {timestamp}\n\n'
    )


class Optimizer:
    def __init__(
        self,
        target: Optimizable4,
        method: OptimizerMethod,
        trajectory=None,
        restartfile=None,
        comm=world,
        logfile='-',
        step: Step | None = None,
        maxstep: float = 0.2,
    ):
        from ase.optimize.optimize import Log

        self.log = Log(logfile, comm)
        self.comm = comm
        self.target = target
        self.method = method
        self.trajectory = trajectory
        if restartfile is not None:
            restartfile = Path(restartfile)
        self.restartfile = restartfile
        # TODO We need both "restart from" and "save restart to", somehow.
        # Altough maybe that feature can come via a classmethod
        self.maxstep = maxstep
        self.step = step

        # If we don't have self.step, then restarting would need to
        # pass the step to run(), which may be awkward.
        # But having self.step is also awkward.

    def run(self, *, steps: int | None = None) -> Step:
        for _ in self.irun(steps=steps):
            pass
        # Note: This may take zero steps, but self.step is always non-None
        assert self.step is not None
        return self.step

    def irun(self, *, steps: int | None = None) -> Iterator[Step]:
        self.log.write(toplevel_header(self.target, self.method))
        if self.step is None:
            # XXX does not allow the method to write anything.
            if hasattr(self.target, 'log_headers'):
                self.log.write(self.target.log_headers())
                self.log.write('\n')
            self.step = Step.start(self.target)
            self._writefiles(self.step)
            yield self.step

        if steps is not None:
            # (We may start from "step 17" if we restart)
            steps += self.step.i

        def check_stop(step):
            # Would it be better to raise an error if we don't converge,
            # or should there be an option for doing so?
            return step.gradient_obj.converged or step.i == steps

        while not check_stop(self.step):
            # (Both method and target change in this update)
            self.step = next_step(
                self.target, self.method, self.step, self.maxstep
            )
            self._writefiles(self.step)
            yield self.step

        self.log.write(f'\nOptimization ended with step {self.step.i}.')

    def _writefiles(self, step):
        # XXX Also need to pass method, at least.
        # Or maybe method should log to a separate line.
        if hasattr(self.target, 'step_to_string'):
            self.log.write(self.target.step_to_string(step))
            self.log.write('\n')
        else:
            write_to_log(self.method, self.log, self.step)
        if self.trajectory is not None:
            write_to_traj(self.target, self.trajectory, self.comm)
        if self.restartfile is not None and self.comm.rank == 0:
            write_restartfile(self.restartfile, self.method, self.target, step)

    @classmethod
    def restart(cls, restartfile, calc, **kwargs):
        # We need these read behaviours:
        #  * Start from scratch
        #  * Read this file
        #  * Read this file if it exists, else start from scratch
        #
        # ... and these write behaviours:
        #
        #  * Do not write
        #  * Write to this file
        #  * Write to file we restarted from

        # Since this method has "calc", it doesn't really belong on this
        # class (we know about Targets etc. but not calcs).
        # Maybe therefore this should be a standalone function.
        target, method, step = read_restartfile(restartfile, calc)
        return cls(target=target, method=method, step=step, **kwargs)


@dataclass
class Step:
    i: int
    x: np.ndarray
    gradient_obj: Any
    value: float

    @classmethod
    def start(cls, target: Optimizable4) -> Step:
        step = cls(0, target.get_x(), target.get_gradient(), target.get_value())
        assert step.gradient_obj.gradient.shape == (len(step.x),)
        return step

    def datafy(self) -> dict[str, Any]:
        return {
            'i': self.i,
            'x': self.x.tolist(),
            'gradient_obj': self.gradient_obj.datafy(),
            'value': self.value,
        }

    @classmethod
    def undatafy(cls, dct, gradient_obj):
        return cls(
            i=dct['i'],
            x=np.array(dct['x']),
            gradient_obj=gradient_obj,
            value=dct['value'],
        )


def next_step(
    target: Optimizable4,
    method: OptimizerMethod,
    step: Step,
    maxstep: float | None,
) -> Step:
    dx = method.compute_step(step.gradient_obj.gradient)
    # TODO We do not have maxstep right now.  This will not run the same
    # as legacy optimizations until we apply a maxstep.

    # Questionable: We are using the function for computing norm of
    # the gradient to compute norm of another vector.
    if maxstep is not None:
        longest = target.gradient_norm(dx)
        if longest > maxstep:
            scale = maxstep / longest
            dx *= scale

    # Target may apply constraints or other magic, so we may not
    # get the same x back as the one we set.
    target.set_x(step.x + dx)

    newstep = Step(
        i=step.i + 1,
        x=target.get_x(),
        gradient_obj=target.get_gradient(),
        value=target.get_value(),
    )

    method.update(
        newstep.x,
        newstep.gradient_obj.gradient,
        step.x,
        step.gradient_obj.gradient,
    )
    return newstep


def write_to_log(method, log, step):
    loginfo = step.gradient_obj.loginfo()
    name = method.methodname
    txt = ' '.join(f'{key}={value:e}' for key, value in loginfo.items())
    msg = f'{name} i={step.i:4d} e={step.value:f} {txt}\n'
    log.write(msg)


def write_to_traj(target, trajpath, comm):
    with Trajectory(trajpath, comm=comm, mode='a') as traj:
        # TODO we are not setting metadata (like old optimizers)
        traj.write(target)


def read_images(trajpath):
    with Trajectory(trajpath) as traj:
        return [*traj]


def write_restartfile(restartfile, method, target, step):
    # TODO Unsafe if we just overwrite, we should backup/delete to prevent
    # accidental partial save

    # Still need some things, like maximum iterations.
    # How about trajectory writing, logfile settings, etc.?
    # General observers obviously cannot be saved.
    savedata = {
        'method': [method.iotype, method.datafy()],
        'target': [target.iotype, target.datafy()],
        'step': step.datafy(),
    }
    json_text = json.dumps(savedata, default=default)
    restartfile.write_text(json_text)


def read_restartfile(restartfile, calc) -> tuple:
    json_text = restartfile.read_text()
    dct = json.loads(json_text, object_hook=object_hook)

    assert {*dct} == {'method', 'target', 'step'}
    method_iotype, method_data = dct['method']
    target_iotype, target_dct = dct['target']

    if method_iotype == 'bfgs':
        from ase._4.optimize.bfgs import BFGSMethod

        method = BFGSMethod.undatafy(method_data)
    elif method_iotype == 'lbfgs':
        from ase._4.optimize.lbfgs import LBFGSMethod

        method = LBFGSMethod.undatafy(method_data)
    else:
        raise ValueError(f'No such method: {method_iotype!r}')

    if target_iotype == 'frechet':
        from ase._4.optimize.frechet import FrechetOptimizable

        target = FrechetOptimizable.undatafy(target_dct, calc)
    elif target_iotype == 'symopt':
        from ase._4.optimize.symopt import SymOpt

        target = SymOpt.undatafy(target_dct, calc)
    else:
        raise ValueError(f'No such target: {target_iotype!r}')

    gradient_obj = target.undatafy_gradient(dct['step']['gradient_obj'])
    step = Step.undatafy(dct['step'], gradient_obj)

    return target, method, step
