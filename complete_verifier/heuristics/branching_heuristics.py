import arguments

from heuristics.base import RandomNeuronBranching, InterceptBranching
from heuristics.babsr import BabsrBranching
from heuristics.fsb import FsbBranching
from heuristics.kfsb import KfsbBranching
from heuristics.nonlinear import NonlinearBranching


def get_branching_heuristic(net):
    branching_method = arguments.Config['bab']['branching']['method']
    branching_obj = None
    if branching_method == 'random':
        branching_obj = RandomNeuronBranching(net)
    elif branching_method == 'intercept':
        branching_obj = InterceptBranching(net)
    elif branching_method == 'nonlinear':
        branching_args = arguments.Config['bab']['branching']['nonlinear_split']
        branching_obj = NonlinearBranching(net, **branching_args)
    elif branching_method == 'babsr':
        branching_obj = BabsrBranching(net)
    elif branching_method == 'fsb':
        branching_obj = FsbBranching(net)
    elif branching_method.startswith('kfsb'):
        branching_obj = KfsbBranching(net)
    else:
        raise ValueError(f'Unsupported branching method "{branching_method}" '
                         'for activation splits.')
    return branching_obj
