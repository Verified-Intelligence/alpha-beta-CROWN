import multiprocessing
import multiprocessing.pool
import sys
import os

import torch
from .bound_ops import *


def build_solver_module(self, x=None, C=None, intermediate_layer_bounds=None,
                        final_node_name=None, model_type="mip", solver_pkg="gurobi"):
    r"""build lp/mip solvers in general graph.

    Args:
        x: inputs, a list of BoundedTensor. If set to None, we reuse exisint bounds that
        were previously computed in compute_bounds().
        C (Tensor): The specification matrix that can map the output of the model with an
        additional linear layer. This is usually used for maping the logits output of the
        model to classification margins.
        intermediate_layer_bounds: if specified, will replace existing intermediate layer bounds.
        Otherwise we reuse exising intermediate bounds.

        final_node_name (String): the name for the target layer to optimize

        solver_pkg (String): the backbone of the solver, default gurobi, also support scipy

    Returns:
        output vars (list): a list of final nodes to optimize
    """
    # self.root_name: list of root node name
    # self.final_name: list of output node name
    # self.final_node: output module
    # <module>.input: a list of input modules of this layer module
    # <module>.solver_vars: a list of gurobi vars of every layer module
    #       list with conv shape if conv layers, otherwise flattened
    # if last layer we need to be careful with:
    #       C: specification matrix
    #       <module>.is_input_perturbed(1)

    if x is not None:
        assert intermediate_layer_bounds is not None
        # Set the model to use new intermediate layer bounds, ignore the original ones.
        self._set_input(x, intermediate_layer_bounds=intermediate_layer_bounds)
 
    root = [self[name] for name in self.root_name]

    # create interval ranges for input and other weight parameters
    for i in range(len(root)):
        value = root[i].forward()
        # if isinstance(root[i], BoundInput) and not isinstance(root[i], BoundParams):
        if type(root[i]) is BoundInput:
            # create input vars for gurobi self.model
            inp_gurobi_vars = self._build_solver_input(root[i])
        else:
            # regular weights
            root[i].solver_vars = value

    final = self.final_node() if final_node_name is None else self[final_node_name]

    # backward propagate every layer including last layer
    self._build_solver_general(node=final, C=C, model_type=model_type, solver_pkg=solver_pkg)

    # a list of output solver vars
    return final.solver_vars


def _build_solver_general(self, node, C=None, model_type="mip", solver_pkg="gurobi"):
    if not hasattr(node, 'solver_vars'):
        for n in node.inputs:
            self._build_solver_general(n, C=C, model_type=model_type, solver_pkg=solver_pkg)
        inp = [n_pre.solver_vars for n_pre in node.inputs]
        # print(node, node.inputs)
        if C is not None and isinstance(node, BoundLinear) and\
                not node.is_input_perturbed(1) and self.final_name == node.name:
            # when node is the last layer
            # merge the last BoundLinear node with the specification, 
            # available when weights of this layer are not perturbed
            solver_vars = node.build_solver(*inp, model=self.model, C=C, 
                    model_type=model_type, solver_pkg=solver_pkg)
        else:
            solver_vars = node.build_solver(*inp, model=self.model, C=None, 
                    model_type=model_type, solver_pkg=solver_pkg)
        # just return output node gurobi vars
        return solver_vars
    

def _build_solver_input(self, node):
    ## Do the input layer, which is a special case
    assert isinstance(node, BoundInput)
    assert node.perturbation is not None
    assert node.perturbation.norm == float("inf")
    inp_gurobi_vars = []
    # zero var will be shared within the solver model
    zero_var = self.model.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name='zero')
    x_L = node.value - node.perturbation.eps if node.perturbation.x_L is None else node.perturbation.x_L
    x_U = node.value + node.perturbation.eps if node.perturbation.x_U is None else node.perturbation.x_U
    x_L = x_L.squeeze(0)
    x_U = x_U.squeeze(0)
    # x_L, x_U = node.lower.squeeze(0), node.upper.squeeze(0)

    if x_L.ndim == 1:
        # This is a linear input.
        for dim, (lb, ub) in enumerate(zip(x_L, x_U)):
            v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                    vtype=grb.GRB.CONTINUOUS,
                                    name=f'inp_{dim}')
            inp_gurobi_vars.append(v)
    else:
        assert x_L.ndim == 3, f"x_L ndim  {x_L.ndim}"
        dim = 0
        for chan in range(x_L.shape[0]):
            chan_vars = []
            for row in range(x_L.shape[1]):
                row_vars = []
                for col in range(x_L.shape[2]):
                    lb = x_L[chan, row, col]
                    ub = x_U[chan, row, col]
                    v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'inp_{dim}')
                                            # name=f'inp_[{chan},{row},{col}]')
                    row_vars.append(v)
                    dim += 1
                chan_vars.append(row_vars)
            inp_gurobi_vars.append(chan_vars)

    node.solver_vars = inp_gurobi_vars
    # save the gurobi input variables so that we can later extract primal values in input space easily
    self.input_vars = inp_gurobi_vars
    self.model.update()
    return inp_gurobi_vars

