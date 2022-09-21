import torch
from .bound_ops import *


def IBP_general(self, node=None, C=None, delete_bounds_after_use=False):

    def _delete_unused_bounds(node_list):
        """Delete bounds from input layers after use to save memory. Used when
        sparse_intermediate_bounds_with_ibp is true."""
        if delete_bounds_after_use:
            for n in node_list:
                del n.interval
                del n.lower
                del n.upper

    if self.bound_opts.get('loss_fusion', False):
        res = self._IBP_loss_fusion(node, C)
        if res is not None:
            return res

    if not node.perturbed and hasattr(node, 'forward_value'):
        node.lower, node.upper = node.interval = (
            node.forward_value, node.forward_value)

    to_be_deleted_bounds = []
    if not hasattr(node, 'interval'):
        for n in node.inputs:
            if not hasattr(n, 'interval'):
                # Node n does not have interval bounds; we must compute it.
                self.IBP_general(
                    n, delete_bounds_after_use=delete_bounds_after_use)
                to_be_deleted_bounds.append(n)
        inp = [n_pre.interval for n_pre in node.inputs]
        if (C is not None and isinstance(node, BoundLinear)
                and not node.is_input_perturbed(1)):
            # merge the last BoundLinear node with the specification, available
            # when weights of this layer are not perturbed
            ret = node.interval_propagate(*inp, C=C)
            _delete_unused_bounds(to_be_deleted_bounds)
            return ret
        else:
            node.interval = node.interval_propagate(*inp)

        node.lower, node.upper = node.interval
        if isinstance(node.lower, torch.Size):
            node.lower = torch.tensor(node.lower)
            node.interval = (node.lower, node.upper)
        if isinstance(node.upper, torch.Size):
            node.upper = torch.tensor(node.upper)
            node.interval = (node.lower, node.upper)

    if C is not None:
        _delete_unused_bounds(to_be_deleted_bounds)
        return BoundLinear.interval_propagate(None, node.interval, C=C)
    else:
        _delete_unused_bounds(to_be_deleted_bounds)
        return node.interval

def _IBP_loss_fusion(self, node, C):
    """Merge BoundLinear, BoundGatherElements and BoundSub.

    Improvement when loss fusion is used in training.
    """

    # not using loss fusion
    if not self.bound_opts.get('loss_fusion', False):
        return None

    # Currently this function has issues in more complicated networks.
    if self.bound_opts.get('no_ibp_loss_fusion', False):
        return None

    if (C is None and isinstance(node, BoundSub)
            and isinstance(node.inputs[1], BoundGatherElements)
            and isinstance(node.inputs[0], BoundLinear)):
        node_gather = node.inputs[1]
        node_linear = node.inputs[0]
        node_start = node_linear.inputs[0]
        w = node_linear.inputs[1].param
        b = node_linear.inputs[2].param
        labels = node_gather.inputs[1]
        if not hasattr(node_start, 'interval'):
            self.IBP_general(node_start)
        for n in node_gather.inputs:
            if not hasattr(n, 'interval'):
                self.IBP_general(n)
        if torch.isclose(labels.lower, labels.upper, 1e-8).all():
            labels = labels.lower
            batch_size = labels.shape[0]
            w = w.expand(batch_size, *w.shape)
            w = w - torch.gather(
                w, dim=1,
                index=labels.unsqueeze(-1).repeat(1, w.shape[1], w.shape[2]))
            b = b.expand(batch_size, *b.shape)
            b = b - torch.gather(b, dim=1,
                                    index=labels.repeat(1, b.shape[1]))
            lower, upper = node_start.interval
            lower, upper = lower.unsqueeze(1), upper.unsqueeze(1)
            node.lower, node.upper = node_linear.interval_propagate(
                (lower, upper), (w, w), (b.unsqueeze(1), b.unsqueeze(1)))
            node.interval = node.lower, node.upper = (
                node.lower.squeeze(1), node.upper.squeeze(1))
            return node.interval

    return None


def check_IBP_intermediate(self, node):
    """ Check if we use IBP bounds to compute intermediate bounds on this node.
        Basically we check if we can get bounds by only visiting operators in
        `self.ibp_intermediate`.

        Currently, assume all eligible operators have exactly one input. """
    nodes = []
    while not hasattr(node, 'lower') or not hasattr(node, 'upper'):
        if type(node) not in self.ibp_intermediate:
            return False
        nodes.append(node)
        node = node.inputs[0]
    nodes.reverse()
    for n in nodes:
        node.interval = self.IBP_general(n)
    return True


def check_IBP_first_linear(self, node):
    """Here we avoid creating a big C matrix in the first linear layer.
    Disable this optimization when we have beta for intermediate layer bounds.
    Disable this optimization when we need the A matrix of the first nonlinear
    layer, forcibly use CROWN to record A matrix.
    """
    # This is the list of all intermediate layers where we need to refine.
    if self.intermediate_constr is not None:
        intermediate_beta_enabled_layers = [
            k for v in self.intermediate_constr.values() for k in v]
    else:
        intermediate_beta_enabled_layers = []

    if (node.name not in self.needed_A_dict.keys()
            and (type(node) == BoundLinear
                or type(node) == BoundConv
                and node.name not in intermediate_beta_enabled_layers)):
        if type(node.inputs[0]) == BoundInput:
            node.lower, node.upper = self.IBP_general(node)
            return True

    return False
