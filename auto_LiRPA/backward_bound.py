import torch
from torch import Tensor
from collections import deque, defaultdict
from tqdm import tqdm
from .patches import Patches
from .utils import *
from .bound_ops import *
import warnings


def batched_backward(
        self, node, C, unstable_idx, batch_size, bound_lower=True,
        bound_upper=True):
    crown_batch_size = self.bound_opts['crown_batch_size']
    unstable_size = get_unstable_size(unstable_idx)
    print(f'Batched CROWN: unstable size {unstable_size}')
    num_batches = (unstable_size + crown_batch_size - 1) // crown_batch_size
    output_shape = node.output_shape[1:]
    dim = int(prod(output_shape))
    ret = []
    for i in tqdm(range(num_batches)):
        if isinstance(unstable_idx, tuple):
            unstable_idx_batch = tuple(
                u[i*crown_batch_size:(i+1)*crown_batch_size]
                for u in unstable_idx
            )
            unstable_size_batch = len(unstable_idx_batch[0])
        else:
            unstable_idx_batch = unstable_idx[i*crown_batch_size:(i+1)*crown_batch_size]
            unstable_size_batch = len(unstable_idx_batch)
        if node.patches_start and node.mode == "patches":
            assert C in ['Patches', None]
            C_batch = Patches(shape=[
                unstable_size_batch, batch_size, *node.output_shape[1:-2], 1, 1],
                identity=1, unstable_idx=unstable_idx_batch,
                output_shape=[batch_size, *node.output_shape[1:]])
        elif isinstance(node, BoundLinear) or isinstance(node, BoundMatMul):
            assert C in ['OneHot', None]
            C_batch = OneHotC(
                [batch_size, unstable_size_batch, *node.output_shape[1:]],
                self.device, unstable_idx_batch, None)
        else:
            assert C in ['eye', None]
            C_batch = torch.zeros([1, unstable_size_batch, dim], device=self.device)
            C_batch[0, torch.arange(unstable_size_batch), unstable_idx_batch] = 1.0
            C_batch = C_batch.expand(batch_size, -1, -1).view(
                batch_size, unstable_size_batch, *output_shape)
        ret.append(self.backward_general(
            C=C_batch, node=node,
            bound_lower=bound_lower, bound_upper=bound_upper,
            average_A=False, need_A_only=False, unstable_idx=unstable_idx_batch,
            verbose=False))
    if bound_lower:
        lb = torch.cat([item[0].view(batch_size, -1) for item in ret], dim=1)
    else:
        lb = None
    if bound_upper:
        ub = torch.cat([item[1].view(batch_size, -1) for item in ret], dim=1)
    else:
        ub = None
    return lb, ub


def backward_general(
        self, C, node=None, bound_lower=True, bound_upper=True, average_A=False,
        need_A_only=False, unstable_idx=None, unstable_size=0, update_mask=None, verbose=True):
    if verbose:
        logger.debug(f'Bound backward from {node.__class__.__name__}({node.name})')
        if isinstance(C, str):
            logger.debug(f'  C: {C}')
        elif C is not None:
            logger.debug(f'  C: shape {C.shape}, type {type(C)}')
    _print_time = False

    if isinstance(C, str):
        # If C is a str, use batched CROWN. If batched CROWN is not intended to
        # be enabled, C must be a explicitly provided non-str object for this function.
        if need_A_only or self.return_A or average_A:
            raise ValueError(
                'Batched CROWN is not compatible with '
                f'need_A_only={need_A_only}, return_A={self.return_A}, '
                f'average_A={average_A}')
        node.lower, node.upper = self.batched_backward(
            node, C, unstable_idx,
            batch_size=self.root[0].value.shape[0],
            bound_lower=bound_lower, bound_upper=bound_upper,
        )
        return node.lower, node.upper

    for l in self._modules.values():
        l.lA = l.uA = None
        l.bounded = True

    degree_out = get_degrees(node, self.backward_from)
    all_nodes_before = list(degree_out.keys())

    C, batch_size, output_dim, output_shape = preprocess_C(C, node)

    node.lA = C if bound_lower else None
    node.uA = C if bound_upper else None
    lb = ub = torch.tensor(0., device=self.device)


    # Save intermediate layer A matrices when required.
    A_record = {}

    queue = deque([node])
    while len(queue) > 0:
        l = queue.popleft()  # backward from l
        l.bounded = True

        if l.name in self.root_name: continue

        # if all the succeeds are done, then we can turn to this node in the
        # next iteration.
        for l_pre in l.inputs:
            degree_out[l_pre.name] -= 1
            if degree_out[l_pre.name] == 0:
                queue.append(l_pre)

        # Initially, l.lA or l.uA will be set to C for this node.
        if l.lA is not None or l.uA is not None:
            if verbose:
                logger.debug(f'  Bound backward to {l}'
                            f' (lA shape {l.lA.shape if l.lA is not None else None},'
                            f' uA shape {l.uA.shape if l.uA is not None else None},'
                            f' out shape {l.output_shape})')

            if _print_time:
                start_time = time.time()

            if not l.perturbed:
                if not hasattr(l, 'forward_value'):
                    self.get_forward_value(l)
                lb, ub = add_constant_node(lb, ub, l)
                continue

            if l.zero_uA_mtx and l.zero_lA_mtx:
                # A matrices are all zero, no need to propagate.
                continue

            if isinstance(l, BoundRelu):
                # TODO: unify this interface.
                A, lower_b, upper_b = l.bound_backward(
                    l.lA, l.uA, *l.inputs, start_node=node, unstable_idx=unstable_idx,
                    beta_for_intermediate_layers=self.intermediate_constr is not None)
            elif isinstance(l, BoundOptimizableActivation):
                # For other optimizable activation functions (TODO: unify with ReLU).
                if node.name != self.final_node_name:
                    start_shape = node.output_shape[1:]
                else:
                    start_shape = C.shape[0]
                A, lower_b, upper_b = l.bound_backward(
                    l.lA, l.uA, *l.inputs, start_shape=start_shape, start_node=node)
            else:
                A, lower_b, upper_b = l.bound_backward(l.lA, l.uA, *l.inputs)
            # After propagation through this node, we delete its lA, uA variables.
            if not self.return_A and node.name != self.final_name:
                del l.lA, l.uA
            if _print_time:
                time_elapsed = time.time() - start_time
                if time_elapsed > 1e-3:
                    print(l, time_elapsed)
            if lb.ndim > 0 and type(lower_b) == Tensor and self.conv_mode == 'patches':
                lb, ub, lower_b, upper_b = check_patch_biases(lb, ub, lower_b, upper_b)
            lb = lb + lower_b
            ub = ub + upper_b
            if self.return_A and self.needed_A_dict and node.name in self.needed_A_dict:
                # FIXME remove [0][0] and [0][1]?
                if len(self.needed_A_dict[node.name]) == 0 or l.name in self.needed_A_dict[node.name]:
                    A_record.update({l.name: {
                        "lA": A[0][0].transpose(0, 1).detach() if A[0][0] is not None else None,
                        "uA": A[0][1].transpose(0, 1).detach() if A[0][1] is not None else None,
                        # When not used, lb or ub is tensor(0).
                        "lbias": lb.transpose(0, 1).detach() if lb.ndim > 1 else None,
                        "ubias": ub.transpose(0, 1).detach() if ub.ndim > 1 else None,
                        "unstable_idx": unstable_idx
                    }})
                # FIXME: solve conflict with the following case
                self.A_dict.update({node.name: A_record})
                if need_A_only and set(self.needed_A_dict[node.name]) == set(A_record.keys()):
                    # We have collected all A matrices we need. We can return now!
                    self.A_dict.update({node.name: A_record})
                    # Do not concretize to save time. We just need the A matrices.
                    # return A matrix as a dict: {node.name: [A_lower, A_upper]}
                    return None, None, self.A_dict


            for i, l_pre in enumerate(l.inputs):
                add_bound(l, l_pre, lA=A[i][0], uA=A[i][1])

    if lb.ndim >= 2:
        lb = lb.transpose(0, 1)
    if ub.ndim >= 2:
        ub = ub.transpose(0, 1)

    if self.return_A and self.needed_A_dict and node.name in self.needed_A_dict:
        save_A_record(
            node, A_record, self.A_dict, self.root, self.needed_A_dict[node.name],
            lb=lb, ub=ub, unstable_idx=unstable_idx)

    # TODO merge into `concretize`
    if self.cut_used and getattr(self, 'cut_module', None) is not None and self.cut_module.x_coeffs is not None:
        # propagate input neuron in cut constraints
        self.root[0].lA, self.root[0].uA = self.cut_module.input_cut(
            node, self.root[0].lA, self.root[0].uA, self.root[0].lower.size()[1:], unstable_idx,
            batch_mask=update_mask)

    lb, ub = concretize(
        lb, ub, node, self.root, batch_size, output_dim,
        bound_lower, bound_upper, average_A=average_A)

    # TODO merge into `concretize`
    if self.cut_used and getattr(self, "cut_module", None) is not None and self.cut_module.cut_bias is not None:
        # propagate cut bias in cut constraints
        lb, ub = self.cut_module.bias_cut(node, lb, ub, unstable_idx, batch_mask=update_mask)
        if lb is not None and ub is not None and ((lb-ub)>0).sum().item() > 0:
            # make sure there is no bug for cut constraints propagation
            print(f"Warning: lb is larger than ub with diff: {(lb-ub)[(lb-ub)>0].max().item()}")

    lb = lb.view(batch_size, *output_shape) if bound_lower else None
    ub = ub.view(batch_size, *output_shape) if bound_upper else None

    if verbose:
        logger.debug('')

    if self.return_A:
        return lb, ub, self.A_dict
    else:
        return lb, ub


def get_unstable_size(unstable_idx):
    if isinstance(unstable_idx, tuple):
        return unstable_idx[0].numel()
    else:
        return unstable_idx.numel()


def check_optimized_variable_sparsity(self, node):
    alpha_sparsity = None  # unknown.
    for relu in self.relus:
        if hasattr(relu, 'alpha_lookup_idx') and node.name in relu.alpha_lookup_idx:
            if relu.alpha_lookup_idx[node.name] is not None:
                # This node was created with sparse alpha
                alpha_sparsity = True
            else:
                alpha_sparsity = False
            break
    # print(f'node {node.name} alpha sparsity {alpha_sparsity}')
    return alpha_sparsity


def get_sparse_C(
        self, node, sparse_intermediate_bounds=True, ref_intermediate_lb=None,
        ref_intermediate_ub=None):
    sparse_conv_intermediate_bounds = self.bound_opts.get('sparse_conv_intermediate_bounds', False)
    minimum_sparsity = self.bound_opts.get('minimum_sparsity', 0.9)
    crown_batch_size = self.bound_opts.get('crown_batch_size', 1e9)
    dim = int(prod(node.output_shape[1:]))
    batch_size = self.batch_size

    reduced_dim = False  # Only partial neurons (unstable neurons) are bounded.
    unstable_idx = None
    unstable_size = np.inf
    newC = None

    alpha_is_sparse = self.check_optimized_variable_sparsity(node)

    # NOTE: batched CROWN is so far only supported for some of the cases below

    # FIXME: C matrix shape incorrect for BoundParams.
    if (isinstance(node, BoundLinear) or isinstance(node, BoundMatMul)) and int(
            os.environ.get('AUTOLIRPA_USE_FULL_C', 0)) == 0:
        if sparse_intermediate_bounds:
            # If we are doing bound refinement and reference bounds are given, we only refine unstable neurons.
            # Also, if we are checking against LP solver we will refine all neurons and do not use this optimization.
            # For each batch element, we find the unstable neurons.
            unstable_idx, unstable_size = self.get_unstable_locations(
                ref_intermediate_lb, ref_intermediate_ub)
            if unstable_size == 0:
                # Do nothing, no bounds will be computed.
                reduced_dim = True
                unstable_idx = []
            elif unstable_size > crown_batch_size:
                # Create C in batched CROWN
                newC = 'OneHot'
                reduced_dim = True
            elif (unstable_size <= minimum_sparsity * dim and unstable_size > 0 and alpha_is_sparse is None) or alpha_is_sparse:
                # When we already have sparse alpha for this layer, we always use sparse C. Otherwise we determine it by sparsity.
                # Create an abstract C matrix, the unstable_idx are the non-zero elements in specifications for all batches.
                newC = OneHotC(
                    [batch_size, unstable_size, *node.output_shape[1:]],
                    self.device, unstable_idx, None)
                reduced_dim = True
            else:
                unstable_idx = None
                del ref_intermediate_lb, ref_intermediate_ub
        if not reduced_dim:
            newC = eyeC([batch_size, dim, *node.output_shape[1:]], self.device)
    elif node.patches_start and node.mode == "patches":
        if sparse_intermediate_bounds:
            unstable_idx, unstable_size = self.get_unstable_locations(
                ref_intermediate_lb, ref_intermediate_ub, conv=True)
            if unstable_size == 0:
                # Do nothing, no bounds will be computed.
                reduced_dim = True
                unstable_idx = []
            elif unstable_size > crown_batch_size:
                # Create C in batched CROWN
                newC = 'Patches'
                reduced_dim = True
            # We sum over the channel direction, so need to multiply that.
            elif (sparse_conv_intermediate_bounds and unstable_size <= minimum_sparsity * dim and alpha_is_sparse is None) or alpha_is_sparse:
                # When we already have sparse alpha for this layer, we always use sparse C. Otherwise we determine it by sparsity.
                # Create an abstract C matrix, the unstable_idx are the non-zero elements in specifications for all batches.
                # The shape of patches is [unstable_size, batch, C, H, W].
                newC = Patches(
                    shape=[unstable_size, batch_size, *node.output_shape[1:-2], 1, 1],
                    identity=1, unstable_idx=unstable_idx,
                    output_shape=[batch_size, *node.output_shape[1:]])
                reduced_dim = True
            else:
                unstable_idx = None
                del ref_intermediate_lb, ref_intermediate_ub
        # Here we create an Identity Patches object
        if not reduced_dim:
            newC = Patches(
                None, 1, 0, [node.output_shape[1], batch_size, *node.output_shape[2:],
                *node.output_shape[1:-2], 1, 1], 1,
                output_shape=[batch_size, *node.output_shape[1:]])
    elif isinstance(node, (BoundAdd, BoundSub)) and node.mode == "patches":
        # FIXME: BoundAdd does not always have patches. Need to use a better way to determine patches mode.
        # FIXME: We should not hardcode BoundAdd here!
        if sparse_intermediate_bounds:
            if crown_batch_size < 1e9:
                warnings.warn('Batched CROWN is not supported in this case')
            unstable_idx, unstable_size = self.get_unstable_locations(
                ref_intermediate_lb, ref_intermediate_ub, conv=True)
            if unstable_size == 0:
                # Do nothing, no bounds will be computed.
                reduced_dim = True
                unstable_idx = []
            elif (sparse_conv_intermediate_bounds and unstable_size <= minimum_sparsity * dim and alpha_is_sparse is None) or alpha_is_sparse:
                # When we already have sparse alpha for this layer, we always use sparse C. Otherwise we determine it by sparsity.
                num_channel = node.output_shape[-3]
                # Identity patch size: (ouc_c, 1, 1, 1, out_c, 1, 1).
                patches = (
                    torch.eye(num_channel, device=self.device,
                    dtype=list(self.parameters())[0].dtype)).view(
                        num_channel, 1, 1, 1, num_channel, 1, 1)
                # Expand to (out_c, 1, unstable_size, out_c, 1, 1).
                patches = patches.expand(-1, 1, node.output_shape[-2], node.output_shape[-1], -1, 1, 1)
                patches = patches[unstable_idx[0], :, unstable_idx[1], unstable_idx[2]]
                # Expand with the batch dimension. Final shape (unstable_size, batch_size, out_c, 1, 1).
                patches = patches.expand(-1, batch_size, -1, -1, -1)
                newC = Patches(
                    patches, 1, 0, patches.shape, unstable_idx=unstable_idx,
                    output_shape=[batch_size, *node.output_shape[1:]])
                reduced_dim = True
            else:
                unstable_idx = None
                del ref_intermediate_lb, ref_intermediate_ub
        if not reduced_dim:
            num_channel = node.output_shape[-3]
            # Identity patch size: (ouc_c, 1, 1, 1, out_c, 1, 1).
            patches = (
                torch.eye(num_channel, device=self.device,
                dtype=list(self.parameters())[0].dtype)).view(
                    num_channel, 1, 1, 1, num_channel, 1, 1)
            # Expand to (out_c, batch, out_h, out_w, out_c, 1, 1).
            patches = patches.expand(-1, batch_size, node.output_shape[-2], node.output_shape[-1], -1, 1, 1)
            newC = Patches(patches, 1, 0, patches.shape, output_shape=[batch_size, *node.output_shape[1:]])
    else:
        if sparse_intermediate_bounds:
            unstable_idx, unstable_size = self.get_unstable_locations(
                ref_intermediate_lb, ref_intermediate_ub)
            if unstable_size == 0:
                # Do nothing, no bounds will be computed.
                reduced_dim = True
                unstable_idx = []
            elif unstable_size > crown_batch_size:
                # Create in C in batched CROWN
                newC = 'eye'
                reduced_dim = True
            elif (unstable_size <= minimum_sparsity * dim and alpha_is_sparse is None) or alpha_is_sparse:
                newC = torch.zeros([1, unstable_size, dim], device=self.device)
                # Fill the corresponding elements to 1.0
                newC[0, torch.arange(unstable_size), unstable_idx] = 1.0
                newC = newC.expand(batch_size, -1, -1).view(batch_size, unstable_size, *node.output_shape[1:])
                reduced_dim = True
            else:
                unstable_idx = None
                del ref_intermediate_lb, ref_intermediate_ub
        if not reduced_dim:
            if dim > 1000:
                warnings.warn(f"Creating an identity matrix with size {dim}x{dim} for node {node}. This may indicate poor performance for bound computation. If you see this message on a small network please submit a bug report.", stacklevel=2)
            newC = torch.eye(dim, device=self.device, dtype=list(self.parameters())[0].dtype) \
                .unsqueeze(0).expand(batch_size, -1, -1) \
                .view(batch_size, dim, *node.output_shape[1:])

    return newC, reduced_dim, unstable_idx, unstable_size


def restore_sparse_bounds(
        self, node, unstable_idx, unstable_size, ref_intermediate_lb, ref_intermediate_ub,
        new_lower=None, new_upper=None):
    batch_size = self.batch_size
    if unstable_size == 0:
        # No unstable neurons. Skip the update.
        node.lower = ref_intermediate_lb.detach().clone()
        node.upper = ref_intermediate_ub.detach().clone()
    else:
        if new_lower is None:
            new_lower = node.lower
        if new_upper is None:
            new_upper = node.upper
        # If we only calculated unstable neurons, we need to scatter the results back based on reference bounds.
        if isinstance(unstable_idx, tuple):
            lower = ref_intermediate_lb.detach().clone()
            upper = ref_intermediate_ub.detach().clone()
            # Conv layer with patches, the unstable_idx is a 3-element tuple for 3 indices (C, H,W) of unstable neurons.
            if len(unstable_idx) == 3:
                lower[:, unstable_idx[0], unstable_idx[1], unstable_idx[2]] = new_lower
                upper[:, unstable_idx[0], unstable_idx[1], unstable_idx[2]] = new_upper
            elif len(unstable_idx) == 4:
                lower[:, unstable_idx[0], unstable_idx[1], unstable_idx[2], unstable_idx[3]] = new_lower
                upper[:, unstable_idx[0], unstable_idx[1], unstable_idx[2], unstable_idx[3]] = new_upper
        else:
            # Other layers.
            lower = ref_intermediate_lb.detach().clone().view(batch_size, -1)
            upper = ref_intermediate_ub.detach().clone().view(batch_size, -1)
            lower[:, unstable_idx] = new_lower.view(batch_size, -1)
            upper[:, unstable_idx] = new_upper.view(batch_size, -1)
        node.lower = lower.view(batch_size, *node.output_shape[1:])
        node.upper = upper.view(batch_size, *node.output_shape[1:])


def get_degrees(node_start, backward_from):
    degrees = {}
    queue = deque([node_start])
    node_start.bounded = False
    while len(queue) > 0:
        l = queue.popleft()
        backward_from[l.name].append(node_start)
        for l_pre in l.inputs:
            degrees[l_pre.name] = degrees.get(l_pre.name, 0) + 1
            if l_pre.bounded:
                l_pre.bounded = False
                queue.append(l_pre)
    return degrees


def preprocess_C(C, node):
    if isinstance(C, Patches):
        if C.unstable_idx is None:
            # Patches have size (out_c, batch, out_h, out_w, c, h, w).
            if len(C.shape) == 7:
                out_c, batch_size, out_h, out_w = C.shape[:4]
                output_dim = out_c * out_h * out_w
            else:
                out_dim, batch_size, out_c, out_h, out_w = C.shape[:5]
                output_dim = out_dim * out_c * out_h * out_w
        else:
            # Patches have size (unstable_size, batch, c, h, w).
            output_dim, batch_size = C.shape[:2]
    else:
        batch_size, output_dim = C.shape[:2]

    # The C matrix specified by the user has shape (batch, spec) but internally we have (spec, batch) format.
    if not isinstance(C, (eyeC, Patches, OneHotC)):
        C = C.transpose(0, 1)
    elif isinstance(C, eyeC):
        C = C._replace(shape=(C.shape[1], C.shape[0], *C.shape[2:]))
    elif isinstance(C, OneHotC):
        C = C._replace(
            shape=(C.shape[1], C.shape[0], *C.shape[2:]),
            index=C.index.transpose(0,-1),
            coeffs=None if C.coeffs is None else C.coeffs.transpose(0,-1))

    if isinstance(C, Patches) and C.unstable_idx is not None:
        # Sparse patches; the output shape is (unstable_size, ).
        output_shape = [C.shape[0]]
    elif prod(node.output_shape[1:]) != output_dim and not isinstance(C, Patches):
        # For the output node, the shape of the bound follows C
        # instead of the original output shape
        #
        # TODO Maybe don't set node.lower and node.upper in this case?
        # Currently some codes still depend on node.lower and node.upper
        output_shape = [-1]
    else:
        # Generally, the shape of the bounds match the output shape of the node
        output_shape = node.output_shape[1:]

    return C, batch_size, output_dim, output_shape


def concretize(lb, ub, node, root, batch_size, output_dim, bound_lower=True, bound_upper=True, average_A=False):

    for i in range(len(root)):
        if root[i].lA is None and root[i].uA is None: continue
        if average_A and isinstance(root[i], BoundParams):
            lA = root[i].lA.mean(node.batch_dim + 1, keepdim=True).expand(root[i].lA.shape) if bound_lower else None
            uA = root[i].uA.mean(node.batch_dim + 1, keepdim=True).expand(root[i].uA.shape) if bound_upper else None
        else:
            lA, uA = root[i].lA, root[i].uA
        if not isinstance(root[i].lA, eyeC) and not isinstance(root[i].lA, Patches):
            lA = root[i].lA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_lower else None
        if not isinstance(root[i].uA, eyeC) and not isinstance(root[i].uA, Patches):
            uA = root[i].uA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_upper else None
        if hasattr(root[i], 'perturbation') and root[i].perturbation is not None:
            if isinstance(root[i], BoundParams):
                # add batch_size dim for weights node
                lb = lb + root[i].perturbation.concretize(
                    root[i].center.unsqueeze(0), lA,
                    sign=-1, aux=root[i].aux) if bound_lower else None
                ub = ub + root[i].perturbation.concretize(
                    root[i].center.unsqueeze(0), uA,
                    sign=+1, aux=root[i].aux) if bound_upper else None
            else:
                lb = lb + root[i].perturbation.concretize(
                    root[i].center, lA, sign=-1, aux=root[i].aux) if bound_lower else None
                ub = ub + root[i].perturbation.concretize(
                    root[i].center, uA, sign=+1, aux=root[i].aux) if bound_upper else None
        else:
            fv = root[i].forward_value
            if type(root[i]) == BoundInput:
                # Input node with a batch dimension
                batch_size_ = batch_size
            else:
                # Parameter node without a batch dimension
                batch_size_ = 1

            if bound_lower:
                if isinstance(lA, eyeC):
                    lb = lb + fv.view(batch_size_, -1)
                elif isinstance(lA, Patches):
                    lb = lb + lA.matmul(fv, input_shape=root[0].center.shape)
                elif type(root[i]) == BoundInput:
                    lb = lb + lA.matmul(fv.view(batch_size_, -1, 1)).squeeze(-1)
                else:
                    lb = lb + lA.matmul(fv.view(-1, 1)).squeeze(-1)
            else:
                lb = None

            if bound_upper:
                if isinstance(uA, eyeC):
                    ub = ub + fv.view(batch_size_, -1)
                elif isinstance(uA, Patches):
                    ub = ub + uA.matmul(fv, input_shape=root[0].center.shape)
                elif type(root[i]) == BoundInput:
                    ub = ub + uA.matmul(fv.view(batch_size_, -1, 1)).squeeze(-1)
                else:
                    ub = ub + uA.matmul(fv.view(-1, 1)).squeeze(-1)
            else:
                ub = None

    return lb, ub


def addA(A1, A2):
    """ Add two A (each of them is either Tensor or Patches) """
    if type(A1) == type(A2):
        return A1 + A2
    elif type(A1) == Patches:
        return A1 + A2
    elif type(A2) == Patches:
        return A2 + A1
    else:
        raise NotImplementedError(f'Unsupported types for A1 ({type(A1)}) and A2 ({type(A2)}')


def add_bound(node, node_pre, lA, uA):
    """Propagate lA and uA to a preceding node."""
    if lA is not None:
        if node_pre.lA is None:
            # First A added to this node.
            node_pre.zero_lA_mtx = node.zero_backward_coeffs_l
            node_pre.lA = lA
        else:
            node_pre.zero_lA_mtx = node_pre.zero_lA_mtx and node.zero_backward_coeffs_l
            new_node_lA = addA(node_pre.lA, lA)
            node_pre.lA = new_node_lA
    if uA is not None:
        if node_pre.uA is None:
            # First A added to this node.
            node_pre.zero_uA_mtx = node_pre.zero_backward_coeffs_u
            node_pre.uA = uA
        else:
            node_pre.zero_uA_mtx = node_pre.zero_uA_mtx and node.zero_backward_coeffs_u
            node_pre.uA = addA(node_pre.uA, uA)


def get_beta_watch_list(intermediate_constr, all_nodes_before):
    beta_watch_list = defaultdict(dict)
    if intermediate_constr is not None:
        # Intermediate layer betas are handled in two cases.
        # First, if the beta split is before this node, we don't need to do anything special;
        # it will done in BoundRelu.
        # Second, if the beta split after this node, we need to modify the A matrix
        # during bound propagation to reflect beta after this layer.
        for k in intermediate_constr:
            if k not in all_nodes_before:
                # The second case needs special care: we add all such splits in a watch list.
                # However, after first occurance of a layer in the watchlist,
                # beta_watch_list will be deleted and the A matrix from split constraints
                # has been added and will be propagated to later layers.
                for kk, vv in intermediate_constr[k].items():
                    beta_watch_list[kk][k] = vv
    return beta_watch_list


def add_constant_node(lb, ub, node):
    new_lb = node.get_bias(node.lA, node.forward_value)
    new_ub = node.get_bias(node.uA, node.forward_value)
    if isinstance(lb, Tensor) and isinstance(new_lb, Tensor) and lb.ndim > 0 and lb.ndim != new_lb.ndim:
        new_lb = new_lb.reshape(lb.shape)
    if isinstance(ub, Tensor) and isinstance(new_ub, Tensor) and ub.ndim > 0 and ub.ndim != new_ub.ndim:
        new_ub = new_ub.reshape(ub.shape)
    lb = lb + new_lb # FIXME (09/16): shape for the bias of BoundConstant.
    ub = ub + new_ub
    return lb, ub


def save_A_record(node, A_record, A_dict, root, needed_A_dict, lb, ub, unstable_idx):
    root_A_record = {}
    for i in range(len(root)):
        if root[i].lA is None and root[i].uA is None: continue
        if root[i].name in needed_A_dict:
            if root[i].lA is not None:
                if isinstance(root[i].lA, Patches):
                    _lA = root[i].lA
                else:
                    _lA = root[i].lA.transpose(0, 1).detach()
            else:
                _lA = None

            if root[i].uA is not None:
                if isinstance(root[i].uA, Patches):
                    _uA = root[i].uA
                else:
                    _uA = root[i].uA.transpose(0, 1).detach()
            else:
                _uA = None
            root_A_record.update({root[i].name: {
                "lA": _lA,
                "uA": _uA,
                # When not used, lb or ub is tensor(0). They have been transposed above.
                "lbias": lb.detach() if lb.ndim > 1 else None,
                "ubias": ub.detach() if ub.ndim > 1 else None,
                "unstable_idx": unstable_idx
            }})
    root_A_record.update(A_record)  # merge to existing A_record
    A_dict.update({node.name: root_A_record})


def select_unstable_idx(ref_intermediate_lb, ref_intermediate_ub, unstable_locs, max_crown_size):
    """When there are too many unstable neurons, only bound those
    with the loosest reference bounds."""
    gap = (
        ref_intermediate_ub[:, unstable_locs]
        - ref_intermediate_lb[:, unstable_locs]).sum(dim=0)
    indices = torch.argsort(gap, descending=True)
    indices_selected = indices[:max_crown_size]
    indices_selected, _ = torch.sort(indices_selected)
    print(f'{len(indices_selected)}/{len(indices)} unstable neurons selected for CROWN')
    return indices_selected


def get_unstable_locations(
        self, ref_intermediate_lb, ref_intermediate_ub, conv=False, channel_only=False):
    max_crown_size = self.bound_opts.get('max_crown_size', int(1e9))
    # For conv layer we only check the case where all neurons are active/inactive.
    unstable_masks = torch.logical_and(ref_intermediate_lb < 0, ref_intermediate_ub > 0)
    # For simplicity, merge unstable locations for all elements in this batch. TODO: use individual unstable mask.
    # It has shape (H, W) indicating if a neuron is unstable/stable.
    # TODO: so far we merge over the batch dimension to allow easier implementation.
    if channel_only:
        # Only keep channels with unstable neurons. Used for initializing alpha.
        unstable_locs = unstable_masks.sum(dim=(0,2,3)).bool()
        # Shape is consistent with linear layers: a list of unstable neuron channels (no batch dim).
        unstable_idx = unstable_locs.nonzero().squeeze(1)
    else:
        if not conv and unstable_masks.ndim > 2:
            # Flatten the conv layer shape.
            unstable_masks = unstable_masks.view(unstable_masks.size(0), -1)
            ref_intermediate_lb = ref_intermediate_lb.view(ref_intermediate_lb.size(0), -1)
            ref_intermediate_ub = ref_intermediate_ub.view(ref_intermediate_ub.size(0), -1)
        unstable_locs = unstable_masks.sum(dim=0).bool()
        if conv:
            # Now converting it to indices for these unstable nuerons.
            # These are locations (i,j) of unstable neurons.
            unstable_idx = unstable_locs.nonzero(as_tuple=True)
        else:
            unstable_idx = unstable_locs.nonzero().squeeze(1)

    unstable_size = get_unstable_size(unstable_idx)
    if unstable_size > max_crown_size:
        indices_seleted = select_unstable_idx(
            ref_intermediate_lb, ref_intermediate_ub, unstable_locs, max_crown_size)
        if isinstance(unstable_idx, tuple):
            unstable_idx = tuple(u[indices_seleted] for u in unstable_idx)
        else:
            unstable_idx = unstable_idx[indices_seleted]
    unstable_size = get_unstable_size(unstable_idx)

    return unstable_idx, unstable_size


def get_alpha_crown_start_nodes(
        self, node, c=None,  share_slopes=False, final_node_name=None):
    # When use_full_conv_alpha is True, conv layers do not share alpha.
    sparse_intermediate_bounds = self.bound_opts.get('sparse_intermediate_bounds', False)
    use_full_conv_alpha_thresh = self.bound_opts.get('use_full_conv_alpha_thresh', 512)

    start_nodes = []
    for nj in self.backward_from[node.name]:  # Pre-activation layers.
        unstable_idx = None
        use_sparse_conv = None
        use_full_conv_alpha = self.bound_opts.get('use_full_conv_alpha', False)
        if (sparse_intermediate_bounds and isinstance(node, BoundRelu)
                and nj.name != final_node_name and not share_slopes):
            # Create sparse optimization variables for intermediate neurons.
            if ((isinstance(nj, BoundLinear) or isinstance(nj, BoundMatMul))
                    and int(os.environ.get('AUTOLIRPA_USE_FULL_C', 0)) == 0):
                # unstable_idx has shape [neuron_size_of_nj]. Batch dimension is reduced.
                unstable_idx, _ = self.get_unstable_locations(nj.lower, nj.upper)
            elif isinstance(nj, (BoundConv, BoundAdd, BoundSub, BoundBatchNormalization)) and nj.mode == 'patches':
                if nj.name in node.patch_size:
                    # unstable_idx has shape [channel_size_of_nj]. Batch and spatial dimensions are reduced.
                    unstable_idx, _ = self.get_unstable_locations(
                        nj.lower, nj.upper, channel_only=not use_full_conv_alpha, conv=True)
                    use_sparse_conv = False  # alpha is shared among channels. Sparse-spec alpha in hw dimension not used.
                    if use_full_conv_alpha and unstable_idx[0].size(0) > use_full_conv_alpha_thresh:
                        # Too many unstable neurons. Using shared alpha per channel.
                        unstable_idx, _ = self.get_unstable_locations(
                            nj.lower, nj.upper, channel_only=True, conv=True)
                        use_full_conv_alpha = False
                else:
                    # matrix mode for conv layers.
                    # unstable_idx has shape [c_out * h_out * w_out]. Batch dimension is reduced.
                    unstable_idx, _ = self.get_unstable_locations(nj.lower, nj.upper)
                    use_sparse_conv = True  # alpha is not shared among channels, and is sparse in spec dimension.
        if nj.name == final_node_name:
            size_final = self[final_node_name].output_shape[-1] if c is None else c.size(1)
            start_nodes.append((final_node_name, size_final, None))
            continue
        if share_slopes:
            # all intermediate neurons from the same layer share the same set of slopes.
            output_shape = 1
        elif isinstance(node, BoundOptimizableActivation) and node.patch_size and nj.name in node.patch_size:
            # Patches mode. Use output channel size as the spec size. This still shares some alpha, but better than no sharing.
            if use_full_conv_alpha:
                # alphas not shared among channels, so the spec dim shape is c,h,w
                # The patch size is [out_ch, batch, out_h, out_w, in_ch, H, W]. We use out_ch as the output shape.
                output_shape = node.patch_size[nj.name][0], node.patch_size[nj.name][2], node.patch_size[nj.name][3]
            else:
                # The spec dim is c only, and is shared among h, w.
                output_shape = node.patch_size[nj.name][0]
            assert not sparse_intermediate_bounds or use_sparse_conv is False  # Double check our assumption holds. If this fails, then we created wrong shapes for alpha.
        else:
            # Output is linear layer, or patch converted to matrix.
            assert not sparse_intermediate_bounds or use_sparse_conv is not False  # Double check our assumption holds. If this fails, then we created wrong shapes for alpha.
            output_shape = nj.lower.shape[1:]  # FIXME: for non-relu activations it's still expecting a prod.
        start_nodes.append((nj.name, output_shape, unstable_idx))
    return start_nodes
