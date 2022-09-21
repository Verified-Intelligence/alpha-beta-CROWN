""" Convolution, pooling and padding operators"""
from multiprocessing.sharedctypes import Value
from .base import *
from .activation import BoundOptimizableActivation
import numpy as np
from .solver_utils import grb


class BoundMaxPool(BoundOptimizableActivation):
    #FIXME clean up needed
    
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        assert ('pads' not in attr) or (attr['pads'][0] == attr['pads'][2])
        assert ('pads' not in attr) or (attr['pads'][1] == attr['pads'][3])

        self.requires_input_bounds = [0]
        self.kernel_size = attr['kernel_shape']
        self.stride = attr['strides']
        self.padding = [attr['pads'][0], attr['pads'][1]]
        self.ceil_mode = False
        self.use_default_ibp = True
        self.alpha = None
        self.init = {}
        self.alpha_batch_dim = 2

    def forward(self, x):
        output, _ = F.max_pool2d(x, self.kernel_size, self.stride, self.padding, return_indices=True, ceil_mode=self.ceil_mode)
        return output

    def project_simplex(self, patches):
        sorted = torch.flatten(patches, -2)
        sorted, _ = torch.sort(sorted, -1, descending=True)
        rho_sum = torch.cumsum(sorted, -1)
        rho_value = 1 - rho_sum
        rho_value = (sorted + rho_value/torch.tensor(range(1, sorted.size(-1)+1), dtype=torch.float, device=sorted.device)) > 0
        _, rho_index = torch.max(torch.cumsum(rho_value, -1), -1)
        rho_sum = torch.gather(rho_sum, -1, rho_index.unsqueeze(-1)).squeeze(-1)
        lbd = 1/(rho_index+1)* (1-rho_sum)

        return torch.clamp(patches + lbd.unsqueeze(-1).unsqueeze(-1), min=0)

    def init_opt_parameters(self, start_nodes):
        self.alpha = OrderedDict()
        ref = self.inputs[0].lower # a reference variable for getting the shape
        for ns, size_s, unstable_idx in start_nodes:
            if ns == '_forward':
                warnings.warn("MaxPool's optimization is not supported for forward mode")
                continue
            self.alpha[ns] = torch.empty(
                [1, size_s, self.input_shape[0], self.input_shape[1], 
                self.output_shape[-2], self.output_shape[-1], 
                self.kernel_size[0], self.kernel_size[1]], 
                dtype=torch.float, device=ref.device, requires_grad=True)
            self.init[ns] = False

    @staticmethod
    @torch.jit.script
    def jit_mutiply(Apos, Aneg, pos, neg):
        return pos.contiguous() * Apos + neg.contiguous() * Aneg

    def bound_backward(self, last_lA, last_uA, x, start_node=None, start_shape=None, unstable_idx=None):
        # self.padding is a tuple of two elements: (height dimension padding, width dimension padding).
        paddings = tuple((self.padding[0], self.padding[0], self.padding[1], self.padding[1]))

        if self.stride[0] != self.kernel_size[0]:
            raise ValueError("self.stride ({}) != self.kernel_size ({})".format(self.stride, self.kernel_size))
        # if self.padding[0] != 0:
        #     raise ValueError("BoundMaxPool doesn't support padding != 0")

        shape = self.input_shape
        batch_size = x.lower.shape[0]
        shape = list(shape[:-2]) + [a + 2*b for a, b in zip(self.input_shape[-2:], self.padding)]
        shape[0] = batch_size
        # Lower and upper D matrices. They have size (batch_size, input_c, x, y) which will be multiplied on enlarges the A matrices via F.interpolate.
        upper_d = torch.zeros(shape, device=x.device)
        lower_d = None

        # Size of upper_b and lower_b: (batch_size, output_c, h, w).
        upper_b = torch.zeros(batch_size, *self.output_shape[1:], device=x.device)
        lower_b = torch.zeros(batch_size, *self.output_shape[1:], device=x.device)

        # Find the maxpool neuron whose input bounds satisfy l_i > max_j u_j for all j != i. In this case, the maxpool neuron is linear, and we can set upper_d = lower_d = 1.
        # We first find which indices has the largest lower bound.
        max_lower, max_lower_index = F.max_pool2d(x.lower, self.kernel_size, self.stride, self.padding, return_indices=True, ceil_mode=self.ceil_mode)
        # Set the upper bound of the i-th input to -inf so it will not be selected as the max.

        if paddings == (0,0,0,0):
            delete_upper = torch.scatter(
                torch.flatten(x.upper, -2), -1, 
                torch.flatten(max_lower_index, -2), -np.inf).view(upper_d.shape)
        else:
            delete_upper = torch.scatter(torch.flatten(F.pad(x.upper, paddings), -2), -1, torch.flatten(max_lower_index, -2), -np.inf).view(upper_d.shape)
        # Find the the max upper bound over the remaining ones.
        max_upper, _ = F.max_pool2d(delete_upper, self.kernel_size, self.stride, 0, return_indices=True, ceil_mode=self.ceil_mode)

        # The upper bound slope for maxpool is either 1 on input satisfies l_i > max_j u_j (linear), or 0 everywhere. Upper bound is not optimized.
        values = torch.zeros_like(max_lower)
        values[max_lower >= max_upper] = 1.0
        upper_d = torch.scatter(torch.flatten(upper_d, -2), -1, torch.flatten(max_lower_index, -2), torch.flatten(values, -2)).view(upper_d.shape)

        if self.opt_stage == 'opt':
            if unstable_idx is not None and self.alpha[start_node.name].size(1) != 1:
                if unstable_idx.ndim == 1:
                    # Only unstable neurons of the start_node neurons are used.
                    alpha = self.non_deter_index_select(self.alpha[start_node.name], index=unstable_idx, dim=1)
                elif unstable_idx.ndim == 2:
                    # Each element in the batch selects different neurons.
                    alpha = batched_index_select(self.alpha[start_node.name], index=unstable_idx, dim=1)
                else:
                    raise ValueError
            else:
                alpha = self.alpha[start_node.name]

            if not self.init[start_node.name]:
                lower_d = torch.zeros((shape), device=x.device)
                # [batch, C, H, W]
                lower_d = torch.scatter(torch.flatten(lower_d, -2), -1, torch.flatten(max_lower_index, -2), 1.0).view(upper_d.shape)
                # shape [batch, C*k*k, L]
                lower_d_unfold = F.unfold(lower_d, self.kernel_size, 1, stride=self.stride)

                # [batch, C, k, k, out_H, out_W]
                alpha_data = lower_d_unfold.view(lower_d.shape[0], lower_d.shape[1], self.kernel_size[0], self.kernel_size[1], self.output_shape[-2], self.output_shape[-1])
                
                # [batch, C, out_H, out_W, k, k]
                alpha.data.copy_(alpha_data.permute((0,1,4,5,2,3)).clone().detach())
                self.init[start_node.name] = True
                # In optimization mode, we use the same lower_d once builded.
                if self.padding[0] > 0 or self.padding[1] > 0:
                    lower_d = lower_d[...,self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
            # The lower bound coefficients must be positive and projected to an unit simplex.
            alpha.data = self.project_simplex(alpha.data).clone().detach()  # TODO: don't do this, never re-assign the .data property. Use copy_ instead.
            # permute the last 6 dimensions of alpha to [batch, C, k, k, out_H, out_W], which prepares for the unfold operation.
            alpha = alpha.permute((0,1,2,3,6,7,4,5))
            alpha_shape = alpha.shape
            alpha = alpha.reshape((alpha_shape[0]*alpha_shape[1]*alpha_shape[2], -1, alpha_shape[-2]*alpha_shape[-1]))
            lower_d = F.fold(alpha, self.input_shape[-2:], self.kernel_size, 1, self.padding, self.stride)
            lower_d = lower_d.view(alpha_shape[0], alpha_shape[1], alpha_shape[2], *lower_d.shape[1:])
            lower_d = lower_d.squeeze(0)
        else:
            lower_d = torch.zeros((shape), device=x.device)
            # Not optimizable bounds. We simply set \hat{z} >= z_i where i is the input element with largest lower bound.
            lower_d = torch.scatter(torch.flatten(lower_d, -2), -1, torch.flatten(max_lower_index, -2), 1.0).view(upper_d.shape)
            if self.padding[0] > 0 or self.padding[1] > 0:
                lower_d = lower_d[...,self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]

        # For the upper bound, we set the bias term to concrete upper bounds for maxpool neurons that are not linear.
        max_upper_, _ = F.max_pool2d(x.upper, self.kernel_size, self.stride, self.padding, return_indices=True, ceil_mode=self.ceil_mode)
        upper_b[max_upper > max_lower] = max_upper_[max_upper > max_lower]
        
        def _bound_oneside(last_A, d_pos, d_neg, b_pos, b_neg):
            if last_A is None:
                return None, 0

            bias = 0

            if isinstance(last_A, torch.Tensor):
                pos_A = last_A.clamp(min=0)
                neg_A = last_A.clamp(max=0)

                if b_pos is not None:
                    # This is matrix mode, and padding is considered in the previous layers
                    bias = bias + self.get_bias(pos_A, b_pos)
                if b_neg is not None:
                    bias = bias + self.get_bias(neg_A, b_neg)

                # Here we should comfirm that the maxpool patches are not overlapped.
                shape = last_A.size()

                padding = [self.padding[0], self.padding[0], self.padding[1], self.padding[1]]
                d_pos = F.pad(d_pos, padding)
                d_neg = F.pad(d_neg, padding)

                pos_A = F.interpolate(pos_A.view(shape[0] * shape[1], *shape[2:]), scale_factor=self.kernel_size)
                if d_pos.shape[-2] > pos_A.shape[-2] or d_pos.shape[-1] > pos_A.shape[-1]:
                    if not (d_pos.shape[-2] > pos_A.shape[-2] and d_pos.shape[-1] > pos_A.shape[-1]):
                        raise NotImplementedError("Asymmetric padding of maxpool not implemented.")
                    pos_A = F.pad(pos_A, (0, d_pos.shape[-2] - pos_A.shape[-2], 0, d_pos.shape[-1] - pos_A.shape[-1]))
                else:
                    d_pos = F.pad(d_pos, (0, pos_A.shape[-2] - d_pos.shape[-2], 0, pos_A.shape[-1] - d_pos.shape[-1]))
                pos_A = pos_A.view(shape[0], shape[1], *pos_A.shape[1:])

                neg_A = F.interpolate(neg_A.view(shape[0] * shape[1], *shape[2:]), scale_factor=self.kernel_size)
                if d_neg.shape[-2] > neg_A.shape[-2] or d_neg.shape[-1] > neg_A.shape[-1]:
                    if not (d_neg.shape[-2] > neg_A.shape[-2] and d_neg.shape[-1] > neg_A.shape[-1]):
                        raise NotImplementedError("Asymmetric padding of maxpool not implemented.")
                    neg_A = F.pad(neg_A, (0, d_neg.shape[-2] - neg_A.shape[-2], 0, d_neg.shape[-1] - neg_A.shape[-1]))
                else:
                    d_neg = F.pad(d_neg, (0, neg_A.shape[-2] - d_neg.shape[-2], 0, neg_A.shape[-1] - d_neg.shape[-1]))
                neg_A = neg_A.view(shape[0], shape[1], *neg_A.shape[1:])

                next_A = self.jit_mutiply(pos_A, neg_A, d_pos, d_neg)
                if self.padding[0] > 0 or self.padding[1] > 0:
                    next_A = next_A[...,self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
            elif isinstance(last_A, Patches):
                # The last_A.patches was not padded, so we need to pad them here.
                # If this Conv layer is followed by a ReLU layer, then the padding was already handled there and there is no need to pad again.
                one_d = torch.ones(tuple(1 for i in self.output_shape[1:]), device=last_A.patches.device, dtype=last_A.patches.dtype).expand(self.output_shape[1:])
                # Add batch dimension.
                one_d = one_d.unsqueeze(0)
                # After unfolding, the shape is (1, out_h, out_w, in_c, h, w)
                one_d_unfolded = inplace_unfold(one_d, kernel_size=last_A.patches.shape[-2:], stride=last_A.stride, padding=last_A.padding, inserted_zeros=last_A.inserted_zeros, output_padding=last_A.output_padding)
                if last_A.unstable_idx is not None:
                    # Move out_h, out_w dimension to the front for easier selection.
                    one_d_unfolded_r = one_d_unfolded.permute(1, 2, 0, 3, 4, 5)
                    # for sparse patches the shape is (unstable_size, batch, in_c, h, w). Batch size is 1 so no need to select here.
                    one_d_unfolded_r = one_d_unfolded_r[last_A.unstable_idx[1], last_A.unstable_idx[2]]
                else:
                    # Append the spec dimension.
                    one_d_unfolded_r = one_d_unfolded.unsqueeze(0)
                patches = last_A.patches * one_d_unfolded_r

                if b_pos is not None:
                    patch_pos = Patches(
                        patches.clamp(min=0), last_A.stride, last_A.padding, 
                        last_A.shape, unstable_idx=last_A.unstable_idx, 
                        output_shape=last_A.output_shape)
                    bias = bias + self.get_bias(patch_pos, b_pos)
                if b_neg is not None:
                    patch_neg = Patches(
                        patches.clamp(max=0), last_A.stride, last_A.padding, 
                        last_A.shape, unstable_idx=last_A.unstable_idx, 
                        output_shape=last_A.output_shape)                    
                    bias = bias + self.get_bias(patch_neg, b_neg)
                
                # bias = bias.transpose(0,1)
                shape = last_A.shape
                pos_A = last_A.patches.clamp(min=0)
                neg_A = last_A.patches.clamp(max=0)

                def upsample(last_patches, last_A):
                    if last_A.unstable_idx is None:
                        patches = F.interpolate(
                            last_patches.view(shape[0] * shape[1] * shape[2], *shape[3:]), 
                            scale_factor=[1,]+self.kernel_size)
                        patches = patches.view(shape[0], shape[1], shape[2], *patches.shape[1:])
                    else:
                        patches = F.interpolate(
                            last_patches, scale_factor=[1,] + self.kernel_size)
                    return Patches(
                        patches, stride=last_A.stride, padding=last_A.padding, 
                        shape=patches.shape, unstable_idx=last_A.unstable_idx, 
                        output_shape=last_A.output_shape)

                pos_A = upsample(pos_A, last_A)
                neg_A = upsample(neg_A, last_A)
                
                stride = self.stride[0] * last_A.stride
                if isinstance(last_A.padding, int):
                    padding = last_A.padding * self.stride[0] + self.padding[0]
                else:
                    # Here we need to unfold the d_pos to match pos_A and neg_A patches
                    # And we compute the padding and stride of pos_A and neg_A
                    padding = tuple(a * self.stride[0] + self.padding[0] for a in last_A.padding)

                padding, stride, output_padding = compute_patches_stride_padding(
                self.input_shape, last_A.padding, last_A.stride, self.padding, self.stride, last_A.inserted_zeros, last_A.output_padding)

                pos_A.padding, pos_A.stride, pos_A.output_padding = padding, stride, output_padding
                neg_A.padding, neg_A.stride, neg_A.output_padding = padding, stride, output_padding

                # unsqueeze for the spec dimension
                d_pos  = maybe_unfold_patches(d_pos.unsqueeze(0), pos_A)
                d_neg  = maybe_unfold_patches(d_neg.unsqueeze(0), neg_A)


                next_A_patches = self.jit_mutiply(pos_A.patches, neg_A.patches, d_pos, d_neg)

                if start_node is not None:
                    self.patch_size[start_node.name] = next_A_patches.size()


                next_A = Patches(
                    next_A_patches, stride, padding, next_A_patches.shape,
                    unstable_idx=last_A.unstable_idx, output_shape=last_A.output_shape,
                    inserted_zeros=last_A.inserted_zeros, output_padding=output_padding)

            return next_A, bias

        if self.padding[0] > 0:
            upper_d = upper_d[...,self.padding[0]:-self.padding[0], self.padding[0]:-self.padding[0]]

        uA, ubias = _bound_oneside(last_uA, upper_d, lower_d, upper_b, lower_b)
        lA, lbias = _bound_oneside(last_lA, lower_d, upper_d, lower_b, upper_b)

        return [(lA, uA)], lbias, ubias

    def bound_forward(self, dim_in, x):
        lower_d, lower_b, upper_d, upper_b = self.bound_relax(x)

        def _bound_oneside(w_pos, b_pos, w_neg, b_neg, d, b):
            d_pos, d_neg = d.clamp(min=0), d.clamp(max=0)
            w_new = d_pos.unsqueeze(1) * w_pos + d_neg.unsqueeze(1) * w_neg
            b_new = d_pos * b_pos + d_neg * b_neg
            if isinstance(self.kernel_size, list) and len(self.kernel_size) == 2:
                tot_kernel_size = prod(self.kernel_size)
            elif isinstance(self.kernel_size, int):
                tot_kernel_size = self.kernel_size ** 2
            else:
                raise ValueError(f'Unsupported kernel size {self.kernel_size}')
            w_pooled = (F.avg_pool2d(w_new.view(-1, *w_new.shape[2:]), 
                self.kernel_size, self.stride, self.padding, 
                ceil_mode=self.ceil_mode) * tot_kernel_size)
            w_pooled = w_pooled.reshape(w_new.shape[0], -1, *w_pooled.shape[1:])
            b_pooled = F.avg_pool2d(b_new, self.kernel_size, self.stride, self.padding, 
                ceil_mode=self.ceil_mode) * tot_kernel_size + b
            return w_pooled, b_pooled

        lw, lb = _bound_oneside(x.lw, x.lb, x.uw, x.ub, lower_d, lower_b)
        uw, ub = _bound_oneside(x.uw, x.ub, x.lw, x.lb, upper_d, upper_b)

        return LinearBound(lw, lb, uw, ub)

    def bound_relax(self, x):
        # Only used by forward mode
        paddings = tuple(self.padding + self.padding)
        self.upper, self.lower = x.upper, x.lower

        # A_shape = last_lA.shape if last_lA is not None else last_uA.shape
        # batch_size, input_c, x, y
        upper_d = torch.zeros_like(x.lower)
        lower_d = torch.zeros_like(x.lower)

        upper_d = F.pad(upper_d, paddings)
        lower_d = F.pad(lower_d, paddings)

        # batch_size, output_c, x, y
        upper_b = torch.zeros((list(self.output_shape))).to(x.lower)
        lower_b = torch.zeros((list(self.output_shape))).to(x.lower)

        # 1. find the index i where li > uj for all j, then set upper_d = lower_d = 1
        max_lower, max_lower_index = F.max_pool2d(x.lower, self.kernel_size, self.stride, self.padding, return_indices=True, ceil_mode=self.ceil_mode)
        delete_upper = torch.scatter(torch.flatten(F.pad(x.upper, paddings), -2), -1, torch.flatten(max_lower_index, -2), -np.inf).view(upper_d.shape)
        max_upper, _ = F.max_pool2d(delete_upper, self.kernel_size, self.stride, 0, return_indices=True, ceil_mode=self.ceil_mode)

        values = torch.zeros_like(max_lower)
        values[max_lower >= max_upper] = 1.0
        upper_d = torch.scatter(torch.flatten(upper_d, -2), -1, torch.flatten(max_lower_index, -2), torch.flatten(values, -2)).view(upper_d.shape)

        # FIXME shape error
        if False and self.opt_stage == 'opt':
            alpha = self.alpha[self._start]

            if self.init[self._start] == False:
                lower_d = torch.scatter(torch.flatten(lower_d, -2), -1, torch.flatten(max_lower_index, -2), 1.0).view(upper_d.shape)
                lower_d_unfold = F.unfold(lower_d, self.kernel_size, 1, stride=self.stride)

                alpha_data = lower_d_unfold.view(lower_d.shape[0], lower_d.shape[1], self.kernel_size[0], self.kernel_size[1], self.output_shape[-2], self.output_shape[-1])
                alpha.data.copy_(alpha_data.permute((0,1,4,5,2,3)).clone().detach())
                self.init[self._start] = True
                if self.padding[0] > 0:
                    lower_d = lower_d[...,self.padding[0]:-self.padding[0], self.padding[0]:-self.padding[0]]

            alpha.data = self.project_simplex(alpha.data).clone().detach()
            alpha = alpha.permute((0,1,2,3,6,7,4,5))
            alpha_shape = alpha.shape
            alpha = alpha.reshape((alpha_shape[0]*alpha_shape[1]*alpha_shape[2], -1, alpha_shape[-2]*alpha_shape[-1]))
            lower_d = F.fold(alpha, self.input_shape[-2:], self.kernel_size, 1, self.padding, self.stride)
            lower_d = lower_d.view(alpha_shape[0], alpha_shape[1], alpha_shape[2], *lower_d.shape[1:])
            lower_d = lower_d.squeeze(0)
        else:
            lower_d = torch.scatter(torch.flatten(lower_d, -2), -1, torch.flatten(max_lower_index, -2), 1.0).view(upper_d.shape)
            if self.padding[0] > 0:
                lower_d = lower_d[...,self.padding[0]:-self.padding[0], self.padding[0]:-self.padding[0]]

        values[:] = 0.0
        max_upper_, _ = F.max_pool2d(x.upper, self.kernel_size, self.stride, self.padding, return_indices=True, ceil_mode=self.ceil_mode)
        values[max_upper > max_lower] = max_upper_[max_upper > max_lower]
        upper_b = values

        if self.padding[0] > 0:
            upper_d = upper_d[...,self.padding[0]:-self.padding[0], self.padding[0]:-self.padding[0]]

        return lower_d, lower_b, upper_d, upper_b

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        # e.g., last layer input gurobi vars (3,32,32)
        gvars_array = np.array(v[0])
        # pre_layer_shape (1,32,27,27)
        pre_layer_shape = np.expand_dims(gvars_array, axis=0).shape
        # this layer shape (1,32,6,6)
        this_layer_shape = self.output_shape
        assert this_layer_shape[2] ==  ((2 * self.padding[0] + pre_layer_shape[2] - (self.stride[0] - 1))//self.stride[0])

        new_layer_gurobi_vars = []
        neuron_idx = 0
        pre_ubs = self.forward(self.inputs[0].upper).detach().cpu().numpy()

        for out_chan_idx in range(this_layer_shape[1]):
            out_chan_vars = []
            for out_row_idx in range(this_layer_shape[2]):
                out_row_vars = []
                for out_col_idx in range(this_layer_shape[3]):
                    a_sum = 0.0
                    v = model.addVar(lb=-float('inf'), ub=float('inf'),
                                            obj=0, vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{self.name}_{neuron_idx}')
                    for ker_row_idx in range(self.kernel_size[0]):
                        in_row_idx = -self.padding[0] + self.stride[0] * out_row_idx + ker_row_idx
                        if (in_row_idx < 0) or (in_row_idx == len(gvars_array[out_chan_idx][ker_row_idx])):
                            # This is padding -> value of 0
                            continue
                        for ker_col_idx in range(self.kernel_size[1]):
                            in_col_idx = -self.padding[1] + self.stride[1] * out_col_idx + ker_col_idx
                            if (in_col_idx < 0) or (in_col_idx == pre_layer_shape[3]):
                                # This is padding -> value of 0
                                continue
                            var = gvars_array[out_chan_idx][in_row_idx][in_col_idx]
                            a = model.addVar(vtype=grb.GRB.BINARY)
                            a_sum += a
                            model.addConstr(v >= var)
                            model.addConstr(v <= var + (1 - a) * pre_ubs[0, out_chan_idx, out_row_idx, out_col_idx])
                    model.addConstr(a_sum == 1, name=f'lay{self.name}_{neuron_idx}_eq')
                    out_row_vars.append(v)
                out_chan_vars.append(out_row_vars)
            new_layer_gurobi_vars.append(out_chan_vars)

        self.solver_vars = new_layer_gurobi_vars
        model.update()



class BoundGlobalAveragePool(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x):
        output = nn.AdaptiveAvgPool2d((1, 1)).forward(x)  # adaptiveAveragePool with output size (1, 1)
        return output

    def bound_backward(self, last_lA, last_uA, x):
        H, W = self.input_shape[-2], self.input_shape[-1]

        lA = (last_lA.expand(list(last_lA.shape[:-2]) + [H, W]) / (H * W)) if last_lA is not None else None
        uA = (last_uA.expand(list(last_uA.shape[:-2]) + [H, W]) / (H * W)) if last_uA is not None else None

        return [(lA, uA)], 0, 0

    def interval_propagate(self, *v):
        h_L, h_U = v[0]
        h_L = F.adaptive_avg_pool2d(h_L, (1, 1))
        h_U = F.adaptive_avg_pool2d(h_U, (1, 1))
        return h_L, h_U


class BoundAveragePool(Bound):
    def __init__(self, attr, inputs, output_index, options):
        # assumptions: ceil_mode=False, count_include_pad=True
        super().__init__(attr, inputs, output_index, options)

        assert ('pads' not in attr) or (attr['pads'][0] == attr['pads'][2])
        assert ('pads' not in attr) or (attr['pads'][1] == attr['pads'][3])

        self.kernel_size = attr['kernel_shape']
        assert len(self.kernel_size) == 2
        self.stride = attr['strides']
        assert len(self.stride) == 2
        # FIXME (22/07/02): padding is inconsistently handled. Should use 4-tuple.

        if 'pads' not in attr:
            self.padding = [0, 0]
        else:
            self.padding = [attr['pads'][0], attr['pads'][1]]
        self.ceil_mode = False
        self.count_include_pad = True
        self.use_default_ibp = True

    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            if isinstance(last_A, torch.Tensor):
                shape = last_A.size()
                # propagate A to the next layer, with batch concatenated together
                next_A = F.interpolate(last_A.view(shape[0] * shape[1], *shape[2:]),
                    scale_factor=self.kernel_size) / (prod(self.kernel_size))
                next_A = F.pad(next_A, (0, self.input_shape[-2] - next_A.shape[-2], 0, self.input_shape[-1] - next_A.shape[-1]))
                next_A = next_A.view(shape[0], shape[1], *next_A.shape[1:])
            elif isinstance(last_A, Patches):
                patches = last_A.patches
                shape = patches.size()
                # When the number of inserted zeros can cancel out the stride, we use a shortcut that can reduce computation.
                simplify_patch = (last_A.inserted_zeros + 1 == self.kernel_size[0]) and (self.kernel_size[0] == self.kernel_size[1])
                padding, stride, output_padding = compute_patches_stride_padding(
                        self.input_shape, last_A.padding, last_A.stride, self.padding, self.stride,
                        inserted_zeros=last_A.inserted_zeros, output_padding=last_A.output_padding, simplify=not simplify_patch)
                inserted_zeros = last_A.inserted_zeros
                if last_A.inserted_zeros == 0:
                    # No inserted zeros, can be handled using interpolate.
                    if last_A.unstable_idx is None:
                        # shape is: [out_C, batch, out_H, out_W, in_c, patch_H, patch_W]
                        up_sampled_patches = F.interpolate(patches.view(shape[0] * shape[1], shape[2] * shape[3], *shape[4:]), scale_factor=[1,] + self.kernel_size)
                        # The dimension of patch-H and patch_W has changed.
                        up_sampled_patches = up_sampled_patches.view(*shape[:-2], up_sampled_patches.size(-2), up_sampled_patches.size(-1))
                    else:
                        # shape is: [spec, batch, in_c, patch_H, patch_W]
                        up_sampled_patches = F.interpolate(patches, scale_factor=[1,] + self.kernel_size)
                    # Divided by the averaging factor.
                    up_sampled_patches = up_sampled_patches / prod(self.kernel_size)
                elif simplify_patch:
                    padding = tuple(p // s - o for p, s, o in zip(padding, stride, output_padding))
                    output_padding = (0, 0, 0, 0)
                    stride = 1  # Stride and inserted zero canceled out. No need to insert zeros and add output_padding.
                    inserted_zeros = 0
                    value = 1. / prod(self.kernel_size)
                    # In the case where the stride and adding_zeros cancel out, we do not need to insert zeros.
                    weight = torch.full(size=(self.input_shape[1], 1, *self.kernel_size), fill_value=value, dtype=patches.dtype, device=patches.device)
                    if last_A.unstable_idx is None:
                        # shape is: [out_C, batch, out_H, out_W, in_c, patch_H, patch_W]
                        up_sampled_patches = F.conv_transpose2d(patches.reshape(shape[0] * shape[1] * shape[2] * shape[3], *shape[4:]), weight, stride=1, groups=self.input_shape[1])
                    else:
                        # shape is: [spec, batch, in_c, patch_H, patch_W]
                        up_sampled_patches = F.conv_transpose2d(patches.reshape(shape[0] * shape[1], *shape[2:]), weight, stride=1, groups=self.input_shape[1])
                    up_sampled_patches = up_sampled_patches.view(*shape[:-2], up_sampled_patches.size(-2), up_sampled_patches.size(-1))
                else:
                    # With inserted zeros, must be handled by treating pooling as general convolution.
                    value = 1. / prod(self.kernel_size)
                    weight = torch.full(size=(self.input_shape[1], 1, *self.kernel_size), fill_value=value, dtype=patches.dtype, device=patches.device)
                    weight = insert_zeros(weight, last_A.inserted_zeros)
                    if last_A.unstable_idx is None:
                        # shape is: [out_C, batch, out_H, out_W, in_c, patch_H, patch_W]
                        up_sampled_patches = F.conv_transpose2d(patches.reshape(shape[0] * shape[1] * shape[2] * shape[3], *shape[4:]), weight, stride=self.kernel_size, groups=self.input_shape[1])
                    else:
                        # shape is: [spec, batch, in_c, patch_H, patch_W]
                        up_sampled_patches = F.conv_transpose2d(patches.reshape(shape[0] * shape[1], *shape[2:]), weight, stride=self.kernel_size, groups=self.input_shape[1])
                    up_sampled_patches = up_sampled_patches.view(*shape[:-2], up_sampled_patches.size(-2), up_sampled_patches.size(-1))
                next_A = last_A.create_similar(up_sampled_patches, stride=stride, padding=padding, output_padding=output_padding, inserted_zeros=inserted_zeros)
            else:
                raise ValueError(f'last_A has unexpected shape {type(last_A)}')
            return next_A, 0.

        lA, lbias = _bound_oneside(last_lA)
        uA, ubias = _bound_oneside(last_uA)
        return [(lA, uA)], lbias, ubias
    
    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        # e.g., last layer input gurobi vars (3,32,32)
        gvars_array = np.array(v[0])
        # pre_layer_shape (1,32,27,27)
        pre_layer_shape = np.expand_dims(gvars_array, axis=0).shape
        # this layer shape (1,32,6,6)
        this_layer_shape = self.output_shape
        assert this_layer_shape[2] ==  ((2 * self.padding[0] + pre_layer_shape[2] - (self.stride[0] - 1))//self.stride[0])
        
        value = 1.0/(self.kernel_size[0] * self.kernel_size[1])
        new_layer_gurobi_vars = []
        neuron_idx = 0
        for out_chan_idx in range(this_layer_shape[1]):
            out_chan_vars = []
            for out_row_idx in range(this_layer_shape[2]):
                out_row_vars = []
                for out_col_idx in range(this_layer_shape[3]):
                    # print(self.bias.shape, out_chan_idx, out_lbs.size(1))
                    lin_expr = 0.0
                    for ker_row_idx in range(self.kernel_size[0]):
                        in_row_idx = -self.padding[0] + self.stride[0] * out_row_idx + ker_row_idx
                        if (in_row_idx < 0) or (in_row_idx == len(gvars_array[out_chan_idx][ker_row_idx])):
                            # This is padding -> value of 0
                            continue
                        for ker_col_idx in range(self.kernel_size[1]):
                            in_col_idx = -self.padding[1] + self.stride[1] * out_col_idx + ker_col_idx
                            if (in_col_idx < 0) or (in_col_idx == pre_layer_shape[3]):
                                # This is padding -> value of 0
                                continue
                            coeff = value
                            lin_expr += coeff * gvars_array[out_chan_idx][in_row_idx][in_col_idx]
                    v = model.addVar(lb=-float('inf'), ub=float('inf'),
                                            obj=0, vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{self.name}_{neuron_idx}')
                    model.addConstr(lin_expr == v, name=f'lay{self.name}_{neuron_idx}_eq')
                    neuron_idx += 1

                    out_row_vars.append(v)
                out_chan_vars.append(out_row_vars)
            new_layer_gurobi_vars.append(out_chan_vars)

        self.solver_vars = new_layer_gurobi_vars
        model.update()
