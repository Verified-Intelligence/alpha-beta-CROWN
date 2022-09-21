""" Convolution and padding operators"""
from .base import *
import numpy as np
from .solver_utils import grb
from ..patches import unify_shape, compute_patches_stride_padding, is_shape_used


class BoundConv(Bound):
    def __init__(self, attr, inputs, output_index, options):
        assert (attr['pads'][0] == attr['pads'][2])
        assert (attr['pads'][1] == attr['pads'][3])

        super().__init__(attr, inputs, output_index, options)

        self.stride = attr['strides']
        self.padding = [attr['pads'][0], attr['pads'][1]]
        self.dilation = attr['dilations']
        self.groups = attr['group']
        if len(inputs) == 3:
            self.has_bias = True
        else:
            self.has_bias = False
        self.relu_followed = False
        self.patches_start = True
        self.mode = options.get("conv_mode", "matrix")
        # denote whether this Conv is followed by a ReLU
        # if self.relu_followed is False, we need to manually pad the conv patches.
        # If self.relu_followed is True, the patches are padded in the ReLU layer and the manual padding is not needed.

    def forward(self, *x):
        # x[0]: input, x[1]: weight, x[2]: bias if self.has_bias
        bias = x[2] if self.has_bias else None
        output = F.conv2d(x[0], x[1], bias, self.stride, self.padding, self.dilation, self.groups)
        return output

    def bound_backward(self, last_lA, last_uA, *x):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        lA_y = uA_y = lA_bias = uA_bias = None
        weight = x[1].lower

        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            if type(last_A) is OneHotC:
                # Conv layer does not support the OneHotC fast path. We have to create a dense matrix instead.
                shape = last_A.shape  # [spec, batch, C, H, W]
                dim = int(prod(shape[2:]))
                dense_last_A = torch.zeros(size=(shape[0], shape[1], dim), device=last_A.device, dtype=weight.dtype)
                # last_A.index has size (spec, batch), its values are the index of the one-hot non-zero elements in A.
                # last_A.coeffs is the value of the non-zero element.
                dense_last_A = torch.scatter(dense_last_A, dim=2, index=last_A.index.unsqueeze(-1), src=last_A.coeffs.unsqueeze(-1))
                # We created a large A matrix and it will be handled below.
                last_A = dense_last_A.view(shape[0], shape[1], *shape[2:])

            if type(last_A) == Tensor:
                shape = last_A.size()
                # when (W−F+2P)%S != 0, construct the output_padding
                output_padding0 = int(self.input_shape[2]) - (int(self.output_shape[2]) - 1) * self.stride[0] + 2 * \
                                self.padding[0] - 1 - (int(weight.size()[2] - 1) * self.dilation[0])
                output_padding1 = int(self.input_shape[3]) - (int(self.output_shape[3]) - 1) * self.stride[1] + 2 * \
                                self.padding[1] - 1 - (int(weight.size()[3] - 1) * self.dilation[0])
                next_A = F.conv_transpose2d(last_A.reshape(shape[0] * shape[1], *shape[2:]), weight, None,
                                            stride=self.stride, padding=self.padding, dilation=self.dilation,
                                            groups=self.groups, output_padding=(output_padding0, output_padding1))
                next_A = next_A.view(shape[0], shape[1], *next_A.shape[1:])
                if self.has_bias:
                    # sum_bias = (last_A.sum((3, 4)) * x[2].lower).sum(2)
                    sum_bias = torch.einsum('sbchw,c->sb', last_A, x[2].lower)
                else:
                    sum_bias = 0
                return next_A, sum_bias
            elif type(last_A) == Patches:
                # Here we build and propagate a Patch object with (patches, stride, padding)
                assert type(last_A) == Patches
                if last_A.identity == 0:
                    if not self.relu_followed:  # FIXME (09/20): Don't call it relu_followed. Instead, make this a property of A, called "padded" and propagate this property.
                        # The last_A.patches was not padded, so we need to pad them here.
                        # If this Conv layer is followed by a ReLU layer, then the padding was already handled there and there is no need to pad again.
                        one_d = torch.ones(tuple(1 for i in self.output_shape[1:]), device=last_A.patches.device, dtype=weight.dtype).expand(self.output_shape[1:])
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
                    else:
                        patches = last_A.patches

                    if self.has_bias:
                        # bias is x[2] (lower and upper are the same), and has shape (c,).
                        # Patches either has [out_c, batch, out_h, out_w, c, h, w] or [unstable_size, batch, c, h, w].
                        sum_bias = torch.einsum('sb...chw,c->sb...', patches, x[2].lower)
                        # sum_bias has shape (out_c, batch, out_h, out_w) or (unstable_size, batch).
                    else:
                        sum_bias = 0

                    flattened_patches = patches.reshape(-1, patches.size(-3), patches.size(-2), patches.size(-1))
                    pieces = F.conv_transpose2d(flattened_patches, insert_zeros(weight, last_A.inserted_zeros), stride=self.stride)
                    # New patch size: (out_c, batch, out_h, out_w, c, h, w) or (unstable_size, batch, c, h, w).
                    pieces = pieces.view(*patches.shape[:-3], pieces.size(-3), pieces.size(-2), pieces.size(-1))

                elif last_A.identity == 1:
                    # New patches have size [out_c, batch, out_h, out_w, c, h, w] if it is not sparse.
                    # New patches have size [unstable_size, batch, c, h, w] if it is sparse.
                    if last_A.unstable_idx is not None:
                        pieces = weight.view(weight.size(0), 1, weight.size(1), weight.size(2), weight.size(3))
                        # Select based on the output channel (out_h and out_w are irrelevant here).
                        pieces = pieces[last_A.unstable_idx[0]]
                        # Expand the batch dimnension.
                        pieces = pieces.expand(-1, last_A.shape[1], -1, -1, -1)
                        # Do the same for the bias.
                        if self.has_bias:
                            sum_bias = x[2].lower[last_A.unstable_idx[0]].unsqueeze(-1)
                            # bias has shape (unstable_size, batch).
                            sum_bias = sum_bias.expand(-1, last_A.shape[1])
                        else:
                            sum_bias = 0
                    else:
                        assert weight.size(0) == last_A.shape[0]
                        pieces = weight.view(weight.size(0), 1, 1, 1, weight.size(1), weight.size(2), weight.size(3)).expand(-1, *last_A.shape[1:4], -1, -1, -1)
                        # The bias (x[2].lower) has shape (out_c,) need to make it (out_c, batch, out_h, out_w).
                        # Here we should transpose sum_bias to set the batch dim to 1, aiming to keep it consistent with the matrix version
                        if self.has_bias:
                            sum_bias = x[2].lower.view(-1, 1, 1, 1).expand(-1, *last_A.shape[1:4])
                        else:
                            sum_bias = 0
                else:
                    raise NotImplementedError()
                padding = last_A.padding if last_A is not None else (0, 0, 0, 0)  # (left, right, top, bottom)
                stride = last_A.stride if last_A is not None else (1, 1)
                inserted_zeros = last_A.inserted_zeros if last_A is not None else 0
                output_padding = last_A.output_padding if last_A is not None else (0, 0, 0, 0)

                padding, stride, output_padding = compute_patches_stride_padding(self.input_shape, padding, stride, self.padding, self.stride, inserted_zeros, output_padding)

                if inserted_zeros == 0 and not is_shape_used(output_padding) and pieces.shape[-1] > self.input_shape[-1]:  # the patches is too large and from now on, we will use matrix mode instead of patches mode.
                    # This is our desired matrix: the input will be flattend to (batch_size, input_channel*input_x * input_y) and multiplies on this matrix.
                    # After multiplication, the desired output is (batch_size, out_channel*output_x*output_y).
                    # A_matrix has size (batch, out_c*out_h*out_w, in_c*in_h*in_w)
                    A_matrix = patches_to_matrix(pieces, self.input_shape[1:], stride, padding, last_A.output_shape, last_A.unstable_idx)
                    # print(f'Converting patches to matrix: old shape {pieces.shape}, size {pieces.numel()}; new shape {A_matrix.shape}, size {A_matrix.numel()}')
                    if isinstance(sum_bias, Tensor) and last_A.unstable_idx is None:
                        sum_bias = sum_bias.transpose(0, 1)
                        sum_bias = sum_bias.reshape(sum_bias.size(0), -1).transpose(0,1)
                    A_matrix = A_matrix.transpose(0,1)  # Spec dimension at the front.
                    return A_matrix, sum_bias
                # print(f'Conv returns patches with size={pieces.size()}, stride={stride}, padding={padding}, inserted_zeros={inserted_zeros}, output_padding={output_padding}')
                return Patches(pieces, stride, padding, pieces.shape, unstable_idx=last_A.unstable_idx, output_shape=last_A.output_shape, inserted_zeros=last_A.inserted_zeros, output_padding=output_padding), sum_bias
            else:
                raise NotImplementedError()

        lA_x, lbias = _bound_oneside(last_lA)
        uA_x, ubias = _bound_oneside(last_uA)
        return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        assert self.dilation == (1, 1) or self.dilation == [1, 1]
        # e.g., last layer input gurobi vars (3,32,32)
        gvars_array = np.array(v[0])
        # pre_layer_shape (1,3,32,32)
        pre_layer_shape = np.expand_dims(gvars_array, axis=0).shape
        # this layer shape (1,8,16,16)
        this_layer_shape = self.output_shape
        out_lbs, out_ubs = None, None
        if hasattr(self, "lower"):
            # self.lower shape (1,8,16,16)
            out_lbs = self.lower.cpu().numpy()
            out_ubs = self.upper.cpu().numpy()

        # current layer weight (8,3,4,4)
        this_layer_weight = v[1].detach().cpu().numpy()
        # current layer bias (8,)
        this_layer_bias = None
        if self.has_bias:
            this_layer_bias = v[2].detach().cpu().numpy()
        weight_shape2, weight_shape3 = this_layer_weight.shape[2], this_layer_weight.shape[3]
        padding0, padding1 = self.padding[0], self.padding[1]
        stride0, stride1 = self.stride[0], self.stride[1]

        new_layer_gurobi_vars = []
        new_layer_gurobi_constrs = []

        neuron_idx = 0
        for out_chan_idx in range(this_layer_shape[1]):
            out_chan_vars = []
            for out_row_idx in range(this_layer_shape[2]):
                out_row_vars = []
                for out_col_idx in range(this_layer_shape[3]):
                    # print(this_layer_bias.shape, out_chan_idx, out_lbs.size(1))
                    lin_expr = 0
                    if self.has_bias:
                        lin_expr = this_layer_bias[out_chan_idx]

                    for in_chan_idx in range(this_layer_weight.shape[1]):

                        # new version of conv layer for building mip by skipping kernel loops
                        ker_row_min, ker_row_max = 0, weight_shape2
                        in_row_idx_min = -padding0 + stride0 * out_row_idx
                        in_row_idx_max = in_row_idx_min + weight_shape2 - 1
                        if in_row_idx_min < 0:
                            ker_row_min = -in_row_idx_min
                        if in_row_idx_max >= pre_layer_shape[2]:
                            ker_row_max = ker_row_max - in_row_idx_max + pre_layer_shape[2] -1
                        in_row_idx_min, in_row_idx_max = max(in_row_idx_min, 0), min(in_row_idx_max, pre_layer_shape[2] - 1)

                        ker_col_min, ker_col_max = 0, weight_shape3
                        in_col_idx_min = -padding1 + stride1 * out_col_idx
                        in_col_idx_max = in_col_idx_min + weight_shape3 - 1
                        if in_col_idx_min < 0:
                            ker_col_min = -in_col_idx_min
                        if in_col_idx_max >= pre_layer_shape[3]:
                            ker_col_max = ker_col_max - in_col_idx_max + pre_layer_shape[3] -1
                        in_col_idx_min, in_col_idx_max = max(in_col_idx_min, 0), min(in_col_idx_max, pre_layer_shape[3] - 1)

                        coeffs = this_layer_weight[out_chan_idx, in_chan_idx, ker_row_min:ker_row_max, ker_col_min:ker_col_max].reshape(-1)

                        gvars = gvars_array[in_chan_idx, in_row_idx_min:in_row_idx_max+1, in_col_idx_min:in_col_idx_max+1].reshape(-1)
                        if solver_pkg == 'gurobi':
                            lin_expr += grb.LinExpr(coeffs, gvars)
                        else:
                            # lin_expr += coeffs@gvars

                            for i in range(len(coeffs)):
                                try:
                                    lin_expr += coeffs[i] * gvars[i]
                                except TypeError:
                                    lin_expr += coeffs[i] * gvars[i].var


                    out_lb = out_lbs[0, out_chan_idx, out_row_idx, out_col_idx] if out_lbs is not None else -float('inf')
                    out_ub = out_ubs[0, out_chan_idx, out_row_idx, out_col_idx] if out_ubs is not None else float('inf')
                    var = model.addVar(lb=out_lb, ub=out_ub,
                                            obj=0, vtype=grb.GRB.CONTINUOUS,
                                            # name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                                            name=f'lay{self.name}_{neuron_idx}')
                    # model.addConstr(lin_expr == var, name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]_eq')
                    # new_layer_gurobi_constrs.append(
                        # model.addConstr(lin_expr == var, name=f'lay{self.name}_{neuron_idx}_eq'))
                    model.addConstr(lin_expr == var, name=f'lay{self.name}_{neuron_idx}_eq')
                    neuron_idx += 1

                    out_row_vars.append(var)
                out_chan_vars.append(out_row_vars)
            new_layer_gurobi_vars.append(out_chan_vars)

        self.solver_vars = new_layer_gurobi_vars
        # self.solver_constrs = new_layer_gurobi_constrs
        model.update()

    def interval_propagate(self, *v, C=None):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        norm = Interval.get_perturbation(v[0])
        norm = norm[0]

        h_L, h_U = v[0]
        weight = v[1][0]
        bias = v[2][0] if self.has_bias else None

        if norm == np.inf:
            mid = (h_U + h_L) / 2.0
            diff = (h_U - h_L) / 2.0
            weight_abs = weight.abs()
            deviation = F.conv2d(diff, weight_abs, None, self.stride, self.padding, self.dilation, self.groups)
        elif norm > 0:
            norm, eps = Interval.get_perturbation(v[0])
            # L2 norm, h_U and h_L are the same.
            mid = h_U
            # TODO: padding
            deviation = torch.mul(weight, weight).sum((1, 2, 3)).sqrt() * eps
            deviation = deviation.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        else: # Here we calculate the L0 norm IBP bound using the bound proposed in [Certified Defenses for Adversarial Patches, ICLR 2020]
            norm, eps, ratio = Interval.get_perturbation(v[0])
            mid = h_U
            k = int(eps)
            weight_sum = torch.sum(weight.abs(), 1)
            deviation = torch.sum(torch.topk(weight_sum.view(weight_sum.shape[0], -1), k)[0], dim=1) * ratio

            if self.has_bias:
                center = F.conv2d(mid, weight, v[2][0], self.stride, self.padding, self.dilation, self.groups)
            else:
                center = F.conv2d(mid, weight, None, self.stride, self.padding, self.dilation, self.groups)

            ss = center.shape
            deviation = deviation.repeat(ss[2] * ss[3]).view(-1, ss[1]).t().view(ss[1], ss[2], ss[3])

        center = F.conv2d(mid, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        upper = center + deviation
        lower = center - deviation
        return lower, upper

    def bound_dynamic_forward(self, *x, max_dim=None, offset=0):
        if self.is_input_perturbed(1) or self.is_input_perturbed(2):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")
        weight = x[1].lb
        bias = x[2].lb if self.has_bias else None
        x = x[0]
        w = x.lw
        b = x.lb
        shape = w.shape
        shape_wconv = [shape[0] * shape[1]] + list(shape[2:])
        def conv2d(input, weight, bias, stride, padding, dilation, groups):
            """ There may be some CUDA error (illegal memory access) when
            the batch size is too large. Thus split the input into several
            batches when needed. """
            max_batch_size = 50
            if input.device != torch.device('cpu') and input.shape[0] > max_batch_size:
                ret = []
                for i in range((input.shape[0] + max_batch_size - 1) // max_batch_size):
                    ret.append(F.conv2d(
                        input[i*max_batch_size:(i+1)*max_batch_size],
                        weight, bias, stride, padding, dilation, groups))
                return torch.cat(ret, dim=0)
            else:
                return F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        w_new = conv2d(
            w.reshape(shape_wconv), weight, None, self.stride, self.padding,
            self.dilation, self.groups)
        w_new = w_new.reshape(shape[0], -1, *w_new.shape[1:])
        b_new = conv2d(
            b, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return LinearBound(w_new, b_new, w_new, b_new, x_L=x.x_L, x_U=x.x_U, tot_dim=x.tot_dim)

    def bound_forward(self, dim_in, *x):
        if self.is_input_perturbed(1) or self.is_input_perturbed(2):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        weight = x[1].lb
        bias = x[2].lb if self.has_bias else None
        x = x[0]

        mid_w = (x.lw + x.uw) / 2
        mid_b = (x.lb + x.ub) / 2
        diff_w = (x.uw - x.lw) / 2
        diff_b = (x.ub - x.lb) / 2
        weight_abs = weight.abs()
        shape = mid_w.shape
        shape_wconv = [shape[0] * shape[1]] + list(shape[2:])
        deviation_w = F.conv2d(
            diff_w.reshape(shape_wconv), weight_abs, None,
            self.stride, self.padding, self.dilation, self.groups)
        deviation_b = F.conv2d(
            diff_b, weight_abs, None,
            self.stride, self.padding, self.dilation, self.groups)
        center_w = F.conv2d(
            mid_w.reshape(shape_wconv), weight, None,
            self.stride, self.padding, self.dilation, self.groups)
        center_b = F.conv2d(
            mid_b, weight, bias,
            self.stride, self.padding, self.dilation, self.groups)
        deviation_w = deviation_w.reshape(shape[0], -1, *deviation_w.shape[1:])
        center_w = center_w.reshape(shape[0], -1, *center_w.shape[1:])

        return LinearBound(
            lw = center_w - deviation_w,
            lb = center_b - deviation_b,
            uw = center_w + deviation_w,
            ub = center_b + deviation_b)

class BoundConvTranspose(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        assert (attr['pads'][0] == attr['pads'][2])
        assert (attr['pads'][1] == attr['pads'][3])

        self.stride = attr['strides']
        self.padding = [attr['pads'][0], attr['pads'][1]]
        self.dilation = attr['dilations']
        self.groups = attr['group']
        self.output_padding = [attr.get('output_padding', [0, 0])[0], attr.get('output_padding', [0, 0])[1]]
        if len(inputs) == 3:
            self.has_bias = True
        else:
            self.has_bias = False
        self.mode = options.get("conv_mode", "matrix")
        assert self.output_padding == [0, 0]
        assert self.padding == [0, 0]
        assert self.dilation == [1, 1]
        assert self.stride[0] == self.stride[1]
        assert self.groups == 1

    def forward(self, *x):
        # x[0]: input, x[1]: weight, x[2]: bias if self.has_bias
        bias = x[2] if self.has_bias else None
        output = F.conv_transpose2d(x[0], x[1], bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, output_padding=self.output_padding)
        return output


    def bound_backward(self, last_lA, last_uA, *x):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        lA_y = uA_y = lA_bias = uA_bias = None
        weight = x[1].lower
        assert weight.size(-1) == weight.size(-2)

        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            if type(last_A) is OneHotC:
                # Conv layer does not support the OneHotC fast path. We have to create a dense matrix instead.
                shape = last_A.shape  # [spec, batch, C, H, W]
                dim = int(prod(shape[2:]))
                dense_last_A = torch.zeros(size=(shape[0], shape[1], dim), device=last_A.device, dtype=weight.dtype)
                # last_A.index has size (spec, batch), its values are the index of the one-hot non-zero elements in A.
                # last_A.coeffs is the value of the non-zero element.
                dense_last_A = torch.scatter(dense_last_A, dim=2, index=last_A.index.unsqueeze(-1), src=last_A.coeffs.unsqueeze(-1))
                # We created a large A matrix and it will be handled below.
                last_A = dense_last_A.view(shape[0], shape[1], *shape[2:])

            if type(last_A) == Tensor:
                shape = last_A.size()
                next_A = F.conv2d(last_A.reshape(shape[0] * shape[1], *shape[2:]), weight, None,
                                            stride=self.stride, padding=self.padding, dilation=self.dilation,
                                            groups=self.groups)
                next_A = next_A.view(shape[0], shape[1], *next_A.shape[1:])
                if self.has_bias:
                    sum_bias = (last_A.sum((3, 4)) * x[2].lower).sum(2)
                else:
                    sum_bias = 0
                return next_A, sum_bias
            elif type(last_A) == Patches:
                # Here we build and propagate a Patch object with (patches, stride, padding)
                assert type(last_A) == Patches
                if last_A.identity == 0:
                    patches = last_A.patches

                    # FIXME: so far, assume there will be a relu layer in its input.

                    if self.has_bias:
                        # bias is x[2] (lower and upper are the same), and has shape (c,).
                        # Patches either has [out_c, batch, out_h, out_w, c, h, w] or [unstable_size, batch, c, h, w].
                        sum_bias = torch.einsum('sb...chw,c->sb...', patches, x[2].lower)
                        # sum_bias has shape (out_c, batch, out_h, out_w) or (unstable_size, batch).
                    else:
                        sum_bias = 0

                    flattened_patches = patches.reshape(-1, patches.size(-3), patches.size(-2), patches.size(-1))
                    # Merge patches with this layer's weights. Weight must be flipped here; and if stride != 1, we must insert zeros in the input image.
                    # For conv_transpose2d, the weight matrix is in the (in, out, k, k) shape.
                    pieces = F.conv_transpose2d(flattened_patches, weight.transpose(0,1).flip(-1,-2), stride=last_A.inserted_zeros + 1)
                    # New patch size: (out_c, batch, out_h, out_w, c, h, w) or (unstable_size, batch, c, h, w).
                    pieces = pieces.view(*patches.shape[:-3], pieces.size(-3), pieces.size(-2), pieces.size(-1))

                elif last_A.identity == 1:
                    # New patches have size [out_c, batch, out_h, out_w, c, h, w] if it is not sparse.
                    # New patches have size [unstable_size, batch, c, h, w] if it is sparse.
                    if last_A.unstable_idx is not None:
                        raise NotImplementedError()
                        pieces = weight.view(weight.size(0), 1, weight.size(1), weight.size(2), weight.size(3))
                        # Select based on the output channel (out_h and out_w are irrelevant here).
                        pieces = pieces[last_A.unstable_idx[0]]
                        # Expand the batch dimnension.
                        pieces = pieces.expand(-1, last_A.shape[1], -1, -1, -1)
                        # Do the same for the bias.
                        sum_bias = x[2].lower[last_A.unstable_idx[0]].unsqueeze(-1)
                        # bias has shape (unstable_size, batch).
                        sum_bias = sum_bias.expand(-1, last_A.shape[1])
                    else:
                        assert weight.size(0) == last_A.shape[0]
                        pieces = weight.view(weight.size(0), 1, 1, 1, weight.size(1), weight.size(2), weight.size(3)).expand(-1, *last_A.shape[1:4], -1, -1, -1)
                        # The bias (x[2].lower) has shape (out_c,) need to make it (out_c, batch, out_h, out_w).
                        # Here we should transpose sum_bias to set the batch dim to 1, aiming to keep it consistent with the matrix version
                        sum_bias = x[2].lower.view(-1, 1, 1, 1).expand(-1, *last_A.shape[1:4])
                else:
                    raise NotImplementedError()
                padding = last_A.padding if last_A is not None else (0, 0, 0, 0)  # (left, right, top, bottom)
                output_padding = last_A.output_padding if last_A is not None else (0, 0, 0, 0)  # (left, right, top, bottom)
                inserted_zeros = last_A.inserted_zeros
                assert self.padding == [0, 0]
                assert self.stride[0] == self.stride[1]

                # Unify the shape to 4-tuple.
                output_padding = unify_shape(output_padding)
                padding = unify_shape(padding)
                this_stride = unify_shape(self.stride)
                this_padding = unify_shape(self.padding)

                # Compute new padding.
                padding = tuple(p + (weight.size(3 - j//2) - 1) for j, p in enumerate(padding))

                # Compute new output padding
                output_padding = tuple(p * this_stride[j] + this_padding[j] for j, p in enumerate(output_padding))
                # When we run insert_zeros, it's missing the right most column and the bottom row.
                # padding = (padding[0], padding[1] + inserted_zeros, padding[2], padding[3] + inserted_zeros)

                # If no transposed conv so far, inserted_zero is 0.
                # We a transposed conv is encountered, stride is multiplied on it.
                inserted_zeros = (inserted_zeros + 1) * this_stride[0] - 1

                # FIXME: disabled patches_to_matrix because not all parameters are supported.
                if inserted_zeros == 0 and not is_shape_used(output_padding) and pieces.shape[-1] > self.input_shape[-1]:  # the patches is too large and from now on, we will use matrix mode instead of patches mode.
                    # This is our desired matrix: the input will be flattend to (batch_size, input_channel*input_x * input_y) and multiplies on this matrix.
                    # After multiplication, the desired output is (batch_size, out_channel*output_x*output_y).
                    # A_matrix has size (batch, out_c*out_h*out_w, in_c*in_h*in_w)
                    assert inserted_zeros == 0
                    A_matrix = patches_to_matrix(pieces, self.input_shape[1:], last_A.stride, padding, last_A.output_shape, last_A.unstable_idx)
                    if isinstance(sum_bias, Tensor) and last_A.unstable_idx is None:
                        sum_bias = sum_bias.transpose(0, 1)
                        sum_bias = sum_bias.reshape(sum_bias.size(0), -1).transpose(0,1)
                    A_matrix = A_matrix.transpose(0,1)  # Spec dimension at the front.
                    return A_matrix, sum_bias
                return Patches(pieces, last_A.stride, padding, pieces.shape, unstable_idx=last_A.unstable_idx,
                        output_shape=last_A.output_shape, inserted_zeros=inserted_zeros, output_padding=output_padding), sum_bias
            else:
                raise NotImplementedError()

        lA_x, lbias = _bound_oneside(last_lA)
        uA_x, ubias = _bound_oneside(last_uA)
        return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

    def interval_propagate(self, *v, C=None):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        norm = Interval.get_perturbation(v[0])
        norm = norm[0]

        h_L, h_U = v[0]
        weight = v[1][0]
        bias = v[2][0] if self.has_bias else None

        if norm == np.inf:
            mid = (h_U + h_L) / 2.0
            diff = (h_U - h_L) / 2.0
            weight_abs = weight.abs()
            deviation = F.conv_transpose2d(diff, weight_abs, None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, output_padding=self.output_padding)
        elif norm > 0:
            raise NotImplementedError()
            norm, eps = Interval.get_perturbation(v[0])
            # L2 norm, h_U and h_L are the same.
            mid = h_U
            # TODO: padding
            deviation = torch.mul(weight, weight).sum((1, 2, 3)).sqrt() * eps
            deviation = deviation.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        else: # Here we calculate the L0 norm IBP bound using the bound proposed in [Certified Defenses for Adversarial Patches, ICLR 2020]
            raise NotImplementedError()
            norm, eps, ratio = Interval.get_perturbation(v[0])
            mid = h_U
            k = int(eps)
            weight_sum = torch.sum(weight.abs(), 1)
            deviation = torch.sum(torch.topk(weight_sum.view(weight_sum.shape[0], -1), k)[0], dim=1) * ratio

            if self.has_bias:
                center = F.conv2d(mid, weight, v[2][0], self.stride, self.padding, self.dilation, self.groups)
            else:
                center = F.conv2d(mid, weight, None, self.stride, self.padding, self.dilation, self.groups)

            ss = center.shape
            deviation = deviation.repeat(ss[2] * ss[3]).view(-1, ss[1]).t().view(ss[1], ss[2], ss[3])

        center = F.conv_transpose2d(mid, weight, bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, output_padding=self.output_padding)

        upper = center + deviation
        lower = center - deviation
        return lower, upper

class BoundPad(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        if hasattr(attr, 'pads'):
            self.padding = attr['pads'][2:4] + attr['pads'][6:8]
        else:
            self.padding = [0, 0, 0, 0]
        self.value = attr.get('value', 0.0)
        assert self.padding == [0, 0, 0, 0]

    def forward(self, x, pad, value=0.0):
        # TODO: padding for 3-D or more dimensional inputs.
        assert x.ndim == 4
        # x[1] should be [0,0,pad_top,pad_left,0,0,pad_bottom,pad_right]
        assert pad[0] == pad[1] == pad[4] == pad[5] == 0
        pad = [int(pad[3]), int(pad[7]), int(pad[2]), int(pad[6])]
        final = F.pad(x, pad, value=value)
        self.padding, self.value = pad, value
        return final

    def interval_propagate(self, *v):
        l, u = zip(*v)
        return Interval.make_interval(self.forward(*l), self.forward(*u), v[0])

    def bound_backward(self, last_lA, last_uA, *x):
        # TODO: padding for 3-D or more dimensional inputs.
        left, right, top, bottom = self.padding
        def _bound_oneside(last_A):
            if last_A is None:
                return None
            assert type(last_A) is Patches or last_A.ndim == 5
            if type(last_A) is Patches:
                if isinstance(last_A.padding, tuple):
                    new_padding = (last_A.padding[0] + left, last_A.padding[1] + right, last_A.padding[2] + top, last_A.padding[3] + bottom)
                else:
                    new_padding = (last_A.padding + left, last_A.padding + right, last_A.padding + top, last_A.padding + bottom)
                return last_A.create_similar(padding=new_padding)
            else:
                shape = last_A.size()
                return last_A[:, :, :, top:(shape[3] - bottom), left:(shape[4] - right)]
        last_lA = _bound_oneside(last_lA)
        last_uA = _bound_oneside(last_uA)
        return [(last_lA, last_uA), (None, None), (None, None)], 0, 0

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        # e.g., last layer input gurobi vars (3,32,32)
        gvars_array = np.array(v[0])
        # pre_layer_shape (1,3,32,32)
        pre_layer_shape = np.expand_dims(gvars_array, axis=0).shape
        # this layer shape (1,3,35,35)
        this_layer_shape = self.output_shape
        # v1 = tensor([0, 0, 1, 1, 0, 0, 2, 2])
        # [0,0,pad_top,pad_left,0,0,pad_bottom,pad_right]
        # => [left, right, top, bottom]
        padding = [int(v[1][3]), int(v[1][7]), int(v[1][2]), int(v[1][6])]
        left, right, top, bottom = padding
        assert pre_layer_shape[2] + padding[0] + padding[1] == this_layer_shape[2]
        assert pre_layer_shape[3] + padding[2] + padding[3] == this_layer_shape[3]

        new_layer_gurobi_vars = []
        neuron_idx = 0
        for out_chan_idx in range(this_layer_shape[1]):
            out_chan_vars = []
            for out_row_idx in range(this_layer_shape[2]):
                out_row_vars = []
                row_pad = out_row_idx < left or out_row_idx >= this_layer_shape[2] - right
                for out_col_idx in range(this_layer_shape[3]):
                    col_pad = out_col_idx < top or out_col_idx >= this_layer_shape[3] - bottom
                    if row_pad or col_pad:
                        v = model.addVar(lb=0, ub=0,
                                    obj=0, vtype=grb.GRB.CONTINUOUS,
                                    name=f'pad{self.name}_{neuron_idx}')
                    else:
                        v = gvars_array[out_chan_idx, out_row_idx - left, out_col_idx - top]
                    # print(out_chan_idx, out_row_idx, out_col_idx, row_pad, col_pad, v.LB, v.UB)
                    neuron_idx += 1

                    out_row_vars.append(v)
                out_chan_vars.append(out_row_vars)
            new_layer_gurobi_vars.append(out_chan_vars)

        self.solver_vars = new_layer_gurobi_vars
        model.update()
