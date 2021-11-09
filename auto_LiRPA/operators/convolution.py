""" Convolution, pooling and padding operators"""
from .base import *
from .activation import BoundOptimizableActivation


class BoundConv(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        assert (attr['pads'][0] == attr['pads'][2])
        assert (attr['pads'][1] == attr['pads'][3])

        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

        self.stride = attr['strides']
        self.padding = [attr['pads'][0], attr['pads'][1]]
        self.dilation = attr['dilations']
        self.groups = attr['group']
        if len(inputs) == 3:
            self.has_bias = True
        else:
            self.has_bias = False
        self.to(device)
        self.mode = options.get("conv_mode", "matrix")
        self.relu_followed = False 
        # denote whether this Conv is followed by a ReLU
        # if self.relu_followed is False, we need to manually pad the conv patches. 
        # If self.relu_followed is True, the patches are padded in the ReLU layer and the manual padding is not needed.

    @Bound.save_io_shape
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
                dense_last_A = torch.zeros(size=(shape[0], shape[1], dim), device=last_A.device)
                # last_A.index has size (spec, batch), its values are the index of the one-hot non-zero elements in A.
                # last_A.coeffs is the value of the non-zero element.
                dense_last_A = torch.scatter(dense_last_A, dim=2, index=last_A.index.unsqueeze(-1), src=last_A.coeffs.unsqueeze(-1))
                # We created a large A matrix and it will be handled below.
                last_A = dense_last_A.view(shape[0], shape[1], *shape[2:])

            if type(last_A) == torch.Tensor:
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
                    sum_bias = (last_A.sum((3, 4)) * x[2].lower).sum(2)
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
                        one_d = torch.ones(tuple(1 for i in self.output_shape), device=last_A.patches.device).expand(self.output_shape)
                        # After unfolding, the shape is (batch, out_h, out_w, in_c, h, w)
                        one_d_unfolded = inplace_unfold(one_d, kernel_size=last_A.patches.shape[-2:], stride=last_A.stride, padding=last_A.padding)
                        if last_A.unstable_idx is not None:
                            # Move out_h, out_w dimension to the front for easier selection.
                            one_d_unfolded_r = one_d_unfolded.permute(1, 2, 0, 3, 4, 5)
                            # for sparse patches the shape is (unstable_size, batch, in_c, h, w).
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
                    pieces = F.conv_transpose2d(flattened_patches, weight, stride=self.stride)
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
                stride = last_A.stride if last_A is not None else 1

                if type(padding) == int:
                    padding = padding * self.stride[0] + self.padding[0]
                else:
                    padding = tuple(p * self.stride[0] + self.padding[0] for p in padding)
                stride *= self.stride[0]

                if pieces.shape[-1] > self.input_shape[-1]:  # the patches is too large and from now on, we will use matrix mode instead of patches mode.
                    # This is our desired matrix: the input will be flattend to (batch_size, input_channel*input_x * input_y) and multiplies on this matrix.
                    # After multiplication, the desired output is (batch_size, out_channel*output_x*output_y).
                    # A_matrix has size (batch, out_c*out_h*out_w, in_c*in_h*in_w)
                    A_matrix = patches_to_matrix(pieces, self.input_shape[1:], stride, padding, last_A.output_shape, last_A.unstable_idx)
                    if isinstance(sum_bias, torch.Tensor) and last_A.unstable_idx is None:
                        sum_bias = sum_bias.transpose(0, 1)
                        sum_bias = sum_bias.reshape(sum_bias.size(0), -1).transpose(0,1)
                    A_matrix = A_matrix.transpose(0,1)  # Spec dimension at the front.
                    return A_matrix, sum_bias
                return Patches(pieces, stride, padding, pieces.shape, unstable_idx=last_A.unstable_idx, output_shape=last_A.output_shape), sum_bias
            else:
                raise NotImplementedError()

        lA_x, lbias = _bound_oneside(last_lA)
        uA_x, ubias = _bound_oneside(last_uA)
        return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

    def bound_forward(self, dim_in, *x):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        weight = x[1].lb
        bias = x[2].lb if self.has_bias else None
        x = x[0]
        input_dim = x.lb.shape[-2] * x.lb.shape[-1]
        wshape = x.lw.shape
        eye = torch.eye(input_dim).view(input_dim, 1, *x.lb.shape[-2:])
        weight = F.conv2d(eye, weight, None, self.stride, self.padding, self.dilation, self.groups)
        weight = weight.view(input_dim, -1)
        output_dim = weight.shape[-1]
        bias = bias.view(1, -1, 1).repeat(1, 1, output_dim // bias.shape[0]).view(*self.output_shape[1:])
        batch_size = x.lb.shape[0]

        lw = (x.lw.reshape(batch_size, dim_in, -1).matmul(weight.clamp(min=0)) + 
            x.uw.reshape(batch_size, dim_in, -1).matmul(weight.clamp(max=0)))\
            .reshape(batch_size, dim_in, *self.output_shape[1:])
        uw = (x.uw.reshape(batch_size, dim_in, -1).matmul(weight.clamp(min=0)) + 
            x.lw.reshape(batch_size, dim_in, -1).matmul(weight.clamp(max=0)))\
            .reshape(batch_size, dim_in, *self.output_shape[1:])
        
        lb = (x.lb.reshape(batch_size, -1).matmul(weight.clamp(min=0)) + 
            x.ub.reshape(batch_size, -1).matmul(weight.clamp(max=0)))\
            .reshape(batch_size, *self.output_shape[1:]) + bias
        ub = (x.ub.reshape(batch_size, -1).matmul(weight.clamp(min=0)) + 
            x.lb.reshape(batch_size, -1).matmul(weight.clamp(max=0)))\
            .reshape(batch_size, *self.output_shape[1:]) + bias

        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v, C=None):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        norm = Interval.get_perturbation(v[0])
        norm = norm[0]

        if Interval.use_relative_bounds(*v):
            bias = v[2].nominal if self.has_bias else None
            if norm == np.inf:
                weight = v[1].nominal
                nominal = F.conv2d(
                    v[0].nominal, weight, bias, 
                    self.stride, self.padding, self.dilation, self.groups)
                lower_offset = (F.conv2d(
                                    v[0].lower_offset, weight.clamp(min=0), None,
                                    self.stride, self.padding, self.dilation, self.groups) + 
                                F.conv2d(
                                    v[0].upper_offset, weight.clamp(max=0), None,
                                    self.stride, self.padding, self.dilation, self.groups))
                upper_offset = (F.conv2d(
                                    v[0].upper_offset, weight.clamp(min=0), None,
                                    self.stride, self.padding, self.dilation, self.groups) + 
                                F.conv2d(
                                    v[0].lower_offset, weight.clamp(max=0), None,
                                    self.stride, self.padding, self.dilation, self.groups))
                return Interval(
                    None, None, nominal=nominal, 
                    lower_offset=lower_offset, upper_offset=upper_offset
                )
            else:
                raise NotImplementedError

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

    def bound_forward(self, dim_in, *x):
        if self.is_input_perturbed(1):
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
        center_b =  F.conv2d(
            mid_b, weight, bias, 
            self.stride, self.padding, self.dilation, self.groups)
        deviation_w = deviation_w.reshape(shape[0], -1, *deviation_w.shape[1:])
        center_w = center_w.reshape(shape[0], -1, *center_w.shape[1:])

        return LinearBound(
            lw = center_w - deviation_w,
            lb = center_b - deviation_b,
            uw = center_w + deviation_w,
            ub = center_b + deviation_b)


class BoundMaxPool(BoundOptimizableActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        assert ('pads' not in attr) or (attr['pads'][0] == attr['pads'][2])
        assert ('pads' not in attr) or (attr['pads'][1] == attr['pads'][3])

        self.nonlinear = True
        self.kernel_size = attr['kernel_shape']
        self.stride = attr['strides']
        self.padding = [attr['pads'][0], attr['pads'][1]]
        self.ceil_mode = False
        self.use_default_ibp = True
        self.alpha = None
        self.init = {}

    @Bound.save_io_shape
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
        batch_size, channel, h, w = self.input_shape
        o_h, o_w = self.output_shape[-2:]
        # batch_size, out_c, out_h, out_w, k, k

        self.alpha = OrderedDict()
        ref = self.inputs[0].lower # a reference variable for getting the shape
        for ns, size_s in start_nodes:
            self.alpha[ns] = torch.empty([1, size_s, self.input_shape[0], self.input_shape[1], self.output_shape[-2], self.output_shape[-1], self.kernel_size[0], self.kernel_size[1]], 
                dtype=torch.float, device=ref.device, requires_grad=True)
            self.init[ns] = False

    def bound_backward(self, last_lA, last_uA, x, start_node=None, start_shape=None, unstable_idx=None):
        paddings = tuple(self.padding + self.padding)

        A_shape = last_lA.shape if last_lA is not None else last_uA.shape
        # batch_size, input_c, x, y
        upper_d = torch.zeros((list(self.input_shape)), device=x.device)
        lower_d = torch.zeros((list(self.input_shape)), device=x.device)

        upper_d = F.pad(upper_d, paddings)
        lower_d = F.pad(lower_d, paddings)

        # batch_size, output_c, x, y
        upper_b = torch.zeros((list(self.output_shape)), device=x.device)
        lower_b = torch.zeros((list(self.output_shape)), device=x.device)

        # 1. find the index i where li > uj for all j, then set upper_d = lower_d = 1
        max_lower, max_lower_index = F.max_pool2d(x.lower, self.kernel_size, self.stride, self.padding, return_indices=True, ceil_mode=self.ceil_mode)
        delete_upper = torch.scatter(torch.flatten(F.pad(x.upper, paddings), -2), -1, torch.flatten(max_lower_index, -2), -np.inf).view(upper_d.shape)
        max_upper, _ = F.max_pool2d(delete_upper, self.kernel_size, self.stride, 0, return_indices=True, ceil_mode=self.ceil_mode)
        
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

            if self.init[start_node.name] == False:
                lower_d = torch.scatter(torch.flatten(lower_d, -2), -1, torch.flatten(max_lower_index, -2), 1.0).view(upper_d.shape)
                lower_d_unfold = F.unfold(lower_d, self.kernel_size, 1, stride=self.stride)

                alpha_data = lower_d_unfold.view(lower_d.shape[0], lower_d.shape[1], self.kernel_size[0], self.kernel_size[1], self.output_shape[-2], self.output_shape[-1])
                alpha.data.copy_(alpha_data.permute((0,1,4,5,2,3)).clone().detach())
                self.init[start_node.name] = True
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

        assert type(last_lA) == torch.Tensor or type(last_uA) == torch.Tensor
        def _bound_oneside(last_A, d_pos, d_neg, b_pos, b_neg):
            if last_A is None:
                return None, 0
            pos_A = last_A.clamp(min=0)
            neg_A = last_A.clamp(max=0)

            bias = 0
            if b_pos is not None:
                bias = bias + self.get_bias(pos_A, b_pos)
            if b_neg is not None:
                bias = bias + self.get_bias(neg_A, b_neg)

            shape = last_A.size()
            pos_A = F.interpolate(pos_A.view(shape[0] * shape[1], *shape[2:]), scale_factor=self.kernel_size)
            pos_A = F.pad(pos_A, (0, self.input_shape[-2] - pos_A.shape[-2], 0, self.input_shape[-1] - pos_A.shape[-1]))
            pos_A = pos_A.view(shape[0], shape[1], *pos_A.shape[1:])

            neg_A = F.interpolate(neg_A.view(shape[0] * shape[1], *shape[2:]), scale_factor=self.kernel_size)
            neg_A = F.pad(neg_A, (0, self.input_shape[-2] - neg_A.shape[-2], 0, self.input_shape[-1] - neg_A.shape[-1]))
            neg_A = neg_A.view(shape[0], shape[1], *neg_A.shape[1:])

            next_A = pos_A * d_pos + neg_A * d_neg
            return next_A, bias

        if self.padding[0] > 0:
            upper_d = upper_d[...,self.padding[0]:-self.padding[0], self.padding[0]:-self.padding[0]]

        uA, ubias = _bound_oneside(last_uA, upper_d, lower_d, upper_b, lower_b)
        lA, lbias = _bound_oneside(last_lA, lower_d, upper_d, lower_b, upper_b)

        return [(lA, uA)], lbias, ubias

class BoundPad(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        if len(attr) == 1:
            self.padding = [0, 0, 0, 0]
            self.value = 0.0
        else:
            self.padding = attr['pads'][2:4] + attr['pads'][6:8]
            self.value = attr['value']
        assert self.padding == [0, 0, 0, 0]
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, x, pad, value=0.0):
        # TODO: padding for 3-D or more dimensional inputs.
        assert x.ndim == 4
        # x[1] should be [0,0,pad_top,pad_left,0,0,pad_bottom,pad_right]
        pad = [int(pad[3]), int(pad[7]), int(pad[2]), int(pad[6])]
        final = F.pad(x, pad, value=value)
        self.padding, self.value = pad, value
        return final

    def interval_propagate(self, *v):
        l, u = zip(*v)
        return Interval.make_interval(self.forward(*l), self.forward(*u), v[0])

    def bound_backward(self, last_lA, last_uA, *x):
        # TODO: padding for 3-D or more dimensional inputs.
        pad = self.padding
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
                return Patches(last_A.patches, last_A.stride, new_padding, last_A.shape, last_A.identity, unstable_idx=last_A.unstable_idx, output_shape=last_A.output_shape)
            else:
                shape = last_A.size()
                return last_A[:, :, :, top:(shape[3] - bottom), left:(shape[4] - right)]
        last_lA = _bound_oneside(last_lA)
        last_uA = _bound_oneside(last_uA)
        return [(last_lA, last_uA), (None, None), (None, None)], 0, 0

class BoundGlobalAveragePool(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, x):
        output = nn.AdaptiveAvgPool2d((1, 1)).forward(x)  # adaptiveAveragePool with output size (1, 1)
        return output

    def bound_backward(self, last_lA, last_uA, x):
        H, W = self.input_shape[-2], self.input_shape[-1]

        lA = last_lA.expand(list(last_lA.shape[:-2]) + [H, W]) / (H * W)
        uA = last_uA.expand(list(last_lA.shape[:-2]) + [H, W]) / (H * W)

        return [(lA, uA)], 0, 0

    def interval_propagate(self, *v):
        h_L, h_U = v[0]
        h_L = F.adaptive_avg_pool2d(h_L, (1, 1))
        h_U = F.adaptive_avg_pool2d(h_U, (1, 1))
        return h_L, h_U

class BoundAveragePool(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        # assumptions: ceil_mode=False, count_include_pad=True
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

        assert ('pads' not in attr) or (attr['pads'][0] == attr['pads'][2])
        assert ('pads' not in attr) or (attr['pads'][1] == attr['pads'][3])
      
        self.kernel_size = attr['kernel_shape']
        self.stride = attr['strides']
        self.padding = [attr['pads'][0], attr['pads'][1]]
        self.ceil_mode = False
        self.count_include_pad = True
        self.use_default_ibp = True

    @Bound.save_io_shape
    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            shape = last_A.size()
            # propagate A to the next layer, with batch concatenated together
            next_A = F.interpolate(last_A.view(shape[0] * shape[1], *shape[2:]), 
                scale_factor=self.kernel_size) / (prod(self.kernel_size))
            next_A = F.pad(next_A, (0, self.input_shape[-2] - next_A.shape[-2], 0, self.input_shape[-1] - next_A.shape[-1]))
            next_A = next_A.view(shape[0], shape[1], *next_A.shape[1:])
            return next_A, 0

        lA, lbias = _bound_oneside(last_lA)
        uA, ubias = _bound_oneside(last_uA)
        return [(lA, uA)], lbias, ubias        