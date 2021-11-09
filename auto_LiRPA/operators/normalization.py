""" Normalization operators"""
from .base import *

class BoundBatchNormalization(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device, training):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.eps = attr['epsilon']
        self.momentum = round(1 - attr['momentum'], 5)  # take care!
        self.mode = options.get("conv_mode", "matrix")
        self.options = options.get("bn", {})
        # modes:
        #   - forward: use mean and variance estimated from clean forward pass
        #   - ibp: use mean and variance estimated from ibp
        self.bn_mode = self.options.get("mode", "forward") 
        self.use_mean = self.options.get("mean", True)
        self.use_var = self.options.get("var", True)
        self.use_affine = self.options.get("affine", True)
        self.training = training
        if not self.use_mean or not self.use_var:
            logger.info(f'Batch normalization node {self.name}: use_mean {self.use_mean}, use_var {self.use_var}')        

    def _check_unused_mean_or_var(self):
        # Check if either mean or var is opted out
        if not self.use_mean:
            self.current_mean = torch.zeros_like(self.current_mean)
        if not self.use_var:
            self.current_var = torch.ones_like(self.current_var)

    @Bound.save_io_shape
    def forward(self, x, w, b, m, v):
        if self.training:
            dim = [0] + list(range(2, x.ndim))
            self.current_mean = x.mean(dim)
            self.current_var = x.var(dim, unbiased=False)
        else:
            self.current_mean = m.data
            self.current_var = v.data
        self._check_unused_mean_or_var() 
        if not self.use_affine:
            w = torch.ones_like(w)
            b = torch.zeros_like(b)                     
        result = F.batch_norm(x, m, v, w, b, self.training, self.momentum, self.eps)
        if not self.use_mean or not self.use_var:
            # If mean or variance is disabled, recompute the output from self.current_mean
            # and self.current_var instead of using standard F.batch_norm.
            w = w / torch.sqrt(self.current_var + self.eps)
            b = b - self.current_mean * w
            shape = (1, -1) + (1,) * (x.ndim - 2)
            result = w.view(*shape) * x + b.view(*shape)
        return result

    def bound_backward(self, last_lA, last_uA, *x):
        assert not self.is_input_perturbed(1) and not self.is_input_perturbed(2), \
            'Weight perturbation is not supported for BoundBatchNormalization'

        # x[0]: input, x[1]: weight, x[2]: bias, x[3]: running_mean, x[4]: running_var
        weight, bias = x[1].param, x[2].param
        if not self.training:
            self.current_mean = x[3].value
            self.current_var = x[4].value
        self._check_unused_mean_or_var()
        if not self.use_affine:
            weight = torch.ones_like(weight)
            bias = torch.zeros_like(bias)                     
        
        tmp_bias = bias - self.current_mean / torch.sqrt(self.current_var + self.eps) * weight
        tmp_weight = weight / torch.sqrt(self.current_var + self.eps)

        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            if type(last_A) == torch.Tensor:
                next_A = last_A * tmp_weight.view(*((1, 1, -1) + (1,) * (last_A.ndim - 3)))
                if last_A.ndim > 3:
                    sum_bias = (last_A.sum(tuple(range(3, last_A.ndim))) * tmp_bias).sum(2)
                else:
                    sum_bias = (last_A * tmp_bias).sum(2)
            elif type(last_A) == Patches:
                # TODO Only 4-dim BN supported in the Patches mode
                if last_A.identity == 0:
                    # FIXME (09/17): Need to check if it has already been padding.
                    # Patch has dimension (out_c, batch, out_h, out_w, c, h, w) or (unstable_size, batch, c, h, w)
                    patches = last_A.patches

                    # tmp_weight has shape (c,), it will be applied on the (c,) dimension.
                    patches = patches * tmp_weight.view(*([1] * (patches.ndim - 3)), -1, 1, 1)  # Match with sparse or non-sparse patches.
                    next_A = Patches(patches, last_A.stride, last_A.padding, last_A.shape, identity=0, unstable_idx=last_A.unstable_idx, output_shape=last_A.output_shape)
                    
                    # bias to size (c,), need expansion before unfold.
                    bias = tmp_bias.view(-1,1,1).expand(self.input_shape[1:]).unsqueeze(0)
                    # Unfolded bias has shape (1, out_h, out_w, in_c, H, W).
                    bias_unfolded = inplace_unfold(bias, kernel_size=last_A.patches.shape[-2:], padding=last_A.padding, stride=last_A.stride)
                    if last_A.unstable_idx is not None:
                        # Sparse bias has shape (unstable_size, batch, in_c, H, W).
                        bias_unfolded = bias_unfolded[:, last_A.unstable_idx[1], last_A.unstable_idx[2]]
                        sum_bias = torch.einsum('bschw,sbchw->sb', bias_unfolded, last_A.patches)
                        # Output sum_bias has shape (unstable_size, batch).
                    else:
                        # Patch has dimension (out_c, batch, out_h, out_w, c, h, w).
                        sum_bias = torch.einsum('bijchw,sbijchw->sbij', bias_unfolded, last_A.patches)
                        # Output sum_bias has shape (out_c, batch, out_h, out_w).
                else:
                    # we should create a real identity Patch
                    num_channel = tmp_weight.numel()
                    # desired Shape is (c, batch, out_w, out_h, c, 1, 1) or (unstable_size, batch, c, 1, 1).
                    patches = (torch.eye(num_channel, device=tmp_weight.device) * tmp_weight.view(-1)).view(num_channel, 1, 1, 1, num_channel, 1, 1)
                    # Expand out_h, out_w dimensions but not for batch dimension.
                    patches = patches.expand(-1, -1, last_A.output_shape[2], last_A.output_shape[3], -1, 1, 1)
                    if last_A.unstable_idx is not None:
                        # Select based on unstable indices.
                        patches = patches[last_A.unstable_idx[0], :, last_A.unstable_idx[1], last_A.unstable_idx[2]]
                    # Expand the batch dimension.
                    patches = patches.expand(-1, last_A.shape[1], *([-1] * (patches.ndim - 2)))
                    next_A = Patches(patches, 1, 0, last_A.shape, unstable_idx=last_A.unstable_idx, output_shape=last_A.output_shape)
                    if last_A.unstable_idx is not None:
                        # Need to expand the bias and choose the selected ones.
                        bias = tmp_bias.view(-1,1,1,1).expand(-1, 1, last_A.output_shape[2], last_A.output_shape[3])
                        bias = bias[last_A.unstable_idx[0], :, last_A.unstable_idx[1], last_A.unstable_idx[2]]
                        # Expand the batch dimension, and final output shape is (unstable_size, batch).
                        sum_bias = bias.expand(-1, last_A.shape[1])
                    else:
                        # Output sum_bias has shape (out_c, batch, out_h, out_w).
                        sum_bias = tmp_bias.view(-1, 1, 1, 1).expand(-1, *last_A.shape[1:4])
            else:
                raise NotImplementedError()
            return next_A, sum_bias

        lA, lbias = _bound_oneside(last_lA)
        uA, ubias = _bound_oneside(last_uA)

        return [(lA, uA), (None, None), (None, None), (None, None), (None, None)], lbias, ubias

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1) and not self.is_input_perturbed(2), \
            'Weight perturbation is not supported for BoundBatchNormalization'

        h_L, h_U = v[0]
        weight, bias = v[1][0], v[2][0]

        mid = (h_U + h_L) / 2.0
        diff = (h_U - h_L) / 2.0

        # Use `mid` in IBP to compute mean and variance for BN.
        # In this case, `forward` should not have been called.
        if self.bn_mode == 'ibp' and not hasattr(self, 'forward_value'):
            m, v, w, b = tuple(self.inputs[i].forward() for i in range(1, 5))
            self.forward(mid, m, v, w, b)

        if not self.training:
            assert not (self.is_input_perturbed(3) or self.is_input_perturbed(4))
            self.current_mean = v[3][0]
            self.current_var = v[4][0]
        self._check_unused_mean_or_var()
        if not self.use_affine:
            weight = torch.ones_like(weight)
            bias = torch.zeros_like(bias)                     

        tmp_weight = weight / torch.sqrt(self.current_var + self.eps)
        tmp_weight_abs = tmp_weight.abs()
        tmp_bias = bias - self.current_mean * tmp_weight

        shape = (1, -1) + (1,) * (mid.ndim - 2)
        center = tmp_weight.view(*shape) * mid + tmp_bias.view(*shape)
        deviation = tmp_weight_abs.view(*shape) * diff
        lower = center - deviation
        upper = center + deviation

        return lower, upper

