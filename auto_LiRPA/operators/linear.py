""" Linear (possibly with weight perturbation) or Dot product layers """
from .base import *
from .bivariate import BoundMul
from ..patches import Patches
from .solver_utils import grb
from torch import Tensor
from ..patches import inplace_unfold

class BoundLinear(Bound):
    def __init__(self, attr, inputs, output_index, options):
        # Gemm:
        # A = A if transA == 0 else A.T
        # B = B if transB == 0 else B.T
        # C = C if C is not None else np.array(0)
        # Y = alpha * np.dot(A, B) + beta * C
        # return Y

        super().__init__(attr, inputs, output_index, options)

        # Defaults in ONNX
        self.transA = 0
        self.transB = 0
        self.alpha = 1.0
        self.beta = 1.0
        if attr is not None:
            self.transA = attr['transA'] if 'transA' in attr else self.transA
            self.transB = attr['transB'] if 'transB' in attr else self.transB
            self.alpha = attr['alpha'] if 'alpha' in attr else self.alpha
            self.beta = attr['beta'] if 'beta' in attr else self.beta

        self.opt_matmul = options.get('matmul')

    """Handle tranpose and linear coefficients."""
    def _preprocess(self, a, b, c=None):
        if self.transA and isinstance(a, Tensor):
            a = a.transpose(-2,-1)
        if self.alpha != 1.0:
            a = self.alpha * a
        if not self.transB and isinstance(b, Tensor):
            # our code assumes B is transposed (common case), so we transpose B only when it is not transposed in gemm.
            b = b.transpose(-2, -1)
        if c is not None:
            if self.beta != 1.0:
                c = self.beta * c
        return a, b, c

    def forward(self, x, w, b=None):
        x, w, b = self._preprocess(x, w, b)
        self.input_shape = self.x_shape = x.shape
        self.y_shape = w.t().shape
        res = x.matmul(w.t())
        if b is not None:
            res += b
        return res

    """Multiply weight matrix with a diagonal matrix with selected rows."""
    def onehot_mult(self, weight, bias, C, batch_size):

        if C is None:
            return None, 0.0

        new_weight = None
        new_bias = 0.0

        if C.index.ndim == 2:
            # Shape is [spec, batch]
            index = C.index.transpose(0, 1)
            coeffs = C.coeffs.transpose(0, 1)
        else:
            index = C.index
            coeffs = C.coeffs

        if C.index.ndim == 1:
            # Every element in the batch shares the same rows.
            if weight is not None:
                new_weight = self.non_deter_index_select(weight, dim=0, index=index).unsqueeze(1).expand([-1, batch_size] + [-1] * (weight.ndim - 1))
            if bias is not None:
                new_bias = self.non_deter_index_select(bias, dim=0, index=index).unsqueeze(1).expand(-1, batch_size)
        elif C.index.ndim == 2:
            # Every element in the batch has different rows, but the number of rows are the same. This essentially needs a batched index_select function.
            if weight is not None:
                new_weight = batched_index_select(weight.unsqueeze(0), dim=1, index=index)
            if bias is not None:
                new_bias = batched_index_select(bias.unsqueeze(0), dim=1, index=index)
        if C.coeffs is not None:
            if weight is not None:
                new_weight = new_weight * coeffs.unsqueeze(-1)
            if bias is not None:
                new_bias = new_bias * coeffs
        if C.index.ndim == 2:
            # Eventually, the shape of A is [spec, batch, *node] so need a transpose.
            new_weight = new_weight.transpose(0, 1)
            new_bias = new_bias.transpose(0, 1)
        return new_weight, new_bias

    def bound_backward(self, last_lA, last_uA, *x):
        assert len(x) == 2 or len(x) == 3
        has_bias = len(x) == 3
        # x[0]: input node, x[1]: weight, x[2]: bias
        input_lb = [getattr(xi, 'lower', None) for xi in x]
        input_ub = [getattr(xi, 'upper', None) for xi in x]
        # transpose and scale each term if necessary.
        input_lb = self._preprocess(*input_lb)
        input_ub = self._preprocess(*input_ub)
        lA_y = uA_y = lA_bias = uA_bias = None
        lbias = ubias = 0
        batch_size = last_lA.shape[1] if last_lA is not None else last_uA.shape[1]

        # Case #1: No weight/bias perturbation, only perturbation on input.
        if not self.is_input_perturbed(1) and (not has_bias or not self.is_input_perturbed(2)):
            weight = input_lb[1]
            bias = input_lb[2] if has_bias else None
            # If last_lA and last_uA are indentity matrices.
            if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):  # FIXME (12/28): we should check last_lA and last_uA separately. Same applies to the weight perturbed, bias perturbed settings.
                # Use this layer's W as the next bound matrices. Duplicate the batch dimension. Other dimensions are kept 1.
                # Not perturbed, so we can use either lower or upper.
                assert last_lA.shape == last_uA.shape
                shape_others = prod(last_lA.shape[2:-1])
                A_identity = torch.eye(shape_others).to(weight).view(shape_others, 1, 1, shape_others, 1)
                assert last_lA.shape[0] == weight.size(0) * shape_others
                w = weight.view(1, weight.size(0), *[1] * (len(last_lA.shape) - 2), weight.size(1))
                w = w * A_identity

                # expand the batch_size dim
                lA_x = uA_x = w.view(last_lA.shape[0], 1, *last_lA.shape[2:-1], weight.size(1)).expand(last_lA.shape[0], *last_lA.shape[1:-1], weight.size(1))
                if has_bias:
                    lbias = ubias = bias.unsqueeze(1).repeat(1, batch_size)
            elif isinstance(last_lA, OneHotC) or isinstance(last_uA, OneHotC):
                # We need to select several rows from the weight matrix (its shape is output_size * input_size).
                lA_x, lbias = self.onehot_mult(weight, bias, last_lA, batch_size)
                if last_lA is last_uA:
                    uA_x = lA_x
                    ubias = lbias
                else:
                    uA_x, ubias = self.onehot_mult(weight, bias, last_uA, batch_size)
            else:
                def _bound_oneside(last_A):
                    if last_A is None:
                        return None, 0
                    if isinstance(last_A, torch.Tensor):
                        # Matrix mode.
                        # Just multiply this layer's weight into bound matrices, and produce biases.
                        next_A = last_A.to(weight).matmul(weight)
                        sum_bias = (last_A.to(bias).matmul(bias)
                            if has_bias else 0.0)
                    elif isinstance(last_A, Patches):
                        # Patches mode. After propagating through this layer, it will become a matrix.
                        # Reshape the weight matrix as a conv image.
                        # Weight was in (linear_output_shape, linear_input_shape)
                        # Reshape it to (linear_input_shape, c, h, w)
                        reshaped_weight = weight.transpose(0,1).view(-1, *last_A.input_shape[1:])
                        # After unfolding the shape is (linear_input_shape, output_h, output_w, in_c, patch_h, patch_w)
                        unfolded_weight = inplace_unfold(
                            reshaped_weight,
                            kernel_size=last_A.patches.shape[-2:],
                            stride=last_A.stride, padding=last_A.padding,
                            inserted_zeros=last_A.inserted_zeros,
                            output_padding=last_A.output_padding)
                        if has_bias:
                            # Do the same for the bias.
                            reshaped_bias = bias.view(*last_A.input_shape[1:]).unsqueeze(0)
                            # After unfolding the bias shape is (1, output_h, output_w, in_c, patch_h, patch_w)
                            unfolded_bias = inplace_unfold(reshaped_bias, kernel_size=last_A.patches.shape[-2:], stride=last_A.stride, padding=last_A.padding, inserted_zeros=last_A.inserted_zeros, output_padding=last_A.output_padding)
                        if last_A.unstable_idx is not None:
                            # Reshape our weight to (output_h, output_w, 1, in_c, patch_h, patch_w, linear_input_shape), 1 is the inserted batch dim.
                            unfolded_weight_r = unfolded_weight.permute(1, 2, 3, 4, 5, 0).unsqueeze(2)
                            # for sparse patches the shape is (unstable_size, batch, in_c, patch_h, patch_w). Batch size is 1 so no need to select here.
                            # We select in the (output_h, out_w) dimension.
                            selected_weight = unfolded_weight_r[last_A.unstable_idx[1], last_A.unstable_idx[2]]
                            next_A = torch.einsum('sbchw,sbchwi->sbi', last_A.patches, selected_weight)
                            if has_bias:
                                # Reshape our bias to (output_h, output_w, 1, in_c, patch_h, patch_w). We already have the batch dim.
                                unfolded_bias_r = unfolded_bias.permute(1, 2, 0, 3, 4, 5)
                                selected_bias = unfolded_bias_r[last_A.unstable_idx[1], last_A.unstable_idx[2]]
                                sum_bias = torch.einsum('sbchw,sbchw->sb', last_A.patches, selected_bias)
                        else:
                            # Reshape our weight to (1, 1, output_h, output_w, in_c, patch_h, patch_w, linear_input_shape), 1 is the spec and batch.
                            selected_weight = unfolded_weight.permute(1, 2, 3, 4, 5, 0).unsqueeze(0).unsqueeze(0)
                            next_A_r = torch.einsum('sbpqchw,sbpqchwi->spqbi', last_A.patches, selected_weight)
                            # We return a matrix with flattened spec dimension (corresponding to out_c * out_h * out_w).
                            next_A = next_A_r.reshape(-1, next_A_r.size(-2), next_A_r.size(-1))
                            if has_bias:
                                # Reshape our bias to (1, 1, output_h, output_w, in_c, patch_h, patch_w)
                                selected_bias = unfolded_bias.unsqueeze(0)
                                sum_bias_r = torch.einsum('sbpqchw,sbpqchw->spqb', last_A.patches, selected_bias)
                                sum_bias = sum_bias_r.reshape(-1, sum_bias_r.size(-1))
                    return next_A, sum_bias if has_bias else 0.0

                lA_x, lbias = _bound_oneside(last_lA)
                uA_x, ubias = _bound_oneside(last_uA)

        # Case #2: weight is perturbed. bias may or may not be perturbed.
        elif self.is_input_perturbed(1):
            # Obtain relaxations for matrix multiplication.
            [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias = self.bound_backward_with_weight(last_lA, last_uA, input_lb, input_ub, x[0], x[1])
            if has_bias:
                if x[2].perturbation is not None:
                    # Bias is also perturbed. Since bias is directly added to the output, in backward mode it is treated
                    # as an input with last_lA and last_uA as associated bounds matrices.
                    # It's okay if last_lA or last_uA is eyeC, as it will be handled in the perturbation object.
                    lA_bias = last_lA
                    uA_bias = last_uA
                else:
                    # Bias not perturbed, so directly adding the bias of this layer to the final bound bias term.
                    if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                        # Bias will be directly added to output.
                        lbias += input_lb[2].unsqueeze(1).repeat(1, batch_size)
                        ubias += input_lb[2].unsqueeze(1).repeat(1, batch_size)
                    else:
                        if last_lA is not None:
                            lbias += last_lA.matmul(input_lb[2])
                        if last_uA is not None:
                            ubias += last_uA.matmul(input_lb[2])
            # If not has_bias, no need to compute lA_bias and uA_bias

        # Case 3: Only bias is perturbed, weight is not perturbed.
        elif not self.is_input_perturbed(1) and has_bias and self.is_input_perturbed(2):
            if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                # Use this layer's W as the next bound matrices. Duplicate the batch dimension. Other dimensions are kept 1.
                lA_x = uA_x = input_lb[1].unsqueeze(1).repeat([1, batch_size] + [1] * (input_lb[1].ndim - 1))
            else:
                lA_x = last_lA.matmul(input_lb[1])
                uA_x = last_uA.matmul(input_lb[1])
            # It's okay if last_lA or last_uA is eyeC, as it will be handled in the perturbation object.
            lA_bias = last_lA
            uA_bias = last_uA

        return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias

    def _reshape(self, x_l, x_u, y_l, y_u):
        x_shape, y_shape = self.input_shape, self.y_shape

        # (x_1, x_2, ..., x_{n-1}, -1, x_n) # FIXME
        x_l = x_l.unsqueeze(-2)
        x_u = x_u.unsqueeze(-2)

        # FIXME merge these two cases
        if len(x_shape) == len(y_shape):
            # (x_1, x_2, ..., -1, y_n, y_{n-1})
            y_l = y_l.unsqueeze(-3)
            y_u = y_u.unsqueeze(-3)
        elif len(y_shape) == 2:
            # (x_1, x_2, ..., -1, y_2, y_1)
            y_l = y_l.reshape(*([1] * (len(x_shape) - 2)), *y_shape).unsqueeze(-3)
            y_u = y_u.reshape(*([1] * (len(x_shape) - 2)), *y_shape).unsqueeze(-3)
        else:
            raise ValueError(f'Unsupported shapes: x_shape {x_shape}, y_shape {y_shape}')

        return x_l, x_u, y_l, y_u

    def _relax(self, input_lb, input_ub):
        return BoundMul.get_bound_mul(*self._reshape(input_lb[0], input_ub[0], input_lb[1], input_ub[1]))

    # FIXME This is nonlinear. Move to `bivariate.py`.
    def bound_backward_with_weight(self, last_lA, last_uA, input_lb, input_ub, x, y):
        # Note: x and y are not tranposed or scaled, and we should avoid using them directly.
        # Use input_lb and input_ub instead.
        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = self._relax(input_lb, input_ub)
        alpha_l, alpha_u = alpha_l.unsqueeze(0), alpha_u.unsqueeze(0)
        beta_l, beta_u = beta_l.unsqueeze(0), beta_u.unsqueeze(0)
        x_shape, y_shape = input_lb[0].size(), input_lb[1].size()
        gamma_l = torch.sum(gamma_l, dim=-1).reshape(x_shape[0], -1, 1)
        gamma_u = torch.sum(gamma_u, dim=-1).reshape(x_shape[0], -1, 1)

        if len(x.output_shape) != 2 and len(x.output_shape) == len(y.output_shape):
            dim_y = [-3]
        elif len(y.output_shape) == 2:
            dim_y = list(range(2, 2 + len(x_shape) - 2))
        else:
            raise NotImplementedError

        def _bound_oneside(last_A, alpha_pos, beta_pos, gamma_pos, alpha_neg, beta_neg, gamma_neg):
            if last_A is None:
                return None, None, 0
            if isinstance(last_A, eyeC):  # FIXME (12/28): Handle the OneHotC case.
                #FIXME previous implementation is incorrect
                #      expanding eyeC for now
                last_A = (torch.eye(last_A.shape[0], device=last_A.device)
                    .view(last_A.shape[0], 1, *last_A.shape[2:]).expand(last_A.shape))

            # last_uA has size (batch, spec, output)
            last_A_pos = last_A.clamp(min=0).unsqueeze(-1)
            last_A_neg = last_A.clamp(max=0).unsqueeze(-1)
            # alpha_u has size (batch, spec, output, input)
            # uA_x has size (batch, spec, input).
            A_x = (alpha_pos.transpose(-1, -2).matmul(last_A_pos) + \
                    alpha_neg.transpose(-1, -2).matmul(last_A_neg)).squeeze(-1)
            # beta_u has size (batch, spec, output, input)
            # uA_y is for weight matrix, with parameter size (output, input)
            # uA_y has size (batch, spec, output, input). This is an element-wise multiplication.
            A_y = last_A_pos * beta_pos + last_A_neg * beta_neg
            if len(dim_y) != 0:
                A_y = torch.sum(A_y, dim=dim_y)
            # last_uA has size (batch, spec, output)
            _last_A_pos = last_A_pos.reshape(last_A.shape[0], last_A.shape[1], -1)
            _last_A_neg = last_A_neg.reshape(last_A.shape[0], last_A.shape[1], -1)
            # gamma_u has size (batch, output, 1)
            # ubias has size (batch, spec, 1)
            bias = _last_A_pos.transpose(0, 1).matmul(gamma_pos).transpose(0, 1) + \
                    _last_A_neg.transpose(0, 1).matmul(gamma_neg).transpose(0, 1)

            bias = bias.squeeze(-1)
            return A_x, A_y, bias

        lA_x, lA_y, lbias = _bound_oneside(last_lA, alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u)
        uA_x, uA_y, ubias = _bound_oneside(last_uA, alpha_u, beta_u, gamma_u, alpha_l, beta_l, gamma_l)

        return [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias

    @staticmethod
    def _propagate_Linf(x, w):
        h_L, h_U = x
        mid = (h_L + h_U) / 2
        diff = (h_U - h_L) / 2
        w_abs = w.abs()
        if mid.ndim == 2 and w.ndim == 3:
            center = torch.bmm(mid.unsqueeze(1), w.transpose(-1, -2)).squeeze(1)
            deviation = torch.bmm(diff.unsqueeze(1), w_abs.transpose(-1, -2)).squeeze(1)
        else:
            center = mid.matmul(w.transpose(-1, -2))
            deviation = diff.matmul(w_abs.transpose(-1, -2))
        return center, deviation

    def interval_propagate(self, *v, C=None, w=None):
        has_bias = self is not None and len(v) == 3
        if self is not None:
            # This will convert an Interval object to tuple. We need to add perturbation property later.
            v_lb, v_ub = zip(*v)
            v_lb = self._preprocess(*v_lb)
            v_ub = self._preprocess(*v_ub)
            # After preprocess the lower and upper bounds, we make them Intervals again.
            v = [Interval.make_interval(bounds[0], bounds[1], bounds[2]) for bounds in zip(v_lb, v_ub, v)]
        if w is None and self is None:
            # Use C as the weight, no bias.
            w, lb, ub = C, torch.tensor(0., device=C.device), torch.tensor(0., device=C.device)
        else:
            if w is None:
                # No specified weight, use this layer's weight.
                if self.is_input_perturbed(1):  # input index 1 is weight.
                    # w is a perturbed tensor. Use IBP with weight perturbation.
                    # C matrix merging not supported.
                    assert C is None
                    res = self.interval_propagate_with_weight(*v)
                    l, u = res
                    if has_bias:
                        return l + v[2][0], u + v[2][1]
                    else:
                        return l, u
                else:
                    # Use weight
                    w = v[1][0]
            if has_bias:
                lb, ub = v[2]
            else:
                lb = ub = 0.0

            if C is not None:
                w = C.matmul(w)
                lb = C.matmul(lb) if not isinstance(lb, float) else lb
                ub = C.matmul(ub) if not isinstance(ub, float) else ub

        # interval_propagate() of the Linear layer may encounter input with different norms.
        norm, eps = Interval.get_perturbation(v[0])[:2]
        if norm == np.inf:
            interval = BoundLinear._propagate_Linf(v[0], w)
            center, deviation = interval
        elif norm > 0:
            # General Lp norm.
            norm, eps = Interval.get_perturbation(v[0])
            mid = v[0][0]
            dual_norm = np.float64(1.0) / (1 - 1.0 / norm)
            if w.ndim == 3:
                # Extra batch dimension.
                # mid has dimension [batch, input], w has dimension [batch, output, input].
                center = w.matmul(mid.unsqueeze(-1)).squeeze(-1)
            else:
                # mid has dimension [batch, input], w has dimension [output, input].
                center = mid.matmul(w.t())
            deviation = w.norm(dual_norm, dim=-1) * eps
        else: # here we calculate the L0 norm IBP bound of Linear layers, using the bound proposed in [Certified Defenses for Adversarial Patches, ICLR 2020]
            norm, eps, ratio = Interval.get_perturbation(v[0])
            mid = v[0][0]
            weight_abs = w.abs()
            if w.ndim == 3:
                # Extra batch dimension.
                # mid has dimension [batch, input], w has dimension [batch, output, input].
                center = w.matmul(mid.unsqueeze(-1)).squeeze(-1)
            else:
                # mid has dimension [batch, input], w has dimension [output, input].
                center = mid.matmul(w.t())
            # L0 norm perturbation
            k = int(eps)
            deviation = torch.sum(torch.topk(weight_abs, k)[0], dim=1) * ratio

        lower, upper = center - deviation + lb, center + deviation + ub

        return (lower, upper)

    def interval_propagate_with_weight(self, *v):
        input_norm, input_eps = Interval.get_perturbation(v[0])
        weight_norm, weight_eps = Interval.get_perturbation(v[1])

        if input_norm == np.inf and weight_norm == np.inf:
            # A memory-efficient implementation without expanding all the elementary multiplications
            if self.opt_matmul == 'economic':
                x_l, x_u = v[0][0], v[0][1]
                y_l, y_u = v[1][0].transpose(-1, -2), v[1][1].transpose(-1, -2)

                dx, dy = F.relu(x_u - x_l), F.relu(y_u - y_l)
                base = x_l.matmul(y_l)

                mask_xp, mask_xn = (x_l > 0).to(x_l.dtype), (x_u < 0).to(x_u.dtype)
                mask_xpn = 1 - mask_xp - mask_xn
                mask_yp, mask_yn = (y_l > 0).to(y_l.dtype), (y_u < 0).to(y_u.dtype)
                mask_ypn = 1 - mask_yp - mask_yn

                lower, upper = base.clone(), base.clone()

                lower += dx.matmul(y_l.clamp(max=0)) - (dx * mask_xn).matmul(y_l * mask_ypn)
                upper += dx.matmul(y_l.clamp(min=0)) + (dx * mask_xp).matmul(y_l * mask_ypn)

                lower += x_l.clamp(max=0).matmul(dy) - (x_l * mask_xpn).matmul(dy * mask_yn)
                upper += x_l.clamp(min=0).matmul(dy) + (x_l * mask_xpn).matmul(dy * mask_yp)

                lower += (dx * mask_xn).matmul(dy * mask_yn)
                upper += (dx * (mask_xpn + mask_xp)).matmul(dy * (mask_ypn + mask_yp))
            else:
                # Both input data and weight are Linf perturbed (with upper and lower bounds).
                # We need a x_l, x_u for each row of weight matrix.
                x_l, x_u = v[0][0].unsqueeze(-2), v[0][1].unsqueeze(-2)
                y_l, y_u = v[1][0].unsqueeze(-3), v[1][1].unsqueeze(-3)
                # Reuse the multiplication bounds and sum over results.
                lower, upper = BoundMul.interval_propagate_both_perturbed(*[(x_l, x_u), (y_l, y_u)])
                lower, upper = torch.sum(lower, -1), torch.sum(upper, -1)

            return lower, upper
        elif input_norm == np.inf and weight_norm == 2:
            # This eps is actually the epsilon per row, as only one row is involved for each output element.
            eps = weight_eps
            # Input data and weight are Linf perturbed (with upper and lower bounds).
            h_L, h_U = v[0]
            # First, handle non-perturbed weight with Linf perturbed data.
            center, deviation = BoundLinear._propagate_Linf(v[0], v[1][0])
            # Compute the maximal L2 norm of data. Size is [batch, 1].
            max_l2 = torch.max(h_L.abs(), h_U.abs()).norm(2, dim=-1).unsqueeze(-1)
            # Add the L2 eps to bounds.
            lb, ub = center - deviation - max_l2 * eps, center + deviation + max_l2 * eps
            return lb, ub
        else:
            raise NotImplementedError(
                "Unsupported perturbation combination: data={}, weight={}".format(input_norm, weight_norm))

    @staticmethod
    @torch.jit.script
    def bound_forward_mul(x_lw: Tensor, x_lb: Tensor, x_uw: Tensor, x_ub: Tensor, w: Tensor):
        w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
        lw = x_lw.matmul(w_pos) + x_uw.matmul(w_neg)
        uw = x_uw.matmul(w_pos) + x_lw.matmul(w_neg)
        lb = x_lb.matmul(w_pos) + x_ub.matmul(w_neg)
        ub = x_ub.matmul(w_pos) + x_lb.matmul(w_neg)
        return lw, lb, uw, ub

    # w: an optional argument which can be utilized by BoundMatMul
    def bound_dynamic_forward(self, x, w=None, b=None, C=None, max_dim=None, offset=0):
        assert not self.transA and self.alpha == 1.0 and self.transB and self.beta == 1.0
        assert not self.is_input_perturbed(1)
        assert not self.is_input_perturbed(2)

        weight = w.lb
        bias = b.lb if b is not None else None
        if C is not None:
            weight = C.to(weight).matmul(weight).transpose(-1, -2)
            if bias is not None:
                bias = C.to(bias).matmul(bias)
            lb = x.lb.unsqueeze(1)
        else:
            weight = weight.t()
            lb = x.lb

        w_new = x.lw.matmul(weight)
        b_new = lb.matmul(weight)
        if C is not None:
            b_new = b_new.squeeze(1)
        if bias is not None:
            b_new += bias

        return LinearBound(w_new, b_new, w_new, b_new, x_L=x.x_L, x_U=x.x_U, tot_dim=x.tot_dim)

    # w: an optional argument which can be utilized by BoundMatMul
    def bound_forward(self, dim_in, x, w=None, b=None, C=None):
        has_bias = b is not None
        #FIXME _preprocess can only be applied to tensors so far but not linear bounds.
        x, w, b = self._preprocess(x, w, b)

        # Case #1: No weight/bias perturbation, only perturbation on input.
        if not self.is_input_perturbed(1) and (not has_bias or not self.is_input_perturbed(2)):
            if isinstance(w, LinearBound):
                w = w.lower
            if isinstance(b, LinearBound):
                b = b.lower
            if C is not None:
                w = C.to(w).matmul(w).transpose(-1, -2)
                if b is not None:
                    b = C.to(b).matmul(b)
                x_lb, x_ub = x.lb.unsqueeze(1), x.ub.unsqueeze(1)
            else:
                w = w.t()
                x_lb, x_ub = x.lb, x.ub
            lw, lb, uw, ub = BoundLinear.bound_forward_mul(x.lw, x_lb, x.uw, x_ub, w)

            if C is not None:
                lb, ub = lb.squeeze(1), ub.squeeze(1)

            if b is not None:
                lb += b
                ub += b
        # Case #2: weight is perturbed. bias may or may not be perturbed.
        elif self.is_input_perturbed(1):
            if C is not None:
                raise NotImplementedError
            res = self.bound_forward_with_weight(dim_in, x, w)
            if has_bias:
                raise NotImplementedError
            lw, lb, uw, ub = res.lw, res.lb, res.uw, res.ub
        # Case 3: Only bias is perturbed, weight is not perturbed.
        elif not self.is_input_perturbed(1) and has_bias and self.is_input_perturbed(2):
            raise NotImplementedError

        return LinearBound(lw, lb, uw, ub)

    def bound_forward_with_weight(self, dim_in, x, y):
        x_unsqueeze = LinearBound(
            x.lw.unsqueeze(-2),
            x.lb.unsqueeze(-2),
            x.uw.unsqueeze(-2),
            x.ub.unsqueeze(-2),
            x.lower.unsqueeze(-2),
            x.upper.unsqueeze(-2),
        )
        y_unsqueeze = LinearBound(
            y.lw.unsqueeze(-3),
            y.lb.unsqueeze(-3),
            y.uw.unsqueeze(-3),
            y.ub.unsqueeze(-3),
            y.lower.unsqueeze(-3),
            y.upper.unsqueeze(-3),
        )
        res_mul = BoundMul.bound_forward_both_perturbed(dim_in, x_unsqueeze, y_unsqueeze)
        return LinearBound(
            res_mul.lw.sum(dim=-1) if res_mul.lw is not None else None,
            res_mul.lb.sum(dim=-1),
            res_mul.uw.sum(dim=-1) if res_mul.uw is not None else None,
            res_mul.ub.sum(dim=-1)
        )

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        has_bias = self is not None and len(v) == 3
        # e.g., last layer gurobi vars (1024,)
        gvars_array = np.array(v[0])
        # pre_layer_shape (1024,)
        pre_layer_shape = gvars_array.shape
        # this layer shape (100,)
        # if last layer, this layer shape (9,) instead of (10,)!!!
        this_layer_shape = self.lower.squeeze(0).shape
        out_lbs = self.lower.squeeze(0).detach().cpu().numpy() if self.lower is not None else None
        out_ubs = self.upper.squeeze(0).detach().cpu().numpy() if self.upper is not None else None

        # current layer weight (100, 1024)
        this_layer_weight = v[1]
        #### make sure if this is correct for per-label operations
        if C is not None:
            # merge specification C into last layer weights
            # only last layer has C not None
            this_layer_weight = C.squeeze(0).mm(this_layer_weight)
        # if last layer, this layer weight (9,100) instead of (10,100)!!!
        this_layer_weight = this_layer_weight.detach().cpu().numpy()

        this_layer_bias = None
        if has_bias:
            # current layer bias (100,)
            this_layer_bias = v[2]
            if C is not None:
                this_layer_bias = C.squeeze(0).mm(this_layer_bias.unsqueeze(-1)).view(-1)
            # if last layer, this layer bias (9,) instead of (10,)!!!
            this_layer_bias = this_layer_bias.detach().cpu().numpy()

        new_layer_gurobi_vars = []

        for neuron_idx in range(this_layer_shape[0]):
            out_lb = out_lbs[neuron_idx] if out_lbs is not None else -float('inf')
            out_ub = out_ubs[neuron_idx] if out_ubs is not None else float('inf')

            lin_expr = 0
            if has_bias:
                lin_expr = this_layer_bias[neuron_idx].item()
            coeffs = this_layer_weight[neuron_idx, :]

            if solver_pkg == 'gurobi':
                lin_expr += grb.LinExpr(coeffs, v[0])
            else:
                # FIXME (01/12/22): This is slow, must be fixed using addRow() or similar.
                for i in range(len(coeffs)):
                    try:
                        lin_expr += coeffs[i] * v[0][i]
                    except TypeError:
                        lin_expr += coeffs[i] * v[0][i].var

            var = model.addVar(lb=out_lb, ub=out_ub, obj=0,
                                    vtype=grb.GRB.CONTINUOUS,
                                    name=f'lay{self.name}_{neuron_idx}')
            model.addConstr(lin_expr == var, name=f'lay{self.name}_{neuron_idx}_eq')
            new_layer_gurobi_vars.append(var)

        self.solver_vars = new_layer_gurobi_vars
        model.update()


class BoundMatMul(BoundLinear):
    # Reuse most functions from BoundLinear.
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.transA = 0
        self.transB = 0
        self.requires_input_bounds = [0, 1]

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        self.x = x
        self.y = y
        return x.matmul(y)

    def interval_propagate(self, *v):
        lower, upper = super().interval_propagate(*v)
        return lower, upper

    def bound_backward(self, last_lA, last_uA, *x):
        assert len(x) == 2
        results = super().bound_backward(last_lA, last_uA, *x)
        lA_y = results[0][1][0].transpose(-1, -2) if results[0][1][0] is not None else None
        uA_y = results[0][1][1].transpose(-1, -2) if results[0][1][1] is not None else None
        return [results[0][0], (lA_y, uA_y), results[0][2]], results[1], results[2]

    def bound_forward(self, dim_in, x, y):
        return super().bound_forward(dim_in, x, LinearBound(
            y.lw.transpose(-1, -2) if y.lw is not None else None,
            y.lb.transpose(-1, -2) if y.lb is not None else None,
            y.uw.transpose(-1, -2) if y.uw is not None else None,
            y.ub.transpose(-1, -2) if y.ub is not None else None,
            y.lower.transpose(-1, -2) if y.lower is not None else None,
            y.upper.transpose(-1, -2) if y.upper is not None else None
        ))

class BoundNeg(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x):
        return -x

    def bound_backward(self, last_lA, last_uA, x):
        if type(last_lA) == Tensor or type(last_uA) == Tensor:
            return [(-last_lA if last_lA is not None else None,
                 -last_uA if last_uA is not None else None)], 0, 0
        elif type(last_lA) == Patches or type(last_uA) == Patches:
            if last_lA is not None:
                lA = Patches(-last_lA.patches, last_lA.stride, last_lA.padding, last_lA.shape, unstable_idx=last_lA.unstable_idx, output_shape=last_lA.output_shape)
            else:
                lA = None

            if last_uA is not None:
                uA = Patches(-last_uA.patches, last_uA.stride, last_uA.padding, last_uA.shape, unstable_idx=last_uA.unstable_idx, output_shape=last_uA.output_shape)
            else:
                uA = None
            return [(lA, uA)], 0, 0
        else:
            raise NotImplementedError

    def bound_forward(self, dim_in, x):
        return LinearBound(-x.uw, -x.ub, -x.lw, -x.lb)

    def interval_propagate(self, *v):
        return -v[0][1], -v[0][0]


class BoundCumSum(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True

    def forward(self, x, axis):
        self.axis = axis
        return torch.cumsum(x, axis)

    def infer_batch_dim(self, batch_size, *x):
        assert self.axis != x[0]
        return x[0]


class BoundIdentity(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True

    def forward(self, x):
        return x

    def bound_backward(self, last_lA, last_uA, x):
        return [(last_lA, last_uA)], 0, 0

    def bound_forward(self, dim_in, x):
        return x
