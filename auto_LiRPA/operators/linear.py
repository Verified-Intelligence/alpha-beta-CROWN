""" Linear (possibly with weight perturbation) or Dot product layers """
from .base import *
from .bivariate import BoundMul

class BoundLinear(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        # Gemm:
        # A = A if transA == 0 else A.T
        # B = B if transB == 0 else B.T
        # C = C if C is not None else np.array(0)
        # Y = alpha * np.dot(A, B) + beta * C
        # return Y

        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

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
        if self.transA and isinstance(a, torch.Tensor):
            a = a.transpose(-2,-1)
        if self.alpha != 1.0:
            a = self.alpha * a
        if not self.transB and isinstance(b, torch.Tensor):
            # our code assumes B is transposed (common case), so we transpose B only when it is not transposed in gemm.
            b = b.transpose(-2,-1)
        if c is not None:
            if self.beta != 1.0:
                c = self.beta * c
        return a, b, c

    @Bound.save_io_shape
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
            index = C.index.transpose(0,1)
            coeffs = C.coeffs.transpose(0,1)
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
        input_lb = [xi.lower if hasattr(xi, 'lower') else None for xi in x]
        input_ub = [xi.upper if hasattr(xi, 'upper') else None for xi in x]
        # transpose and scale each term if necessary.
        input_lb = self._preprocess(*input_lb)
        input_ub = self._preprocess(*input_ub)
        lA_y = uA_y = lA_bias = uA_bias = None
        lbias = ubias = 0
        batch_size = last_lA.shape[1] if last_lA is not None else last_uA.shape[1]

        # Case #1: No weight/bias perturbation, only perturbation on input.
        if not self.is_input_perturbed(1) and (not has_bias or not self.is_input_perturbed(2)):
            # If last_lA and last_uA are indentity matrices.
            if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                # Use this layer's W as the next bound matrices. Duplicate the batch dimension. Other dimensions are kept 1.
                # Not perturbed, so we can use either lower or upper.
                lA_x = uA_x = input_lb[1].unsqueeze(1).repeat([1, batch_size] + [1] * (input_lb[1].ndim - 1))
                # Bias will be directly added to output.
                if has_bias:
                    lbias = ubias = input_lb[2].unsqueeze(1).repeat(1, batch_size)
            elif isinstance(last_lA, OneHotC) or isinstance(last_uA, OneHotC):
                # We need to select several rows from the weight matrix (its shape is output_size * input_size).
                lA_x, lbias = self.onehot_mult(input_lb[1], input_lb[2] if has_bias else None, last_lA, batch_size)
                if last_lA is last_uA:
                    uA_x = lA_x
                    ubias = lbias
                else:
                    uA_x, ubias = self.onehot_mult(input_lb[1], input_lb[2] if has_bias else None, last_uA, batch_size)
            else:
                def _bound_oneside(last_A):
                    if last_A is None:
                        return None, 0
                    # Just multiply this layer's weight into bound matrices, and produce biases.
                    next_A = last_A.to(input_lb[1]).matmul(input_lb[1])
                    sum_bias = (last_A.to(input_lb[2]).matmul(input_lb[2]) 
                        if has_bias else 0.0)
                    return next_A, sum_bias

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

        # (x_1, x_2, ..., x_{n-1}, y_2, x_n)
        x_l = x_l.unsqueeze(-2)
        x_u = x_u.unsqueeze(-2)

        if len(x_shape) == len(y_shape):
            # (x_1, x_2, ..., x_{n-1}, y_n, y_{n-1})
            shape = x_shape[:-1] + (y_shape[-1], y_shape[-2])
            y_l = y_l.unsqueeze(-3)
            y_u = y_u.unsqueeze(-3)
        elif len(y_shape) == 2:
            # (x_1, x_2, ..., x_{n-1}, y_2, y_1)
            shape = x_shape[:-1] + y_shape[1:] + y_shape[:1]
            y_l = y_l.reshape(*([1] * (len(x_shape) - 2)), *y_shape).unsqueeze(-3)
            y_u = y_u.reshape(*([1] * (len(x_shape) - 2)), *y_shape).unsqueeze(-3)
        return x_l, x_u, y_l, y_u

    def _relax(self, input_lb, input_ub):
        return BoundMul.get_bound_mul(*self._reshape(input_lb[0], input_ub[0], input_lb[1], input_ub[1]))

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
            if isinstance(last_A, eyeC):
                A_x = alpha_pos.squeeze(0).permute(1, 0, 2).repeat(1, last_A.shape[1], 1)
                A_y = beta_pos * torch.eye(last_A.shape[2], device=last_A.device) \
                    .view((last_A.shape[2], 1, last_A.shape[2], 1))
                if len(dim_y) != 0:
                    A_y = torch.sum(beta_pos, dim=dim_y)
                bias = gamma_pos.transpose(0, 1)
            else:
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
        if Interval.use_relative_bounds(x):
            if len(x.nominal.shape) == 2 and w.ndim == 3:
                nominal = torch.bmm(x.nominal.unsqueeze(1), w.transpose(-1, -2)).squeeze(1)
                lower_offset = (
                    torch.bmm(x.lower_offset.unsqueeze(1), w.clamp(min=0).transpose(-1, -2)) + 
                    torch.bmm(x.upper_offset.unsqueeze(1), w.clamp(max=0).transpose(-1, -2))).squeeze(1)
                upper_offset = (
                    torch.bmm(x.lower_offset.unsqueeze(1), w.clamp(max=0).transpose(-1, -2)) + 
                    torch.bmm(x.upper_offset.unsqueeze(1), w.clamp(min=0).transpose(-1, -2))).squeeze(1)
            else:
                nominal = x.nominal.matmul(w.transpose(-1, -2))
                lower_offset = (
                    x.lower_offset.matmul(w.clamp(min=0).transpose(-1, -2)) + 
                    x.upper_offset.matmul(w.clamp(max=0).transpose(-1, -2)))
                upper_offset = (
                    x.lower_offset.matmul(w.clamp(max=0).transpose(-1, -2)) + 
                    x.upper_offset.matmul(w.clamp(min=0).transpose(-1, -2)))
            return Interval(None, None, nominal, lower_offset, upper_offset)
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
            if Interval.use_relative_bounds(v[0]):
                v_nominal = self._preprocess(v[0].nominal, v[1].nominal, v[2].nominal)
                v_lower_offset = self._preprocess(v[0].lower_offset, v[1].lower_offset, v[2].lower_offset)
                v_upper_offset = self._preprocess(v[0].upper_offset, v[1].upper_offset, v[2].upper_offset)
                v = [Interval(None, None, bounds[0], bounds[1], bounds[2]) 
                    for bounds in zip(v_nominal, v_lower_offset, v_upper_offset)]
            else:
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
                    if Interval.use_relative_bounds(res):
                        if has_bias:
                            raise NotImplementedError
                        else:
                            return res
                    else:
                        l, u = res
                        if has_bias:
                            return l + v[2][0], u + v[2][1]
                        else:
                            return l, u
                else:
                    # Use weight 
                    if Interval.use_relative_bounds(v[1]):
                        w = v[1].nominal
                    else:
                        w = v[1][0]
            if has_bias:
                lb, ub = (v[2].lower, v[2].upper) if Interval.use_relative_bounds(v[2]) else v[2]
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
            if isinstance(interval, Interval):
                b_center = (lb + ub) / 2
                interval.nominal += b_center
                interval.lower_offset += lb - b_center
                interval.upper_offset += ub - b_center
                return interval
            else:
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

        if Interval.use_relative_bounds(*v):
            assert input_norm == weight_norm == np.inf
            assert self.opt_matmul == 'economic'
            
            x, y = v[0], v[1]

            nominal = x.nominal.matmul(y.nominal.transpose(-1, -2))

            matmul_offset = torch.matmul(
                torch.max(x.lower_offset.abs(), x.upper_offset.abs()),
                torch.max(y.upper_offset.abs(), y.lower_offset.abs()).transpose(-1, -2))

            lower_offset = (
                x.nominal.clamp(min=0).matmul(y.lower_offset.transpose(-1, -2)) + 
                x.nominal.clamp(max=0).matmul(y.upper_offset.transpose(-1, -2)) + 
                x.lower_offset.matmul(y.nominal.clamp(min=0).transpose(-1, -2)) + 
                x.upper_offset.matmul(y.nominal.clamp(max=0).transpose(-1, -2)) - matmul_offset)
            
            upper_offset = (
                x.nominal.clamp(min=0).matmul(y.upper_offset.transpose(-1, -2)) + 
                x.nominal.clamp(max=0).matmul(y.lower_offset.transpose(-1, -2)) + 
                x.upper_offset.matmul(y.nominal.clamp(min=0).transpose(-1, -2)) + 
                x.lower_offset.matmul(y.nominal.clamp(max=0).transpose(-1, -2)) + matmul_offset)

            return Interval(None, None, nominal, lower_offset, upper_offset)

        self.x_shape = v[0][0].shape
        self.y_shape = v[1][0].shape

        if input_norm == np.inf and weight_norm == np.inf:
            # A memory-efficient implementation without expanding all the elementary multiplications
            if self.opt_matmul == 'economic':
                x_l, x_u = v[0][0], v[0][1]
                y_l, y_u = v[1][0].transpose(-1, -2), v[1][1].transpose(-1, -2)

                dx, dy = F.relu(x_u - x_l), F.relu(y_u - y_l)
                base = x_l.matmul(y_l)

                mask_xp, mask_xn = (x_l > 0).float(), (x_u < 0).float()
                mask_xpn = 1 - mask_xp - mask_xn
                mask_yp, mask_yn = (y_l > 0).float(), (y_u < 0).float()
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
                lower, upper = BoundMul.interval_propagate(*[(x_l, x_u), (y_l, y_u)])
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

    # w: an optional argument which can be utilized by BoundMatMul
    def bound_forward(self, dim_in, x, w=None, b=None, C=None):
        has_bias = b is not None
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
                w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
                lb = (x.lb.unsqueeze(1).matmul(w_pos) + x.ub.unsqueeze(1).matmul(w_neg)).squeeze(1)
                ub = (x.ub.unsqueeze(1).matmul(w_pos) + x.lb.unsqueeze(1).matmul(w_neg)).squeeze(1)
            else:               
                w = w.t()
                w_pos, w_neg = w.clamp(min=0), w.clamp(max=0)
                lb = x.lb.matmul(w_pos) + x.ub.matmul(w_neg)
                ub = x.ub.matmul(w_pos) + x.lb.matmul(w_neg)
            lw = x.lw.matmul(w_pos) + x.uw.matmul(w_neg)
            uw = x.uw.matmul(w_pos) + x.lw.matmul(w_neg)
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
        res_mul = BoundMul.bound_forward(dim_in, x_unsqueeze, y_unsqueeze)
        return LinearBound(
            res_mul.lw.sum(dim=-1) if res_mul.lw is not None else None,
            res_mul.lb.sum(dim=-1),
            res_mul.uw.sum(dim=-1) if res_mul.uw is not None else None,
            res_mul.ub.sum(dim=-1)
        )


class BoundMatMul(BoundLinear):
    # Reuse most functions from BoundLinear.
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.transA = 0
        self.transB = 1  # MatMul assumes B is transposed.
        self.nonlinear = True

    @Bound.save_io_shape
    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        self.x = x
        self.y = y
        return x.matmul(y)

    def interval_propagate(self, *v):
        w_l = v[1][0].transpose(-1, -2)
        w_u = v[1][1].transpose(-1, -2)
        lower, upper = super().interval_propagate(v[0], (w_l, w_u))
        return lower, upper   

    def bound_backward(self, last_lA, last_uA, *x):
        assert len(x) == 2
        # BoundLinear has W transposed.
        x[1].lower = x[1].lower.transpose(-1, -2)
        x[1].upper = x[1].upper.transpose(-1, -2)
        results = super().bound_backward(last_lA, last_uA, *x)
        # Transpose input back.
        x[1].lower = x[1].lower.transpose(-1, -2)
        x[1].upper = x[1].upper.transpose(-1, -2)
        lA_y = results[0][1][0].transpose(-1, -2) if results[0][1][0] is not None else None
        uA_y = results[0][1][1].transpose(-1, -2) if results[0][1][1] is not None else None
        # Transpose result on A.
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

    def infer_batch_dim(self, batch_size, *x):
        return BoundMul.infer_batch_dim(batch_size, *x)      

class BoundNeg(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, x):
        return -x

    def bound_backward(self, last_lA, last_uA, x):
        return [(-last_lA if last_lA is not None else None,
                 -last_uA if last_uA is not None else None)], 0, 0

    def bound_forward(self, dim_in, x):
        return LinearBound(-x.uw, -x.ub, -x.lw, -x.lb)

    def interval_propagate(self, *v):
        return -v[0][1], -v[0][0]    

class BoundCumSum(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.use_default_ibp = True

    @Bound.save_io_shape
    def forward(self, x, axis):
        self.axis = axis
        return torch.cumsum(x, axis)

    def infer_batch_dim(self, batch_size, *x):
        assert self.axis != x[0]
        return x[0]