""" Shape operators """
from .base import *
from ..patches import Patches, patches_to_matrix
from .linear import BoundLinear

class BoundReshape(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        # It can be set to `view`, so that `view` instead of `reshape` will be used.
        self.option = options.get('reshape', 'reshape')

    def forward(self, x, shape):
        shape = list(shape)
        for i in range(len(shape)):
            if shape[i] == -1:
                shape[i] = prod(x.shape) // int(prod(shape[:i]) * prod(shape[(i + 1):]))
        self.shape = shape
        if self.option == 'view':
            return x.contiguous().view(shape)
        else:
            return x.reshape(shape)

    def bound_backward(self, last_lA, last_uA, x, shape):
        def _bound_oneside(A):
            if A is None:
                return None
            if type(A) == Patches:
                if type(self.inputs[0]) == BoundLinear:
                    # Save the shape and it will be converted to matrix in Linear layer.
                    return A.create_similar(input_shape=self.output_shape)
                if A.unstable_idx is None:
                    patches = A.patches
                    # non-sparse: [batch, out_dim, out_c, out_H, out_W, out_dim, in_c, H, W]
                    # [batch, out_dim*out_c, out_H, out_W, out_dim*in_c, H, W]
                    patches = patches.reshape(
                        patches.shape[0],
                        patches.shape[1]*patches.shape[2], patches.shape[3], patches.shape[4],
                        patches.shape[5]*patches.shape[6], patches.shape[7], patches.shape[8])
                    # expected next_A shape [batch, spec, in_c, in_H , in_W].
                    next_A = patches_to_matrix(
                        patches, [
                            self.input_shape[0]*self.input_shape[1],
                            patches.shape[-3],
                            int(math.sqrt(self.input_shape[-1]//A.patches.shape[-3])),
                            int(math.sqrt(self.input_shape[-1]//A.patches.shape[-3]))],
                        A.stride, A.padding)
                else:
                    # sparse: [spec, batch, in_c, patch_H, patch_W] (specs depends on the number of unstable neurons).
                    patches = A.patches
                    # expected next_A shape [batch, spec, input_c, in_H, in_W].
                    next_A = patches_to_matrix(patches, [self.input_shape[0]*self.input_shape[1], patches.shape[-3], int(math.sqrt(self.input_shape[-1]//patches.shape[-3])), int(math.sqrt(self.input_shape[-1]//patches.shape[-3]))], A.stride, A.padding, output_shape=A.output_shape, unstable_idx=A.unstable_idx)
                # Reshape it to [batch, spec, *input_shape]  (input_shape is the shape before Reshape operation).
                next_A = next_A.reshape(A.shape[1], -1, *self.input_shape[1:])
                return next_A.transpose(0,1)
            else:
                return A.reshape(A.shape[0], A.shape[1], *self.input_shape[1:])
        #FIXME check reshape or view
        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, x, shape):
        batch_size = x.lw.shape[0]
        lw = x.lw.reshape(batch_size, dim_in, *self.shape[1:])
        uw = x.uw.reshape(batch_size, dim_in, *self.shape[1:])
        lb = x.lb.reshape(batch_size, *self.shape[1:])
        ub = x.ub.reshape(batch_size, *self.shape[1:])
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        return Interval.make_interval(
            self.forward(v[0][0], v[1][0]),
            self.forward(v[0][1], v[1][0]), v[0])

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        elif self.input_shape[x[0]] == self.shape[x[0]]:
            return x[0]
        raise NotImplementedError('input shape {}, new shape {}, input batch dim {}'.format(
            self.input_shape, self.shape, x[0]
        ))

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        if isinstance(v[0], Tensor):
            self.solver_vars = self.forward(*v)
            return
        gvar_array = np.array(v[0])
        gvar_array = gvar_array.reshape(v[1].detach().cpu().numpy())[0]
        self.solver_vars = gvar_array.tolist()


class BoundUnsqueeze(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.axes = attr['axes']
        assert (len(self.axes) == 1)
        self.axes = self.axes[0]
        self.use_default_ibp = True

    def forward(self, x):
        return x.unsqueeze(self.axes)

    def bound_backward(self, last_lA, last_uA, x):
        self.axes = self.make_axis_non_negative(self.axes, 'output')
        if self.axes == 0:
            # TODO: unsqueeze on batch dimension can be problematic.
            return [(last_lA, last_uA)], 0, 0
        else:
            if type(last_lA) == Patches:
                lA = Patches(last_lA.patches.squeeze(self.axes - 5), last_lA.stride, last_lA.padding, last_lA.shape, last_lA.identity, last_lA.unstable_idx, last_lA.output_shape)
            elif last_lA is not None:
                lA = last_lA.squeeze(self.axes+1)
            else:
                lA = None
            if type(last_uA) == Patches:
                uA = Patches(last_uA.patches.squeeze(self.axes - 5), last_uA.stride, last_uA.padding, last_uA.shape, last_uA.identity, last_uA.unstable_idx, last_uA.output_shape)
            elif last_uA is not None:
                uA = last_uA.squeeze(self.axes+1)
            else:
                uA = None
            return [(lA, uA)], 0, 0

    def bound_forward(self, dim_in, x):
        self.axes = self.make_axis_non_negative(self.axes, 'output')
        if len(self.input_shape) == 0:
            lw, lb = x.lw.unsqueeze(1), x.lb.unsqueeze(0)
            uw, ub = x.uw.unsqueeze(1), x.ub.unsqueeze(0)
        else:
            lw, lb = x.lw.unsqueeze(self.axes + 1), x.lb.unsqueeze(self.axes)
            uw, ub = x.uw.unsqueeze(self.axes + 1), x.ub.unsqueeze(self.axes)
        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        elif self.axes > x[0]:
            return x[0]
        raise NotImplementedError

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(v[0])

class BoundSqueeze(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.axes = attr['axes']
        assert (len(self.axes) == 1)
        self.axes = self.axes[0]
        self.use_default_ibp = True

    def forward(self, x):
        return x.squeeze(self.axes)

    def bound_backward(self, last_lA, last_uA, x):
        assert (self.axes != 0)
        return [(last_lA.unsqueeze(self.axes + 1) if last_lA is not None else None,
                 last_uA.unsqueeze(self.axes + 1) if last_uA is not None else None)], 0, 0

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        elif x[0] < self.axes:
            return x[0]
        elif x[0] > self.axes:
            return x[0] - 1
        else:
            assert 0


class BoundFlatten(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True
        self.axis = attr['axis']

    def forward(self, x):
        return torch.flatten(x, self.axis)

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(A):
            if A is None:
                return None
            return A.reshape(A.shape[0], A.shape[1], *self.input_shape[1:])
        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_dynamic_forward(self, x, max_dim=None, offset=0):
        w = torch.flatten(x.lw, self.axis + 1)
        b = torch.flatten(x.lb, self.axis)
        x_L = torch.flatten(x.x_L, self.axis)
        x_U = torch.flatten(x.x_U, self.axis)
        return LinearBound(w, b, w, b, x_L=x_L, x_U=x_U, tot_dim=x.tot_dim)

    def bound_forward(self, dim_in, x):
        self.axis = self.make_axis_non_negative(self.axis)
        assert self.axis > 0
        return LinearBound(
            torch.flatten(x.lw, self.axis + 1),
            torch.flatten(x.lb, self.axis),
            torch.flatten(x.uw, self.axis + 1),
            torch.flatten(x.ub, self.axis),
        )

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        # e.g., v[0] input shape (16, 8, 8) => output shape (1024,)
        self.solver_vars = np.array(v[0]).reshape(-1).tolist()
        model.update()



class BoundConcat(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr['axis']
        self.IBP_rets = None

    def forward(self, *x):  # x is a list of tensors
        x = [(item if isinstance(item, Tensor) else torch.tensor(item)) for item in x]
        self.input_size = [item.shape[self.axis] for item in x]
        self.axis = self.make_axis_non_negative(self.axis)
        return torch.cat(x, dim=int(self.axis))

    def interval_propagate(self, *v):
        norms = []
        eps = []
        # Collect perturbation information for all inputs.
        for i, _v in enumerate(v):
            if self.is_input_perturbed(i):
                n, e = Interval.get_perturbation(_v)
                norms.append(n)
                eps.append(e)
            else:
                norms.append(None)
                eps.append(0.0)
        eps = np.array(eps)
        # Supporting two cases: all inputs are Linf norm, or all inputs are L2 norm perturbed.
        # Some inputs can be constants without perturbations.
        all_inf = all(map(lambda x: x is None or x == np.inf, norms))
        all_2 = all(map(lambda x: x is None or x == 2, norms))

        h_L = [_v[0] for _v in v]
        h_U = [_v[1] for _v in v]
        if all_inf:
            # Simply returns a tuple. Every subtensor has its own lower and upper bounds.
            return self.forward(*h_L), self.forward(*h_U)
        elif all_2:
            # Sum the L2 norm over all subtensors, and use that value as the new L2 norm.
            # This will be an over-approximation of the original perturbation (we can prove it).
            max_eps = np.sqrt(np.sum(eps * eps))
            # For L2 norm perturbed inputs, lb=ub and for constants lb=ub. Just propagate one object.
            r = self.forward(*h_L)
            ptb = PerturbationLpNorm(norm=2, eps=max_eps)
            return Interval(r, r, ptb=ptb)
        else:
            raise RuntimeError("BoundConcat does not support inputs with norm {}".format(norms))

    def bound_backward(self, last_lA, last_uA, *x):
        self.axis = self.make_axis_non_negative(self.axis, 'output')
        assert self.axis > 0

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            if isinstance(last_A, torch.Tensor):
                return torch.split(last_A, self.input_size, dim=self.axis + 1)
            elif isinstance(last_A, Patches):
                assert len(self.input_shape) == 4 and self.axis == 1, "Split channel dimension is supported; others are unimplemented."
                # Patches shape can be [out_c, batch, out_h, out_w, in_c, patch_h, patch_w]
                # Or [spec, batch, in_c, patch_h, patch_w]  (sparse)
                new_patches = torch.split(last_A.patches, self.input_size, dim=-3)  # split the in_c dimension is easy.
                return [last_A.create_similar(p) for p in new_patches]
            else:
                raise RuntimeError(f'Unsupported type for last_A: {type(last_A)}')

        uA = _bound_oneside(last_uA)
        lA = _bound_oneside(last_lA)
        if uA is None:
            return [(lA[i] if lA is not None else None, None) for i in range(len(lA))], 0, 0
        if lA is None:
            return [(None, uA[i] if uA is not None else None) for i in range(len(uA))], 0, 0
        return [(lA[i], uA[i]) for i in range(len(lA))], 0, 0

    def bound_forward(self, dim_in, *x):
        self.axis = self.make_axis_non_negative(self.axis)
        assert (self.axis == 0 and not self.from_input or self.from_input)
        lw = torch.cat([item.lw for item in x], dim=self.axis + 1)
        lb = torch.cat([item.lb for item in x], dim=self.axis)
        uw = torch.cat([item.uw for item in x], dim=self.axis + 1)
        ub = torch.cat([item.ub for item in x], dim=self.axis)
        return LinearBound(lw, lb, uw, ub)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(*v)

    def infer_batch_dim(self, batch_size, *x):
        assert np.min(x) == np.max(x)
        assert x[0] != self.axis
        return x[0]


BoundConcatFromSequence = BoundConcat

class BoundShape(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.use_default_ibp = True

    @staticmethod
    def shape(x):
        return x.shape if isinstance(x, Tensor) else torch.tensor(x).shape

    def forward(self, x):
        self.from_input = False
        return BoundShape.shape(x)

    def bound_forward(self, dim_in, x):
        return self.forward_value

    def infer_batch_dim(self, batch_size, *x):
        return -1

    def interval_propagate(self, *v):
        return super().interval_propagate(*v)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        if not isinstance(v[0], Tensor):
            # e.g., v[0] input shape (8, 7, 7) => output its shape (1, 8, 7, 7)
            gvars_array = np.array(v[0])
            self.solver_vars = torch.tensor(np.expand_dims(gvars_array, axis=0).shape).long()
        else:
            self.solver_vars = torch.tensor(self.forward(v[0])).long()


class BoundGather(Bound):
    def __init__(self, attr, x, output_index, options):
        super().__init__(attr, x, output_index, options)
        self.axis = attr['axis'] if 'axis' in attr else 0

    def forward(self, x, indices):
        self.indices = indices
        if self.axis == -1:
            self.axis = len(x.shape) - 1
        x = x.to(self.indices.device)  # BoundShape.shape() will return value on cpu only
        if indices.ndim == 0:
            return torch.index_select(x, dim=self.axis, index=indices).squeeze(self.axis)
        elif self.axis == 0:
            return torch.index_select(x, dim=self.axis, index=indices.reshape(-1)) \
                .reshape(*indices.shape, x.shape[-1])
        elif self.indices.ndim == 1:
            # `index_select` requires `indices` to be a 1-D tensor
            return torch.index_select(x, dim=self.axis, index=indices)

        raise ValueError('Unsupported shapes in Gather: data {}, indices {}, axis {}'.format(x.shape, indices.shape, self.axis))

    def bound_backward(self, last_lA, last_uA, x, indices):
        assert self.from_input

        def _expand_A_with_zeros(A, axis, idx, max_axis_size):
            # Need to recreate A with three parts: before the gathered element, gathered element, and after gathered element.
            tensors = []
            if idx > 0:
                shape_pre = list(A.shape)
                shape_pre[axis] *= idx
                # Create the same shape as A, except for the dimension to be gathered.
                tensors.append(torch.zeros(shape_pre, device=A.device))
            # The gathered element itself, in the middle.
            tensors.append(A)
            if max_axis_size - idx - 1 > 0:
                shape_next = list(A.shape)
                shape_next[axis] *= max_axis_size - idx - 1
                # Create the rest part of A.
                tensors.append(torch.zeros(shape_next, device=A.device))
            # Concatenate all three parts together.
            return torch.cat(tensors, dim=axis)

        def _bound_oneside(A):
            if A is None:
                return None

            if isinstance(A, torch.Tensor):
                if self.indices.ndim == 0:
                    A = A.unsqueeze(self.axis + 1)
                    idx = int(self.indices)
                    return _expand_A_with_zeros(A, self.axis + 1, idx, self.input_shape[self.axis])
                else:
                    shape = list(A.shape)
                    final_A = torch.zeros(*shape[:self.axis + 1], self.input_shape[self.axis], *shape[self.axis + 2:], device=A.device)
                    idx = self.indices.view([*[1]*(self.axis+1), -1, *[1]*len(shape[self.axis + 2:])])
                    idx = idx.repeat([*A.shape[:self.axis+1], 1, *A.shape[self.axis+2:]])
                    final_A.scatter_(dim=self.axis+1, index=idx, src=A)
                    return final_A
            elif isinstance(A, Patches):
                if self.indices.ndim == 0:
                    idx = int(self.indices)
                    assert len(self.input_shape) == 4 and self.axis == 1, "Gather is only supported on the channel dimension for Patches mode."
                    # For gather in the channel dimension, we only need to deal with the in_c dimension (-3) in patches.
                    patches = A.patches
                    # -3 is the in_c dimension.
                    new_patches = _expand_A_with_zeros(patches, axis=-3, idx=idx, max_axis_size=self.input_shape[self.axis])
                    return A.create_similar(new_patches)
                else:
                    raise NotImplementedError
            else:
                raise ValueError(f'Unknown last_A type {type(A)}')

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, x, indices):
        assert self.indices.ndim == 0  # TODO

        if isinstance(x, torch.Size):
            lw = uw = torch.zeros(dim_in, device=self.device)
            lb = ub = torch.index_select(
                torch.tensor(x, device=self.device),
                dim=self.axis, index=self.indices).squeeze(self.axis)
        else:
            axis = self.axis + 1
            lw = torch.index_select(x.lw, dim=self.axis + 1, index=self.indices).squeeze(axis)
            uw = torch.index_select(x.uw, dim=self.axis + 1, index=self.indices).squeeze(axis)
            lb = torch.index_select(x.lb, dim=self.axis, index=self.indices).squeeze(self.axis)
            ub = torch.index_select(x.ub, dim=self.axis, index=self.indices).squeeze(self.axis)
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1)
        return self.forward(v[0][0], v[1][0]), self.forward(v[0][1], v[1][0])

    def infer_batch_dim(self, batch_size, *x):
        if x[0] != -1:
            assert self.axis != x[0]
            return x[0]
        else:
            return x[1]

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(v[0], v[1])


class BoundGatherElements(Bound):
    def __init__(self, attr, input, output_index, options):
        super().__init__(attr, input, output_index, options)
        self.axis = attr['axis']

    def forward(self, x, index):
        self.index = index
        return torch.gather(x, dim=self.axis, index=index)

    def bound_backward(self, last_lA, last_uA, x, index):
        assert self.from_input

        dim = self._get_dim()

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            A = torch.zeros(
                last_A.shape[0], last_A.shape[1], *x.output_shape[1:], device=last_A.device)
            A.scatter_(
                dim=dim + 1,
                index=self.index.unsqueeze(0).repeat(A.shape[0], *([1] * (A.ndim - 1))),
                src=last_A)
            return A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1)
        return self.forward(v[0][0], v[1][0]), \
               self.forward(v[0][1], v[1][1])

    def bound_forward(self, dim_in, x, index):
        assert self.axis != 0
        dim = self._get_dim()
        return LinearBound(
            torch.gather(x.lw, dim=dim + 1, index=self.index.unsqueeze(1).repeat(1, dim_in, 1)),
            torch.gather(x.lb, dim=dim, index=self.index),
            torch.gather(x.uw, dim=dim + 1, index=self.index.unsqueeze(1).repeat(1, dim_in, 1)),
            torch.gather(x.ub, dim=dim, index=self.index))

    def infer_batch_dim(self, batch_size, *x):
        assert self.axis != x[0]
        return x[0]

    def _get_dim(self):
        dim = self.axis
        if dim < 0:
            dim = len(self.output_shape) + dim
        return dim

class BoundTranspose(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.perm = attr['perm']
        self.perm_inv_inc_one = [-1] * (len(self.perm) + 1)
        self.perm_inv_inc_one[0] = 0
        for i in range(len(self.perm)):
            self.perm_inv_inc_one[self.perm[i] + 1] = i + 1
        self.use_default_ibp = True

    def forward(self, x):
        return x.permute(*self.perm)

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(last_A):
            if last_A is None:
                return None
            return last_A.permute(self.perm_inv_inc_one)

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        if self.input_shape[0] != 1:
            perm = [0] + [(p + 1) for p in self.perm]
        else:
            assert (self.perm[0] == 0)
            perm = [0, 1] + [(p + 1) for p in self.perm[1:]]
        lw, lb = x.lw.permute(*perm), x.lb.permute(self.perm)
        uw, ub = x.uw.permute(*perm), x.ub.permute(self.perm)

        return LinearBound(lw, lb, uw, ub)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(*v)

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        else:
            return self.perm.index(x[0])


class BoundSlice(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.start = attr["starts"][0] if "starts" in attr else None
        self.end = attr["ends"][0] if "ends" in attr else None
        self.axes = attr["axes"][0] if "axes" in attr else None
        self.use_default_ibp = False

    def _fixup_params(self, shape, start, end, axes, steps):
        if start < 0:
            start += shape[axes]
        if end < 0:
            if end == -9223372036854775807:  # -inf in ONNX
                end = 0  # only possible when step == -1
            else:
                end += shape[axes]
        if steps == -1:
            start, end = end, start + 1  # TODO: more test more negative step size.
        end = min(end, shape[axes])
        return start, end

    # Older Pytorch version only passes steps as input.
    def forward(self, x, start=None, end=None, axes=None, steps=1):
        start = self.start if start is None else start
        end = self.end if end is None else end
        axes = self.axes if axes is None else axes
        assert (steps == 1 or steps == -1) and axes == int(axes) and start == int(start) and end == int(end)
        shape = x.shape if isinstance(x, Tensor) else [len(x)]
        start, end = self._fixup_params(shape, start, end, axes, steps)
        final = torch.narrow(x, dim=int(axes), start=int(start), length=int(end - start))
        if steps == -1:
            final = torch.flip(final, dims=tuple(axes))
        return final

    def interval_propagate(self, *v):
        lb = tuple(map(lambda x:x[0],v))
        ub = tuple(map(lambda x:x[1],v))
        return Interval.make_interval(self.forward(*lb), self.forward(*ub))

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(*v)

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        else:
            assert self.axes != x[0]
            return x[0]

    def bound_backward(self, last_lA, last_uA, *x):
        def _bound_oneside(A, start, end, axes, steps):
            if A is None:
                return None
            if isinstance(A, torch.Tensor):
                # Reuse the batch and spec dimension of A, and replace other shapes with input.
                A_shape = A.shape[:2] + self.input_shape[1:]
                new_A = torch.zeros(size=A_shape, device=A.device, requires_grad=A.requires_grad)
                # Fill part of the new_A based on start, end, axes and steps.
                # Skip the spec dimension at the front (axes + 1).
                dim = axes if axes < 0 else axes + 1
                indices = torch.arange(start, end, device=A.device)
                new_A = torch.index_copy(new_A, dim=dim, index=indices, source=A)
            elif isinstance(A, Patches):
                assert A.unstable_idx is None
                assert len(self.input_shape) == 4 and axes == 1, "Slice is only supported on channel dimension."
                patches = A.patches
                # patches shape is [out_c, batch, out_h, out_w, in_c, patch_h, patch_w].
                new_patches_shape = patches.shape[:4] + (self.input_shape[1], ) + patches.shape[-2:]
                new_patches = torch.zeros(size=new_patches_shape, device=patches.device, requires_grad=patches.requires_grad)
                indices = torch.arange(start, end, device=patches.device)
                new_patches = torch.index_copy(new_patches, dim=-3, index=indices, source=patches)
                # Only the in_c dimension is changed.
                new_A = A.create_similar(new_patches)
            else:
                raise ValueError(f'Unsupport A type {type(A)}')
            return new_A

        start, end, axes = x[1].value.item(), x[2].value.item(), x[3].value.item()
        steps = x[4].value.item() if len(x) == 5 else 1  # If step is not specified, it is 1.
        # Other step size untested, do not enable for now.
        assert steps == 1 and axes == int(axes) and start == int(start) and end == int(end)
        start, end = self._fixup_params(self.input_shape, start, end, axes, steps)
        # Find the original shape of A.
        lA = _bound_oneside(last_lA, start, end, axes, steps)
        uA = _bound_oneside(last_uA, start, end, axes, steps)
        return [(lA, uA), (None, None), (None, None), (None, None), (None, None)], 0, 0

    def bound_forward(self, dim_in, *inputs):
        assert len(inputs) == 5 or len(inputs) == 4
        start = inputs[1].lb.item()
        end = inputs[2].lb.item()
        axis = self.make_axis_non_negative(inputs[3].lb.item())
        assert axis > 0, "Slicing along the batch dimension is not supported yet"
        steps = inputs[4].lb.item() if len(inputs) == 5 else 1  # If step is not specified, it is 1.
        assert steps in [1, -1]
        x = inputs[0]
        shape = x.lb.shape
        start, end = self._fixup_params(shape, start, end, axis, steps)
        lw = torch.narrow(x.lw, dim=axis+1, start=start, length=end - start)
        uw = torch.narrow(x.uw, dim=axis+1, start=start, length=end - start)
        lb = torch.narrow(x.lb, dim=axis, start=start, length=end - start)
        ub = torch.narrow(x.ub, dim=axis, start=start, length=end - start)
        if steps == -1:
            lw = torch.flip(lw, dims=tuple(axis+1))
            uw = torch.flip(uw, dims=tuple(axis+1))
            lb = torch.flip(lb, dims=tuple(axis))
            ub = torch.flip(ub, dims=tuple(axis))
        return LinearBound(lw, lb, uw, ub)


class BoundExpand(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x, y):
        y = y.clone()
        assert y.ndim == 1
        n, m = x.ndim, y.shape[0]
        assert n <= m
        for i in range(n):
            if y[m - n + i] == 1:
                y[m - n + i] = x.shape[i]
            else:
                assert x.shape[i] == 1 or x.shape[i] == y[m - n + i]
        return x.expand(*list(y))

    def infer_batch_dim(self, batch_size, *x):
        # FIXME should avoid referring to batch_size
        if self.forward_value.shape[0] != batch_size:
            return -1
        else:
            raise NotImplementedError('forward_value shape {}'.format(self.forward_value.shape))


class BoundSplit(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr['axis']
        self.split = attr['split']
        self.use_default_ibp = True

    def forward(self, x):
        if self.axis == -1:
            self.axis = len(x.shape) - 1
        return torch.split(x, self.split, dim=self.axis)[self.output_index]

    def bound_backward(self, last_lA, last_uA, x):
        assert self.axis > 0
        pre = sum(self.split[:self.output_index])
        suc = sum(self.split[(self.output_index + 1):])

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            A = []
            if pre > 0:
                A.append(torch.zeros(
                    *last_A.shape[:(self.axis + 1)], pre, *last_A.shape[(self.axis + 2):],
                    device=last_A.device))
            A.append(last_A)
            if suc > 0:
                A.append(torch.zeros(
                    *last_A.shape[:(self.axis + 1)], suc, *last_A.shape[(self.axis + 2):],
                    device=last_A.device))
            return torch.cat(A, dim=self.axis + 1)

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        assert (self.axis > 0 and self.from_input)
        lw = torch.split(x.lw, self.split, dim=self.axis + 1)[self.output_index]
        uw = torch.split(x.uw, self.split, dim=self.axis + 1)[self.output_index]
        lb = torch.split(x.lb, self.split, dim=self.axis)[self.output_index]
        ub = torch.split(x.ub, self.split, dim=self.axis)[self.output_index]
        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] != self.axis
        return x[0]
