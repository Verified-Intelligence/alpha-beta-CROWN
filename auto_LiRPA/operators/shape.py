""" Shape operators """
from .base import *

class BoundReshape(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
    def forward(self, x, shape):
        shape = list(shape)
        for i in range(len(shape)):
            if shape[i] == -1:
                shape[i] = prod(x.shape) // int(prod(shape[:i]) * prod(shape[(i + 1):]))
        self.shape = shape
        return x.reshape(shape)

    def bound_backward(self, last_lA, last_uA, x, shape):
        def _bound_oneside(A):
            if A is None:
                return None
            # A shape is (spec, batch, *node_shape)
            return A.reshape(A.shape[0], A.shape[1], *self.input_shape[1:])

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, x, shape):
        batch_size = x.lw.shape[0]
        lw = x.lw.reshape(batch_size, dim_in, *self.shape[1:])
        uw = x.uw.reshape(batch_size, dim_in, *self.shape[1:])
        lb = x.lb.reshape(batch_size, *self.shape[1:])
        ub = x.ub.reshape(batch_size, *self.shape[1:])
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        if Interval.use_relative_bounds(*v):
            return Interval(
                None, None,
                v[0].nominal.reshape(self.shape),
                v[0].lower_offset.reshape(self.shape),
                v[0].upper_offset.reshape(self.shape),
                ptb=v[0].ptb
            )
        return Interval.make_interval(v[0][0].reshape(*v[1][0]), v[0][1].reshape(*v[1][0]), v[0])

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        elif self.input_shape[x[0]] == self.shape[x[0]]:
            return x[0]
        raise NotImplementedError('input shape {}, new shape {}, input batch dim {}'.format(
            self.input_shape, self.shape, x[0]
        ))


class BoundUnsqueeze(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axes = attr['axes']
        assert (len(self.axes) == 1)
        self.axes = self.axes[0]
        self.use_default_ibp = True

    @Bound.save_io_shape
    def forward(self, x):
        if self.axes < 0:
            self.axes = len(self.input_shape) + self.axes + 1
        return x.unsqueeze(self.axes) 

    def bound_backward(self, last_lA, last_uA, x):
        if self.axes == 0:
            return last_lA, 0, last_uA, 0
        else:
            return [(last_lA.squeeze(self.axes + 1) if last_lA is not None else None,
                     last_uA.squeeze(self.axes + 1) if last_uA is not None else None)], 0, 0

    def bound_forward(self, dim_in, x):
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
    

class BoundSqueeze(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axes = attr['axes']
        assert (len(self.axes) == 1)
        self.axes = self.axes[0]
        self.use_default_ibp = True

    @Bound.save_io_shape
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
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.use_default_ibp = True
        self.axis = attr['axis']

    @Bound.save_io_shape
    def forward(self, x):
        return torch.flatten(x, self.axis)

    def bound_backward(self, last_lA, last_uA, x):
        def _bound_oneside(A):
            if A is None:
                return None
            return A.reshape(A.shape[0], A.shape[1], *self.input_shape[1:])

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0


class BoundConcat(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axis']
        self.IBP_rets = None

    @Bound.save_io_shape
    def forward(self, *x):  # x is a list of tensors
        x = [(item if isinstance(item, torch.Tensor) else torch.tensor(item)) for item in x]
        self.input_size = [item.shape[self.axis] for item in x]
        if self.axis < 0:
            self.axis = x[0].ndim + self.axis
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

        if Interval.use_relative_bounds(*v):
            assert all_inf # Only LINF supported for now
            return Interval(
                None, None,
                self.forward(*[_v.nominal for _v in v]),
                self.forward(*[_v.lower_offset for _v in v]),
                self.forward(*[_v.upper_offset for _v in v]),
            )

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
        if self.axis < 0:
            self.axis = len(self.output_shape) + self.axis
        assert (self.axis > 0)

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            return torch.split(last_A, self.input_size, dim=self.axis + 1)

        uA = _bound_oneside(last_uA)
        lA = _bound_oneside(last_lA)
        if uA is None:
            return [(lA[i] if lA is not None else None, None) for i in range(len(lA))], 0, 0
        if lA is None:
            return [(None, uA[i] if uA is not None else None) for i in range(len(uA))], 0, 0
        return [(lA[i], uA[i]) for i in range(len(lA))], 0, 0

    def bound_forward(self, dim_in, *x):
        if self.axis < 0:
            self.axis = x[0].lb.ndim + self.axis
        assert (self.axis == 0 and not self.from_input or self.from_input)
        lw = torch.cat([item.lw for item in x], dim=self.axis + 1)
        lb = torch.cat([item.lb for item in x], dim=self.axis)
        uw = torch.cat([item.uw for item in x], dim=self.axis + 1)
        ub = torch.cat([item.ub for item in x], dim=self.axis)
        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        assert np.min(x) == np.max(x)
        assert x[0] != self.axis
        return x[0]

class BoundShape(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.use_default_ibp = True        

    @staticmethod
    def shape(x):
        return x.shape if isinstance(x, torch.Tensor) else torch.tensor(x).shape

    def forward(self, x):
        self.from_input = False
        return BoundShape.shape(x)

    def bound_forward(self, dim_in, x):
        return self.forward_value

    def infer_batch_dim(self, batch_size, *x):
        return -1

    def interval_propagate(self, *v):
        if Interval.use_relative_bounds(*v):
            shape = self.forward(v[0].nominal)
            if not isinstance(shape, torch.Tensor):
                shape = torch.tensor(shape, device=self.device)
            return Interval(
                None, None,
                shape, torch.zeros_like(shape), torch.zeros_like(shape)
            )

        return super().interval_propagate(*v)


class BoundGather(Bound):
    def __init__(self, input_name, name, ori_name, attr, x, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, x, output_index, options, device)
        self.axis = attr['axis'] if 'axis' in attr else 0
        self.nonlinear = False  # input shape required

    @Bound.save_io_shape
    def forward(self, x, indices):
        self.indices = indices
        x = x.to(self.indices.device)  # BoundShape.shape() will return value on cpu only
        if indices.ndim == 0:
            # `index_select` requires `indices` to be a 1-D tensor
            return torch.index_select(x, dim=self.axis, index=indices).squeeze(self.axis)
        elif self.axis == 0:
            return torch.index_select(x, dim=self.axis, index=indices.reshape(-1)) \
                .reshape(*indices.shape, x.shape[-1])
        raise ValueError(
            'Unsupported shapes in Gather: data {}, indices {}, axis {}'.format(x.shape, indices.shape, self.axis))

    def bound_backward(self, last_lA, last_uA, x, indices):
        assert self.from_input
        assert self.indices.ndim == 0  # TODO

        def _bound_oneside(A):
            if A is None:
                return None
            assert (self.indices.ndim == 0)

            A = A.unsqueeze(self.axis + 1)
            idx = int(self.indices)
            tensors = []
            if idx > 0:
                shape_pre = list(A.shape)
                shape_pre[self.axis + 1] *= idx
                tensors.append(torch.zeros(shape_pre, device=self.device))
            tensors.append(A)
            if self.input_shape[self.axis] - idx - 1 > 0:
                shape_next = list(A.shape)
                shape_next[self.axis + 1] *= self.input_shape[self.axis] - idx - 1
                tensors.append(torch.zeros(shape_next, device=self.device))
            return torch.cat(tensors, dim=self.axis + 1)

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

        if Interval.use_relative_bounds(*v):
            return Interval(
                None, None,
                self.forward(v[0].nominal, v[1].nominal),
                self.forward(v[0].lower_offset, v[1].nominal),
                self.forward(v[0].upper_offset, v[1].nominal)
            )
        
        return self.forward(v[0][0], v[1][0]), self.forward(v[0][1], v[1][0])

    def infer_batch_dim(self, batch_size, *x):
        if x[0] != -1:
            assert self.axis != x[0]
            return x[0]
        else:
            return x[1]

    def infer_batch_dim(self, batch_size, *x):
        if x[0] != -1:
            assert self.axis != x[0]
            return x[0]
        else:
            return x[1]


class BoundGatherElements(Bound):
    def __init__(self, input_name, name, ori_name, attr, input, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, input, output_index, options, device)
        self.axis = attr['axis']

    @Bound.save_io_shape
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

        if Interval.use_relative_bounds(*v):
            return Interval(
                None, None,
                self.forward(v[0].nominal, v[1].nominal),
                self.forward(v[0].lower_offset, v[1].nominal),
                self.forward(v[0].upper_offset, v[1].nominal)
            )

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
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.perm = attr['perm']
        self.perm_inv_inc_one = [-1] * (len(self.perm) + 1)
        self.perm_inv_inc_one[0] = 0
        for i in range(len(self.perm)):
            self.perm_inv_inc_one[self.perm[i] + 1] = i + 1
        self.use_default_ibp = True            

    @Bound.save_io_shape
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

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        else:
            return self.perm.index(x[0])


class BoundSlice(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.start = attr["starts"][0] if "starts" in attr else None
        self.end = attr["ends"][0] if "ends" in attr else None
        self.axes = attr["axes"][0] if "axes" in attr else None
        self.use_default_ibp = False

    # Older Pytorch version only passes steps as input.
    @Bound.save_io_shape
    def forward(self, x, start=None, end=None, axes=None, steps=1):
        start = self.start if start is None else start
        end = self.end if end is None else end
        axes = self.axes if axes is None else axes
        assert (steps == 1 or steps == -1) and axes == int(axes) and start == int(start) and end == int(end)
        shape = x.shape if isinstance(x, torch.Tensor) else [len(x)]
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
        final = torch.narrow(x, dim=int(axes), start=int(start), length=int(end - start))
        if steps == -1:
            final = torch.flip(final, dims=tuple(axes))
        return final
    
    def interval_propagate(self, *v):       
        lb = tuple(map(lambda x:x[0],v))
        ub = tuple(map(lambda x:x[1],v))
        return Interval.make_interval(self.forward(*lb), self.forward(*ub))

    def infer_batch_dim(self, batch_size, *x):
        if x[0] == -1:
            return -1
        else:
            assert self.axes != x[0]
            return x[0]


class BoundExpand(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

    @Bound.save_io_shape
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
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axis']
        self.split = attr['split']
        self.use_default_ibp = True

    @Bound.save_io_shape
    def forward(self, x):
        return torch.split(x, self.split, dim=self.axis)[self.output_index]

    def bound_backward(self, last_lA, last_uA, x):
        assert (self.axis > 0)
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
