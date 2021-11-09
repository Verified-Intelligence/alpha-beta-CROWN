""" Reduce operators"""
from .base import *


class BoundReduceMax(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axes']
        # for torch.max, `dim` must be an int
        if isinstance(self.axis, list):
            assert len(self.axis) == 1
            self.axis = self.axis[0]
        self.keepdim = bool(attr['keepdims']) if 'keepdims' in attr else True
        self.use_default_ibp = True      

        """Assume that the indexes with the maximum values are not perturbed. 
        This generally doesn't hold true, but can still be used for the input shift 
        in Softmax of Transformers."""   
        self.fixed_max_index = options.get('fixed_reducemax_index', False)

    @Bound.save_io_shape
    def forward(self, x):
        if self.axis < 0:
            self.axis += len(self.input_shape)
        assert self.axis > 0
        res = torch.max(x, dim=self.axis, keepdim=self.keepdim)
        self.indices = res.indices
        return res.values

    def infer_batch_dim(self, batch_size, *x):
        assert x[0] != self.axis
        return x[0]

    def bound_backward(self, last_lA, last_uA, x):
        if self.fixed_max_index:	
            def _bound_oneside(last_A):	
                if last_A is None:	
                    return None	
                indices = self.indices.unsqueeze(0)	
                if not self.keepdim:	
                    assert (self.from_input)	
                    last_A = last_A.unsqueeze(self.axis + 1)	
                    indices = indices.unsqueeze(self.axis + 1)	
                shape = list(last_A.shape)	
                shape[self.axis + 1] *= self.input_shape[self.axis]	
                A = torch.zeros(shape, device=last_A.device)	
                A.scatter_(dim=self.axis + 1, index=indices, src=last_A)	
                return A	

            return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0	
        else:
            raise NotImplementedError('`bound_backward` for BoundReduceMax with perturbed maximum indexes is not implemented.')


class BoundReduceMean(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axes']
        self.keepdim = bool(attr['keepdims']) if 'keepdims' in attr else True
        self.use_default_ibp = True        

    @Bound.save_io_shape
    def forward(self, x):
        return torch.mean(x, dim=self.axis, keepdim=self.keepdim)

    def bound_backward(self, last_lA, last_uA, x):
        for i in range(len(self.axis)):
            if self.axis[i] < 0:
                self.axis[i] = len(self.input_shape) + self.axis[i]
                assert self.axis[i] > 0

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            if not self.keepdim:
                assert (self.from_input)
                for axis in self.axis:
                    if axis > 0:
                        last_A = last_A.unsqueeze(axis + 1)
            for axis in self.axis:
                shape = list(last_A.shape)
                size_axis = self.input_shape[axis]
                shape[axis + 1] *= size_axis 
                last_A = last_A.expand(*shape) / size_axis
            return last_A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        assert (self.keepdim)
        assert (len(self.axis) == 1)
        axis = self.axis[0]
        if axis < 0:
            axis = len(self.input_shape) + axis
        assert (axis > 0)
        size = self.input_shape[axis]
        lw = x.lw.sum(dim=axis + 1, keepdim=True) / size
        lb = x.lb.sum(dim=axis, keepdim=True) / size
        uw = x.uw.sum(dim=axis + 1, keepdim=True) / size
        ub = x.ub.sum(dim=axis, keepdim=True) / size
        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        if x[0] in self.axis:
            assert not self.perturbed
            return -1
        return x[0]

class BoundReduceSum(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axes'] if 'axes' in attr else None
        self.keepdim = bool(attr['keepdims'])
        self.use_default_ibp = True        

    @Bound.save_io_shape
    def forward(self, x):
        if self.axis is not None:
            return torch.sum(x, dim=self.axis, keepdim=self.keepdim)
        else:
            return torch.sum(x)
            
    def bound_backward(self, last_lA, last_uA, x):
        for i in range(len(self.axis)):
            if self.axis[i] < 0:
                self.axis[i] = len(self.input_shape) + self.axis[i]
                assert self.axis[i] > 0

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            if not self.keepdim:
                assert (self.from_input)
                for axis in self.axis:
                    if axis > 0:
                        last_A = last_A.unsqueeze(axis + 1)
            for axis in self.axis:
                shape = list(last_A.shape)
                shape[axis + 1] *= self.input_shape[axis]
                last_A = last_A.expand(*shape)
            return last_A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA))], 0, 0

    def bound_forward(self, dim_in, x):
        assert self.keepdim
        assert len(self.axis) == 1
        axis = self.axis[0]
        if axis < 0:
            axis = len(self.input_shape) + axis
        assert (axis > 0)
        lw, lb = x.lw.sum(dim=axis + 1, keepdim=True), x.lb.sum(dim=axis, keepdim=True)
        uw, ub = x.uw.sum(dim=axis + 1, keepdim=True), x.ub.sum(dim=axis, keepdim=True)
        return LinearBound(lw, lb, uw, ub)

    def infer_batch_dim(self, batch_size, *x):
        assert not x[0] in self.axis
        return x[0]