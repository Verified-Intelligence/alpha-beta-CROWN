#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
## Copyright (C) 2021-2022, Huan Zhang <huan@huan-zhang.com>           ##
##                     Kaidi Xu, Zhouxing Shi, Shiqi Wang              ##
##                     Linyi Li, Jinqi (Kathryn) Chen                  ##
##                     Zhuolin Yang, Yihan Wang                        ##
##                                                                     ##
##      See CONTRIBUTORS for author contacts and affiliations.         ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import torch

class TensorStorage(object):
    """
    Fast managed dynamic sized tensor storage.
    """
    def __init__(self, full_shape, initial_size=1024, switching_size=65536,
                 dtype=None, device='cpu', concat_dim=0):
        """
        full_shape is the tensor shape you want to store using this object, including the "batch" dimension.
        dtype is tensor type (default float32).
        initial_size is the initial size of the storage. It will go up exponentially until reaching a batch size of "switching_size".
        switching_size is the point where exponential growth changes to linear growth.
        device is storage device, and for CPU memory it will be pinned.
        concat_dim is the axis of batch dimension (default is 0).
        """
        if isinstance(full_shape, torch.Tensor):
            data = full_shape
            full_shape = data.shape
        else:
            data = None
        if dtype is None:
            dtype = torch.get_default_dtype()
        self.shape = list(full_shape)  # Full shape, with batch size that will become dynamic.
        self.dtype = dtype
        self.device = device
        self.concat_dim = concat_dim
        self.num_used = 0
        self.switching_size = switching_size
        self._storage = self._allocate(initial_size)

        if data is not None:
            self.append(data)

    def _allocate(self, new_size):
        allocate_shape = self.shape.copy()
        allocate_shape[self.concat_dim] = new_size
        if self.device == 'cpu' and torch.cuda.is_available():
            # Pin CPU memory if cuda is available.
            return torch.empty(allocate_shape, dtype=self.dtype, device=self.device, pin_memory=True)
        else:
            return torch.empty(allocate_shape, dtype=self.dtype, device=self.device)

    def _get_new_size(self, request_size):
        """Compute new size of storage given the current request."""
        if self._storage.size(self.concat_dim) < self.switching_size:
            # Tensor is small. Exponential growth.
            return max(self._storage.size(self.concat_dim) * 2, self.num_used + request_size)
        else:
            # Tensor is big. Linear growth.
            return self._storage.size(self.concat_dim) + request_size * 32

    @torch.no_grad()
    def append(self, appended_tensor):
        """Append a new tensor to the storage object."""
        if self.num_used + appended_tensor.size(self.concat_dim) > self._storage.size(self.concat_dim):
            # Reallocate a new tensor, copying the existing contents over.
            new_size = self._get_new_size(appended_tensor.size(self.concat_dim))
            new_tensor = self._allocate(new_size)
            new_tensor.narrow(dim=self.concat_dim, start=0, length=self.num_used).copy_(
                self._storage.narrow(dim=self.concat_dim, start=0, length=self.num_used))
            # And then remove the old storage object.
            del self._storage
            self._storage = new_tensor
        self._storage.narrow(self.concat_dim, self.num_used, appended_tensor.size(self.concat_dim)).copy_(appended_tensor)
        self.num_used += appended_tensor.size(self.concat_dim)
        return self

    @torch.no_grad()
    def pop(self, size):
        """Remove tensors with 'size' at the end of the storage."""
        size = max(min(size, self.num_used), 0)
        ret = self._storage.narrow(self.concat_dim, self.num_used - size, size)
        self.num_used -= size
        return ret

    def tensor(self):
        return self._storage.narrow(self.concat_dim, 0, self.num_used)

    def __getattr__(self, attr):
        """Proxy all tensor attributes."""
        return getattr(self.tensor(), attr)

    def __getitem__(self, idx):
        return self.tensor()[idx]

    def __len__(self):
        return self.num_used

    def __sub__(self, o):
        return self._storage.narrow(self.concat_dim, 0, self.num_used) - o._storage.narrow(o.concat_dim, 0, o.num_used)


def _test():
    for concat_dim in [0, 1, 2]:
        shape = [1,1,1]
        shape[concat_dim] = -1 # does no matter.
        zero_shape = shape.copy()
        zero_shape[concat_dim] = 0
        make_tensor = lambda x: torch.arange(1,x+1, dtype=torch.float32).view(*shape)
        s = TensorStorage(full_shape=shape, initial_size=16, switching_size=65536, concat_dim=concat_dim)
        s.append(make_tensor(1))
        assert s.sum() == 1, print(s)
        s.append(make_tensor(3))
        assert s.sum() == 1 + 6, print(s)
        s.append(make_tensor(5))
        assert s.sum() == 1 + 6 + 15, print(s)
        t = s.pop(5)
        assert torch.allclose(t.squeeze(), torch.tensor([1,2,3,4,5], dtype=torch.float32)), print(t)
        t = s.pop(0)
        assert t.shape == torch.Size(zero_shape)
        t = s.pop(-1)
        assert t.shape == torch.Size(zero_shape)
        s.append(make_tensor(100))
        assert s.sum() == 1 + 6 + 50*101
        t = s.pop(5)
        assert torch.allclose(t.squeeze(), torch.tensor([96,97,98,99,100], dtype=torch.float32)), print(t)
        assert s.size(concat_dim) == 99, print(s.size())
        assert s._storage.size(concat_dim) == 104, print(s._storage.size())
        s.append(make_tensor(10))
        assert s.size(concat_dim) == 109, print(s.size())
        assert s._storage.size(concat_dim) == 208, print(s._storage.size())
        s.append(make_tensor(32768))
        assert s.size(concat_dim) == 32877, print(s.size())
        assert s._storage.size(concat_dim) == 32877, print(s._storage.size())
        s.pop(1)
        s.append(make_tensor(2))
        assert s.size(concat_dim) == 32878, print(s.size())
        assert s._storage.size(concat_dim) == 32877*2, print(s._storage.size())
        s.append(make_tensor(32800))
        s.append(make_tensor(100))
        assert s._storage.size(concat_dim) == 32877*2+100*32, print(s._storage.size())
        s.pop(100000)
        assert s._storage.size(concat_dim) == 32877*2+100*32, print(s._storage.size())
        assert s.size(concat_dim) == 0, print(s.size())
        t = s.pop(1)
        assert t.shape == torch.Size(zero_shape)
        t = s.pop(0)
        assert t.shape == torch.Size(zero_shape)
        t = s.pop(-1)
        assert t.shape == torch.Size(zero_shape)

if __name__ == "__main__":
    _test()
