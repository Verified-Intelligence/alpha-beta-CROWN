from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class View(nn.Module):
    '''
    This is necessary in order to reshape "flat activations" such as used by
    nn.Linear with those that comes from MaxPooling
    '''
    def __init__(self, out_shape):
        super(View, self).__init__()
        self.out_shape = out_shape

    def forward(self, inp):
        # We make the assumption that all the elements in the tuple have
        # the same batchsize and need to be brought to the same size

        # We assume that the first dimension is the batch size
        batch_size = inp.size(0)
        out_size = (batch_size, ) + self.out_shape
        out = inp.view(out_size)
        return out
