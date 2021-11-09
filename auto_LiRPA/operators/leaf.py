""" Leaf nodes (indepedent nodes in the auto_LiRPA paper),
including input, parameter, buffer, etc."""
from .base import *


class BoundInput(Bound):
    def __init__(self, input_name, name, ori_name, value, perturbation=None):
        super().__init__(input_name, name, ori_name)
        self.value = value
        self.perturbation = perturbation
        self.from_input = True

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        # Update perturbed property based on the perturbation set.
        if key == "perturbation":
            if self.perturbation is not None:
                self.perturbed = True
            else:
                self.perturbed = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        local_name_params = chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            if len(prefix.split('.')) == 2:
                key = prefix + name
            else:
                # change key to prefix + self.ori_name when calling load_state_dict()
                key = '.'.join(prefix.split('.')[:-2]) + '.' + self.ori_name
            if key in state_dict:
                input_param = state_dict[key]

                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if param.ndim == 0 and input_param.ndim == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue

                try:
                    with torch.no_grad():
                        param.copy_(input_param)
                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}, '
                                      'an exception occured : {}.'
                                      .format(key, param.size(), input_param.size(), ex.args))
            elif strict:
                missing_keys.append(key)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                if len(prefix.split('.')) == 2:
                    destination[self.ori_name] = param if keep_vars else param.detach()
                else:
                    # change parameters' name to self.ori_name when calling state_dict()
                    destination[
                        '.'.join(prefix.split('.')[:-2]) + '.' + self.ori_name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None:
                if len(prefix.split('.')) == 2:
                    destination[self.ori_name] = buf if keep_vars else buf.detach()
                else:
                    # change buffers' name to self.ori_name when calling state_dict()
                    destination[
                        '.'.join(prefix.split('.')[:-2]) + '.' + self.ori_name] = buf if keep_vars else buf.detach()

    @Bound.save_io_shape
    def forward(self):
        return self.value

    def bound_forward(self, dim_in):
        assert 0

    def bound_backward(self, last_lA, last_uA):
        raise ValueError('{} is a BoundInput node and should not be visited here'.format(
            self.name))

    def interval_propagate(self, *v):
        raise ValueError('{} is a BoundInput node and should not be visited here'.format(
            self.name))

    def infer_batch_dim(self, batch_size, *x):
        shape = self.forward_value.shape
        for i in range(len(shape)):
            if shape[i] == batch_size:
                return i
        return -1


class BoundParams(BoundInput):
    def __init__(self, input_name, name, ori_name, value, perturbation=None):
        super().__init__(input_name, name, ori_name, None, perturbation)
        self.register_parameter('param', value)
        self.from_input = False
        self.initializing = False

    """Override register_parameter() hook to register only needed parameters."""

    def register_parameter(self, name, param):
        if name == 'param':
            # self._parameters[name] = param  # cannot contain '.' in name, it will cause error when loading state_dict
            return super().register_parameter(name, param)
        else:
            # Just register it as a normal property of class.
            object.__setattr__(self, name, param)

    def init(self, initializing=False):
        self.initializing = initializing

    @Bound.save_io_shape
    def forward(self):
        if self.initializing:
            return self.param_init
        else:
            return self.param

    def infer_batch_dim(self, batch_size, *x):
        return -1


class BoundBuffers(BoundInput):
    def __init__(self, input_name, name, ori_name, value, perturbation=None):
        super().__init__(input_name, name, ori_name, None, perturbation)
        self.register_buffer('buffer', value.clone().detach())

    @Bound.save_io_shape
    def forward(self):
        return self.buffer