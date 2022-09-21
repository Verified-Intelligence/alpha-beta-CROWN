from torch.nn import DataParallel
from .perturbations import *
from .bounded_tensor import BoundedTensor
from itertools import chain

class BoundDataParallel(DataParallel):
    # https://github.com/huanzhang12/CROWN-IBP/blob/master/bound_layers.py
    # This is a customized DataParallel class for our project
    def __init__(self, *inputs, **kwargs):
        super(BoundDataParallel, self).__init__(*inputs, **kwargs)
        self._replicas = None

    # Overide the forward method
    def forward(self, *inputs, **kwargs):
        disable_multi_gpu = False  # forward by single GPU
        no_replicas = False  # forward by multi GPUs but without replicate
        if "disable_multi_gpu" in kwargs:
            disable_multi_gpu = kwargs["disable_multi_gpu"]
            kwargs.pop("disable_multi_gpu")

        if "no_replicas" in kwargs:
            no_replicas = kwargs["no_replicas"]
            kwargs.pop("no_replicas")

        if not self.device_ids or disable_multi_gpu:
            if kwargs.pop("get_property", False):
                return self.get_property(self, *inputs, **kwargs)
            return self.module(*inputs, **kwargs)

        if kwargs.pop("get_property", False):
            if self._replicas is None:
                assert 0, 'please call IBP/CROWN before get_property'
            if len(self.device_ids) == 1:
                return self.get_property(self.module, **kwargs)
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            kwargs = list(kwargs)
            for i in range(len(kwargs)):
                kwargs[i]['model'] = self._replicas[i]
            outputs = self.parallel_apply([self.get_property] * len(kwargs), inputs, kwargs)
            return self.gather(outputs, self.output_device)

        # Only replicate during forward/IBP propagation. Not during interval bounds
        # and CROWN-IBP bounds, since weights have not been updated. This saves 2/3
        # of communication cost.
        if not no_replicas:
            if self._replicas is None:  # first time
                self._replicas = self.replicate(self.module, self.device_ids)
            elif kwargs.get("method_opt", "forward") == "forward":
                self._replicas = self.replicate(self.module, self.device_ids)
            elif kwargs.get("x") is not None and kwargs.get("IBP") is True:  #
                self._replicas = self.replicate(self.module, self.device_ids)
            # Update the input nodes to the ones within each replica respectively
            for bounded_module in self._replicas:
                for node in bounded_module._modules.values():
                    node.inputs = [bounded_module[name] for name in node.input_name]

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        # TODO: can be done in parallel, only support same ptb for all inputs per forward/IBP propagation
        if len(inputs) > 0 and hasattr(inputs[0], 'ptb') and inputs[0].ptb is not None:
            # compute bounds without x
            # inputs_scatter is a normal tensor, we need to assign ptb to it if inputs is a BoundedTensor
            inputs_scatter, kwargs = self.scatter((inputs, inputs[0].ptb.x_L, inputs[0].ptb.x_U), kwargs,
                                                  self.device_ids)
            # inputs_scatter = inputs_scatter[0]
            bounded_inputs = []
            for input_s in inputs_scatter:  # GPU numbers
                # FIXME other perturbations are not supported yet
                assert isinstance(inputs[0].ptb, PerturbationLpNorm)
                ptb = PerturbationLpNorm(norm=inputs[0].ptb.norm, eps=inputs[0].ptb.eps, x_L=input_s[1], x_U=input_s[2])
                input_s = list(input_s[0])
                input_s[0] = BoundedTensor(input_s[0], ptb)
                input_s = tuple(input_s)
                bounded_inputs.append(input_s)

            # bounded_inputs = tuple(bounded_inputs)
        elif kwargs.get("x") is not None and hasattr(kwargs.get("x")[0], 'ptb') and kwargs.get("x")[0].ptb is not None:
            # compute bounds with x
            # kwargs['x'] is a normal tensor, we need to assign ptb to it
            x = kwargs.get("x")[0]
            bounded_inputs = []
            inputs_scatter, kwargs = self.scatter((inputs, x.ptb.x_L, x.ptb.x_U), kwargs, self.device_ids)
            for input_s, kw_s in zip(inputs_scatter, kwargs):  # GPU numbers
                # FIXME other perturbations are not supported yet
                assert isinstance(x.ptb, PerturbationLpNorm)
                ptb = PerturbationLpNorm(norm=x.ptb.norm, eps=x.ptb.eps, x_L=input_s[1], x_U=input_s[2])
                kw_s['x'] = list(kw_s['x'])
                kw_s['x'][0] = BoundedTensor(kw_s['x'][0], ptb)
                kw_s['x'] = (kw_s['x'])
                bounded_inputs.append(tuple(input_s[0], ))
        else:
            # normal forward
            inputs_scatter, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            bounded_inputs = inputs_scatter

        if len(self.device_ids) == 1:
            return self.module(*bounded_inputs[0], **kwargs[0])
        outputs = self.parallel_apply(self._replicas[:len(bounded_inputs)], bounded_inputs, kwargs)
        return self.gather(outputs, self.output_device)

    @staticmethod
    def get_property(model, node_class=None, att_name=None, node_name=None):
        if node_name:
            # Find node by name
            # FIXME If we use `model.named_modules()`, the nodes have the
            # `BoundedModule` type rather than bound nodes.
            for node in model._modules.values():
                if node.name == node_name:
                    return getattr(node, att_name)    
        else:
            # Find node by class
            for _, node in model.named_modules():
                # Find the Exp neuron in computational graph
                if isinstance(node, node_class):
                    return getattr(node, att_name)
             
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # add 'module.' here before each keys in self.module.state_dict() if needed
        return self.module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        return self.module._named_members(get_members_fn, prefix, recurse)

