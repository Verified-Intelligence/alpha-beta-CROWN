import time
import os
import numpy as np
from collections import OrderedDict, deque, defaultdict

import torch
import torch.optim as optim
from torch.nn import DataParallel, Parameter, parameter

from auto_LiRPA.bound_op_map import bound_op_map
from auto_LiRPA.bound_ops import *
from auto_LiRPA.bounded_tensor import BoundedTensor, BoundedParameter
from auto_LiRPA.parse_graph import parse_module
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import *
from auto_LiRPA.adam_element_lr import AdamElementLR

import warnings

warnings.simplefilter("once")

Check_against_base_lp = False  # A debugging option, used for checking against LPs. Will be removed.
Check_against_base_lp_layer = '/21'  # Check for bounds in this layer ('/9', '/11', '/21')

class BoundedModule(nn.Module):
    """Bounded module with support for automatically computing bounds.

    Args:
        model (nn.Module): The original model to be wrapped by BoundedModule.

        global_input (tuple): A dummy input to the original model. The shape of 
        the dummy input should be consistent with the actual input to the model 
        except for the batch dimension.

        bound_opts (dict): Options for bounds. See 
        `Bound Options <bound_opts.html>`_.

        device (str or torch.device): Device of the bounded module. 
        If 'auto', the device will be automatically inferred from the device of 
        parameters in the original model or the dummy input.

        custom_ops (dict): A dictionary of custom operators. 
        The dictionary maps operator names to their corresponding bound classes 
        (subclasses of `Bound`).

    """
    def __init__(self, model, global_input, bound_opts={}, auto_batch_dim=True, device='auto',
                 verbose=False, custom_ops={}):
        super(BoundedModule, self).__init__()
        if isinstance(model, BoundedModule):
            for key in model.__dict__.keys():
                setattr(self, key, getattr(model, key))
            return
        self.verbose = verbose
        self.bound_opts = bound_opts
        self.custom_ops = custom_ops
        self.auto_batch_dim = auto_batch_dim
        if device == 'auto':
            try:
                self.device = next(model.parameters()).device
            except StopIteration:  # Model has no parameters. We use the device of input tensor.
                self.device = global_input.device
        else:
            self.device = device
        self.global_input = global_input
        self.ibp_relative = bound_opts.get('ibp_relative', False)   
        self.conv_mode = bound_opts.get("conv_mode", "patches")
        if auto_batch_dim:
            # logger.warning('Using automatic batch dimension inferring, which may not be correct')
            self.init_batch_size = -1

        state_dict_copy = copy.deepcopy(model.state_dict())
        object.__setattr__(self, 'ori_state_dict', state_dict_copy)
        model.to(self.device)
        self.final_shape = model(*unpack_inputs(global_input, device=self.device)).shape
        self.bound_opts.update({'final_shape': self.final_shape})
        self._convert(model, global_input)
        self._mark_perturbed_nodes()

        # set the default values here
        optimize_bound_args = {'ob_iteration': 20, 'ob_beta': False, 'ob_alpha': True, 'ob_alpha_share_slopes': False,
                               'ob_opt_coeffs': False, 'ob_opt_bias': False,
                               'ob_optimizer': "adam", 'ob_verbose': 0,
                               'ob_keep_best': True, 'ob_update_by_layer': True, 'ob_lr': 0.5,
                               'ob_lr_beta': 0.05, 'ob_init': True,
                               'ob_single_node_split': True, 'ob_lr_intermediate_beta': 0.1,
                               'ob_lr_coeffs': 0.01, 'ob_intermediate_beta': False, 'ob_intermediate_refinement_layers': [-1],
                               'ob_loss_reduction_func': reduction_sum, 
                               'ob_stop_criterion_func': lambda x: False,
                               'ob_input_grad': False,
                               'ob_lr_decay': 0.98 }
        # change by bound_opts
        optimize_bound_args.update(self.bound_opts.get('optimize_bound_args', {}))
        self.bound_opts.update({'optimize_bound_args': optimize_bound_args})

        self.next_split_hint = []  # Split hints, used in beta optimization.
        self.relus = []  # save relu layers for convenience
        for l in self._modules.values():
            if isinstance(l, BoundRelu):
                self.relus.append(l)
        self.optimizable_activations = []
        for l in self._modules.values():
            if isinstance(l, BoundOptimizableActivation):
                self.optimizable_activations.append(l)

        # Beta values for all intermediate bounds. Set to None (not used) by default.
        self.best_intermediate_betas = None
        # Initialization value for intermediate betas.
        self.init_intermediate_betas = None

    """Some operations are non-deterministic and deterministic mode will fail. So we temporary disable it."""
    def non_deter_wrapper(self, op, *args, **kwargs):
        if self.bound_opts.get('deterministic', False):
            torch.use_deterministic_algorithms(False)
        ret = op(*args, **kwargs)
        if self.bound_opts.get('deterministic', False):
            torch.use_deterministic_algorithms(True)
        return ret

    def non_deter_scatter_add(self, *args, **kwargs):
        return self.non_deter_wrapper(torch.scatter_add, *args, **kwargs)

    def non_deter_index_select(self, *args, **kwargs):
        return self.non_deter_wrapper(torch.index_select, *args, **kwargs)

    def set_bound_opts(self, new_opts):
        for k, v in new_opts.items():
            assert v is not dict, 'only support change optimize_bound_args'
            self.bound_opts[k].update(v)

    def __call__(self, *input, **kwargs):

        if "method_opt" in kwargs:
            opt = kwargs["method_opt"]
            kwargs.pop("method_opt")
        else:
            opt = "forward"
        for kwarg in [
            'disable_multi_gpu', 'no_replicas', 'get_property',
            'node_class', 'att_name']:
            if kwarg in kwargs:
                kwargs.pop(kwarg)
        if opt == "compute_bounds":
            return self.compute_bounds(**kwargs)
        else:
            return self.forward(*input, **kwargs)

    def register_parameter(self, name, param):
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter): parameter to be added to the module.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(torch.typename(param), name))
        elif param.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.".format(name))
        else:
            self._parameters[name] = param

    def load_state_dict(self, state_dict, strict=False):
        new_dict = OrderedDict()
        # translate name to ori_name
        for k, v in state_dict.items():
            if k in self.node_name_map:
                new_dict[self.node_name_map[k]] = v
        return super(BoundedModule, self).load_state_dict(new_dict, strict=strict)

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                # translate name to ori_name
                if name in self.node_name_map:
                    name = self.node_name_map[name]
                yield name, v

    def train(self, mode=True):
        super().train(mode)
        for node in self._modules.values():
            node.train(mode=mode)

    def eval(self):
        super().eval()
        for node in self._modules.values():
            node.eval()

    def forward(self, *x, final_node_name=None):
        self._set_input(*x)

        degree_in = {}
        queue = deque()
        for key in self._modules.keys():
            l = self._modules[key]
            degree_in[l.name] = len(l.input_name)
            if degree_in[l.name] == 0:
                queue.append(l)
        forward_values = {}

        final_output = None
        while len(queue) > 0:
            l = queue.popleft()

            inp = [forward_values[l_pre] for l_pre in l.input_name]
            for l_pre in l.input_name:
                l.from_input = l.from_input or self._modules[l_pre].from_input

            fv = l.forward(*inp)
            if isinstance(fv, torch.Size) or isinstance(fv, tuple):
                fv = torch.tensor(fv, device=self.device)
            object.__setattr__(l, 'forward_value', fv)
            # infer batch dimension
            if not hasattr(l, 'batch_dim'):
                inp_batch_dim = [self._modules[l_pre].batch_dim for l_pre in l.input_name]
                try:
                    l.batch_dim = l.infer_batch_dim(self.init_batch_size, *inp_batch_dim)
                    try:
                        logger.debug(
                            'Batch dimension of ({})[{}]: forward_value shape {}, infered {}, input batch dimensions {}'.format(
                                l, l.name, l.forward_value.shape, l.batch_dim, inp_batch_dim
                            ))
                    except:
                        pass
                except:
                    raise Exception(
                        'Fail to infer the batch dimension of ({})[{}]: forward_value shape {}, input batch dimensions {}'.format(
                            l, l.name, l.forward_value.shape, inp_batch_dim
                        ))

            # if isinstance(l.forward_value, torch.Tensor):
            #     l.default_shape = l.forward_value.shape
            forward_values[l.name] = l.forward_value
            logger.debug('Forward at {}[{}], forward_value shape {}'.format(l, l.name, fv.shape))

            # Unperturbed node but it is not a root node. Save forward_value to value.
            # (Can be used in forward bounds.)
            if not l.from_input and len(l.inputs) > 0:
                l.value = l.forward_value

            for l_next in l.output_name:
                degree_in[l_next] -= 1
                if degree_in[l_next] == 0:  # all inputs of this node have already set
                    queue.append(self._modules[l_next])

        if final_node_name:
            return forward_values[final_node_name]
        else:
            out = deque([forward_values[n] for n in self.output_name])

            def _fill_template(template):
                if template is None:
                    return out.popleft()
                elif isinstance(template, list) or isinstance(template, tuple):
                    res = []
                    for t in template:
                        res.append(_fill_template(t))
                    return tuple(res) if isinstance(template, tuple) else res
                elif isinstance(template, dict):
                    res = {}
                    for key in template:
                        res[key] = _fill_template(template[key])
                    return res
                else:
                    raise NotImplementedError

            return _fill_template(self.output_template)

    """Mark the graph nodes and determine which nodes need perturbation."""
    def _mark_perturbed_nodes(self):
        degree_in = {}
        queue = deque()
        # Initially the queue contains all "root" nodes.
        for key in self._modules.keys():
            l = self._modules[key]
            degree_in[l.name] = len(l.input_name)
            if degree_in[l.name] == 0:
                queue.append(l)  # in_degree ==0 -> root node

        while len(queue) > 0:
            l = queue.popleft()
            # Obtain all output node, and add the output nodes to the queue if all its input nodes have been visited.
            # the initial "perturbed" property is set in BoundInput or BoundParams object, depending on ptb.
            for name_next in l.output_name:
                node_next = self._modules[name_next]
                if isinstance(l, BoundShape):
                    # Some nodes like Shape, even connected, do not really propagate bounds.
                    # TODO: make this a property of node?
                    pass
                else:
                    # The next node is perturbed if it is already perturbed, or this node is perturbed.
                    node_next.perturbed = node_next.perturbed or l.perturbed
                degree_in[name_next] -= 1
                if degree_in[name_next] == 0:  # all inputs of this node have been visited, now put it in queue.
                    queue.append(node_next)
        return

    def _clear_and_set_new(self, new_interval):
        for l in self._modules.values():
            if hasattr(l, 'linear'):
                if isinstance(l.linear, tuple):
                    for item in l.linear:
                        del (item)
                delattr(l, 'linear')
            for attr in ['lower', 'upper', 'interval', 'forward_value']:
                if hasattr(l, attr):
                    delattr(l, attr)
            # Given an interval here to make IBP/CROWN start from this node
            if new_interval is not None and l.name in new_interval.keys():
                l.interval = tuple(new_interval[l.name][:2])
                l.lower = new_interval[l.name][0]
                l.upper = new_interval[l.name][1]
            # Mark all nodes as non-perturbed except for weights.
            if not hasattr(l, 'perturbation') or l.perturbation is None:
                l.perturbed = False

    def _set_input(self, *x, new_interval=None):
        self._clear_and_set_new(new_interval=new_interval)
        inputs_unpacked = unpack_inputs(x)
        for name, index in zip(self.input_name, self.input_index):
            node = self._modules[name]
            node.value = inputs_unpacked[index]
            if isinstance(node.value, (BoundedTensor, BoundedParameter)):
                node.perturbation = node.value.ptb
            else:
                node.perturbation = None
        # Mark all perturbed nodes.
        self._mark_perturbed_nodes()
        if self.init_batch_size == -1:
            # Automatic batch dimension inferring: get the batch size from 
            # the first dimension of the first input tensor.
            self.init_batch_size = inputs_unpacked[0].shape[0]

    def _get_node_input(self, nodesOP, nodesIn, node):
        ret = []
        ori_names = []
        for i in range(len(node.inputs)):
            found = False
            for op in nodesOP:
                if op.name == node.inputs[i]:
                    ret.append(op.bound_node)
                    break
            if len(ret) == i + 1:
                continue
            for io in nodesIn:
                if io.name == node.inputs[i]:
                    ret.append(io.bound_node)
                    ori_names.append(io.ori_name)
                    break
            if len(ret) <= i:
                raise ValueError('cannot find inputs of node: {}'.format(node.name))
        return ret, ori_names

    # move all tensors in the object to a specified device
    def _to(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, tuple):
            return tuple([self._to(item, device) for item in obj])
        elif isinstance(obj, list):
            return list([self._to(item, device) for item in obj])
        elif isinstance(obj, dict):
            res = {}
            for key in obj:
                res[key] = self._to(obj[key], device)
            return res
        else:
            raise NotImplementedError(type(obj))

    def _convert_nodes(self, model, global_input):
        global_input_cpu = self._to(global_input, 'cpu')
        model.train()
        model.to('cpu')
        nodesOP, nodesIn, nodesOut, template = parse_module(model, global_input_cpu)
        model.to(self.device)
        for i in range(0, len(nodesIn)):
            if nodesIn[i].param is not None:
                nodesIn[i] = nodesIn[i]._replace(param=nodesIn[i].param.to(self.device))
        global_input_unpacked = unpack_inputs(global_input)

        # Convert input nodes and parameters.
        for i, n in enumerate(nodesIn):
            if n.input_index is not None:
                nodesIn[i] = nodesIn[i]._replace(bound_node=BoundInput(
                    nodesIn[i].inputs, nodesIn[i].name, nodesIn[i].ori_name,
                    value=global_input_unpacked[nodesIn[i].input_index],
                    perturbation=nodesIn[i].perturbation))
            else:
                bound_class = BoundParams if isinstance(nodesIn[i].param, nn.Parameter) else BoundBuffers 
                nodesIn[i] = nodesIn[i]._replace(bound_node=bound_class(
                    nodesIn[i].inputs, nodesIn[i].name, nodesIn[i].ori_name,
                    value=nodesIn[i].param, perturbation=nodesIn[i].perturbation))

        unsupported_ops = []

        # Convert other operation nodes.
        for n in range(len(nodesOP)):
            attr = nodesOP[n].attr
            inputs, ori_names = self._get_node_input(nodesOP, nodesIn, nodesOP[n])

            try:
                if nodesOP[n].op in bound_op_map:
                    op = bound_op_map[nodesOP[n].op]
                elif nodesOP[n].op.startswith('aten::ATen'):
                    op = eval('BoundATen{}'.format(attr['operator'].capitalize()))
                elif nodesOP[n].op.startswith('onnx::'):
                    op = eval('Bound{}'.format(nodesOP[n].op[6:]))
                elif nodesOP[n].op in self.custom_ops:
                    op = self.custom_ops[nodesOP[n].op]
                else:
                    raise KeyError
            except (NameError, KeyError):
                unsupported_ops.append(nodesOP[n])
                logger.error('The node has an unsupported operation: {}'.format(nodesOP[n]))
                continue

            if nodesOP[n].op == 'onnx::BatchNormalization':
                # BatchNormalization node needs model.training flag to set running mean and vars
                # set training=False to avoid wrongly updating running mean/vars during bound wrapper
                nodesOP[n] = nodesOP[n]._replace(
                    bound_node=op(
                        nodesOP[n].inputs, nodesOP[n].name, None, attr,
                        inputs, nodesOP[n].output_index, self.bound_opts, self.device, False))
            else:
                nodesOP[n] = nodesOP[n]._replace(
                    bound_node=op(
                        nodesOP[n].inputs, nodesOP[n].name, None, attr,
                        inputs, nodesOP[n].output_index, self.bound_opts, self.device))

        if unsupported_ops:
            logger.error('Unsupported operations:')
            for n in unsupported_ops:
                logger.error(f'Name: {n.op}, Attr: {n.attr}')
            raise NotImplementedError('There are unsupported operations')

        return nodesOP, nodesIn, nodesOut, template

    def _build_graph(self, nodesOP, nodesIn, nodesOut, template):
        nodes = []
        for node in nodesOP + nodesIn:
            assert (node.bound_node is not None)
            nodes.append(node.bound_node)
        # We were assuming that the original model had only one output node.
        # When there are multiple output nodes, this seems to be the first output element.
        # In this case, we are assuming that we always aim to compute the bounds for the first
        # output element.
        self.final_name = nodesOP[-1].name
        assert self.final_name == nodesOut[0]
        self.input_name, self.input_index, self.root_name = [], [], []
        for node in nodesIn:
            self.root_name.append(node.name)
            if node.input_index is not None:
                self.input_name.append(node.name)
                self.input_index.append(node.input_index)
        self.output_name = nodesOut
        self.output_template = template
        for l in nodes:
            self._modules[l.name] = l
            l.output_name = []
            if isinstance(l.input_name, str):
                l.input_name = [l.input_name]
        for l in nodes:
            for l_pre in l.input_name:
                self._modules[l_pre].output_name.append(l.name)
        for l in nodes:
            if self.conv_mode != 'patches' and len(l.input_name) == 0:
                if not l.name in self.root_name:
                    # Add independent nodes that do not appear in `nodesIn`.
                    # Note that these nodes are added in the last, since 
                    # order matters in the current implementation because 
                    # `root[0]` is used in some places.
                    self.root_name.append(l.name)

    def _split_complex(self, nodesOP, nodesIn):
        found_complex = False
        for n in range(len(nodesOP)):
            if hasattr(nodesOP[n].bound_node, 'complex') and \
                    nodesOP[n].bound_node.complex:
                found_complex = True
                _nodesOP, _nodesIn, _, _ = self._convert_nodes(
                    nodesOP[n].bound_node.model, nodesOP[n].bound_node.input)
                name_base = nodesOP[n].name + '/split'
                rename_dict = {}
                for node in _nodesOP + _nodesIn:
                    rename_dict[node.name] = name_base + node.name

                num_inputs = len(nodesOP[n].bound_node.input)

                # assuming each supported complex operation only has one output
                for i in range(num_inputs):
                    rename_dict[_nodesIn[i].name] = nodesOP[n].inputs[i]
                rename_dict[_nodesOP[-1].name] = nodesOP[n].name

                def rename(node):
                    node = node._replace(name=rename_dict[node.name])
                    node = node._replace(inputs=[rename_dict[name] for name in node.inputs])
                    node.bound_node.name = rename_dict[node.bound_node.name]
                    node.bound_node.input_name = [
                        rename_dict[name] for name in node.bound_node.input_name]
                    return node

                for i in range(len(_nodesOP)):
                    _nodesOP[i] = rename(_nodesOP[i])
                for i in range(len(_nodesIn)):
                    _nodesIn[i] = rename(_nodesIn[i])

                nodesOP = nodesOP[:n] + _nodesOP + nodesOP[(n + 1):]
                nodesIn = nodesIn + _nodesIn[num_inputs:]

                break

        return nodesOP, nodesIn, found_complex

    """build a dict with {ori_name: name, name: ori_name}"""

    def _get_node_name_map(self, ):
        self.node_name_map = {}
        for node in self._modules.values():
            if isinstance(node, BoundInput) or isinstance(node, BoundParams):
                for p in list(node.named_parameters()):
                    if node.ori_name not in self.node_name_map:
                        self.node_name_map[node.ori_name] = node.name + '.' + p[0]
                        self.node_name_map[node.name + '.' + p[0]] = node.ori_name
                for p in list(node.named_buffers()):
                    if node.ori_name not in self.node_name_map:
                        self.node_name_map[node.ori_name] = node.name + '.' + p[0]
                        self.node_name_map[node.name + '.' + p[0]] = node.ori_name

    # convert a Pytorch model to a model with bounds
    def _convert(self, model, global_input):
        if self.verbose:
            logger.info('Converting the model...')

        if not isinstance(global_input, tuple):
            global_input = (global_input,)
        self.num_global_inputs = len(global_input)

        nodesOP, nodesIn, nodesOut, template = self._convert_nodes(model, global_input)
        global_input = self._to(global_input, self.device)

        while True:
            self._build_graph(nodesOP, nodesIn, nodesOut, template)
            self.forward(*global_input)  # running means/vars changed
            nodesOP, nodesIn, found_complex = self._split_complex(nodesOP, nodesIn)
            if not found_complex:
                break

        self._get_node_name_map()

        # load self.ori_state_dict again to avoid the running means/vars changed during forward()
        self.load_state_dict(self.ori_state_dict)
        model.load_state_dict(self.ori_state_dict)
        delattr(self, 'ori_state_dict')

        # The final node used in the last time calling `compute_bounds`
        self.last_final_node = None

        logger.debug('NodesOP:')
        for node in nodesOP:
            logger.debug('{}'.format(node._replace(param=None)))
        logger.debug('NodesIn')
        for node in nodesIn:
            logger.debug('{}'.format(node._replace(param=None)))

        if self.verbose:
            logger.info('Model converted to support bounds')

    def init_slope(self, x, share_slopes=False, method='backward', c=None):
        if method != 'forward':
            assert isinstance(x, tuple)
            assert method == 'backward'
            x = x[0]

        for node in self.optimizable_activations:
            # initialize the parameters
            node.opt_init()          

        with torch.no_grad():
            if method == 'forward':
                l, u = self.compute_bounds(x=(x,), method='forward')
            else:
                l, u = self.compute_bounds(x=(x,), IBP=False, C=c, method='backward', return_A=False)

        for node in self.optimizable_activations:
            if method == 'forward':
                assert not '_forward' in self._modules.keys(), '_forward is a reserved node name'
                assert isinstance(node, BoundRelu), 'Only ReLU is supported for optimizing forward bounds'
                start_nodes = [ ('_forward', 1) ]
            else:
                start_nodes = []
                for nj in self.backward_from[node.name]:
                    if nj.name == self.final_name:
                        size_final = self.final_shape[-1] if c is None else c.size(1)                    
                        start_nodes.append((self.final_name, size_final))       
                        continue                    
                    if share_slopes:
                        # all intermediate neurons from the same layer share the same set of slopes.
                        output_shape = 1
                    elif isinstance(node, BoundRelu) and node.patch_size and nj.name in node.patch_size:
                        # Patches mode. Use output channel size as the spec size. This still shares some alpha, but better than no sharing.
                        # The patch size is [batch, L, out_ch, in_ch, H, W]. We use out_ch as the output shape.
                        output_shape = node.patch_size[nj.name][2]
                    else:
                        output_shape = prod(nj.lower.shape[1:])
                    start_nodes.append((nj.name, output_shape))
            node.init_opt_parameters(start_nodes)
            node.opt_start()

        print("alpha-CROWN optimizable variables initialized.")

    def beta_bias(self):
        batch_size = len(self.relus[-1].split_beta)
        batch = int(batch_size/2)
        bias = torch.zeros((batch_size, 1), device=self.device)
        for m in self.relus:
            if m.split_beta_used:
                bias[:batch] = bias[:batch] + m.split_bias*m.split_beta[:batch]*m.split_c[:batch]
                bias[batch:] = bias[batch:] + m.split_bias*m.split_beta[batch:]*m.split_c[batch:]
            if m.history_beta_used:
                bias = bias + (m.new_history_bias*m.new_history_beta*m.new_history_c).sum(1, keepdim=True)
            # No single node split here, because single node splits do not have bias.
        return bias


    """Intialization for beta optimization of intermediate layer bounds."""
    def _init_intermediate_beta(self, x, opt_coeffs, intermediate_refinement_layers, first_layer_to_refine, partial_new_interval):
        # This disctionary saves the coefficients for beta for each relu layer.
        beta_constraint_specs = {}
        # A list of all optimizable parameters for intermediate betas. Will be passed to the optimizer.
        all_intermediate_betas = []
        # We only need to collect some A matrices for the split constraints, so we keep a dictionary for it.
        needed_A_list = defaultdict(set)

        for layer in self.relus:
            layer.single_intermediate_betas = {}
            layer.history_intermediate_betas = {}
            layer.split_intermediate_betas = {}

        self.best_intermediate_betas = {}
        # In this loop, we (1) create beta variables for all intermediate neurons for each split, and
        # (2) obtain all history coefficients for each layer, and combine them into a matrix (which will be used as specifications).
        # The current split coefficients (which is optimizable) must be handle later, in the optimization loop.
        for layer in self.relus:
            layer_spec = None
            # print(f'layer {layer.name} {layer.max_single_split if hasattr(layer, "max_single_split") else None}')
            if layer.single_beta_used:
                # Single split case.
                assert not layer.history_beta_used and not layer.split_beta_used
                for ll in self.relus:
                    if ll.name not in intermediate_refinement_layers:
                        # Only refine the specific layers. Usually, the last a few layers have bigger room for improvements.
                        # No beta parameters will be created for layers that will not be refined.
                        # print(f'skipping {ll.name}')
                        continue
                    for prev_layer in ll.inputs:
                        # Locate the linear/conv layer before relu (TODO: this works for feedforward only).
                        if isinstance(prev_layer, (BoundLinear, BoundConv, BoundReshape, BoundAdd)):
                            break
                    else:
                        raise RuntimeError("unsupported network architecture")
                    # print(f'creating {ll.name} for {layer.name}')
                    # This layer's intermediate bounds are being optimized. We need the A matrices of the specifications on this layer.
                    needed_A_list[layer.name].add(prev_layer.name)
                    # Remove the corresponding bounds in intervals to be set.
                    if ll.name in partial_new_interval:
                        del partial_new_interval[ll.name]
                    if prev_layer.name in partial_new_interval:
                        del partial_new_interval[prev_layer.name]
                    # layer.beta_mask has shape [batch, *nodes, max_nbeta]
                    layer.single_intermediate_betas.update({prev_layer.name: {
                        "lb": torch.zeros(
                            size=(x[0].size(0),) + ll.shape + (layer.max_single_split,),
                            device=x[0].device, requires_grad=True),
                        "ub": torch.zeros(
                            size=(x[0].size(0),) + ll.shape + (layer.max_single_split,),
                            device=x[0].device, requires_grad=True),
                    }
                    })
                    beta_constraint_specs[layer.name] = OneHotC(shape=(x[0].size(0), layer.max_single_split) + layer.shape, device=x[0].device, index=layer.single_beta_loc, coeffs=-layer.single_beta_sign)
                if Check_against_base_lp:
                    # Add only one layer to optimize; do not optimize all variables jointly.
                    all_intermediate_betas.extend(
                        layer.single_intermediate_betas[Check_against_base_lp_layer].values())
                else:
                    all_intermediate_betas.extend(
                        [beta_lb_ub for ll in layer.single_intermediate_betas.values() for beta_lb_ub
                         in ll.values()])
                continue  # skip the rest of the loop.

            if layer.history_beta_used:
                # Create optimizable beta variables for all intermediate layers.
                # Add the conv/linear layer that is right before a ReLu layer.
                for ll in self.relus:
                    if ll.name not in intermediate_refinement_layers:
                        # Only refine the specific layers. Usually, the last a few layers have bigger room for improvements.
                        # No beta parameters will be created for layers that will not be refined.
                        continue
                    for prev_layer in ll.inputs:
                        # Locate the linear/conv layer before relu (TODO: this works for feedforward only).
                        if isinstance(prev_layer, (BoundLinear, BoundConv, BoundReshape, BoundAdd)):
                            break
                    else:
                        raise RuntimeError("unsupported network architecture")
                    # This layer's intermediate bounds are being optimized. We need the A matrices of the specifications on this layer.
                    needed_A_list[layer.name].add(prev_layer.name)
                    # Remove the corresponding bounds in intervals to be set.
                    if ll.name in partial_new_interval:
                        del partial_new_interval[ll.name]
                    if prev_layer.name in partial_new_interval:
                        del partial_new_interval[prev_layer.name]
                    # layer.new_history_coeffs has shape [batch, *nodes, max_nbeta]
                    layer.history_intermediate_betas.update({prev_layer.name: {
                        "lb": torch.zeros(
                            size=(x[0].size(0),) + ll.shape + (layer.new_history_coeffs.size(-1),),
                            device=x[0].device, requires_grad=True),
                        "ub": torch.zeros(
                            size=(x[0].size(0),) + ll.shape + (layer.new_history_coeffs.size(-1),),
                            device=x[0].device, requires_grad=True),
                    }
                    })
                if Check_against_base_lp:
                    # Add only one layer to optimize; do not optimize all variables jointly.
                    all_intermediate_betas.extend(
                        layer.history_intermediate_betas[Check_against_base_lp_layer].values())
                else:
                    all_intermediate_betas.extend(
                        [beta_lb_ub for ll in layer.history_intermediate_betas.values() for beta_lb_ub
                         in ll.values()])
                # Coefficients of history constraints only, in shape [batch, n_beta - 1, n_nodes].
                # For new_history_c = +1, it is z >= 0, and we need to negate and get the lower bound of -z < 0.
                # For unused beta (dummy padding split) inside a batch, layer_spec will be 0.
                layer_spec = - layer.new_history_coeffs.transpose(-1,
                                                                  -2) * layer.new_history_c.unsqueeze(
                    -1)
            if layer.split_beta_used:
                # Create optimizable beta variables for all intermediate layers. First, we always have the layer after the root (input) node.
                for ll in self.relus:
                    if ll.name not in intermediate_refinement_layers:
                        # Only refine the specific layers. Usually, the last a few layers have bigger room for improvements.
                        # No beta parameters will be created for layers that will not be refined.
                        continue
                    for prev_layer in ll.inputs:
                        # Locate the linear/conv layer before relu (TODO: this works for feedforward only).
                        if isinstance(prev_layer, (BoundLinear, BoundConv, BoundReshape, BoundAdd)):
                            break
                    else:
                        raise RuntimeError("unsupported network architecture")
                    # This layer's intermediate bounds are being optimized. We need the A matrices of the specifications on this layer.
                    needed_A_list[layer.name].add(prev_layer.name)
                    # Remove the corresponding bounds in intervals to be set.
                    if ll.name in partial_new_interval:
                        del partial_new_interval[ll.name]
                    if prev_layer.name in partial_new_interval:
                        del partial_new_interval[prev_layer.name]
                    layer.split_intermediate_betas.update({prev_layer.name: {
                        "lb": torch.zeros(size=(x[0].size(0),) + ll.shape + (1,), device=x[0].device,
                                          requires_grad=True),
                        "ub": torch.zeros(size=(x[0].size(0),) + ll.shape + (1,), device=x[0].device,
                                          requires_grad=True),
                    }
                    })
                if Check_against_base_lp:
                    # Add only one layer to optimize; do not optimize all variables jointly.
                    all_intermediate_betas.extend(
                        layer.split_intermediate_betas[Check_against_base_lp_layer].values())
                else:
                    all_intermediate_betas.extend(
                        [beta_lb_ub for ll in layer.split_intermediate_betas.values() for beta_lb_ub in
                         ll.values()])
            # If split coefficients are not optimized, we can just add current split constraints here - no need to reconstruct every time.
            if layer.split_beta_used and not opt_coeffs:
                assert layer.split_coeffs[
                           "dense"] is not None  # TODO: We only support dense split coefficients.
                # Now we have coefficients of both history constraints and split constraints, in shape [batch, n_nodes, n_beta].
                # split_c is 1 for z>0 split, is -1 for z<0 split, and we negate them here to much the formulation in Lagrangian.
                layer_split_spec = -(
                            layer.split_coeffs["dense"].repeat(2, 1) * layer.split_c).unsqueeze(1)
                if layer_spec is not None:
                    layer_spec = torch.cat((layer_spec, layer_split_spec), dim=1)
                else:
                    layer_spec = layer_split_spec
            if layer_spec is not None:
                beta_constraint_specs[layer.name] = layer_spec.detach().requires_grad_(False)

        # Remove some unused specs.
        for k in list(beta_constraint_specs.keys()):
            if int(k[1:]) < int(first_layer_to_refine[1:]):  # TODO: use a better way to check this.
                # Remove this spec because it is not used.
                print(f'Removing {k} from specs for intermediate beta.')
                del beta_constraint_specs[k]

        # Preset intermediate betas if they are specified as a list.
        if self.init_intermediate_betas is not None:
            # The batch dimension.
            for i, example_int_betas in enumerate(self.init_intermediate_betas):
                if example_int_betas is not None:
                    # The layer with split constraints.
                    for split_layer, all_int_betas_this_layer in example_int_betas.items():
                        # Beta variables for all layers for that split constraints.
                        for intermediate_layer, intermediate_betas in all_int_betas_this_layer.items():
                            saved_n_betas = intermediate_betas['lb'].size(-1)
                            if self._modules[split_layer].single_beta_used:
                                # Only self.single_intermediate_beta is created.
                                assert not self._modules[split_layer].history_beta_used
                                assert not self._modules[split_layer].split_beta_used
                                if intermediate_layer in self._modules[split_layer].single_intermediate_betas:
                                    self._modules[split_layer].single_intermediate_betas[
                                        intermediate_layer]['lb'].data[i, ..., :saved_n_betas] = \
                                    intermediate_betas['lb']
                                    self._modules[split_layer].single_intermediate_betas[
                                        intermediate_layer]['ub'].data[i, ..., :saved_n_betas] = \
                                    intermediate_betas['ub']
                                else:
                                    warnings.warn(f"Warning: the intermediate bounds of sample {i} split {split_layer} layer {intermediate_layer} are not optimized, but initialization contains it with size {saved_n_betas}. It might be a bug.", stacklevel=2)

                            elif intermediate_layer in self._modules[split_layer].history_intermediate_betas:
                                # Here we assume the last intermediate beta is the last split, which will still be 0.
                                # When we create specifications, we used single_beta_loc, which must have the current split at last.
                                self._modules[split_layer].history_intermediate_betas[
                                    intermediate_layer]['lb'].data[i, ..., :saved_n_betas] = \
                                intermediate_betas['lb']
                                self._modules[split_layer].history_intermediate_betas[
                                    intermediate_layer]['ub'].data[i, ..., :saved_n_betas] = \
                                intermediate_betas['ub']
                            else:
                                warnings.warn(f"Warning: the intermediate bounds of sample {i} split {split_layer} layer {intermediate_layer} are not optimized, but initialization contains it. It might be a bug.", stacklevel=2)

        return beta_constraint_specs, all_intermediate_betas, needed_A_list

    def _get_intermediate_beta_specs(self, x, aux, opt_coeffs, beta_constraint_specs, needed_A_list, new_interval):
        beta_spec_coeffs = {}  # Key of the dictionary is the pre-relu node name, value is the A matrices propagated to this pre-relu node. We will directly add it to the initial C matrices when computing intermediate bounds.
        # Run CROWN using existing intermediate layer bounds, to get linear inequalities of beta constraints w.r.t. input.
        for layer_idx, layer in enumerate(self.relus):
            if layer.split_beta_used and opt_coeffs:
                # In this loop, we add the current optimizable split constraint.
                assert layer.split_coeffs["dense"] is not None  # We only use dense split coefficients.
                if layer.name in beta_constraint_specs:
                    # Now we have coefficients of both history constraints and split constraints, in shape [batch, n_nodes, n_beta].
                    spec_C = torch.cat((beta_constraint_specs[layer.name],
                                        -(layer.split_coeffs["dense"].repeat(2, 1) * layer.split_c).unsqueeze(
                                            1)), dim=1)
                else:
                    spec_C = -(layer.split_coeffs["dense"].repeat(2, 1) * layer.split_c).unsqueeze(1)
            else:
                if layer.name in beta_constraint_specs:
                    # This layer only has history constraints, no split constraints. This has already been saved into beta_constraint_specs.
                    spec_C = beta_constraint_specs[layer.name]
                else:
                    # This layer has no beta constraints.
                    spec_C = None
            if spec_C is not None:
                # We now have the specifications, which are just coefficients for beta.
                # Now get A and bias w.r.t. input x for the layer just before Relu.
                # TODO: no concretization needed here.
                prev_layer_name = layer.inputs[0].name
                # Resize spec size in case there are conv layers.
                if not isinstance(spec_C, OneHotC):
                    spec_C = spec_C.view(spec_C.size(0), spec_C.size(1), *layer.shape)
                # spec_C.index has shape (batch, n_max_beta_split). Need to transpose since alpha has output_shape before batch.
                alpha_idx = spec_C.index.transpose(0,1)
                # We need to find which relu layer is this, and set the start_idx accordingly to get the tightest possible bound with optimal alpha for this layer's intermediate bounds.
                # For example, if this is the pre-activation of the last relu layer, we want start_idx = 2; if it is the pre-activation of the second relu layer, we need to use the first set of alpha, so start_idx = len(self.relus).
                # lower_spec_A contains the A matrices propagated from the split layer to all interemdiate layers.
                _, _, lower_spec_A = self.compute_bounds(x, aux, spec_C, IBP=False, forward=False,
                                                         method="CROWN", bound_lower=True, bound_upper=False,
                                                         reuse_ibp=True,
                                                         return_A=True, needed_A_list=needed_A_list[layer.name],
                                                         final_node_name=prev_layer_name, average_A=False,
                                                         new_interval=new_interval, alpha_idx=alpha_idx)
                # For computing the upper bound, the spec vector needs to be negated.
                if not isinstance(spec_C, OneHotC):
                    spec_C_neg = - spec_C
                else:
                    spec_C_neg = spec_C._replace(coeffs = -spec_C.coeffs)
                # spec_C_neg.index has shape (batch, n_max_beta_split). Need to transpose since alpha has output_shape before batch.
                alpha_idx = spec_C_neg.index.transpose(0,1)
                _, _, upper_spec_A = self.compute_bounds(x, aux, spec_C_neg, IBP=False, forward=False,
                                                         method="CROWN", bound_lower=False, bound_upper=True,
                                                         reuse_ibp=True,
                                                         return_A=True, needed_A_list=needed_A_list[layer.name],
                                                         final_node_name=prev_layer_name, average_A=False,
                                                         new_interval=new_interval, alpha_idx=alpha_idx)
                # Merge spec_A matrices for lower and upper bound.
                spec_A = {}
                for k in lower_spec_A[prev_layer_name].keys():
                    spec_A[k] = {}
                    spec_A[k]["lA"] = lower_spec_A[prev_layer_name][k]["lA"]
                    spec_A[k]["lbias"] = lower_spec_A[prev_layer_name][k]["lbias"]
                    spec_A[k]["uA"] = upper_spec_A[prev_layer_name][k]["uA"]
                    spec_A[k]["ubias"] = upper_spec_A[prev_layer_name][k]["ubias"]

                beta_spec_coeffs.update({prev_layer_name: spec_A})
                # del lb, ub, spec_A

        return beta_spec_coeffs

    def get_optimized_bounds(self, x=None, aux=None, C=None, IBP=False, forward=False, method='backward',
                             bound_lower=True, bound_upper=False, reuse_ibp=False, return_A=False, final_node_name=None,
                             average_A=False, new_interval=None, reference_bounds=None):
        # optimize CROWN lower bound by alpha and beta
        opts = self.bound_opts['optimize_bound_args']
        iteration = opts['ob_iteration']; beta = opts['ob_beta']; alpha = opts['ob_alpha']
        opt_coeffs = opts['ob_opt_coeffs']; opt_bias = opts['ob_opt_bias']
        verbose = opts['ob_verbose']; opt_choice = opts['ob_optimizer']
        single_node_split = opts['ob_single_node_split'] 
        keep_best = opts['ob_keep_best']; update_by_layer = opts['ob_update_by_layer']; init = opts['ob_init']
        lr = opts['ob_lr']; lr_beta = opts['ob_lr_beta']
        lr_intermediate_beta = opts['ob_lr_intermediate_beta']
        intermediate_beta_enabled = opts['ob_intermediate_beta']
        lr_decay = opts['ob_lr_decay']; lr_coeffs = opts['ob_lr_coeffs'] 
        loss_reduction_func = opts['ob_loss_reduction_func']
        stop_criterion_func = opts['ob_stop_criterion_func']
        input_grad = opts['ob_input_grad']

        assert bound_lower != bound_upper, 'we can only optimize lower OR upper bound at one time'
        assert alpha or beta, "nothing to optimize, use compute bound instead!"

        if C is not None:
            self.final_shape = C.size()[:2]
            self.bound_opts.update({'final_shape': self.final_shape})
        if init:
            self.init_slope(x, share_slopes=opts['ob_alpha_share_slopes'], method=method, c=C)

        alphas = []
        betas = []
        beta_masks = []
        parameters = []
        dense_coeffs_mask = []

        for m in self.optimizable_activations:
            if alpha:
                alphas.extend(list(m.alpha.values()))

        if alpha:
            # Alpha has shape (2, output_shape, batch_dim, node_shape)
            parameters.append({'params': alphas, 'lr': lr, 'batch_dim': 2})
            # best_alpha is a dictionary of dictionary. Each key is the alpha variable for one relu layer, and each value is a dictionary contains all relu layers after that layer as keys.
            best_alphas = OrderedDict()
            for m in self.optimizable_activations:
                best_alphas[m.name] = {}
                for alpha_m in m.alpha:
                    best_alphas[m.name][alpha_m] = m.alpha[alpha_m].clone().detach()
                    # We will directly replace the dictionary for each relu layer after optimization, so the saved alpha might not have require_grad=True.
                    m.alpha[alpha_m].requires_grad_()

        if beta:
            if len(self.relus) != len(self.optimizable_activations):
                raise NotImplementedError("Beta-CROWN for tanh models is not supported yet")

            if single_node_split:
                for model in self.relus:
                    betas.append(model.sparse_beta)
            else:
                betas = self.beta_params + self.single_beta_params
                if opt_coeffs:
                    coeffs = [dense_coeffs["dense"] for dense_coeffs in self.split_dense_coeffs_params] + self.coeffs_params
                    dense_coeffs_mask = [dense_coeffs["mask"] for dense_coeffs in self.split_dense_coeffs_params]
                    parameters.append({'params': coeffs, 'lr': lr_coeffs})
                    best_coeffs = [coeff.clone().detach() for coeff in coeffs]
                if opt_bias:
                    biases = self.bias_params
                    parameters.append({'params': biases, 'lr': lr_coeffs})
                    best_biases = [bias.clone().detach() for bias in biases]

            # Beta has shape (batch, max_splits_per_layer)
            parameters.append({'params': betas, 'lr': lr_beta, 'batch_dim': 0})
            best_betas = [b.clone().detach() for b in betas]

        start = time.time()

        if beta and intermediate_beta_enabled:
            # The list of layer numbers for refinement, can be positive or negative. -1 means refine the intermediate layer bound before last relu layer.
            intermediate_refinement_layers = opts['ob_intermediate_refinement_layers']
            # Change negative layer number to positive ones.
            intermediate_refinement_layers = [layer if layer > 0 else layer + len(self.relus) for layer in
                                            intermediate_refinement_layers]
            # This is the first layer to refine; we do not need the specs for all layers before it.
            first_layer_to_refine = self.relus[min(intermediate_refinement_layers)].name
            # Change layer number to layer name.
            intermediate_refinement_layers = [self.relus[layer].name for layer in intermediate_refinement_layers]
            print(f'Layers for refinement: {intermediate_refinement_layers}; there are {len(self.relus) - len(intermediate_refinement_layers)} layers NOT being refined.')
            # We only need to set some intermediate layer bounds.
            partial_new_interval = new_interval.copy() if new_interval is not None else None  # Shallow copy.
            # beta_constraint_specs is a disctionary that saves the coefficients for beta for each relu layer.
            # all_intermediate_betas A list of all optimizable parameters for intermediate betas. Will be passed to the optimizer.
            # For each neuron in each layer, we have M intermediate_beta variables where M is the number of constraints.
            # We only need to collect some A matrices for the split constraints, so we keep a dictionary needed_A_list for it.
            beta_constraint_specs, all_intermediate_betas, needed_A_list = self._init_intermediate_beta(x, opt_coeffs, intermediate_refinement_layers, first_layer_to_refine, partial_new_interval)
            # Add all intermediate layer beta to parameters.
            parameters.append({'params': all_intermediate_betas, 'lr': lr_intermediate_beta})

        if opt_choice == "adam-autolr":
            opt = AdamElementLR(parameters, lr=lr)
        elif opt_choice == "adam":
            opt = optim.Adam(parameters, lr=lr)
        elif opt_choices == 'sgd':
            opt = optim.SGD(parameters, lr=lr, momentum=0.9)
        else:
            raise NotImplementedError(opt_choices)
        # Create a weight vector to scale learning rate.
        loss_weight = torch.ones(size=(x[0].size(0),), device=x[0].device)

        scheduler = optim.lr_scheduler.ExponentialLR(opt, lr_decay)

        last_l = math.inf
        last_total_loss = torch.tensor(1e8, device=x[0].device, dtype=x[0].dtype)
        best_l = torch.zeros([x[0].shape[0], 1], device=x[0].device, dtype=x[0].dtype) + 1e8

        if verbose > 0 and intermediate_beta_enabled:
            for layer in self.relus:
                if layer.history_beta_used:
                    for k, v in layer.history_intermediate_betas.items():
                        print(
                            f'hist split layer {layer.name} beta layer {k} lb value {v["lb"].abs().sum(dim=list(range(1, v["lb"].ndim))).detach().cpu().numpy()} ub value {v["ub"].abs().sum(dim=list(range(1, v["ub"].ndim))).detach().cpu().numpy()}')
                if layer.split_beta_used:
                    for k, v in layer.split_intermediate_betas.items():
                        print(
                            f'new  split layer {layer.name} beta layer {k} lb value {v["lb"].abs().sum(dim=list(range(1, v["lb"].ndim))).detach().cpu().numpy()} ub value {v["ub"].abs().sum(dim=list(range(1, v["ub"].ndim))).detach().cpu().numpy()}')
                if layer.single_beta_used:
                    for k, v in layer.single_intermediate_betas.items():
                        print(
                            f'single split layer {layer.name} beta layer {k} lb value {v["lb"].abs().sum(dim=list(range(1, v["lb"].ndim))).detach().cpu().numpy()} ub value {v["ub"].abs().sum(dim=list(range(1, v["ub"].ndim))).detach().cpu().numpy()}')

        for i in range(iteration):
            if beta and intermediate_beta_enabled:
                intermediate_constr = self._get_intermediate_beta_specs(x, aux, opt_coeffs, beta_constraint_specs, needed_A_list, new_interval)
            else:
                intermediate_constr = None

            ret = self.compute_bounds(x, aux, C, method=method, IBP=IBP, forward=forward, 
                                    bound_lower=bound_lower, bound_upper=bound_upper, reuse_ibp=reuse_ibp,
                                    return_A=False, final_node_name=final_node_name, average_A=average_A,
                                    # If we set neuron bounds individually, or if we are optimizing intermediate layer bounds using beta, we do not set new_interval.
                                    # When intermediate betas are used, we must set new_interval to None because we want to recompute all intermediate layer bounds.
                                    new_interval=partial_new_interval if beta and intermediate_beta_enabled else new_interval if update_by_layer else None,
                                    # This is the currently tightest interval, which will be used to pass split constraints when intermediate betas are used.
                                    reference_bounds=new_interval if beta and intermediate_beta_enabled or not update_by_layer else reference_bounds,
                                    # These are intermediate layer beta variables and their corresponding A matrices and biases.
                                    intermediate_constr=intermediate_constr)

            if i == 0:
                best_ret = ret
                best_intermediate_bounds = []
                for model in self.optimizable_activations:
                    best_intermediate_bounds.append([model.inputs[0].lower.clone().detach(), model.inputs[0].upper.clone().detach()])

            ret_l, ret_u = ret[0], ret[1]

            if beta and opt_bias and not single_node_split:
                ret_l = ret_l + self.beta_bias()
                ret = (ret_l, ret_u)

            l = ret_l
            if ret_l is not None and ret_l.shape[1] != 1:  # Reduction over the spec dimension.
                l = loss_reduction_func(ret_l)
            u = ret_u
            if ret_u is not None and ret_u.shape[1] != 1:
                u = loss_reduction_func(ret_u)                

            if beta and intermediate_beta_enabled and Check_against_base_lp:
                # NOTE stop_criterion here is not valid
                stop_criterion = total_loss = loss_ = loss = -self._modules[Check_against_base_lp_layer].lower[0, 0] - \
                                            self._modules[Check_against_base_lp_layer].lower[
                                                0, 1]  # 2.6972426192395584 matched
            else:
                loss_ = l if bound_lower else -u
                stop_criterion = stop_criterion_func(ret_l) if bound_lower else stop_criterion_func(-ret_u)
                total_loss = -1 * loss_
                if type(stop_criterion) == bool:
                    loss = total_loss.sum() * (not stop_criterion)
                else:
                    loss = (total_loss * stop_criterion.logical_not()).sum()

            with torch.no_grad():
                # Save varibles if this is the best iteration.
                if keep_best and (total_loss < best_l).any():
                    # we only pick up the results improved in a batch
                    idx = (total_loss < best_l).squeeze()
                    best_l[idx] = total_loss[idx]

                    if ret[0] is not None:
                        best_ret[0][idx] = ret[0][idx]
                    if ret[1] is not None:
                        best_ret[1][idx] = ret[1][idx]

                    for ii, model in enumerate(self.optimizable_activations):
                        # best_intermediate_bounds.append([model.inputs[0].lower, model.inputs[0].upper])
                        best_intermediate_bounds[ii][0][idx] = model.inputs[0].lower[idx]
                        best_intermediate_bounds[ii][1][idx] = model.inputs[0].upper[idx]
                        if alpha:
                            # each alpha has shape (2, output_shape, batch, *shape)
                            for alpha_m in model.alpha:
                                best_alphas[model.name][alpha_m][:,:,idx] = model.alpha[alpha_m][:,:,idx].clone().detach()
                        if beta and single_node_split:
                            best_betas[ii][idx] = betas[ii][idx].clone().detach()

                    if not single_node_split and beta:
                        for ii, b in enumerate(betas):
                            best_betas[ii][idx] = b[idx].clone().detach()

                        if opt_coeffs:
                            best_coeffs = [co.clone().detach() for co in coeffs]  # TODO: idx-wise
                        if opt_bias:
                            best_biases = [bias.clone().detach() for bias in biases]  # TODO: idx-wise

                    if beta and intermediate_beta_enabled: # # TODO: idx-wise
                        for layer in self.relus:
                            if layer.history_beta_used or layer.split_beta_used or layer.single_beta_used:
                                self.best_intermediate_betas[layer.name] = {}
                            # The history split and current split is handled seperatedly.
                            if layer.history_beta_used:
                                self.best_intermediate_betas[layer.name]['history'] = {}
                                # Each key in history_intermediate_betas for this layer is a dictionary, with all other pre-relu layers' names.
                                for k, v in layer.history_intermediate_betas.items():
                                    self.best_intermediate_betas[layer.name]['history'][k] = {
                                        "lb": v["lb"],
                                        # This is a tensor with shape (batch, *intermediate_layer_shape, number_of_beta)
                                        "ub": v["ub"],
                                    }
                            if layer.split_beta_used:
                                self.best_intermediate_betas[layer.name]['split'] = {}
                                for k, v in layer.split_intermediate_betas.items():
                                    self.best_intermediate_betas[layer.name]['split'][k] = {
                                        "lb": v["lb"],  # This is a tensor with shape (batch, *intermediate_layer_shape, 1)
                                        "ub": v["ub"],
                                    }
                            if layer.single_beta_used:
                                self.best_intermediate_betas[layer.name]['single'] = {}
                                for k, v in layer.single_intermediate_betas.items():
                                    self.best_intermediate_betas[layer.name]['single'][k] = {
                                        "lb": v["lb"],  # This is a tensor with shape (batch, *intermediate_layer_shape, 1)
                                        "ub": v["ub"],
                                    }

            if os.environ.get('AUTOLIRPA_DEBUG_OPT', False):
                print(f"****** iter [{i}]",
                    f"loss: {loss.item()}, lr: {opt.param_groups[0]['lr']}")

            if isinstance(stop_criterion, torch.Tensor) and stop_criterion.all():
                print(f"\nall verified at {i}th iter")
                break

            current_lr = []
            for param_group in opt.param_groups:
                current_lr.append(param_group['lr'])

            opt.zero_grad(set_to_none=True)

            if input_grad and x[0].ptb.x_L.grad is not None:
                x[0].ptb.x_L.grad = None
                x[0].ptb.x_U.grad = None

            loss.backward()

            if verbose > 0:
                print(f"*** iter [{i}]\n", f"loss: {loss.item()}", total_loss.squeeze().detach().cpu().numpy(), "lr: ", current_lr)
                if beta:
                    masked_betas = []
                    for model in self.relus:
                        masked_betas.append(model.masked_beta)
                        if model.history_beta_used:
                            print(f"{model.name} history beta", model.new_history_beta.squeeze())
                        if model.split_beta_used:
                            print(f"{model.name} split beta:", model.split_beta.view(-1))
                            print(f"{model.name} bias:", model.split_bias)
                    if intermediate_beta_enabled:
                        for layer in self.relus:
                            print(
                                f'layer {layer.name} lower {layer.inputs[0].lower.sum().item()}, upper {layer.inputs[0].upper.sum().item()}')
                        for layer in self.relus:
                            if layer.history_beta_used:
                                for k, v in layer.history_intermediate_betas.items():
                                    print(
                                        f'hist split layer {layer.name} beta layer {k} lb value {v["lb"].abs().sum(dim=list(range(1, v["lb"].ndim))).detach().cpu().numpy()} ub value {v["ub"].abs().sum(dim=list(range(1, v["ub"].ndim))).detach().cpu().numpy()}')
                            if layer.split_beta_used:
                                for k, v in layer.split_intermediate_betas.items():
                                    print(
                                        f'new  split layer {layer.name} beta layer {k} lb value {v["lb"].abs().sum(dim=list(range(1, v["lb"].ndim))).detach().cpu().numpy()} ub value {v["ub"].abs().sum(dim=list(range(1, v["ub"].ndim))).detach().cpu().numpy()}')
                            if layer.single_beta_used:
                                for k, v in layer.single_intermediate_betas.items():
                                    print(
                                        f'single split layer {layer.name} beta layer {k} lb value {v["lb"].abs().sum(dim=list(range(1, v["lb"].ndim))).detach().cpu().numpy()} ub value {v["ub"].abs().sum(dim=list(range(1, v["ub"].ndim))).detach().cpu().numpy()}')
                    if opt_coeffs:
                        for co in coeffs:
                            print(f'coeff sum: {co.abs().sum():.5g}')
                if beta and i == 0 and verbose > 0:
                    breakpoint()

            if opt_choice == "adam-autolr":
                opt.step(lr_scale=[loss_weight, loss_weight])
            else:
                opt.step()

            if beta:
                # Clipping to >=0.
                for b in betas:
                    b.data = (b >= 0) * b.data
                if intermediate_beta_enabled:
                    for b in all_intermediate_betas:
                        b.data = torch.clamp(b.data, min=0)
                for dmi in range(len(dense_coeffs_mask)):
                    # apply dense mask to the dense split coeffs matrix
                    coeffs[dmi].data = dense_coeffs_mask[dmi].float() * coeffs[dmi].data

            if alpha:
                for m in self.relus:
                    for v in m.alpha.values():
                        v.data = torch.clamp(v.data, 0., 1.)
                # For tanh, we clip it in bound_ops because clipping depends
                # on pre-activation bounds

            # If loss has become worse for some element, reset those to current best.
            with torch.no_grad():
                if beta and opt_choice == "adam-autolr" and i > iteration * 0.2:
                    for ii, model in enumerate(self.relus):
                        if alpha:
                            # each alpha has shape (2, output_shape, batch, *shape)
                            for alpha_m in model.alpha:
                                model.alpha[alpha_m][:,:,worse_idx] = best_alphas[model.name][alpha_m][:,:,worse_idx].clone().detach()
                        if beta and single_node_split:
                            betas[ii][worse_idx] = best_betas[ii][worse_idx].clone().detach()

            scheduler.step()
            last_l = loss.item()
            last_total_loss = total_loss.detach().clone()

        if beta and intermediate_beta_enabled and verbose > 0:
            breakpoint()

        if keep_best:
            # Set all variables to their saved best values.
            with torch.no_grad():
                for idx, model in enumerate(self.optimizable_activations):
                    if alpha:
                        # Assigns a new dictionary.
                        model.alpha = best_alphas[model.name]
                    model.inputs[0].lower.data = best_intermediate_bounds[idx][0].data
                    model.inputs[0].upper.data = best_intermediate_bounds[idx][1].data
                    if beta:
                        if single_node_split:
                            model.sparse_beta.copy_(best_betas[idx])
                        else:
                            for b, bb in zip(betas, best_betas):
                                b.data = bb.data
                            if opt_coeffs:
                                for co, bco in zip(coeffs, best_coeffs):
                                    co.data = bco.data
                            if opt_bias:
                                for bias, bbias in zip(biases, best_biases):
                                    bias.data = bbias.data

        if new_interval is not None and not update_by_layer:
            for l in self._modules.values():
                if l.name in new_interval.keys() and hasattr(l, "lower"):
                    # l.interval = tuple(new_interval[l.name][:2])
                    l.lower = torch.max(l.lower, new_interval[l.name][0])
                    l.upper = torch.min(l.upper, new_interval[l.name][1])
                    infeasible_neurons = l.lower > l.upper
                    if infeasible_neurons.any():
                        print('infeasible!!!!!!!!!!!!!!', infeasible_neurons.sum().item(), infeasible_neurons.nonzero()[:, 0])
                        # import pdb; pdb.set_trace()

        print("best_l after optimization:", best_l.sum().item(), "with beta sum per layer:", [p.sum().item() for p in betas])
        # np.save('solve_slope.npy', np.array(record))
        print('optimal alpha/beta time:', time.time() - start)
        return best_ret


    def compute_bounds(self, x=None, aux=None, C=None, method='backward', IBP=False, forward=False, 
                       bound_lower=True, bound_upper=True, reuse_ibp=False,
                       return_A=False, needed_A_list=None, final_node_name=None, average_A=False, new_interval=None,
                       return_b=False, b_dict=None, reference_bounds=None, intermediate_constr=None, alpha_idx=None):
        r"""Main function for computing bounds.

        Args:
            x (tuple or None): Input to the model. If it is None, the input from the last 
            `forward` or `compute_bounds` call is reused. Otherwise: the number of elements in the tuple should be 
            equal to the number of input nodes in the model, and each element in the tuple 
            corresponds to the value for each input node respectively. It should look similar 
            as the `global_input` argument when used for creating a `BoundedModule`.

            aux (object, optional): Auxliary information that can be passed to `Perturbation` 
            classes for initializing and concretizing bounds, e.g., additional information 
            for supporting synonym word subsitution perturbaiton. 

            C (Tensor): The specification matrix that can map the output of the model with an 
            additional linear layer. This is usually used for maping the logits output of the 
            model to classification margins.

            method (str): The main method for bound computation. Choices: 
                * `IBP`: purely use Interval Bound Propagation (IBP) bounds.
                * `CROWN-IBP`: use IBP to compute intermediate bounds, but use CROWN (backward mode LiRPA) to compute the bounds of the final node.
                * `CROWN`: purely use CROWN to compute bounds for intermediate nodes and the final node.
                * `Forward`: purely use forward mode LiRPA to compute the bounds.
                * `Forward+Backward`: use forward mode LiRPA to compute bounds for intermediate nodes, but further use CROWN to compute bounds for the final node.
                * `CROWN-Optimized` or `alpha-CROWN`: use CROWN, and also optimize the linear relaxation parameters for activations.

            IBP (bool, optional): If `True`, use IBP to compute the bounds of intermediate nodes.
            It can be automatically set according to `method`.

            forward (bool, optional): If `True`, use the forward mode bound propagation to compute the bounds
            of intermediate nodes. It can be automatically set according to `method`.            

            bound_lower (bool, default `True`): If `True`, the lower bounds of the output needs to be computed.

            bound_upper (bool, default `True`): If `True`, the upper bounds of the output needs to be computed.

            reuse_ibp (bool, optional): If `True` and `method` is None, reuse the previously saved IBP bounds.

        Returns:
            bound (tuple): a tuple of computed lower bound and upper bound respectively.
        """

        # Several shortcuts.
        method = method.lower() if method is not None else method
        if method == 'ibp':
            # Pure IBP bounds.
            method = None
            IBP = True
        elif method == 'ibp+backward' or method == 'ibp+crown' or method == 'crown-ibp':
            method = 'backward'
            IBP = True
        elif method == 'crown':
            method = 'backward'
        elif method == 'forward':
            forward = True
        elif method == 'forward+backward':
            method = 'backward'
            forward = True
        elif method == "crown-optimized" or method == 'alpha-crown':
            if bound_lower:
                ret1 = self.get_optimized_bounds(x=x, IBP=False, C=C, method='backward', new_interval=new_interval, reference_bounds=reference_bounds,
                                                 bound_lower=bound_lower, bound_upper=False, return_A=return_A)
            if bound_upper:
                ret2 = self.get_optimized_bounds(x=x, IBP=False, C=C, method='backward', new_interval=new_interval, reference_bounds=reference_bounds,
                                                 bound_lower=False, bound_upper=bound_upper, return_A=return_A)
            if bound_upper and bound_upper:
                assert return_A is False
                return ret1[0], ret2[1]
            elif bound_lower:
                return ret1
            elif bound_upper:
                return ret2
            else:
                raise NotImplementedError

        # If y in self.backward_node_pairs[x], then node y is visited when 
        # doing backward bound propagation starting from node x.
        self.backward_from = dict([(node, []) for node in self._modules])

        if not bound_lower and not bound_upper:
            raise ValueError('At least one of bound_lower and bound_upper in compute_bounds should be True')
        A_dict = {} if return_A else None

        if x is not None:
            self._set_input(*x, new_interval=new_interval)

        if IBP and method is None and reuse_ibp:
            # directly return the previously saved ibp bounds
            return self.ibp_lower, self.ibp_upper
        root = [self._modules[name] for name in self.root_name]
        batch_size = root[0].value.shape[0]
        dim_in = 0

        for i in range(len(root)):
            value = root[i].forward()
            if hasattr(root[i], 'perturbation') and root[i].perturbation is not None:   
                root[i].linear, root[i].center, root[i].aux = \
                    root[i].perturbation.init(value, aux=aux, forward=forward)
                # This input/parameter has perturbation. Create an interval object.
                if self.ibp_relative:
                    root[i].interval = Interval(
                        None, None, 
                        root[i].linear.nominal, root[i].linear.lower_offset, root[i].linear.upper_offset)
                else:
                    root[i].interval = \
                        Interval(root[i].linear.lower, root[i].linear.upper, ptb=root[i].perturbation)
                if forward:
                    root[i].dim = root[i].linear.lw.shape[1]
                    dim_in += root[i].dim
            else:
                if self.ibp_relative:
                    root[i].interval = Interval(
                        None, None, 
                        value, torch.zeros_like(value), torch.zeros_like(value))                    
                else:
                    # This inpute/parameter does not has perturbation. 
                    # Use plain tuple defaulting to Linf perturbation.
                    root[i].interval = (value, value)
                    root[i].forward_value = root[i].forward_value = root[i].value = root[i].lower = root[i].upper = value

            if self.ibp_relative:
                root[i].lower = root[i].interval.lower
                root[i].upper = root[i].interval.upper
            else:
                root[i].lower, root[i].upper = root[i].interval

        if forward:
            self._init_forward(root, dim_in)

        final = self._modules[self.final_name] if final_node_name is None else self._modules[final_node_name]
        logger.debug('Final node {}[{}]'.format(final, final.name))

        if IBP:
            res = self._IBP_general(node=final, C=C)
            if self.ibp_relative:
                self.ibp_lower, self.ibp_upper = res.lower, res.upper
            else:
                self.ibp_lower, self.ibp_upper = res

        if method is None:
            return self.ibp_lower, self.ibp_upper                

        if C is None:
            # C is an identity matrix by default 
            if final.output_shape is None:
                raise ValueError('C is not provided while node {} has no default shape'.format(final.shape))
            dim_output = int(np.prod(final.output_shape[1:]))
            C = torch.eye(dim_output, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)  # TODO: use an eyeC object here.

        # check whether weights are perturbed and set nonlinear for the BoundMatMul operation
        for n in self._modules.values():
            if isinstance(n, (BoundLinear, BoundConv, BoundBatchNormalization)):
                n.nonlinear = False
                for l_name in n.input_name[1:]:
                    node = self._modules[l_name]
                    if hasattr(node, 'perturbation'):
                        if node.perturbation is not None:
                            n.nonlinear = True

        # BFS to find out whether each node is used given the current final node
        if final != self.last_final_node:
            self.last_final_node = final
            for i in self._modules.values():
                i.used = False
            final.used = True
            queue = deque([final])
            while len(queue) > 0:
                n = queue.popleft()
                for n_pre_name in n.input_name:
                    n_pre = self._modules[n_pre_name]
                    if not n_pre.used:
                        n_pre.used = True
                        queue.append(n_pre)

        for i in self._modules.values():  # for all nodes
            if not i.used:
                continue
            if hasattr(i, 'nonlinear') and i.nonlinear:
                for l_name in i.input_name:
                    node = self._modules[l_name]
                    # print('node', node, 'lower', hasattr(node, 'lower'), 'perturbed', node.perturbed, 'forward_value', hasattr(node, 'forward_value'), 'from_input', node.from_input)
                    if not hasattr(node, 'lower'):
                        assert not IBP, 'There should be no missing intermediate bounds when IBP is enabled'
                        if not node.perturbed and hasattr(node, 'forward_value'):
                            node.interval = node.lower, node.upper = \
                                node.forward_value, node.forward_value
                            continue
                        # FIXME check that weight perturbation is not affected
                        #      (from_input=True should be set for weights)
                        if not node.from_input and hasattr(node, 'forward_value'):
                            node.lower = node.upper = node.forward_value
                            continue
                        if forward:
                            l, u = self._forward_general(
                                node=node, root=root, dim_in=dim_in, concretize=True)
                        else:
                            # assign concretized bound for ReLU layer to save computational cost
                            # FIXME: Put ReLU after reshape will cause problem!
                            if (isinstance(node, BoundActivation) or isinstance(node, BoundTranspose)) and hasattr(
                                    self._modules[node.input_name[0]], 'lower'):
                                node.lower = node.forward(self._modules[node.input_name[0]].lower)
                                node.upper = node.forward(self._modules[node.input_name[0]].upper)
                            elif isinstance(node, BoundReshape) and \
                                    hasattr(self._modules[node.input_name[0]], 'lower') and \
                                    hasattr(self._modules[node.input_name[1]], 'value'):
                                # Node for input value.
                                val_input = self._modules[node.input_name[0]]
                                # Node for input parameter (e.g., shape, permute)
                                arg_input = self._modules[node.input_name[1]]
                                node.lower = node.forward(val_input.lower, arg_input.value)
                                node.upper = node.forward(val_input.upper, arg_input.value)
                            else:
                                first_layer_flag = False
                                # This is the list of all intermediate layers where we need to refine.
                                if intermediate_constr is not None:
                                    intermediate_beta_enabled_layers = [k for v in intermediate_constr.values() for k in v]
                                else:
                                    intermediate_beta_enabled_layers = []
                                # Here we avoid creating a big C matrix in the first linear layer.
                                # Disable this optimization when we have beta for intermediate layer bounds.
                                if type(node) == BoundLinear or type(node) == BoundConv and node.name not in intermediate_beta_enabled_layers:
                                    for l_pre in node.input_name:
                                        if type(self._modules[l_pre]) == BoundInput:
                                            node.lower, node.upper = self._IBP_general(node)
                                            first_layer_flag = True
                                            break
                                if not first_layer_flag:
                                    reduced_dim = False  # Only partial neurons (unstable neurons) are bounded.
                                    unstable_idx = None
                                    dim = int(np.prod(node.output_shape[1:]))
                                    # FIXME: C matrix shape incorrect for BoundParams.
                                    if (isinstance(node, BoundLinear) or isinstance(node, BoundMatMul)) and int(
                                            os.environ.get('AUTOLIRPA_USE_FULL_C', 0)) == 0:
                                        if intermediate_constr is not None and node.name in reference_bounds:
                                            # If we are doing bound refinement and reference bounds are given, we only refine unstable neurons.
                                            # For each batch element, we find the unstable neurons.
                                            unstable_masks = torch.logical_and(reference_bounds[node.name][0] < 0, reference_bounds[node.name][1] > 0)
                                            # unstable_masks = torch.ones(dtype=torch.bool, size=(batch_size, dim), device=self.device)
                                            # For simplicity, merge unstable locations for all elements in this batch. TODO: use individual unstable mask.
                                            unstable_locs = unstable_masks.sum(dim=0).bool()
                                            # This is a 1-d indices, shared by all elements in this batch.
                                            unstable_idx = unstable_locs.nonzero().squeeze()
                                            # Number of unstable neurons after merging.
                                            max_non_zero = unstable_locs.sum()
                                            # Create an abstract C matrix, the unstable_idx are the non-zero elements in specifications for all batches.
                                            newC = OneHotC([batch_size, max_non_zero, *node.output_shape[1:]], self.device, unstable_idx, None)
                                            reduced_dim = True
                                            # print(f'layer {node.name} total {dim} unstable {max_non_zero} newC {newC.shape}')
                                            """
                                            newC = torch.eye(dim, device=self.device) \
                                                .unsqueeze(0).repeat(batch_size, 1, 1) \
                                                .view(batch_size, dim, *node.output_shape[1:])
                                            print(f'creating new C {newC.size()}')
                                            if int(os.environ.get('USE_EYE_C', 0)) == 1:
                                                newC = eyeC([batch_size, dim, *node.output_shape[1:]], self.device)
                                            """
                                        else:
                                            newC = eyeC([batch_size, dim, *node.output_shape[1:]], self.device)
                                    elif (isinstance(node, BoundConv) or isinstance(node,
                                                                                    BoundBatchNormalization)) and node.mode == "patches":
                                        # import pdb; pdb.set_trace()
                                        # Here we create an Identity Patches object 
                                        newC = Patches(None, 1, 0,
                                                       [batch_size, node.output_shape[-2] * node.output_shape[-1],
                                                        node.output_shape[-3], node.output_shape[-3], 1, 1], 1)
                                    elif isinstance(node, BoundAdd) and node.mode == "patches":
                                        num_channel = node.output_shape[-3]
                                        L = node.output_shape[-2] * node.output_shape[-1]
                                        patches = (torch.eye(num_channel, device=self.device)).unsqueeze(0).unsqueeze(
                                            0).unsqueeze(4).unsqueeze(5).expand(batch_size, L, num_channel, num_channel, 1, 1)  # now [1 * 1 * in_C * in_C * 1 * 1]
                                        newC = Patches(patches, 1, 0, [batch_size] + list(patches.shape[1:]))
                                    else:
                                        if intermediate_constr is not None and node.name in reference_bounds:
                                            # If we are doing bound refinement and reference bounds are given, we only refine unstable neurons.
                                            # For each batch element, we find the unstable neurons.
                                            unstable_masks = torch.logical_and(reference_bounds[node.name][0] < 0, reference_bounds[node.name][1] > 0)
                                            # Flatten the conv layer shape.
                                            unstable_masks = unstable_masks.view(unstable_masks.size(0), -1)
                                            # unstable_masks = torch.ones(dtype=torch.bool, size=(batch_size, dim), device=self.device)
                                            # For simplicity, merge unstable locations for all elements in this batch. TODO: use individual unstable mask.
                                            unstable_locs = unstable_masks.sum(dim=0).bool()
                                            # This is always a 1-d indices. For conv layers it's flattened.
                                            unstable_idx = unstable_locs.nonzero().squeeze()
                                            # Number of unstable neurons after merging.
                                            max_non_zero = unstable_locs.sum()
                                            # Create a C matrix.
                                            newC = torch.zeros([1, max_non_zero, dim], device=self.device)
                                            # Fill the corresponding elements to 1.0
                                            newC[0, torch.arange(max_non_zero), unstable_idx] = 1.0
                                            newC = newC.repeat(batch_size, 1, 1).view(batch_size, max_non_zero, *node.output_shape[1:])
                                            reduced_dim = True
                                            # print(f'layer {node.name} total {dim} unstable {max_non_zero} newC {newC.size()}')
                                        else:
                                            if dim > 1000:
                                                warnings.warn(f"Creating an identity matrix with size {dim}x{dim} for node {node}. This may indicate poor performance for bound computation. If you see this message on a small network please submit a bug report.", stacklevel=2)
                                            newC = torch.eye(dim, device=self.device) \
                                                .unsqueeze(0).repeat(batch_size, 1, 1) \
                                                .view(batch_size, dim, *node.output_shape[1:])
                                    # print('Creating new C', type(newC), 'for', node)
                                    if False:  # TODO: only return A_dict of final layer
                                        _, _, A_dict = self._backward_general(C=newC, node=node, root=root,
                                                                              return_A=return_A, A_dict=A_dict, intermedaite_constr=intermediate_constr)
                                    else:
                                        self._backward_general(C=newC, node=node, root=root, return_A=False, intermediate_constr=intermediate_constr, unstable_idx=unstable_idx)

                                    if reduced_dim:
                                        # If we only calculated unstable neurons, we need to scatter the results back based on reference bounds.
                                        new_lower = reference_bounds[node.name][0].detach().clone().view(batch_size, -1)
                                        new_lower[:, unstable_idx] = node.lower.view(batch_size, -1)
                                        node.lower = new_lower.view(batch_size, *node.output_shape[1:])
                                        new_upper = reference_bounds[node.name][1].detach().clone().view(batch_size, -1)
                                        new_upper[:, unstable_idx] = node.upper.view(batch_size, -1)
                                        node.upper = new_upper.view(batch_size, *node.output_shape[1:])
                                    # node.lower and node.upper (intermediate bounds) are computed in the above function.
                                    # If we have bound references, we set them here to always obtain a better set of bounds.
                                    if reference_bounds is not None and node.name in reference_bounds:
                                        # Initially, the reference bound and the computed bound can be exactly the same when intermediate layer beta is 0. This will prevent gradients flow. So we need a small guard here.
                                        if Check_against_base_lp:
                                            if node.name == Check_against_base_lp_layer:
                                                pass
                                                # print(reference_bounds[node.name][0][1,0,0,3].item(), node.lower[1,0,0,3].item())
                                                # node.lower = torch.max(reference_bounds[node.name][0] - 1e-5, node.lower)
                                                # node.upper = torch.min(reference_bounds[node.name][1] + 1e-5, node.upper)
                                            else:
                                                # For LP checking, fix all other intermediate layer bounds.
                                                node.lower = reference_bounds[node.name][0]
                                                node.upper = reference_bounds[node.name][1]
                                        else:
                                            # Setting reference bounds are actually incorrect. Because the split constraints are computed using slightly
                                            # different alpha (not the optimal), they can be slightly worse than original at the beginning.
                                            # So we only update bounds to reference if they cross zero (split constraints).
                                            node.lower = torch.max(reference_bounds[node.name][0] - 1e-5,
                                                                   node.lower)
                                            node.upper = torch.min(reference_bounds[node.name][1] + 1e-5,
                                                                   node.upper)
                                            """
                                            update_lower = reference_bounds[node.name][0] >= 0
                                            node.lower[update_lower] = reference_bounds[node.name][0][update_lower]
                                            update_upper = reference_bounds[node.name][0] <= 0
                                            node.upper[update_upper] = reference_bounds[node.name][1][update_upper]
                                            """

        if method == 'backward':
            # This is for the final output bound. No need to pass in intermediate layer beta constraints.
            return self._backward_general(C=C, node=final, root=root, bound_lower=bound_lower, bound_upper=bound_upper,
                                          return_A=return_A, needed_A_list=needed_A_list, average_A=average_A, A_dict=A_dict,
                                          return_b=return_b, b_dict=b_dict, unstable_idx=alpha_idx)
        elif method == 'forward':
            return self._forward_general(C=C, node=final, root=root, dim_in=dim_in, concretize=True)
        else:
            raise NotImplementedError

    """ improvement on merging BoundLinear, BoundGatherElements and BoundSub
    when loss fusion is used in training"""

    def _IBP_loss_fusion(self, node, C):
        # not using loss fusion
        if not (isinstance(self.bound_opts, dict) and self.bound_opts.get('loss_fusion', False)):
            return None

        # Currently this function has issues in more complicated networks.
        if self.bound_opts.get('no_ibp_loss_fusion', False):
            return None

        if C is None and isinstance(node, BoundSub):
            node_gather = self._modules[node.input_name[1]]
            if isinstance(node_gather, BoundGatherElements) or isinstance(node_gather, BoundGatherAten):
                node_linear = self._modules[node.input_name[0]]
                node_start = self._modules[node_linear.input_name[0]]
                if isinstance(node_linear, BoundLinear):
                    w = self._modules[node_linear.input_name[1]].param
                    b = self._modules[node_linear.input_name[2]].param
                    labels = self._modules[node_gather.input_name[1]]
                    if not hasattr(node_start, 'interval'):
                        self._IBP_general(node_start)
                    for inp in node_gather.input_name:
                        n = self._modules[inp]
                        if not hasattr(n, 'interval'):
                            self._IBP_general(n)
                    if torch.isclose(labels.lower, labels.upper, 1e-8).all():
                        labels = labels.lower
                        batch_size = labels.shape[0]
                        w = w.unsqueeze(0).repeat(batch_size, 1, 1)
                        w = w - torch.gather(w, dim=1,
                                             index=labels.unsqueeze(-1).repeat(1, w.shape[1], w.shape[2]))
                        b = b.unsqueeze(0).repeat(batch_size, 1)
                        b = b - torch.gather(b, dim=1,
                                             index=labels.repeat(1, b.shape[1]))
                        lower, upper = node_start.interval
                        lower, upper = lower.unsqueeze(1), upper.unsqueeze(1)
                        node.lower, node.upper = node_linear.interval_propagate(
                            (lower, upper), (w, w), (b.unsqueeze(1), b.unsqueeze(1)))
                        node.interval = node.lower, node.upper = node.lower.squeeze(1), node.upper.squeeze(1)
                        return node.interval
        return None

    def _IBP_general(self, node=None, C=None):
        if hasattr(node, 'interval'):
            return node.interval
        if not node.perturbed and hasattr(node, 'forward_value'):
            node.lower, node.upper = node.interval = (node.forward_value, node.forward_value)
            
            if self.ibp_relative:
                node.interval = Interval(
                    None, None, 
                    nominal=node.forward_value, 
                    lower_offset=torch.zeros_like(node.forward_value), 
                    upper_offset=torch.zeros_like(node.forward_value))

            return node.interval

        interval = self._IBP_loss_fusion(node, C)
        if interval is not None:
            return interval

        for n_pre in node.input_name:
            n = self._modules[n_pre]
            if not hasattr(n, 'interval'):
                self._IBP_general(n)

        inp = [self._modules[n_pre].interval for n_pre in node.input_name]
        if C is not None:
            if isinstance(node, BoundLinear) and not node.is_input_perturbed(1):
                # merge the output node with the specification, available when weights of this layer are not perturbed
                node.interval = node.interval_propagate(*inp, C=C)
            else:
                interval_before_C = [node.interval_propagate(*inp)]
                node.interval = BoundLinear.interval_propagate(None, *interval_before_C, C=C)
        else:
            node.interval = node.interval_propagate(*inp)
        
        if self.ibp_relative:
            node.lower = node.interval.lower
            node.upper = node.interval.upper
        else:
            node.lower, node.upper = node.interval
            if isinstance(node.lower, torch.Size):
                node.lower = torch.tensor(node.lower)
                node.interval = (node.lower, node.upper)
            if isinstance(node.upper, torch.Size):
                node.upper = torch.tensor(node.upper)
                node.interval = (node.lower, node.upper)

        return node.interval

    def _backward_general(self, C=None, node=None, root=None, bound_lower=True, bound_upper=True,
                          return_A=False, needed_A_list=None, average_A=False, A_dict=None, return_b=False, b_dict=None, intermediate_constr=None, unstable_idx=None):
        logger.debug('Backward from ({})[{}]'.format(node, node.name))
        _print_time = False

        degree_out = {}
        for l in self._modules.values():
            l.bounded = True
            l.lA = l.uA = None
            degree_out[l.name] = 0
        queue = deque([node])
        all_nodes_before = []
        while len(queue) > 0:
            l = queue.popleft()
            self.backward_from[l.name].append(node)
            for l_pre in l.input_name:
                all_nodes_before.append(l_pre)
                degree_out[l_pre] += 1  # calculate the out degree
                if self._modules[l_pre].bounded:
                    self._modules[l_pre].bounded = False
                    queue.append(self._modules[l_pre])
        node.bounded = True
        if isinstance(C, Patches):
            batch_size, L, out_c = C.shape[:3]
            output_dim = L * out_c
        else:
            batch_size, output_dim = C.shape[:2]
            
        if not isinstance(C, (eyeC, Patches, OneHotC)):
            C = C.transpose(0, 1)
        elif isinstance(C, (eyeC, OneHotC)):
            C = C._replace(shape=(C.shape[1], C.shape[0], C.shape[2]))

        node.lA = C if bound_lower else None
        node.uA = C if bound_upper else None
        lb = ub = torch.tensor(0., device=self.device)

        beta_watch_list = defaultdict(dict)
        if intermediate_constr is not None:
            # Intermediate layer betas are handled in two cases.
            # First, if the beta split is before this node, we don't need to do anything special; it will done in BoundRelu.
            # Second, if the beta split after this node, we need to modify the A matrix during bound propagation to reflect beta after this layer.
            for k in intermediate_constr:
                if k not in all_nodes_before:
                    # The second case needs special care: we add all such splits in a watch list.
                    # However, after first occurance of a layer in the watchlist, beta_watch_list will be deleted and the A matrix from split constraints has been added and will be propagated to later layers.
                    for kk, vv in intermediate_constr[k].items():
                        beta_watch_list[kk][k] = vv

        queue = deque([node])
        A_record = {}
        while len(queue) > 0:
            l = queue.popleft()  # backward from l
            l.bounded = True

            if return_b:
                b_dict[l.name] = {
                    'lower_b': lb,
                    'upper_b': ub
                }

            if l.name in self.root_name or l == root: continue

            for l_pre in l.input_name:  # if all the succeeds are done, then we can turn to this node in the next iteration.
                _l = self._modules[l_pre]
                degree_out[l_pre] -= 1
                if degree_out[l_pre] == 0:
                    queue.append(_l)

            # Initially, l.lA or l.uA will be set to C for this node.
            if l.lA is not None or l.uA is not None:
                def bound_add(A, B):
                    if type(A) == torch.Tensor and type(A) == torch.Tensor:
                        return A + B
                    elif type(A) == Patches and type(B) == Patches:
                        # Here we have to merge two patches, and if A.stride != B.stride, the patches will become a matrix, 
                        # in this case, we will avoid using this mode
                        assert A.stride == B.stride, "A.stride should be the same as B.stride, otherwise, please use the matrix mode"

                        # change paddings to merge the two patches
                        if A.padding != B.padding:
                            if A.padding > B.padding:
                                B = B._replace(patches=F.pad(B.patches, (
                                    A.padding - B.padding, A.padding - B.padding, A.padding - B.padding,
                                    A.padding - B.padding)))
                            else:
                                A = A._replace(patches=F.pad(A.patches, (
                                    B.padding - A.padding, B.padding - A.padding, B.padding - A.padding,
                                    B.padding - A.padding)))
                        sum_ret = A.patches + B.patches
                        return Patches(sum_ret, B.stride, max(A.padding, B.padding), sum_ret.shape)
                    else:
                        if type(A) == Patches:
                            pieces = A.patches
                            stride = A.stride
                            padding = A.padding
                            input_shape = B.shape[2:]
                            matrix = B
                        if type(B) == Patches:
                            pieces = B.patches
                            stride = B.stride
                            padding = B.padding
                            input_shape = A.shape[2:]
                            matrix = A

                        A_matrix = patchesToMatrix(pieces, input_shape, stride, padding)
                        return A_matrix.transpose(0,1) + matrix

                def add_bound(node, lA, uA):
                    if lA is not None:
                        node.lA = lA if node.lA is None else bound_add(node.lA, lA)
                    if uA is not None:
                        node.uA = uA if node.uA is None else bound_add(node.uA, uA)

                # TODO can we just use l.inputs?
                input_nodes = [self._modules[l_name] for l_name in l.input_name]
                if _print_time:
                    start_time = time.time()

                # FIXME make fixed nodes have fixed `forward_value` that is never cleaned out
                if not l.perturbed and hasattr(l, 'forward_value'):
                    lb = lb + l.get_bias(l.lA, l.forward_value)
                    ub = ub + l.get_bias(l.uA, l.forward_value)
                    continue

                if not Benchmarking:
                    small_A = 0
                    if l.lA is not None and not isinstance(l.lA, (eyeC, OneHotC, Patches)) and torch.norm(l.lA, p=1) < epsilon:  # FIXME: Why need this???
                        small_A += 1
                    if l.uA is not None and not isinstance(l.uA, (eyeC, OneHotC, Patches)) and torch.norm(l.uA, p=1) < epsilon:
                        small_A += 1
                    if small_A == 2:
                        continue

                    small_A = 0
                    if isinstance(l.lA, Patches) and l.lA.identity == 0 and torch.norm(l.lA.patches, p=1) < epsilon:
                        small_A += 1
                    if isinstance(l.lA, Patches) and l.uA.identity == 0 and torch.norm(l.uA.patches, p=1) < epsilon:
                        small_A += 1
                    if small_A == 2:
                        continue

                if isinstance(l, BoundRelu):
                    A, lower_b, upper_b = l.bound_backward(l.lA, l.uA, *input_nodes, start_node=node, unstable_idx=unstable_idx,
                                                           beta_for_intermediate_layers=intermediate_constr is not None)  # TODO: unify this interface.
                elif isinstance(l, BoundOptimizableActivation):
                    A, lower_b, upper_b = l.bound_backward(l.lA, l.uA, *input_nodes, 
                    start_shape=(prod(node.output_shape[1:]) if node.name != self.final_name
                        else C.shape[0]), start_node=node)
                else:
                    A, lower_b, upper_b = l.bound_backward(l.lA, l.uA, *input_nodes)
                if _print_time:
                    time_elapsed = time.time() - start_time
                    if time_elapsed > 1e-3:
                        print(l, time_elapsed)
                if lb.ndim > 0 and type(lower_b) == torch.Tensor and self.conv_mode == 'patches':
                    if lower_b.ndim < lb.ndim:
                        lb = lb.transpose(0,1)
                        lb = lb.reshape(lb.size(0), -1).transpose(0,1)

                        ub = ub.transpose(0,1)
                        ub = ub.reshape(ub.size(0), -1).transpose(0,1)
                    elif lower_b.ndim > lb.ndim:
                        lower_b = lower_b.transpose(0,1)
                        lower_b = lower_b.reshape(lower_b.size(0), -1).transpose(0,1)

                        upper_b = upper_b.transpose(0,1)
                        upper_b = upper_b.reshape(upper_b.size(0), -1).transpose(0,1)
                lb = lb + lower_b
                ub = ub + upper_b
                if return_A:
                    if isinstance(self._modules[l.output_name[0]], BoundRelu):
                        # We save the A matrices after propagating through layer l if a ReLU follows l. Todo: support a general filter to choose which A to save.
                        # Here we saved the A *after* propagating backwards through this layer.
                        # Note that we return the accumulated bias terms, to maintain linear relationship of this node (TODO: does not support ResNet).
                        A_record.update({l.name: {
                            "lA": A[0][0].transpose(0, 1) if A[0][0] is not None else None,
                            "uA": A[0][1].transpose(0, 1) if A[0][1] is not None else None,
                            "lbias": lb.transpose(0, 1) if lb.ndim > 1 else None,
                            # When not used, lb or ub is tensor(0).
                            "ubias": ub.transpose(0, 1) if ub.ndim > 1 else None,
                        }
                        })
                        if set(needed_A_list) == set(A_record.keys()):
                            # We have collected all A matrices we need. We can return now!
                            A_dict.update({node.name: A_record})
                            # Do not concretize to save time. We just need the A matrices.
                            return None, None, A_dict

                # return A matrix as a dict: {node.name: [A_lower, A_upper]}

                # Check if we have any beta split from a layer after the starting node.
                if l.name in beta_watch_list:
                    # Contribution to the constant term.
                    intermediate_beta_lb = intermediate_beta_ub = 0.0
                    # Contribution to the linear coefficients.
                    intermediate_beta_lA = intermediate_beta_uA = 0.0
                    for split_layer_name, split_coeffs in beta_watch_list[l.name].items():
                        # We may have multiple splits after this layer, and they are enumerated here.
                        # Find the corresponding beta variable, which is in a later layer.
                        # split_layer_name should be the pre-act layer name, and Relu is after it.
                        split_layer = self._modules[self._modules[split_layer_name].output_name[0]]
                        # Concat betas from history split and current split.
                        all_betas_lb = all_betas_ub = None
                        if split_layer.history_beta_used:
                            # The beta has size [batch, *node_shape, n_beta] (i.e., each element, each neuron, has a different beta)
                            all_betas_lb = split_layer.history_intermediate_betas[l.name]["lb"]
                            all_betas_lb = all_betas_lb.view(all_betas_lb.size(0), -1, all_betas_lb.size(-1))
                            all_betas_ub = split_layer.history_intermediate_betas[l.name]["ub"]
                            all_betas_ub = all_betas_ub.view(all_betas_ub.size(0), -1, all_betas_ub.size(-1))
                            if unstable_idx is not None:
                                # Only unstable neuron is considered.
                                all_betas_lb = self.non_deter_index_select(all_betas_lb, index=unstable_idx, dim=1)
                                all_betas_ub = self.non_deter_index_select(all_betas_ub, index=unstable_idx, dim=1)
                        if split_layer.split_beta_used:
                            # Note: we must keep split_intermediate_betas at the last because this is the way we build split_coeffs.
                            split_intermediate_betas_lb = split_layer.split_intermediate_betas[l.name]["lb"]
                            split_intermediate_betas_ub = split_layer.split_intermediate_betas[l.name]["ub"]
                            split_intermediate_betas_lb = split_intermediate_betas_lb.view(split_intermediate_betas_lb.size(0), -1, split_intermediate_betas_lb.size(-1))
                            split_intermediate_betas_ub = split_intermediate_betas_ub.view(split_intermediate_betas_ub.size(0), -1, split_intermediate_betas_ub.size(-1))
                            if unstable_idx is not None:
                                split_intermediate_betas_lb = self.non_deter_index_select(split_intermediate_betas_lb, index=unstable_idx, dim=1)
                                split_intermediate_betas_ub = self.non_deter_index_select(split_intermediate_betas_ub, index=unstable_idx, dim=1)
                            all_betas_lb = torch.cat((all_betas_lb, split_intermediate_betas_lb), dim=-1) if all_betas_lb is not None else split_intermediate_betas_lb
                            all_betas_ub = torch.cat((all_betas_ub, split_intermediate_betas_ub), dim=-1) if all_betas_ub is not None else split_intermediate_betas_ub
                        if split_layer.single_beta_used:
                            single_intermediate_betas_lb = split_layer.single_intermediate_betas[l.name]["lb"]
                            single_intermediate_betas_ub = split_layer.single_intermediate_betas[l.name]["ub"]
                            single_intermediate_betas_lb = single_intermediate_betas_lb.view(single_intermediate_betas_lb.size(0), -1, single_intermediate_betas_lb.size(-1))
                            single_intermediate_betas_ub = single_intermediate_betas_ub.view(single_intermediate_betas_ub.size(0), -1, single_intermediate_betas_ub.size(-1))
                            if unstable_idx is not None:
                                single_intermediate_betas_lb = self.non_deter_index_select(single_intermediate_betas_lb, index=unstable_idx, dim=1)
                                single_intermediate_betas_ub = self.non_deter_index_select(single_intermediate_betas_ub, index=unstable_idx, dim=1)
                            all_betas_lb = torch.cat((all_betas_lb, single_intermediate_betas_lb), dim=-1) if all_betas_lb is not None else single_intermediate_betas_lb
                            all_betas_ub = torch.cat((all_betas_ub, single_intermediate_betas_ub), dim=-1) if all_betas_ub is not None else single_intermediate_betas_ub
                        # beta has been reshaped to to [batch, prod(node_shape), n_beta] , prod(node_shape) is the spec dimension for A.
                        # print(f'Add beta to {l.name} with shape {all_betas_lb.size()}, A shape {A[0][0].size()}, split_coeffs A {split_coeffs["lA"].size()}, split_coeffs bias {split_coeffs["lbias"].size()}')
                        # Constant terms from beta related Lagrangian. split_coeffs['lbias'] is in shape [batch, n_beta].
                        # We got shape [batch, *node_shape], which corresponds to the bias terms caused by beta per batch element per intermediate neuron.
                        intermediate_beta_lb = intermediate_beta_lb + torch.einsum('ijb,ib->ij', all_betas_lb,
                                                                                   split_coeffs['lbias'])
                        intermediate_beta_ub = intermediate_beta_ub + torch.einsum('ijb,ib->ij', all_betas_ub,
                                                                                   split_coeffs['ubias'])
                        # A coefficients from beta related Lagrangian. split_coeffs['lA'] is in shape [batch, n_beta, *preact_shape].
                        # We got shape [batch, prod(node_shape), *preact_shape].
                        # print(f'BEFORE {node.name} split layer {split_layer.name} l {l.name} {torch.tensor(intermediate_beta_lA).abs().sum()} {torch.tensor(intermediate_beta_uA).abs().sum()} \t {all_betas_lb.abs().sum()} {all_betas_lb.size()} {split_coeffs["lA"].abs().sum()} {split_coeffs["lA"].size()}')
                        intermediate_beta_lA = intermediate_beta_lA + torch.einsum('ijb,ib...->ij...', all_betas_lb,
                                                                                   split_coeffs['lA'])
                        intermediate_beta_uA = intermediate_beta_uA + torch.einsum('ijb,ib...->ij...', all_betas_ub,
                                                                                   split_coeffs['uA'])
                        # print(f'AFTER  {node.name} split layer {split_layer.name} l {l.name} {torch.tensor(intermediate_beta_lA).abs().sum()} {torch.tensor(intermediate_beta_uA).abs().sum()} \t {all_betas_ub.abs().sum()} {all_betas_lb.size()} {split_coeffs["uA"].abs().sum()} {split_coeffs["uA"].size()}')
                    # Finished adding beta splits from all layers. Now merge them into the A matrix of this layer.
                    # Our A has spec dimension at the front, so a transpose is needed.
                    A[0] = (A[0][0] + intermediate_beta_lA.transpose(0, 1), A[0][1] + intermediate_beta_uA.transpose(0, 1))
                    lb += intermediate_beta_lb.transpose(0, 1)
                    ub += intermediate_beta_ub.transpose(0, 1)
                    # Only need to add the first encountered. Set the watch list to empty.
                    beta_watch_list = {}

                for i, l_pre in enumerate(l.input_name):
                    try:
                        logger.debug('  {} -> {}, uA shape {}'.format(l.name, l_pre, A[i][1].shape))
                    except:
                        pass
                    _l = self._modules[l_pre]
                    add_bound(_l, lA=A[i][0], uA=A[i][1])

        if lb.ndim >= 2:
            lb = lb.transpose(0, 1)
        if ub.ndim >= 2:
            ub = ub.transpose(0, 1)
        output_shape = node.output_shape[1:]
        if np.prod(node.output_shape[1:]) != output_dim and type(C) != Patches:
            output_shape = [-1]

        if return_A:
            # # return A matrix as a dict: {node.name: [A_lower, A_upper]}
            # this_A_dict = {'bias': [lb.detach(), ub.detach()]}
            # for i in range(len(root)):
            #     if root[i].lA is None and root[i].uA is None: continue
            #     this_A_dict.update({root[i].name: [root[i].lA, root[i].uA]})
            # this_A_dict.update(A_record)
            # A_dict.update({node.name: this_A_dict})
            A_dict.update({node.name: A_record})

        for i in range(len(root)):
            if root[i].lA is None and root[i].uA is None: continue
            # FIXME maybe this one is broken after moving the output dimension to the first
            if average_A and isinstance(root[i], BoundParams):
                A_shape = root[i].lA.shape if bound_lower else root[i].uA.shape
                lA = root[i].lA.mean(0, keepdim=True).repeat(A_shape[0],
                                                             *[1] * len(A_shape[1:])) if bound_lower else None
                uA = root[i].uA.mean(0, keepdim=True).repeat(A_shape[0],
                                                             *[1] * len(A_shape[1:])) if bound_upper else None
            else:
                lA = root[i].lA
                uA = root[i].uA
                    
            if not isinstance(root[i].lA, eyeC) and not isinstance(root[i].lA, Patches):
                lA = root[i].lA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_lower else None
            if not isinstance(root[i].uA, eyeC) and not isinstance(root[i].lA, Patches):
                uA = root[i].uA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_upper else None
            if hasattr(root[i], 'perturbation') and root[i].perturbation is not None:
                if isinstance(root[i], BoundParams):
                    # add batch_size dim for weights node
                    lb = lb + root[i].perturbation.concretize(
                        root[i].center.unsqueeze(0), lA,
                        sign=-1, aux=root[i].aux) if bound_lower else None
                    ub = ub + root[i].perturbation.concretize(
                        root[i].center.unsqueeze(0), uA,
                        sign=+1, aux=root[i].aux) if bound_upper else None
                else:
                    lb = lb + root[i].perturbation.concretize(root[i].center, lA, sign=-1,
                                                              aux=root[i].aux) if bound_lower else None
                    ub = ub + root[i].perturbation.concretize(root[i].center, uA, sign=+1,
                                                              aux=root[i].aux) if bound_upper else None
            # FIXME to simplify
            elif i < self.num_global_inputs:
                if not isinstance(lA, eyeC):
                    lb = lb + lA.bmm(root[i].forward_value.view(batch_size, -1, 1)).squeeze(-1) if bound_lower else None
                else:
                    lb = lb + root[i].forward_value.view(batch_size, -1) if bound_lower else None
                if not isinstance(uA, eyeC):
                    # FIXME looks questionable
                    ub = ub + uA.bmm(root[i].forward_value.view(batch_size, -1, 1)).squeeze(-1) if bound_upper else None
                else:
                    ub = ub + root[i].forward_value.view(batch_size, -1) if bound_upper else None
            else:
                if isinstance(lA, eyeC):
                    lb = lb + root[i].forward_value.view(1, -1) if bound_lower else None
                else:
                    lb = lb + lA.matmul(root[i].forward_value.view(-1, 1)).squeeze(-1) if bound_lower else None                    
                if isinstance(uA, eyeC):
                    ub = ub + root[i].forward_value.view(1, -1) if bound_upper else None
                else:
                    ub = ub + uA.matmul(root[i].forward_value.view(-1, 1)).squeeze(-1) if bound_upper else None
        node.lower = lb.view(batch_size, *output_shape) if bound_lower else None
        node.upper = ub.view(batch_size, *output_shape) if bound_upper else None

        if return_A: return node.lower, node.upper, A_dict
        return node.lower, node.upper

    def _forward_general(self, C=None, node=None, root=None, dim_in=None, concretize=False):
        if hasattr(node, 'lower'):
            return node.lower, node.upper

        if not node.from_input:
            w = None
            b = node.value
            node.linear = LinearBound(w, b, w, b, b, b)
            node.lower = node.upper = b
            node.interval = (node.lower, node.upper)
            return node.interval

        if not hasattr(node, 'linear'):
            for l_pre in node.input_name:
                l = self._modules[l_pre]
                if not hasattr(l, 'linear'):
                    self._forward_general(node=l, root=root, dim_in=dim_in)

            inp = [self._modules[l_pre].linear for l_pre in node.input_name]

            if C is not None and isinstance(node, BoundLinear) and not node.is_input_perturbed(1):
                node.linear = node.bound_forward(dim_in, *inp, C=C)
                C_merged = True
            else:
                node.linear = node.bound_forward(dim_in, *inp)
                C_merged = False

            lw, uw = node.linear.lw, node.linear.uw
            lower, upper = node.linear.lb, node.linear.ub

            if C is not None and not C_merged:
                # FIXME use bound_forward of BoundLinear
                C_pos, C_neg = C.clamp(min=0), C.clamp(max=0)
                _lw = torch.matmul(lw, C_pos.transpose(-1, -2)) + torch.matmul(uw, C_neg.transpose(-1, -2))
                _uw = torch.matmul(uw, C_pos.transpose(-1, -2)) + torch.matmul(lw, C_neg.transpose(-1, -2))
                lw, uw = _lw, _uw
                _lower = torch.matmul(lower.unsqueeze(1), C_pos.transpose(-1, -2)) + \
                         torch.matmul(upper.unsqueeze(1), C_neg.transpose(-1, -2))
                _upper = torch.matmul(upper.unsqueeze(1), C_pos.transpose(-1, -2)) + \
                         torch.matmul(lower.unsqueeze(1), C_neg.transpose(-1, -2))
                lower, upper = _lower.squeeze(1), _upper.squeeze(1)
            # FIXME looks useless?
            # else:
            #     lower, upper = lower.squeeze(1), upper.squeeze(1)
        else:
            lw, uw = node.linear.lw, node.linear.uw
            lower, upper = node.linear.lb, node.linear.ub

        if concretize:
            if node.linear.lw is not None:
                prev_dim_in = 0
                batch_size = lw.shape[0]
                assert (lw.ndim > 1)
                lA = lw.reshape(batch_size, dim_in, -1).transpose(1, 2)
                uA = uw.reshape(batch_size, dim_in, -1).transpose(1, 2)
                for i in range(len(root)):
                    if hasattr(root[i], 'perturbation') and root[i].perturbation is not None:
                        _lA = lA[:, :, prev_dim_in : (prev_dim_in + root[i].dim)]
                        _uA = uA[:, :, prev_dim_in : (prev_dim_in + root[i].dim)]
                        lower = lower + root[i].perturbation.concretize(
                            root[i].center, _lA, sign=-1, aux=root[i].aux).view(lower.shape)
                        upper = upper + root[i].perturbation.concretize(
                            root[i].center, _uA, sign=+1, aux=root[i].aux).view(upper.shape)
                        prev_dim_in += root[i].dim
                if C is None:
                    node.linear = node.linear._replace(lower=lower, upper=upper)
            if C is None:
                node.lower, node.upper = lower, upper
            if not Benchmarking and torch.isnan(lower).any():
                import pdb
                pdb.set_trace()
            return lower, upper

    def _init_forward(self, root, dim_in):
        if dim_in == 0:
            raise ValueError("At least one node should have a specified perturbation")
        prev_dim_in = 0
        # Assumption: root[0] is the input node which implies batch_size
        batch_size = root[0].value.shape[0]
        for i in range(len(root)):
            if hasattr(root[i], 'perturbation') and root[i].perturbation is not None:
                shape = root[i].linear.lw.shape
                device = root[i].linear.lw.device
                dtype = root[i].linear.lw.dtype
                root[i].linear = root[i].linear._replace(
                    lw=torch.cat([
                        torch.zeros(shape[0], prev_dim_in, *shape[2:], device=device, dtype=dtype),
                        root[i].linear.lw,
                        torch.zeros(shape[0], dim_in - shape[1], *shape[2:], device=device, dtype=dtype)
                    ], dim=1),
                    uw=torch.cat([
                        torch.zeros(shape[0], prev_dim_in, *shape[2:], device=device, dtype=dtype),
                        root[i].linear.uw,
                        torch.zeros(shape[0], dim_in - shape[1] - prev_dim_in, *shape[2:], device=device, dtype=dtype)
                    ], dim=1)
                )
                if i >= self.num_global_inputs:
                    root[i].forward_value = root[i].forward_value.unsqueeze(0).repeat(
                        *([batch_size] + [1] * self.forward_value.ndim))
                prev_dim_in += shape[1]
            else:
                fv = root[i].forward_value
                shape = fv.shape
                if root[i].from_input:
                    w = torch.zeros(shape[0], dim_in, *shape[1:], device=self.device)
                else:
                    w = None
                b = fv
                root[i].linear = LinearBound(w, b, w, b, b, b)
                root[i].lower = root[i].upper = b
                root[i].interval = (root[i].lower, root[i].upper)

    """Add perturbation to an intermediate node and it is treated as an independent 
    node in bound computation."""

    def add_intermediate_perturbation(self, node, perturbation):
        node.perturbation = perturbation
        node.perturbed = True
        # NOTE This change is currently inreversible
        if not node.name in self.root_name:
            self.root_name.append(node.name)


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
                ptb = PerturbationLpNorm(norm=inputs[0].ptb.norm, eps=inputs[0].ptb.eps, x_L=input_s[1], x_U=input_s[2])
                # bounded_inputs.append(tuple([(BoundedTensor(input_s[0][0], ptb))]))
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
