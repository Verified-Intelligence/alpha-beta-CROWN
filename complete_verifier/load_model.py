#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENSE file in the repository root.       ##
##                                                                     ##
#########################################################################
import hashlib
import time
from typing import Optional, List
import os
import collections
import gzip
from ast import literal_eval
import torch
import numpy as np

import onnx2pytorch
import onnx
import onnxruntime as ort
import onnxoptimizer
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from onnx_opt import compress_onnx

import warnings
import importlib
from functools import partial

import arguments

from model_defs import *
from utils import expand_path


def Customized(def_file, callable_name, *args, **kwargs):
    """Fully customized model or dataloader."""
    if def_file.endswith('.py'):
        # Use relatively path w.r.t. to the configuration file
        if arguments.Config['general']['root_path']:
            path = os.path.join(
                expand_path(arguments.Config['general']['root_path']), def_file)
        elif arguments.Config.file:
            path = os.path.join(os.path.dirname(arguments.Config.file), def_file)
        else:
            path = def_file
        spec = importlib.util.spec_from_file_location('customized', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        try:
            # Customized loaders should be in "custom" folder.
            module = importlib.import_module(f'custom.{def_file}')
        except ModuleNotFoundError:
            # If not found, try the current folder.
            module = importlib.import_module(f'{def_file}')
            warnings.warn(  # Old config files may refer to custom loaders in the root folder.
                    f'Customized loaders "{def_file}" should be inside the "custom" folder.')
    # Load model from a specified file.
    model_func = getattr(module, callable_name)
    customized_func = partial(model_func, *args, **kwargs)
    # We need to return a Callable which returns the model.
    return customized_func


def deep_update(d, u):
    """Update a dictionary based another dictionary, recursively.

    (https://stackoverflow.com/a/3233356).
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def unzip_and_optimize_onnx(path, onnx_optimization_flags: Optional[List[str]] = None):
    if not onnx_optimization_flags:
        if path.endswith('.gz'):
            onnx_model = onnx.load(gzip.GzipFile(path))
        else:
            onnx_model = onnx.load(path)
        return onnx_model
    else:
        print(f'Onnx optimization with flags: {onnx_optimization_flags}')
        npath = path + '.optimized'
        if os.path.exists(npath):
            print(f'Found existed optimized onnx model at {npath}')
            return onnx.load(npath)
        else:
            print(f'Generate optimized onnx model to {npath}')
            if path.endswith('.gz'):
                onnx_model = onnx.load(gzip.GzipFile(path))
            else:
                onnx_model = onnx.load(path)
            try:   # Capture errors during ONNX optimization, skip errors, and display warnings when failures occur.
                return compress_onnx(onnx_model, path, npath, onnx_optimization_flags, debug=True)
            except Exception as e:
                warnings.warn(f"ONNX optimization failed with flags {onnx_optimization_flags}. Skipping optimization.\nError: {e}")
                return onnx_model  # Fallback to unoptimized model


def _run_onnx_outputs(path, inputs):
    # Workaround for onnx bug, see issue #150
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    sess = ort.InferenceSession(
        unzip_and_optimize_onnx(path).SerializeToString(),
        sess_options=options)

    input_infos = sess.get_inputs()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    assert len(input_infos) == len(inputs)
    feed_dict = {}
    for input_info, input_value in zip(input_infos, inputs):
        if "float" in input_info.type:
            # ONNX input is float, convert to float32
            input_value = input_value.astype(np.float32)
        elif "double" in input_info.type:
            # ONNX input is double, convert to float64
            input_value = input_value.astype(np.float64)
        feed_dict[input_info.name] = input_value
    return sess.run(None, feed_dict)


def inference_onnx(path, input):
    outputs = _run_onnx_outputs(path, input)
    assert len(outputs) == 1
    return outputs[0]


def export_ort_optimized_model(path, output_path=None):
    """Export the ONNX Runtime EXTENDED-optimized graph for ``path``."""
    if output_path is None:
        stem, ext = os.path.splitext(path)
        output_path = f'{stem}.abcrown_ort_extended{ext or ".onnx"}'
    if (os.path.exists(output_path)
            and os.path.getmtime(output_path) >= os.path.getmtime(path)):
        return output_path

    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
    options.optimized_model_filepath = output_path
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    ort.InferenceSession(
        path, sess_options=options, providers=['CPUExecutionProvider'])
    return output_path


def _remove_batch_dim(shape):
    """Return the shape used by verifier after removing a leading batch dim."""
    shape = tuple(shape)
    # ONNX shapes usually include a leading batch dimension. A leading 0 or 1
    # is treated as batch and removed. For nn4sys pensieve parallel, the first
    # dimension can be larger than 1 and is part of the input, so keep it.
    if shape and shape[0] <= 1:
        return shape[1:]
    return shape


def _onnx_feed_input_shapes(onnx_model):
    """Return per-example shapes for non-initializer ONNX graph inputs."""
    # Find real ONNX inputs by excluding initializers from graph.input.
    # https://github.com/onnx/onnx/issues/2657
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    feed_names = set(input_all) - set(input_initializer)
    feed_inputs = [node for node in onnx_model.graph.input if node.name in feed_names]

    if not feed_inputs:
        # If no feed input is found, use the first graph input to get a shape.
        feed_inputs = [onnx_model.graph.input[0]]

    return [
        _remove_batch_dim(tuple(
            d.dim_value for d in feed_input.type.tensor_type.shape.dim))
        for feed_input in feed_inputs
    ]


def _forward_pytorch_model(model, inputs):
    if len(inputs) == 1:
        output = model(inputs[0])
    else:
        output = model(*inputs)
    if isinstance(output, (list, tuple)):
        return list(output)
    return [output]


def _print_conversion_mismatch(output_pytorch, output_onnx):
    print('\n**************************')
    print('Model might not be converted correctly. Please check onnx conversion carefully.')
    print('Output by pytorch:', output_pytorch)
    print('Output by onnx:', output_onnx)
    diff = torch.tensor(output_pytorch - output_onnx).abs().reshape(-1)
    print('Max error:', diff.max())
    index = diff.argmax()
    print('Max error index:', diff.argmax())
    print(f'Output by pytorch at {index}: ',
          torch.tensor(output_pytorch).reshape(-1)[index])
    print(f'Output by onnx at {index}: ',
          torch.tensor(output_onnx).reshape(-1)[index])
    print('**************************\n')


def check_onnx_conversion(
        pytorch_model, model_path, input_shapes, x=None,
        onnx_optimization_flags=None, onnx_model=None):
    conversion_check_result = True
    try:
        # FIXME dtype of dummy may not match the onnx model, which can cause
        # runtime error.
        if x is not None:
            assert len(input_shapes) == 1
            torch_inputs = [x.reshape(1, *input_shapes[0])]
        else:
            torch_inputs = [
                torch.randn([1, *input_shape])
                for input_shape in input_shapes
            ]
        output_pytorch = [
            output.detach().cpu().numpy()
            for output in _forward_pytorch_model(pytorch_model, torch_inputs)
        ]
        onnx_inputs = [
            torch_input.detach().cpu().numpy()
            for torch_input in torch_inputs
        ]
        try:
            output_onnx = _run_onnx_outputs(model_path, onnx_inputs)
        except ort.capi.onnxruntime_pybind11_state.InvalidArgument:
            if len(torch_inputs) != 1:
                raise
            # ONNX model might have shape problems. Remove the batch dimension
            # and try again.
            try:
                output_onnx = _run_onnx_outputs(
                    model_path, onnx_inputs[0].squeeze(0))
            except Exception:
                try:
                    # Still failing: try reshaping the input to
                    # (batch, flattened_dim, 1).
                    reshaped_input = torch_inputs[0].flatten(1).unsqueeze(-1)
                    output_onnx = _run_onnx_outputs(
                        model_path, reshaped_input.cpu().numpy())
                except Exception as e:
                    print(
                        f'Failed to run inference on ONNX model {model_path} '
                        f'with input shape {torch_inputs[0].shape}')
                    print('Please check the ONNX model and input shape.')
                    raise e

        if onnx_optimization_flags is not None and (
                'remove_relu_in_last_layer' in onnx_optimization_flags):
            output_pytorch = [
                output.clip(min=0) for output in output_pytorch
            ]
        conversion_check_result = (
            len(output_pytorch) == len(output_onnx)
            and all(
                np.allclose(torch_output, onnx_output, 1e-4, 1e-5)
                for torch_output, onnx_output in zip(
                    output_pytorch, output_onnx)
            )
        )
    except:  # pylint: disable=broad-except
        warnings.warn('Not able to check model\'s conversion correctness')
        print('\n*************Error traceback*************')
        import traceback; print(traceback.format_exc())
        print('*****************************************\n')
    if not conversion_check_result:
        _print_conversion_mismatch(output_pytorch[0], output_onnx[0])

        if (onnx_model is not None
                and arguments.Config["model"]["debug_onnx"]
                and len(torch_inputs) == 1):
            debug_onnx(onnx_model, pytorch_model, onnx_inputs[0])
    return conversion_check_result


@torch.no_grad()
def load_model_onnx(path, quirks=None, x=None):
    start_time = time.time()
    use_ort_optimized_model = arguments.Config["model"]["use_ort_optimized_model"]
    onnx_optimization_flags = arguments.Config['model']['onnx_optimization_flags']

    if use_ort_optimized_model:
        # Convert the graph exported by ONNX Runtime, and use that same graph
        # as the ONNX Runtime reference for conversion checking.
        conversion_onnx_path = export_ort_optimized_model(path)
        reference_onnx_path = conversion_onnx_path
    else:
        # Convert the original ONNX graph. The reference graph can be the
        # separately optimized file when the existing check_optimized option is set.
        conversion_onnx_path = path
        reference_onnx_path = (
            path + '.optimized'
            if arguments.Config['model']['check_optimized'] else path
        )

    if arguments.Config['model']['cache_onnx_conversion']:
        cached_onnx_suffix = ".cached"
        cached_onnx_filename = f'{conversion_onnx_path}{cached_onnx_suffix}'

        with open(conversion_onnx_path, "rb") as file:
            curfile_sha256 = hashlib.sha256(file.read()).hexdigest()

        if os.path.exists(cached_onnx_filename):
            print(f'Loading cached onnx model from {cached_onnx_filename}')
            read_error = False
            try:
                pytorch_model, onnx_shape, old_file_sha256 = torch.load(cached_onnx_filename, weights_only=False)
            except (Exception, ValueError, EOFError) as e:
                # If the cached file is corrupted or not compatible, we will regenerate it.
                print(f"Error reading cached onnx file {cached_onnx_filename}: {e}")
                read_error = True
            if not read_error:
                if curfile_sha256 == old_file_sha256:
                    pytorch_model.eval()
                    pytorch_model.to(dtype=torch.get_default_dtype())
                    end_time = time.time()
                    print(f'Cached converted model loaded in {end_time - start_time:.4f} seconds')
                    return pytorch_model, onnx_shape
                else:
                    print(f"{cached_onnx_filename} file sha256: {curfile_sha256} does not match the current onnx sha256: {old_file_sha256}. Regenerating...")
        else:
            print(f"{cached_onnx_filename} does not exist.")

    quirks = {} if quirks is None else quirks
    if arguments.Config['model']['onnx_quirks']:
        try:
            config_quirks = literal_eval(arguments.Config['model']['onnx_quirks'])
        except ValueError:
            print('ERROR: onnx_quirks '
                  f'{arguments.Config["model"]["onnx_quirks"]}'
                  'cannot be parsed!')
            raise
        assert isinstance(config_quirks, dict)
        deep_update(quirks, config_quirks)

    print(f'Loading onnx {conversion_onnx_path} with quirks {quirks}')
    onnx_model = unzip_and_optimize_onnx(
        conversion_onnx_path, onnx_optimization_flags)

    if arguments.Config["model"]["input_shape"] is None:
        onnx_input_shapes = _onnx_feed_input_shapes(onnx_model)
    else:
        # User specify input_shape
        onnx_input_shapes = [
            _remove_batch_dim(arguments.Config['model']['input_shape'])]
    # load_model_onnx returns one input shape. Conversion checking below uses
    # all ONNX feed shapes; VNN-LIB 2 loaders use metadata for multi-input layout.
    onnx_shape = onnx_input_shapes[0]

    try:
        pytorch_model = onnx2pytorch.ConvertModel(
            onnx_model, experimental=True, quirks=quirks)
    except TypeError as e:
        print('\n\nA possible onnx2pytorch version error!')
        print('If you see "unexpected keyword argument \'quirks\'", that indicates your onnx2pytorch version is incompatible.')
        print('Please uninstall onnx2pytorch in your python environment (e.g., run "pip uninstall onnx2pytorch"), and then reinstall using:\n')
        print('pip install git+https://github.com/Verified-Intelligence/onnx2pytorch@bae71f32f65534b7b13aa3a40d797919a94ace45\n\n')
        print('The error below may not a bug of alpha-beta-CROWN. See instructions above.')
        raise(e)
    pytorch_model.eval()
    pytorch_model.to(dtype=torch.get_default_dtype())

    check_onnx_conversion(
        pytorch_model, reference_onnx_path, onnx_input_shapes, x=x,
        onnx_optimization_flags=onnx_optimization_flags,
        onnx_model=onnx_model)

    # TODO merge into the unzip_and_optimize_onnx()
    if arguments.Config["model"]["flatten_final_output"]:
        pytorch_model = nn.Sequential(pytorch_model, nn.Flatten())

    if arguments.Config["model"]["cache_onnx_conversion"]:
        torch.save((pytorch_model, onnx_shape, curfile_sha256), cached_onnx_filename)

    end_time = time.time()
    print(f'Finished onnx model loading in {end_time - start_time:.4f} seconds')

    return pytorch_model, onnx_shape


def debug_onnx(onnx_model, pytorch_model, dummy_input):
    path_tmp = '/tmp/debug.onnx'

    output_onnx = {}
    for node in enumerate_model_node_outputs(onnx_model):
        print('Inferencing onnx node:', node)
        save_onnx_model(select_model_inputs_outputs(onnx_model, node), path_tmp)
        optimized_model = onnxoptimizer.optimize(
            onnx.load(path_tmp),
            ["extract_constant_to_initializer",
             "eliminate_unused_initializer"])
        sess = ort.InferenceSession(optimized_model.SerializeToString())
        output_onnx[node] = torch.tensor(sess.run(
            None, {sess.get_inputs()[0].name: dummy_input})[0])

    print('Inferencing the pytorch model')
    output_pytorch = pytorch_model(
        torch.tensor(dummy_input), return_all_nodes=True)

    for k in output_pytorch:
        if k == sess.get_inputs()[0].name:
            continue
        print(k, output_onnx[k].shape)
        close = torch.allclose(output_onnx[k], output_pytorch[k])
        print('  close?', close)
        if not close:
            print('  max error', (output_onnx[k] - output_pytorch[k]).abs().max())

    import pdb; pdb.set_trace()


def load_model(weights_loaded=True):
    """
    Load the model architectures and weights
    """

    assert arguments.Config["model"]["name"] is None or arguments.Config["model"]["onnx_path"] is None, (
        "Conflict detected! User should specify model path by either --model or --onnx_path! "
        "The cannot be both specified.")

    assert arguments.Config["model"]["name"] is not None or arguments.Config["model"]["onnx_path"] is not None, (
        "No model is loaded, please set --model <modelname> for pytorch model or --onnx_path <filename> for onnx model.")

    if arguments.Config['model']['name'] is not None:
        # You can customize this function to load your own model based on model name.
        try:
            model_ori = eval(arguments.Config['model']['name'])()  # pylint: disable=eval-used
        except Exception:  # pylint: disable=broad-except
            print(f'Cannot load pytorch model definition "{arguments.Config["model"]["name"]}()". '
                  f'"{arguments.Config["model"]["name"]}()" must be a callable that returns a torch.nn.Module object.')
            import traceback
            traceback.print_exc()
            exit()
        model_ori.eval()

        if not weights_loaded:
            return model_ori

        if arguments.Config["model"]["path"] is not None:
            # Load pytorch model
            # You can customize this function to load your own model based on model name.
            sd = torch.load(expand_path(arguments.Config["model"]["path"]),
                            map_location=torch.device('cpu'))
            if 'state_dict' in sd:
                sd = sd['state_dict']
            if isinstance(sd, list):
                sd = sd[0]
            if not isinstance(sd, dict):
                raise NotImplementedError("Unknown model format, please modify model loader yourself.")
            try:
                model_ori.load_state_dict(sd)
            except RuntimeError:
                print('Failed to load the model')
                print('Keys in the state_dict of model_ori:')
                print(list(model_ori.state_dict().keys()))
                print('Keys in the state_dict trying to load:')
                print(list(sd.keys()))
                raise

    elif arguments.Config["model"]["onnx_path"] is not None:
        # Load onnx model
        model_ori, _ = load_model_onnx(expand_path(
            arguments.Config["model"]["onnx_path"]))

    else:
        print("Warning: pretrained model path is not given!")

    print(model_ori)
    print('Parameters:')
    for p in model_ori.named_parameters():
        print(f'  {p[0]}: shape {p[1].shape}')

    return model_ori
