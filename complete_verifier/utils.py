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

import os
import gzip
import collections
import csv
import re
from ast import literal_eval
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import onnx2pytorch
import onnx
import onnxruntime as ort
import arguments
import warnings
from attack_pgd import attack_pgd

# Import all model architectures.
from model_defs import *
from read_vnnlib import read_vnnlib
from onnx_opt import compress_onnx


def reshape_bounds(lower_bounds, upper_bounds, y, global_lb=None):
    with torch.no_grad():
        last_lower_bounds = torch.zeros(size=(1, lower_bounds[-1].size(1)+1), dtype=lower_bounds[-1].dtype, device=lower_bounds[-1].device)
        last_upper_bounds = torch.zeros(size=(1, upper_bounds[-1].size(1)+1), dtype=upper_bounds[-1].dtype, device=upper_bounds[-1].device)
        last_lower_bounds[:, :y] = lower_bounds[-1][:, :y]
        last_lower_bounds[:, y+1:] = lower_bounds[-1][:, y:]
        last_upper_bounds[:, :y] = upper_bounds[-1][:, :y]
        last_upper_bounds[:, y+1:] = upper_bounds[-1][:, y:]
        lower_bounds[-1] = last_lower_bounds
        upper_bounds[-1] = last_upper_bounds
        if global_lb is not None:
            last_global_lb = torch.zeros(size=(1, global_lb.size(1)+1), dtype=global_lb.dtype, device=global_lb.device)
            last_global_lb[:, :y] = global_lb[:, :y]
            last_global_lb[:, y+1:] = global_lb[:, y:]
            global_lb = last_global_lb
    return lower_bounds, upper_bounds, global_lb


def convert_mlp_model(model, dummy_input):
    model.eval()
    feature_maps = {}

    def get_feature_map(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach()

        return hook

    def conv_to_dense(conv, inputs):
        b, n, w, h = inputs.shape
        kernel = conv.weight
        bias = conv.bias
        I = torch.eye(n * w * h).view(n * w * h, n, w, h)
        W = F.conv2d(I, kernel, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
        # input_flat = inputs.view(b, -1)
        b1, n1, w1, h1 = W.shape
        # out = torch.matmul(input_flat, W.view(b1, -1)).view(b, n1, w1, h1)
        new_bias = bias.view(1, n1, 1, 1).repeat(1, 1, w1, h1)

        dense_w = W.view(b1, -1).transpose(1, 0)
        dense_bias = new_bias.view(-1)

        new_m = nn.Linear(in_features=dense_w.shape[1], out_features=dense_w.shape[0], bias=m.bias is not None)
        new_m.weight.data.copy_(dense_w)
        new_m.bias.data.copy_(dense_bias)

        return new_m

    new_modules = []
    modules = list(model.named_modules())[1:]
    for mi, (name, m) in enumerate(modules):

        if mi+1 < len(modules) and isinstance(modules[mi+1][-1], nn.Conv2d):
            m.register_forward_hook(get_feature_map(name))
            model(dummy_input)
            pre_conv_input = feature_maps[name]
        elif mi == 0 and isinstance(m, nn.Conv2d):
            pre_conv_input = dummy_input

        if isinstance(m, nn.Linear):
            new_m = nn.Linear(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
            new_m.weight.data.copy_(m.weight.data)
            new_m.bias.data.copy_(m.bias)
            new_modules.append(new_m)
        elif isinstance(m, nn.ReLU):
            new_modules.append(nn.ReLU())
        elif isinstance(m, nn.Flatten):
            pass
            # will flatten at the first layer
            # new_modules.append(nn.Flatten())
        elif isinstance(m, nn.Conv2d):
            new_modules.append(conv_to_dense(m, pre_conv_input))
        else:
            print(m, 'not support in convert_mlp_model')
            raise NotImplementedError

    #  add flatten at the beginning
    new_modules.insert(0, nn.Flatten())
    seq_model = nn.Sequential(*new_modules)

    return seq_model

def deep_update(d, u):
    """Update a dictionary based another dictionary, recursively (https://stackoverflow.com/a/3233356)."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_pgd_acc(model, X, labels, eps, data_min, data_max, batch_size):
    start = arguments.Config["data"]["start"]
    total = arguments.Config["data"]["end"]
    clean_correct = 0
    robust_correct = 0
    model = model.to(device=arguments.Config["general"]["device"])
    X = X.to(device=arguments.Config["general"]["device"])
    labels = labels.to(device=arguments.Config["general"]["device"])
    if isinstance(data_min, torch.Tensor):
        data_min = data_min.to(device=arguments.Config["general"]["device"])
    if isinstance(data_max, torch.Tensor):
        data_max = data_max.to(device=arguments.Config["general"]["device"])
    if isinstance(eps, torch.Tensor):
        eps = eps.to(device=arguments.Config["general"]["device"])
    if arguments.Config["attack"]["pgd_alpha"] == 'auto':
        alpha = eps.mean() / 4 if isinstance(eps, torch.Tensor) else eps / 4
    else:
        alpha = float(arguments.Config["attack"]["pgd_alpha"])
    while start < total:
        end = min(start + batch_size, total)
        batch_X = X[start:end]
        batch_labels = labels[start:end]
        if arguments.Config["specification"]["type"] == "lp":
            # Linf norm only so far.
            data_ub = torch.min(batch_X + eps, data_max)
            data_lb = torch.max(batch_X - eps, data_min)
        else:
            # Per-example, per-element lower and upper bounds.
            data_ub = data_max[start:end]
            data_lb = data_min[start:end]
        clean_output = model(batch_X)

        best_deltas, last_deltas = attack_pgd(model, X=batch_X, y=batch_labels, epsilon=float("inf"), alpha=alpha,
                num_classes=arguments.Config["data"]["num_outputs"],
                attack_iters=arguments.Config["attack"]["pgd_steps"], num_restarts=arguments.Config["attack"]["pgd_restarts"],
                upper_limit=data_ub, lower_limit=data_lb, multi_targeted=True, lr_decay=arguments.Config["attack"]["pgd_lr_decay"],
                target=None, early_stop=arguments.Config["attack"]["pgd_early_stop"])
        attack_images = torch.max(torch.min(batch_X + best_deltas, data_ub), data_lb)
        attack_output = model(attack_images)
        clean_labels = clean_output.argmax(1)
        attack_labels = attack_output.argmax(1)
        batch_clean_correct = (clean_labels == batch_labels).sum().item()
        batch_robust_correct = (attack_labels == batch_labels).sum().item()
        if start == 0:
            print("Clean prediction for first a few examples:")
            print(clean_output[:10].detach().cpu().numpy())
            print("PGD prediction for first a few examples:")
            print(attack_output[:10].detach().cpu().numpy())
        print(f'batch start {start}, batch size {end - start}, clean correct {batch_clean_correct}, robust correct {batch_robust_correct}')
        clean_correct += batch_clean_correct
        robust_correct += batch_robust_correct
        start += batch_size
        del clean_output, best_deltas, last_deltas, attack_images, attack_output
    print(f'data start {arguments.Config["data"]["start"]} end {total}, clean correct {clean_correct}, robust correct {robust_correct}')
    return clean_correct, robust_correct


def get_test_acc(model, input_shape=None, X=None, labels=None, batch_size=256):
    device = arguments.Config["general"]["device"]
    if X is None and labels is None:
        # Load MNIST or CIFAR, used for quickly debugging.
        database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
        mean = torch.tensor(arguments.Config["data"]["mean"])
        std = torch.tensor(arguments.Config["data"]["std"])
        normalize = transforms.Normalize(mean=mean, std=std)
        if input_shape == (3, 32, 32):
            testset = torchvision.datasets.CIFAR10(root=database_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
        elif input_shape == (1, 28, 28):
            testset = torchvision.datasets.MNIST(root=database_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
        else:
            raise RuntimeError("Unable to determine dataset for test accuracy.")
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        testloader = [(X, labels)]
    total = 0
    correct = 0
    if device != 'cpu':
        model = model.to(device)
    print_first_batch = True
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if device != 'cpu':
                images = images.to(device)
                labels = labels.to(device)
            if arguments.Config["model"]["convert_model_to_NCHW"]:
                images = images.permute(0, 2, 3, 1)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if print_first_batch:
                print_first_batch = False
                for i in range(min(outputs.size(0), 10)):
                    print(f"Image {i} norm {images[i].abs().sum().item()} label {labels[i].item()} correct {labels[i].item() == outputs[i].argmax().item()}\nprediction {outputs[i].cpu().numpy()}")
    print(f'correct {correct} of {total}')


def unzip_and_optimize_onnx(path, onnx_optimization_flags='none'):
    if onnx_optimization_flags == 'none':
        if path.endswith('.gz'):
            onnx_model = onnx.load(gzip.GzipFile(path))
        else:
            onnx_model = onnx.load(path)
        return onnx_model
    else:
        print(f"Onnx optimization with flag: {onnx_optimization_flags}")
        npath = path + ".optimized"
        if os.path.exists(npath):
            print(f"Found existed optimized onnx model at {npath}")
            return onnx.load(npath)
        else:
            print(f"Generate optimized onnx model to {npath}")
            if path.endswith('.gz'):
                onnx_model = onnx.load(gzip.GzipFile(path))
            else:
                onnx_model = onnx.load(path)
            return compress_onnx(onnx_model, path, npath, onnx_optimization_flags, debug=True)


def inference_onnx(path, *inputs):
    sess = ort.InferenceSession(unzip_and_optimize_onnx(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    inp = dict(zip(names, inputs))
    res = sess.run(None, inp)
    return res


@torch.no_grad()
def load_model_onnx(path, compute_test_acc=False, quirks=None, input_shape=None):
    onnx_optimization_flags = arguments.Config["model"]["onnx_optimization_flags"]
    if arguments.Config["model"]["cache_onnx_conversion"]:
        path_cache = f'{path}.cache'
        if os.path.exists(path_cache):
            print(f'Loading converted model from {path_cache}')
            return torch.load(path_cache)
    quirks = {} if quirks is None else quirks
    if arguments.Config["model"]["onnx_quirks"]:
        try:
            config_quirks = literal_eval(arguments.Config["model"]["onnx_quirks"])
        except ValueError as e:
            print(f'ERROR: onnx_quirks {arguments.Config["model"]["onnx_quirks"]} cannot be parsed!')
            raise
        assert isinstance(config_quirks, dict)
        deep_update(quirks, config_quirks)
    print(f'Loading onnx {path} wih quirks {quirks}')

    # pip install onnx2pytorch
    onnx_model = unzip_and_optimize_onnx(path, onnx_optimization_flags)

    if arguments.Config["model"]["input_shape"] is None:
        # find the input shape from onnx_model generally
        # https://github.com/onnx/onnx/issues/2657
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        net_feed_input = [node for node in onnx_model.graph.input if node.name in net_feed_input]

        if len(net_feed_input) != 1:
            # in some rare case, we use the following way to find input shape but this is not always true (collins-rul-cnn)
            net_feed_input = [onnx_model.graph.input[0]]

        onnx_input_dims = net_feed_input[0].type.tensor_type.shape.dim
        onnx_shape = tuple(d.dim_value for d in onnx_input_dims[1:])
    else:
        # User specify input_shape
        onnx_shape = arguments.Config["model"]["input_shape"][1:]

    pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=True, quirks=quirks)
    pytorch_model.eval()
    pytorch_model.to(dtype=torch.get_default_dtype())
    # print(pytorch_model)

    conversion_check_result = True
    try:
        # check conversion correctness
        # FIXME dtype of dummy may not match the onnx model, which can cause runtime error
        dummy = torch.randn([1, *onnx_shape])
        output_pytorch = pytorch_model(dummy).numpy()
        output_onnx = inference_onnx(path, dummy.numpy())[0]
        if "remove_relu_in_last_layer" in onnx_optimization_flags:
            output_pytorch = output_pytorch.clip(min=0)
        conversion_check_result = np.allclose(
            output_pytorch, output_onnx, 1e-4, 1e-5)
    except:
        warnings.warn(f'Not able to check model\'s conversion correctness')
        print('\n*************Error traceback*************')
        import traceback; print(traceback.format_exc())
        print('*****************************************\n')
    if not conversion_check_result:
        print('\n**************************')
        print('Model might not be converted correctly. Please check onnx conversion carefully.')
        print('**************************\n')

    if compute_test_acc:
        get_test_acc(pytorch_model, onnx_shape)

    if arguments.Config["model"]["cache_onnx_conversion"]:
        torch.save((pytorch_model, onnx_shape), path_cache)

    return pytorch_model, onnx_shape


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
            model_ori = eval(arguments.Config['model']['name'])()
        except Exception as e:
            print(f'Cannot load pytorch model definition "{arguments.Config["model"]["name"]}()". '
                  f'"{arguments.Config["model"]["name"]}()" must be a callable that returns a torch.nn.Module object.')
            import traceback
            traceback.print_exc()
            exit()
        model_ori.eval()
        print(model_ori)

        if not weights_loaded:
            return model_ori

        if arguments.Config["model"]["path"] is not None:
            # Load pytorch model
            # You can customize this function to load your own model based on model name.
            sd = torch.load(arguments.Config["model"]["path"], map_location=torch.device('cpu'))
            if 'state_dict' in sd:
                sd = sd['state_dict']
            if isinstance(sd, list):
                sd = sd[0]
            if not isinstance(sd, dict):
                raise NotImplementedError("Unknown model format, please modify model loader yourself.")
            model_ori.load_state_dict(sd)

    elif arguments.Config["model"]["onnx_path"] is not None:
        # Load onnx model
        model_ori, _ = load_model_onnx(arguments.Config["model"]["onnx_path"])

    else:
        print("Warning: pretrained model path is not given!")

    return model_ori


########################################
# Preprocess and load the datasets
########################################
def preprocess_cifar(image, inception_preprocess=False, perturbation=False):
    """
    Proprocess images and perturbations.Preprocessing used by the SDP paper.
    """
    MEANS = np.array([125.3, 123.0, 113.9], dtype=np.float32)/255
    STD = np.array([63.0, 62.1, 66.7], dtype=np.float32)/255
    if inception_preprocess:
        # Use 2x - 1 to get [-1, 1]-scaled images
        rescaled_devs = 0.5
        rescaled_means = 0.5
    else:
        rescaled_means = MEANS
        rescaled_devs = STD
    if perturbation:
        return image / rescaled_devs
    else:
        return (image - rescaled_means) / rescaled_devs


def load_cifar_sample_data(normalized=True, MODEL="a_mix"):
    """
    Load sampled cifar data: 100 images that are classified correctly by each MODEL
    """
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/sample100_unnormalized')
    X = np.load(os.path.join(database_path, MODEL, "X.npy"))
    if normalized:
        X = preprocess_cifar(X)
    X = np.transpose(X, (0, 3, 1, 2))
    y = np.load(os.path.join(database_path, MODEL, "y.npy"))
    runnerup = np.load(os.path.join(database_path, MODEL, "runnerup.npy"))
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(int))
    runnerup = torch.from_numpy(runnerup.astype(int))
    print("############################")
    if normalized:
        print("Sampled data loaded. Data already preprocessed!")
    else:
        print("Sampled data loaded. Data not preprocessed yet!")
    print("Shape:", X.shape, y.shape, runnerup.shape)
    print("X range:", X.max(), X.min(), X.mean())
    print("############################")
    return X, y, runnerup


def load_mnist_sample_data(MODEL="mnist_a_adv"):
    """
    Load sampled mnist data: 100 images that are classified correctly by each MODEL
    """
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/sample100_unnormalized')
    X = np.load(os.path.join(database_path, MODEL, "X.npy"))
    X = np.transpose(X, (0, 3, 1, 2))
    y = np.load(os.path.join(database_path, MODEL, "y.npy"))
    runnerup = np.load(os.path.join(database_path, MODEL, "runnerup.npy"))
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(int))
    runnerup = torch.from_numpy(runnerup.astype(int))
    print("############################")
    print("Shape:", X.shape, y.shape, runnerup.shape)
    print("X range:", X.max(), X.min(), X.mean())
    print("############################")
    return X, y, runnerup


def load_dataset():
    """
    Load regular datasets such as MNIST and CIFAR.
    """
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    normalize = transforms.Normalize(mean=arguments.Config["data"]["mean"], std=arguments.Config["data"]["std"])
    if arguments.Config["data"]["dataset"] == 'MNIST':
        loader = datasets.MNIST
    elif arguments.Config["data"]["dataset"] == 'CIFAR':
        loader = datasets.CIFAR10
    elif arguments.Config["data"]["dataset"] == 'CIFAR100':
        loader = datasets.CIFAR100
    else:
        raise ValueError("Dataset {} not supported.".format(arguments.Config["data"]["dataset"]))
    test_data = loader(database_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_data.mean = torch.tensor(arguments.Config["data"]["mean"])
    test_data.std = torch.tensor(arguments.Config["data"]["std"])
    # set data_max and data_min to be None if no clip
    data_max = torch.reshape((1. - test_data.mean) / test_data.std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - test_data.mean) / test_data.std, (1, -1, 1, 1))
    return test_data, data_max, data_min


def load_sampled_dataset():
    """
    Load sampled data and define the robustness region
    """
    if arguments.Config["data"]["dataset"] == "CIFAR_SAMPLE":
        X, labels, runnerup = load_cifar_sample_data(normalized=True, MODEL=arguments.Config['model']['name'])
        data_max = torch.tensor(preprocess_cifar(1.)).reshape(1,-1,1,1)
        data_min = torch.tensor(preprocess_cifar(0.)).reshape(1,-1,1,1)
        eps_temp = 2./255.
        eps_temp = torch.tensor(preprocess_cifar(eps_temp, perturbation=True)).reshape(1,-1,1,1)
    elif arguments.Config["data"]["dataset"] == "MNIST_SAMPLE":
        X, labels, runnerup = load_mnist_sample_data(MODEL=arguments.Config['model']['name'])
        data_max = torch.tensor(1.).reshape(1,-1,1,1)
        data_min = torch.tensor(0.).reshape(1,-1,1,1)
        eps_temp = 0.3
        eps_temp = torch.tensor(eps_temp).reshape(1,-1,1,1)
    return X, labels, data_max, data_min, eps_temp, runnerup


def load_sdp_dataset(eps_temp=None):
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/sdp')
    if arguments.Config["data"]["dataset"] == "CIFAR_SDP":
        X = np.load(os.path.join(database_path, "cifar/X_sdp.npy"))
        X = preprocess_cifar(X)
        X = np.transpose(X, (0,3,1,2))
        y = np.load(os.path.join(database_path, "cifar/y_sdp.npy"))
        runnerup = np.copy(y)
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))

        if eps_temp is None: eps_temp = 2./255.
        eps_temp = torch.tensor(preprocess_cifar(eps_temp, perturbation=True)).reshape(1,-1,1,1)

        data_max = torch.tensor(preprocess_cifar(1.)).reshape(1,-1,1,1)
        data_min = torch.tensor(preprocess_cifar(0.)).reshape(1,-1,1,1)

        print("############################")
        print("Sampled data loaded. Data already preprocessed!")
        print("Shape:", X.shape, y.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        print("############################")
    elif arguments.Config["data"]["dataset"] == "MNIST_SDP":
        X = np.load(os.path.join(database_path, "mnist/X_sdp.npy"))
        X = np.transpose(X, (0,3,1,2))
        y = np.load(os.path.join(database_path, "mnist/y_sdp.npy"))
        runnerup = np.copy(y)
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))

        if eps_temp is None: eps_temp = torch.tensor(0.3)

        data_max = torch.tensor(1.).reshape(1,-1,1,1)
        data_min = torch.tensor(0.).reshape(1,-1,1,1)

        print("############################")
        print("Shape:", X.shape, y.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        print("############################")
    else:
        exit("sdp dataset not supported!")

    return X, y, data_max, data_min, eps_temp, runnerup


def load_generic_dataset(eps_temp=None):
    """Load MNIST/CIFAR test set with normalization."""
    print("Trying generic MNIST/CIFAR data loader.")
    test_data, data_max, data_min = load_dataset()
    if eps_temp is None:
        raise ValueError('You must specify an epsilon')
    testloader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))
    runnerup = None
    # Rescale epsilon.
    eps_temp = torch.reshape(eps_temp / torch.tensor(arguments.Config["data"]["std"], dtype=torch.get_default_dtype()), (1, -1, 1, 1))

    return X, labels, data_max, data_min, eps_temp, runnerup


def load_eran_dataset(eps_temp=None):
    """
    Load sampled data and define the robustness region
    """
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/eran')

    if arguments.Config["data"]["dataset"] == "CIFAR_ERAN":
        X = np.load(os.path.join(database_path, "cifar_eran/X_eran.npy"))
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, -1, 1, 1).astype(np.float32)
        std = np.array([0.2023, 0.1994, 0.201]).reshape(1, -1, 1, 1).astype(np.float32)
        X = (X - mean) / std

        labels = np.load(os.path.join(database_path, "cifar_eran/y_eran.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 2. / 255.

        eps_temp = torch.tensor(eps_temp / std).reshape(1, -1, 1, 1)
        data_max = torch.tensor((1. - mean) / std).reshape(1, -1, 1, 1)
        data_min = torch.tensor((0. - mean) / std).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. Data already preprocessed!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    elif arguments.Config["data"]["dataset"] == "MNIST_ERAN":
        X = np.load(os.path.join(database_path, "mnist_eran/X_eran.npy"))
        mean = 0.1307
        std = 0.3081
        X = (X - mean) / std

        labels = np.load(os.path.join(database_path, "mnist_eran/y_eran.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 0.3

        eps_temp = torch.tensor(eps_temp / std).reshape(1, -1, 1, 1)
        data_max = torch.tensor((1. - mean) / std).reshape(1, -1, 1, 1)
        data_min = torch.tensor((0. - mean) / std).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. Data already preprocessed!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    elif arguments.Config["data"]["dataset"] == "MNIST_ERAN_UN":
        X = np.load(os.path.join(database_path, "mnist_eran/X_eran.npy"))

        labels = np.load(os.path.join(database_path, "mnist_eran/y_eran.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 0.3

        eps_temp = torch.tensor(eps_temp).reshape(1, -1, 1, 1)
        data_max = torch.tensor(1.).reshape(1, -1, 1, 1)
        data_min = torch.tensor(0.).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. No normalization used!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    elif arguments.Config["data"]["dataset"] == "MNIST_MADRY_UN":
        X = np.load(os.path.join(database_path, "mnist_madry/X.npy")).reshape(-1, 1, 28, 28)
        labels = np.load(os.path.join(database_path, "mnist_madry/y.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 0.3

        eps_temp = torch.tensor(eps_temp).reshape(1, -1, 1, 1)
        data_max = torch.tensor(1.).reshape(1, -1, 1, 1)
        data_min = torch.tensor(0.).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. No normalization used!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    else:
        raise(f'Unsupported dataset {arguments.Config["data"]["dataset"]}')

    return X, labels, data_max, data_min, eps_temp, runnerup


def Customized(def_file, callable_name, *args, **kwargs):
    """Fully customized model or dataloader."""
    if def_file.endswith('.py'):
        spec = importlib.util.spec_from_file_location("customized", def_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(def_file)
    # Load model from a specified file.
    model_func = getattr(module, callable_name)
    customized_func = partial(model_func, *args, **kwargs)
    # We need to return a Callable which returns the model.
    return customized_func


def load_verification_dataset(eps_before_normalization):
    if arguments.Config["data"]["dataset"].startswith("Customized("):
        # FIXME (01/10/22): fully document customized data loader.
        # Returns: X, labels, runnerup, data_max, data_min, eps, target_label.
        # X is the data matrix in (batch, ...).
        # labels are the groud truth labels, a tensor of integers.
        # runnerup is the runnerup label used for quickly verify against the runnerup (second largest) label, can be set to None.
        # data_max is the per-example perturbation upper bound, shape (batch, ...) or (1, ...).
        # data_min is the per-example perturbation lower bound, shape (batch, ...) or (1, ...).
        # eps is the Lp norm perturbation epsilon. Can be set to None if element-wise perturbation (specified by data_max and data_min) is used.
        # Target label is the targeted attack label; can be set to None.
        data_config = eval(arguments.Config["data"]["dataset"])(eps=eps_before_normalization)
        if len(data_config) == 5:
            X, labels, data_max, data_min, eps_new = data_config
            runnerup, target_label = None, None
        elif len(data_config) == 6:
            X, labels, data_max, data_min, eps_new, runnerup = data_config
            target_label = None
        elif len(data_config) == 7:
            X, labels, data_max, data_min, eps_new, runnerup, target_label = data_config
        else:
            print("Data config types not correct!")
            exit()
        assert X.size(0) == labels.size(0), "batch size of X and labels should be the same!"
        assert (data_max - data_min).min()>=0, "data_max should always larger or equal to data_min!"
        return X, labels, runnerup, data_max, data_min, eps_new, target_label
    target_label = None
    # Add your customized dataset here.
    if arguments.Config["data"]["pkl_path"] is not None:
        # FIXME (01/10/22): "pkl_path" should not exist in public code!
        # for oval20 base, wide, deep or other datasets saved in .pkl file, we load the pkl file here.
        assert arguments.Config["specification"]["epsilon"] is None, 'will use epsilon saved in .pkl file'
        gt_results = pd.read_pickle(arguments.Config["data"]["pkl_path"])
        test_data, data_max, data_min = load_dataset()
        X, labels = zip(*test_data)
        X = torch.stack(X, dim=0)
        labels = torch.tensor(labels)
        runnerup = None
        idx = gt_results["Idx"].to_list()
        X, labels = X[idx], labels[idx]
        target_label = gt_results['prop'].to_list()
        eps_new = gt_results['Eps'].to_list()
        print('Overwrite epsilon that saved in .pkl file, they should be after normalized!')
        eps_new = [torch.reshape(torch.tensor(i, dtype=torch.get_default_dtype()), (1, -1, 1, 1)) for i in eps_new]
        data_config = (X, labels, data_max, data_min, eps_new, runnerup, target_label)
    # Some special model loaders.
    elif "ERAN" in arguments.Config["data"]["dataset"] or "MADRY" in arguments.Config["data"]["dataset"]:
        data_config = load_eran_dataset(eps_temp=eps_before_normalization)
    elif "SDP" in arguments.Config["data"]["dataset"]:
        data_config = load_sdp_dataset(eps_temp=eps_before_normalization)
    elif "SAMPLE" in arguments.Config["data"]["dataset"]:
        # Sampled datapoints (a small subset of MNIST/CIFAR), only for reproducing some paper results.
        data_config = load_sampled_dataset()
    elif "CIFAR" in arguments.Config["data"]["dataset"] or "MNIST" in arguments.Config["data"]["dataset"]:
        # general MNIST and CIFAR dataset with mean/std defined in config file.
        data_config = load_generic_dataset(eps_temp=eps_before_normalization)
    else:
        exit("Dataset not supported in this file! Please customize load_verification_dataset() function in utils.py.")

    if len(data_config) == 5:
        (X, labels, data_max, data_min, eps_new) = data_config
        runnerup = None
    elif len(data_config) == 6:
        (X, labels, data_max, data_min, eps_new, runnerup) = data_config
    elif len(data_config) == 7:
        (X, labels, data_max, data_min, eps_new, runnerup, target_label) = data_config

    if arguments.Config["specification"]["norm"] != np.inf:
        if isinstance(arguments.Config["data"]["std"], (list, tuple)):
            assert arguments.Config["data"]["std"].count(arguments.Config["data"]["std"][0]) == len(
                arguments.Config["data"]["std"]), ('For non-Linf norm, we support only 1-d eps (all channels with the same perturbation). '
                'If you have more complex, per-channel eps (e.g., an ellipsoid L2 perturbation, you can '
                'add the data normalization into part of the model.')
            arguments.Config["data"]["std"] = arguments.Config["data"]["std"][0]
        else:
             arguments.Config["data"]["std"] = float(arguments.Config["data"]["std"])
        eps_new = eps_new[0, 0, 0, 0]  # only support eps as a scalar for non-Linf norm

    # FIXME (01/10/22): we should have a common interface for dataloader.
    return X, labels, runnerup, data_max, data_min, eps_new, target_label


class Normalization(nn.Module):
    def __init__(self, mean, std, model):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        self.model = model

    def forward(self, x):
        return self.model((x - self.mean)/self.std)


def default_onnx_and_vnnlib_loader(file_root, onnx_path, vnnlib_path):
    vnnlib = read_vnnlib(os.path.join(file_root, vnnlib_path))

    model_ori, onnx_shape = load_model_onnx(os.path.join(file_root, onnx_path))
    shape = (-1, *onnx_shape)  # add the batch dim to onnx_shape

    return model_ori, shape, vnnlib


def construct_vnnlib(X, labels, runnerups, data_max, data_min, perturb_epsilon, target_labels, example_idx_list):

    vnnlib = []
    num_outputs = arguments.Config["data"]["num_outputs"]
    if type(perturb_epsilon) == list:
        # Each example has different perturbations.
        perturb_epsilon = torch.cat(perturb_epsilon)
        perturb_epsilon = perturb_epsilon[example_idx_list]
    elif type(perturb_epsilon) == torch.Tensor:
        # Same perturbation for all examples.
        pass
    else:
        # No perturbation, instead we use lower and upper bounds directly.
        assert arguments.Config["specification"]["type"] == 'bound'

    if arguments.Config["specification"]["type"] == 'bound':
        assert arguments.Config["specification"]["norm"] == float("inf")
        x_lower = data_min.flatten(1)
        x_upper = data_max.flatten(1)
    elif arguments.Config["specification"]["type"] == 'lp':
        if arguments.Config["specification"]["norm"] == float("inf"):
            if data_max is None:
                # perturb_eps is already normalized.
                x_lower = (X[example_idx_list] - perturb_epsilon).flatten(1)
                x_upper = (X[example_idx_list] + perturb_epsilon).flatten(1)
            else:
                x_lower = (X[example_idx_list] - perturb_epsilon).clamp(min=data_min).flatten(1)
                x_upper = (X[example_idx_list] + perturb_epsilon).clamp(max=data_max).flatten(1)
        else:
            x_lower = X[example_idx_list].flatten(1)
            x_upper = X[example_idx_list].flatten(1)
            # Save the actual perturbation epsilon to global variable dictionary.
            arguments.Globals["lp_perturbation_eps"] = perturb_epsilon
    else:
        raise ValueError(f'Unsupported perturbation type {arguments.Config["specification"]["type"]}')


    x_range = torch.stack([x_lower, x_upper], -1).numpy()

    for i in range(len(example_idx_list)):
        label = labels[example_idx_list[i]].view(1, 1)
        this_x_range = x_range[i]

        if arguments.Config["data"]["num_outputs"] > 1:
            # Multi-class.
            if arguments.Config["specification"]["robustness_type"] == "verified-acc":
                c = torch.eye(num_outputs).type_as(x_lower)[label].unsqueeze(1) - torch.eye(num_outputs).type_as(
                    x_lower).unsqueeze(0)
                I = (~(label.unsqueeze(1) == torch.arange(num_outputs).type_as(label.data).unsqueeze(0)))
                c = (c[I].view(1, num_outputs - 1, num_outputs)).numpy()
                new_c = []
                for ii in range(num_outputs - 1):
                    new_c.append((c[:, ii], np.array([arguments.Config["bab"]["decision_thresh"]])))

            elif arguments.Config["specification"]["robustness_type"] == "specify-target":
                target_label = target_labels[example_idx_list[i]]
                c = np.zeros([1, num_outputs])
                c[0, label] = 1
                c[0, target_label] = -1
                new_c = [(c, np.array([arguments.Config["bab"]["decision_thresh"]]))]

            elif arguments.Config["specification"]["robustness_type"] == "runnerup":
                runnerup = runnerups[example_idx_list[i]]
                c = np.zeros([1, num_outputs])
                c[0, label] = 1
                c[0, runnerup] = -1
                new_c = [(c, np.array([arguments.Config["bab"]["decision_thresh"]]))]
        else:
            # Binary class, no target label.
            c = np.ones([1, 1])
            new_c = [(c, np.array([arguments.Config["bab"]["decision_thresh"]]))]

        vnnlib.append([(this_x_range, new_c)])

    return vnnlib


def parse_run_mode():
    """ parse running by vnnlib or customized data
     if using customized data, we convert them to vnnlib format
     """
    file_root = model_ori = vnnlib_all = shape = None

    if arguments.Config["general"]["csv_name"] is not None and arguments.Config["specification"]["vnnlib_path"] is None:
        # A CSV filename is specified, and we will go over all models and specs in this csv file.
        # Used for running VNN-COMP benchmarks in batch mode.
        # In this case, vnnlib_path should not be specified, otherwise we will run only a single model/spec.
        run_mode = 'csv_file'
        file_root = arguments.Config["general"]["root_path"]

        with open(os.path.join(file_root, arguments.Config["general"]["csv_name"]), newline='') as csv_f:
            reader = csv.reader(csv_f, delimiter=',')

            csv_file = []
            for row in reader:
                # In VNN-COMP each line of the csv containts 3 elements: model, vnnlib, timeout
                csv_file.append(row)

        if len(csv_file[0]) == 1:
            # Each row contains only one item, which is the vnnlib spec. So we load and return the model only once here.
            # This case is used when we have a batch of vnnlib specs to verify with one model either pytorch or ONNX.
            model_ori = load_model()

        save_path = 'a-b-crown_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}_cplex_cuts={}_initial_max_domains={}.npz'.format(
                   os.path.splitext(arguments.Config["general"]["csv_name"])[0], arguments.Config["data"]["start"],
                   arguments.Config["data"]["end"], arguments.Config["solver"]["beta-crown"]["iteration"],
                   arguments.Config["solver"]["batch_size"],
                   arguments.Config["bab"]["timeout"], arguments.Config["bab"]["branching"]["method"],
                   arguments.Config["bab"]["branching"]["reduceop"],
                   arguments.Config["bab"]["branching"]["candidates"],
                   arguments.Config["solver"]["alpha-crown"]["lr_alpha"],
                   arguments.Config["solver"]["beta-crown"]["lr_alpha"],
                   arguments.Config["solver"]["beta-crown"]["lr_beta"], arguments.Config["attack"]["pgd_order"],
                   arguments.Config["bab"]["cut"]["cplex_cuts"],
                   arguments.Config["bab"]["initial_max_domains"])

        arguments.Config["data"]["end"] = min(arguments.Config["data"]["end"], reader.line_num)
        if arguments.Config["data"]["start"] != 0 or arguments.Config["data"]["end"] != reader.line_num:
            assert 0 <= arguments.Config["data"]["start"] <= reader.line_num and arguments.Config["data"]["end"] > arguments.Config["data"]["start"], \
                    "specified --start or --end out of range: start={}, end={}, total_in_csv={}".format(arguments.Config["data"]["end"], arguments.Config["data"]["start"], reader.line_num)
            print("customized start/end sample from instance {} to {} in {}".format(arguments.Config["data"]["start"], arguments.Config["data"]["end"], arguments.Config["general"]["csv_name"]))
        else:
            print("no customized start/end sample, testing all samples in {}".format(arguments.Config["general"]["csv_name"]))
            arguments.Config["data"]["start"], arguments.Config["data"]["end"] = 0, reader.line_num
        example_idx_list = csv_file[arguments.Config["data"]["start"]:arguments.Config["data"]["end"]]
    elif arguments.Config["model"]["onnx_path"] is not None and arguments.Config["specification"]["vnnlib_path"] is not None:
        # A onnx file and a vnnlib file is specified, run this onnx file with vnnlib, ignore csv file.
        # Used for VNN-COMP in single instance mode, will be used in run_instance.sh
        run_mode = 'single_vnnlib'
        arguments.Config["data"]["start"], arguments.Config["data"]["end"] = 0, 1
        csv_file = [(arguments.Config["model"]["onnx_path"], arguments.Config["specification"]["vnnlib_path"],
                     arguments.Config["bab"]["timeout"])]
        save_path = arguments.Config["general"]["results_file"]
        file_root = ''
        example_idx_list = csv_file[arguments.Config["data"]["start"]:arguments.Config["data"]["end"]]
    elif arguments.Config["general"]["csv_name"] is None:
        # No CSV specified, we will use specifications defined in yaml file.
        # This part replaces the old robustness_verifier.py interface.
        run_mode = 'customized_data'
        # Load Pytorch or ONNX model depends on the model path or onnx_path is given.
        model_ori = load_model(weights_loaded=True)
        if arguments.Config["specification"]["vnnlib_path"] is None:
            # Lp norm perturbation, replacing robustness_verifier.py
            if arguments.Config["specification"]["epsilon"] is not None:
                perturb_epsilon = torch.tensor(arguments.Config["specification"]["epsilon"], dtype=torch.get_default_dtype())
            else:
                print('No epsilon defined!')
                perturb_epsilon = None
            X, labels, runnerup, data_max, data_min, perturb_epsilon, target_label = load_verification_dataset(perturb_epsilon)

            if arguments.Config["data"]["data_idx_file"] is not None:
                # Go over a list of data indices.
                with open(arguments.Config["data"]["data_idx_file"]) as f:
                    example_idx_list = re.split(r'[;|,|\n|\s]+', f.read().strip())
                    example_idx_list = [int(b_id) for b_id in example_idx_list]
                    print(f'Example indices (total {len(example_idx_list)}): {example_idx_list}')
            else:
                # By default, we go over all data.
                example_idx_list = list(range(X.shape[0]))
            example_idx_list = example_idx_list[arguments.Config["data"]["start"]:  arguments.Config["data"]["end"]]
            vnnlib_all = construct_vnnlib(X, labels, runnerup, data_max, data_min, perturb_epsilon, target_label, example_idx_list)
            shape = [-1] + list(X.shape[1:])
        else:
            # Using vnnlib specification (e.g., loading a pytorch model and use vnnlib to define general specification).
            example_idx_list = [0]
            vnnlib = read_vnnlib(arguments.Config["specification"]["vnnlib_path"])
            assert arguments.Config["model"]["input_shape"] is not None, 'vnnlib does not have shape information, please specify by --input_shape'
            shape = arguments.Config["model"]["input_shape"]
            vnnlib_all = [vnnlib]  # Only 1 vnnlib file.

        if arguments.Config['model']['name'] is None:
            # use onnx model prefix as model_name
            model_name = arguments.Config["model"]["onnx_path"].split('.onnx')[-2].split('/')[-1]
        elif "Customized" in arguments.Config['model']['name']:
            model_name = "Customized_model"
        else:
            model_name = arguments.Config['model']['name']

        save_path = 'Verified_ret_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}_cplex_cuts={}_multiclass={}.npy'.format(
                   model_name, arguments.Config["data"]["start"], arguments.Config["data"]["end"],
                   arguments.Config["solver"]["beta-crown"]["iteration"],
                   arguments.Config["solver"]["batch_size"],
                   arguments.Config["bab"]["timeout"], arguments.Config["bab"]["branching"]["method"],
                   arguments.Config["bab"]["branching"]["reduceop"],
                   arguments.Config["bab"]["branching"]["candidates"],
                   arguments.Config["solver"]["alpha-crown"]["lr_alpha"],
                   arguments.Config["solver"]["beta-crown"]["lr_alpha"],
                   arguments.Config["solver"]["beta-crown"]["lr_beta"],
                   arguments.Config["attack"]["pgd_order"], arguments.Config["bab"]["cut"]["cplex_cuts"],
                   arguments.Config["solver"]["multi_class"]["multi_class_method"])

    else:
        raise NotImplementedError

    print(f'Internal results will be saved to {save_path}.')
    # FIXME_NOW: model_ori should not be handled in this function! Do it in the utility function that loads models for all cases.
    return run_mode, save_path, file_root, example_idx_list, model_ori, vnnlib_all, shape
