#########################################################################
##         This file is part of the alpha-beta-CROWN verifier          ##
##                                                                     ##
## Copyright (C) 2021, Huan Zhang <huan@huan-zhang.com>                ##
##                     Kaidi Xu <xu.kaid@northeastern.edu>             ##
##                     Shiqi Wang <sw3215@columbia.edu>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Yihan Wang <yihanwang@ucla.edu>                 ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
from collections import OrderedDict

import os
import gzip
from functools import partial
import importlib
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import onnx2pytorch
import onnx
import onnxruntime as ort
import arguments
from attack_pgd import attack_pgd

# Import all model architectures.
from model_defs import *


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(pad) is not tuple:
        pad = (pad, pad)
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1
    return h, w


def get_pgd_acc(model, X, labels, eps, data_min, data_max, batch_size):
    start = 0
    total = X.size(0)
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
        print(f'batch size {end - start}, clean correct {batch_clean_correct}, robust correct {batch_robust_correct}')
        clean_correct += batch_clean_correct
        robust_correct += batch_robust_correct
        start += batch_size
        del clean_output, best_deltas, last_deltas, attack_images, attack_output
    print(f'data size {total}, clean correct {clean_correct}, robust correct {robust_correct}')
    return clean_correct, robust_correct


def get_test_acc(model, input_shape=None, X=None, labels=None, is_channel_last=False, batch_size=256):
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
            if is_channel_last:
                images = images.permute(0,2,3,1)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if print_first_batch:
                print_first_batch = False
                for i in range(min(outputs.size(0), 10)):
                    print(f"Image {i} norm {images[i].abs().sum().item()} label {labels[i].item()} correct {labels[i].item() == outputs[i].argmax().item()}\nprediction {outputs[i].cpu().numpy()}")
    print(f'correct {correct} of {total}')


def load_onnx(path):
    if path.endswith('.gz'):
        onnx_model = onnx.load(gzip.GzipFile(path))
    else:
        onnx_model = onnx.load(path)
    return onnx_model


def inference_onnx(path, *inputs):
    print(inputs)
    sess = ort.InferenceSession(load_onnx(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    inp = dict(zip(names, inputs))
    res = sess.run(None, inp)
    return res


def load_model_onnx(path, input_shape, compute_test_acc=False, force_convert=False):
    # pip install onnx2pytorch
    onnx_model = load_onnx(path)

    onnx_input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    onnx_shape = tuple(d.dim_value for d in onnx_input_dims[1:])
    input_shape = tuple(input_shape)
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model)

    if force_convert:
        new_modules = []
        modules = list(pytorch_model.modules())[1:]
        for mi, m in enumerate(modules):
            if isinstance(m, torch.nn.Linear):
                new_m = nn.Linear(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
                new_m.weight.data.copy_(m.weight.data)
                new_m.bias.data.copy_(m.bias)
                new_modules.append(new_m)
            elif isinstance(m, torch.nn.ReLU):
                new_modules.append(torch.nn.ReLU())
            elif isinstance(m, onnx2pytorch.operations.flatten.Flatten):
                new_modules.append(torch.nn.Flatten())
            else:
                raise NotImplementedError

        seq_model = nn.Sequential(*new_modules)
        return seq_model

    if len(input_shape) <= 2:
        return pytorch_model

    # Check model input shape.
    is_channel_last = False
    if onnx_shape != input_shape:
        # Change channel location.
        onnx_shape = onnx_shape[2:] + onnx_shape[:2]
        if onnx_shape == input_shape:
            is_channel_last = True
        else:
            print(f"Unexpected input shape in onnx: {onnx_shape}, given {input_shape}")

    # Fixup converted ONNX model. For ResNet we directly return; for other models, we convert them to a Sequential model.
    # We also need to handle NCHW and NHWC formats here.
    conv_c, conv_h, conv_w = input_shape
    modules = list(pytorch_model.modules())[1:]
    new_modules = []
    need_permute = False
    for mi, m in enumerate(modules):
        if isinstance(m, onnx2pytorch.operations.add.Add):
            # ResNet model. No need to convert to sequential.
            return pytorch_model, is_channel_last
        if isinstance(m, torch.nn.Conv2d):
            # Infer the output size of conv.
            conv_h, conv_w = conv_output_shape((conv_h, conv_w), m.kernel_size, m.stride, m.padding)
            conv_c = m.weight.size(0)
        if isinstance(m, onnx2pytorch.operations.reshape.Reshape):
            # Replace reshape with flatten.
            new_modules.append(nn.Flatten())
            # May need to permute the next linear layer if the model was in NHWC format.
            need_permute = True and is_channel_last
        elif isinstance(m, torch.nn.Linear) and need_permute:
            # The original model is in NHWC format and we now have NCHW format, so the dense layer's weight must be adjusted.
            new_m = nn.Linear(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
            new_m.weight.data.copy_(m.weight.view(m.weight.size(0), conv_h, conv_w, conv_c).permute(0, 3, 1, 2).contiguous().view(m.weight.size(0), -1))
            new_m.bias.data.copy_(m.bias)
            need_permute = False
            new_modules.append(new_m)
        elif isinstance(m, torch.nn.ReLU) and mi == (len(modules)-1):
            # not add relu if last layer is relu
            pass
        else:
            new_modules.append(m)

    seq_model = nn.Sequential(*new_modules)

    if compute_test_acc:
        get_test_acc(seq_model, input_shape)

    return seq_model, is_channel_last


def load_model(weights_loaded=True):
    """
    Load the model architectures and weights
    """
    # You can customize this function to load your own model based on model name.
    model_ori = eval(arguments.Config['model']['name'])()
    model_ori.eval()
    print(model_ori)

    if not weights_loaded:
        return model_ori

    if arguments.Config["model"]["path"] is not None:
        sd = torch.load(arguments.Config["model"]["path"], map_location=torch.device('cpu'))
        if 'state_dict' in sd:
            sd = sd['state_dict']
        if isinstance(sd, list):
            sd = sd[0]
        if not isinstance(sd, dict):
            raise NotImplementedError("Unknown model format, please modify model loader yourself.")
        model_ori.load_state_dict(sd)
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
    upper_limit, lower_limit = 1., 0.
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
    # Load model from a specified file.
    model_func = getattr(importlib.import_module(def_file), callable_name)
    customized_func = partial(model_func, *args, **kwargs)
    # We need to return a Callable which returns the model.
    return customized_func


def load_verification_dataset(eps_before_normalization):
    if arguments.Config["data"]["dataset"].startswith("Customized("):
        # Returns: X, labels, runnerup, data_max, data_min, eps, target_label.
        # X is the data matrix in (batch, ...).
        # labels are the groud truth labels, a tensor of integers.
        # runnerup is the runnerup label used for quickly verify against the runnerup (second largest) label, can be set to None.
        # data_max is the per-example perturbation upper bound, shape (batch, ...) or (1, ...).
        # data_min is the per-example perturbation lower bound, shape (batch, ...) or (1, ...).
        # eps is the Lp norm perturbation epsilon. Can be set to None if element-wise perturbation (specified by data_max and data_min) is used.
        # Target label is the targeted attack label; can be set to None.
        if arguments.Config["specification"]["type"] == "lp":
            data_config = eval(arguments.Config["data"]["dataset"])(eps=eps_before_normalization)
        elif arguments.Config["specification"]["type"] == "bound":
            data_config = eval(arguments.Config["data"]["dataset"])()
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
        assert arguments.Config["data"]["std"].count(arguments.Config["data"]["std"][0]) == len(
            arguments.Config["data"]["std"]), print('For non-Linf norm, we only support 1d eps.')
        arguments.Config["data"]["std"] = arguments.Config["data"]["std"][0]
        eps_new = eps_new[0, 0, 0, 0]  # only support eps as a scalar for non-Linf norm

    return X, labels, runnerup, data_max, data_min, eps_new, target_label


def convert_test_model(model_ori):
    # NOTE: It looks like `in_features` and `out_features` are in the wrong order
    # after converting the onnx model to pytorch model.
    # Swap them below.
    modules = []
    for m in model_ori._modules.values():
        if isinstance(m, nn.Linear):
            layer = nn.Linear(m.in_features, m.out_features)  # Fix a bug in onnx converter for test models.
            layer.weight.data = m.weight.data.to(torch.float)
            layer.bias.data = m.bias.data.to(torch.float) if m.bias is not None else torch.zeros_like(layer.bias.data)
            modules.append(layer)
            # pdb.set_trace()
        else:
            modules.append(m)
    model_ori = nn.Sequential(*modules)

    return model_ori


def convert_nn4sys_model(model_ori):
    model_ori = nn.Sequential(*list(model_ori._modules.values()))
    # Split the model into v1 and v2 models to resolve numerical issues
    modules_v1 = []
    modules_v2 = []
    stage = 1
    for m in model_ori._modules.values():
        if isinstance(m, nn.Linear):
            if m.weight.abs().max() > 1e9:
                stage = 2 if len(modules_v2) == 0 else 3
                continue
        else:
            continue
        if stage == 1:
            modules_v1 += [m, nn.ReLU(inplace=True)]
        elif stage == 2:
            dim = modules_v1[-2].out_features - 1
            lin = nn.Linear(m.in_features - dim, m.out_features - dim)
            lin.weight.data = m.weight[:lin.out_features, :lin.in_features]
            lin.weight = lin.weight.to(dtype=torch.float64)
            lin.bias.data = m.bias[:lin.out_features]
            lin.bias = lin.bias.to(dtype=torch.float64)
            modules_v2 += [lin, nn.ReLU(inplace=True)]
    x = torch.tensor([[119740.8]], dtype=torch.float64)                
    modules_v1 = modules_v1[:-1]
    model_v1 = nn.Sequential(*modules_v1)
    y = model_v1(x)
    dim = y.size(-1) - 1  
    modules_v2 = modules_v2[:-1]
    linear_ident = nn.Linear(1, dim, bias=False)
    linear_ident.weight.data = torch.ones_like(linear_ident.weight, dtype=torch.float64)
    modules_v2.insert(0, linear_ident)
    model_v2 = nn.Sequential(*modules_v2)
    y[:, :-2] *= (y[:, 1:-1] <= 0).int()
    select = (y[:, :-1] > 0).int()
    y2 = model_v2(x)
    y2 = y2[:] * select
    res = y2.sum(dim=-1, keepdim=True)
    res_ref = model_ori(x)
    print(res.item(), res_ref.item())
    # import pdb; pdb.set_trace()
    model_ori = (model_v1, model_v2, model_ori)   
    return model_ori


class Normalization(nn.Module):
    def __init__(self, mean, std, model):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        self.model = model
    
    def forward(self, x):
        return self.model((x - self.mean)/self.std)

