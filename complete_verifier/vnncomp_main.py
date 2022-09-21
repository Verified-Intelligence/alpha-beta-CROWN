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
import argparse
import sys
import os

parser = argparse.ArgumentParser()

parser.add_argument("CATEGORY", type=str)
parser.add_argument("ONNX_FILE", type=str, default=None, help='ONNX_FILE')
parser.add_argument("VNNLIB_FILE", type=str, default=None, help='VNNLIB_FILE')
parser.add_argument("RESULTS_FILE", type=str, default=None, help='RESULTS_FILE')
parser.add_argument("TIMEOUT", type=float, default=180, help='timeout for one property')

args = parser.parse_args()

python_path = sys.executable
library_path = os.path.dirname(os.path.realpath(__file__))

cmd = f"{python_path} {library_path}/abcrown.py --config {library_path}/"


# vnncomp 2021
if args.CATEGORY == "cifar10_resnet":
    cmd += "/exp_configs/vnncomp21/cifar10-resnet.yaml"

elif args.CATEGORY == "eran":
    if 'SIGMOID' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp21/eran_sigmoid.yaml"
    else:
        cmd += "exp_configs/vnncomp21/eran_mlp.yaml"

elif args.CATEGORY == "marabou-cifar10":
    cmd += "exp_configs/vnncomp21/marabou_cifar10.yaml"

elif args.CATEGORY == "verivital":
    cmd += "exp_configs/vnncomp21/verivital.yaml"

# common
elif args.CATEGORY == "acasxu":
    cmd += "exp_configs/vnncomp22/acasxu.yaml"

elif args.CATEGORY == "cifar2020":
    cmd += "exp_configs/vnncomp22/cifar2020_2_255.yaml"

elif args.CATEGORY == "oval21":
    cmd += "exp_configs/vnncomp22/oval22.yaml"

elif args.CATEGORY in ["mnist_fc", "mnistfc"]:  # same benchmark with different names
    if '256x2' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/mnistfc_small.yaml"
    else:
        cmd += "exp_configs/vnncomp22/mnistfc.yaml"

# special case: both vnncomp 2021 and 2022 have nn4sys, but they are different
elif args.CATEGORY == "nn4sys":
    if 'lindex' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/nn4sys_2022_lindex.yaml"
    elif '128d' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/nn4sys_2022_128d.yaml"
    elif '2048d' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/nn4sys_2022_2048d.yaml"
    else:
        # nn4sys in 2021
        cmd += "exp_configs/vnncomp21/nn4sys.yaml"

# vnncomp 2022
elif args.CATEGORY == "carvana_unet_2022":
    if "unet_simp" in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/carvana-unet-simp.yaml"
    else:
        cmd += "exp_configs/vnncomp22/carvana-unet-upsample.yaml"

elif args.CATEGORY == "cifar100_tinyimagenet_resnet":
    if 'CIFAR100_resnet_small' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/cifar100_small_2022.yaml"
    elif 'CIFAR100_resnet_medium' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/cifar100_med_2022.yaml"
    elif 'CIFAR100_resnet_large' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/cifar100_large_2022.yaml"
    elif 'CIFAR100_resnet_super' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/cifar100_super_2022.yaml"
    else:
        cmd += "exp_configs/vnncomp22/tinyimagenet_2022.yaml"

elif args.CATEGORY == "cifar_biasfield":
    cmd += "exp_configs/vnncomp22/cifar_biasfield.yaml"

elif args.CATEGORY == "collins_rul_cnn":
    cmd += "exp_configs/vnncomp22/collins-rul-cnn.yaml"

elif args.CATEGORY == "oval21":
    cmd += "exp_configs/vnncomp22/oval22.yaml"

elif args.CATEGORY == "reach_prob_density":
    if 'gcas' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/reach_probability_gcas.yaml"
    else:
        cmd += "exp_configs/vnncomp22/reach_probability.yaml"

elif args.CATEGORY == "rl_benchmarks":
    if 'cartpole' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/cartpole.yaml"
    elif 'lunarlander' in args.ONNX_FILE:
        cmd += "exp_configs/vnncomp22/lunarlander.yaml"
    else:
        cmd += "exp_configs/vnncomp22/dubins-rejoin.yaml"

elif args.CATEGORY == "sri_resnet_a":
    cmd += "exp_configs/vnncomp22/resnet_A.yaml"

elif args.CATEGORY == "sri_resnet_b":
    cmd += "exp_configs/vnncomp22/resnet_B.yaml"

elif args.CATEGORY == "tllverifybench":
    cmd += "exp_configs/vnncomp22/tllVerifyBench.yaml"

elif args.CATEGORY == "vggnet16_2022":
    cmd += "exp_configs/vnncomp22/vggnet16.yaml"

elif args.CATEGORY == "test":
    pass

else:
    exit("CATEGORY {} not supported yet".format(args.CATEGORY))

# test case may run in other args.CATEGORY at the end of them, so we parse them here to allow correct measurement of overhead.
if os.path.split(args.VNNLIB_FILE)[-1] in ['test_' + f + '.vnnlib' for f in ['nano', 'tiny', 'small']]:
    cmd = f"{python_path} {library_path}/abcrown.py --config {library_path}/exp_configs/vnncomp21/test.yaml"
elif 'test_prop' in args.VNNLIB_FILE:
    cmd = f"{python_path} {library_path}/abcrown.py --config {library_path}/exp_configs/vnncomp22/acasxu.yaml"


cmd += " --precompile_jit"
cmd += " --onnx_path " + str(args.ONNX_FILE)
cmd += " --vnnlib_path " + str(args.VNNLIB_FILE)
cmd += " --results_file " + str(args.RESULTS_FILE)
cmd += " --timeout " + str(args.TIMEOUT)

print("\n------------------------- COMMAND ------------------------------")
print(cmd)
print("----------------------------------------------------------------\n")

os.system(cmd)
