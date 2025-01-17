#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
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
parser.add_argument("--DEBUG", action='store_true', help='whether to run in debug mode (checking saved adv example)')
parser.add_argument("--PREPARE", action='store_true', help='whether in preparation phase')
parser.add_argument("--NOPGD", action='store_true', help='do not use pdg attack')
parser.add_argument("--TRY_CROWN", action='store_true', help='overwrite bound-prop-method to CROWN to save memory')

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
# elif args.CATEGORY == "acasxu":
#     cmd += "exp_configs/vnncomp22/acasxu.yaml"

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
    if 'lindex' in args.ONNX_FILE or '128d' in args.ONNX_FILE or '2048d' in args.ONNX_FILE or 'pensieve' in args.ONNX_FILE:
        # All models in 2022 and 2023 use the same config file.
        cmd += "exp_configs/vnncomp23/nn4sys_2023.yaml"
    else:
        # nn4sys in 2021
        cmd += "exp_configs/vnncomp21/nn4sys.yaml"

# vnncomp 2022
elif args.CATEGORY == "carvana_unet_2022":
    cmd += "exp_configs/vnncomp22/carvana-unet-all.yaml"

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

# vnncomp 2023
elif args.CATEGORY == "acasxu":
    # caution: acasxu is actually a common benchmark across multiple years,
    # we have exactly the same config file in vnncomp23/ and vnncomp22/ folders
    # but for consistency, we point to the config in vnncomp23/ folder
    cmd += "exp_configs/vnncomp23/acasxu.yaml"

elif args.CATEGORY == "cctsdb_yolo":
    cmd += "exp_configs/vnncomp23/cctsdb_yolo.yaml"

elif args.CATEGORY == "cgan":
    cmd += "exp_configs/vnncomp23/cgan.yaml"

elif args.CATEGORY == "collins_rul_cnn":
    cmd += "exp_configs/vnncomp23/collins-rul-cnn.yaml"

elif args.CATEGORY == "collins_yolo_robustness":
    cmd += "exp_configs/vnncomp23/collins_yolo_robustness.yaml"

elif args.CATEGORY == "dist_shift":
    cmd += "exp_configs/vnncomp23/dist-shift.yaml"

elif args.CATEGORY == "metaroom":
    cmd += "exp_configs/vnncomp23/metaroom.yaml"

elif args.CATEGORY == "ml4acopf":
    cmd += "exp_configs/vnncomp23/ml4acopf.yaml"

elif args.CATEGORY == "tllverifybench":
    cmd += "exp_configs/vnncomp23/tllVerifyBench.yaml"

elif args.CATEGORY == "traffic_signs_recognition":
    cmd += "exp_configs/vnncomp23/gtrsb.yaml"

elif args.CATEGORY == "vggnet16":
    # same config as last year
    cmd += "exp_configs/vnncomp23/vggnet16.yaml"

elif args.CATEGORY == "vit":
    cmd += "exp_configs/vnncomp23/vit.yaml"

elif args.CATEGORY == "yolo":
    cmd += "exp_configs/vnncomp23/yolo-xiangruzh.yaml"

# vnncomp 2024

elif args.CATEGORY == "acasxu_2023":
    cmd += "exp_configs/vnncomp23/acasxu.yaml"

elif args.CATEGORY == "cctsdb_yolo_2023":
    cmd += "exp_configs/vnncomp23/cctsdb_yolo.yaml"

elif args.CATEGORY == "cgan_2023":
    cmd += "exp_configs/vnncomp23/cgan.yaml"

elif args.CATEGORY == "cifar100":
    cmd += "exp_configs/vnncomp24/cifar100.yaml"

elif args.CATEGORY == 'collins_aerospace_benchmark':
    cmd += "exp_configs/vnncomp23/collins_yolo_robustness.yaml"

elif args.CATEGORY == "collins_rul_cnn_2023":
    cmd += "exp_configs/vnncomp23/collins-rul-cnn.yaml"

elif args.CATEGORY == 'cora':
    cmd += "exp_configs/vnncomp24/cora.yaml"

elif args.CATEGORY == "dist_shift_2023":
    cmd += "exp_configs/vnncomp23/dist-shift.yaml"

elif args.CATEGORY == "linearizenn":
    cmd += "exp_configs/vnncomp24/linearizenn.yaml"

elif args.CATEGORY == "lsnc":
    cmd += "exp_configs/vnncomp24/lsnc.yaml"

elif args.CATEGORY == "metaroom_2023":
    cmd += "exp_configs/vnncomp23/metaroom.yaml"

elif args.CATEGORY == "ml4acopf_2023":
    cmd += "exp_configs/vnncomp23/ml4acopf.yaml"

elif args.CATEGORY == 'ml4acopf_2024':
    cmd += "exp_configs/vnncomp24/ml4acopf.yaml"

elif args.CATEGORY == "nn4sys_2023":
    cmd += "exp_configs/vnncomp23/nn4sys.yaml"

elif args.CATEGORY.startswith("safenlp"):
    cmd += "exp_configs/vnncomp24/safenlp.yaml"

elif args.CATEGORY == "tinyimagenet":
    cmd += "exp_configs/vnncomp24/tinyimagenet.yaml"

elif args.CATEGORY == "tllverifybench_2023":
    cmd += "exp_configs/vnncomp23/tllVerifyBench.yaml"

elif args.CATEGORY == "traffic_signs_recognition_2023":
    cmd += "exp_configs/vnncomp23/gtrsb.yaml"

elif args.CATEGORY == "vggnet16_2023":
    cmd += "exp_configs/vnncomp23/vggnet16.yaml"

elif args.CATEGORY == "vit_2023":
    cmd += "exp_configs/vnncomp23/vit.yaml"

elif args.CATEGORY == "yolo_2023":
    cmd += "exp_configs/vnncomp24/yolo-xiangruzh.yaml"

elif args.CATEGORY == "test":
    pass

else:
    exit("CATEGORY {} not supported yet".format(args.CATEGORY))

# test case may run in other args.CATEGORY at the end of them, so we parse them here to allow correct measurement of overhead.
if os.path.split(args.VNNLIB_FILE)[-1] in ['test_' + f + '.vnnlib' for f in ['nano', 'tiny', 'small']]:
    cmd = f"{python_path} {library_path}/abcrown.py --config {library_path}/exp_configs/vnncomp21/test.yaml"
elif 'test_prop' in args.VNNLIB_FILE:
    cmd = f"{python_path} {library_path}/abcrown.py --config {library_path}/exp_configs/vnncomp23/acasxu.yaml"


cmd += " --precompile_jit"
cmd += " --onnx_path " + str(args.ONNX_FILE)
cmd += " --vnnlib_path " + str(args.VNNLIB_FILE)
cmd += " --results_file " + str(args.RESULTS_FILE)
cmd += " --timeout " + str(args.TIMEOUT)

# save adv example to args.RESULTS_FILE
cmd += " --save_adv_example"

# use CROWN bound propagation, when original run triggers OOM, run_instance.sh will add this flag
if args.TRY_CROWN:
    # This also disables the use of output constraints. They are only useful for alpha-CROWN
    cmd += " --bound_prop_method crown --apply_output_constraints_to"

# verify the adv example everytime it's saved
if args.DEBUG:
    cmd += " --eval_adv_example"

# do not use pdg attack during verification
if args.NOPGD:
    cmd += " --pgd_order=skip"

if args.PREPARE:
    cmd += " --prepare_only"

print("\n------------------------- COMMAND ------------------------------")
print(cmd)
print("----------------------------------------------------------------\n")

ret = os.system(cmd)
if ret != 0:
    # avoid original return code to be > 255, reserve its non-zero feature
    sys.exit(int(ret) % 255 + 1)
