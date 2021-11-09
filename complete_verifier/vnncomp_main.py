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


if args.CATEGORY == "acasxu":
    cmd = f"{python_path} {library_path}/bab_verification_input_split.py --dataset ACASXU --batch_size 1000 --share_slopes"

elif args.CATEGORY == "cifar10_resnet":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --dataset CIFAR  --batch_size 2000 --branching_reduceop max --lr_beta 0.01 --pgd_order skip"

elif args.CATEGORY == "cifar2020":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --dataset CIFAR --batch_size 200 --branching_candidates 10 --branching_reduceop max --lr_beta 0.01 --pgd_order after"

elif args.CATEGORY == "eran":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --dataset MNIST --batch_size 500 --complete_verifier bab-refine --pgd_order after"

elif args.CATEGORY == "marabou-cifar10":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --dataset CIFAR --batch_size 1000 --branching_candidates 5 --iteration 50 --lr_beta 0.5 --pgd_order before"

elif args.CATEGORY == "mnistfc":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --dataset MNIST --batch_size 500 --branching_candidate 5 --branching_reduceop max --lr_beta 0.003 --complete_verifier bab-refine --pgd_order after"

elif args.CATEGORY == "nn4sys":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --dataset NN4SYS --complete_verifier skip"

elif args.CATEGORY == "oval21":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --dataset CIFAR --batch_size 2000 --branching_candidates 10 --branching_reduceop max --lr_beta 0.01 --pgd_order after"

elif args.CATEGORY == "test":
    if 'test_prop' in args.VNNLIB_FILE:
        cmd = f"{python_path} {library_path}/bab_verification_input_split.py --dataset ACASXU"
    else:
        cmd = f"{python_path} {library_path}/bab_verification_general.py --dataset TEST --pgd_order skip"

elif args.CATEGORY == "verivital":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --dataset MNIST --complete_verifier mip --pgd_order after"

else:
    exit("CATEGORY {} not supported yet".format(args.CATEGORY))

if 'test_prop' in args.VNNLIB_FILE:  # Handle mismatched category name for the test instance, to allow correct measurement of overhead.
        cmd = f"{python_path} {library_path}/bab_verification_input_split.py --dataset ACASXU"
elif 'test_nano' in args.VNNLIB_FILE or 'test_tiny' in args.VNNLIB_FILE or 'test_small' in args.VNNLIB_FILE:
        cmd = f"{python_path} {library_path}/bab_verification_general.py --dataset TEST --pgd_order skip"
cmd += " --onnx_path " + str(args.ONNX_FILE)
cmd += " --vnnlib_path " + str(args.VNNLIB_FILE)
cmd += " --results_file " + str(args.RESULTS_FILE)
cmd += " --timeout " + str(args.TIMEOUT)

print("\n------------------------- COMMAND ------------------------------")
print(cmd)
print("----------------------------------------------------------------\n")

os.system(cmd)
