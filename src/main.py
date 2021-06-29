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
    cmd = f"{python_path} {library_path}/bab_verification_input_split.py --data ACASXU --batch_size 1000 --share_slopes --no_solve_slope"

elif args.CATEGORY == "cifar10_resnet":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --data CIFAR  --batch_size 2000 --branching_reduceop max --lr_beta 0.01 --pgd_order skip"

elif args.CATEGORY == "cifar2020":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --data CIFAR --batch_size 200 --branching_candidates 10 --branching_reduceop max --lr_beta 0.01"

elif args.CATEGORY == "eran":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --data MNIST --batch_size 500 --complete_verifier bab-refine --pgd_order after"

elif args.CATEGORY == "marabou-cifar10":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --data CIFAR --batch_size 1000 --branching_candidates 5 --iteration 50 --lr_beta 0.5 --complete_verifier bab-refine"

elif args.CATEGORY == "mnistfc":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --data MNIST --batch_size 500 --branching_candidate 5 --branching_reduceop max --lr_beta 0.003 --complete_verifier bab-refine --pgd_order after"

elif args.CATEGORY == "nn4sys":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --data NN4SYS --pgd_order after --complete_verifier skip "

elif args.CATEGORY == "oval21":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --batch_size 2000 --branching_candidates 10 --branching_reduceop max --lr_beta 0.01 --pgd_order after"

elif args.CATEGORY == "test":
    if 'test_prop' in args.VNNLIB_FILE:
        cmd = f"{python_path} {library_path}/bab_verification_input_split.py --data ACASXU"
    else:
        cmd = f"{python_path} {library_path}/bab_verification_general.py --data TEST --pgd_order skip"

elif args.CATEGORY == "verivital":
    cmd = f"{python_path} {library_path}/bab_verification_general.py --data MNIST --pgd_order after --complete_verifier mip"

else:
    exit("CATEGORY {} not supported yet".format(args.CATEGORY))

cmd += " --onnx_path " + str(args.ONNX_FILE)
cmd += " --vnnlib_path " + str(args.VNNLIB_FILE)
cmd += " --results_file " + str(args.RESULTS_FILE)
cmd += " --timeout " + str(args.TIMEOUT)

print("\n------------------------- COMMAND ------------------------------")
print(cmd)
print("----------------------------------------------------------------\n")

os.system(cmd)
