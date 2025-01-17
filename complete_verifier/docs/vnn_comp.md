Instructions for running the VNN-COMP benchmarks
----------------

Here we provide general instructions for running our verifier on VNN-COMP
benchmarks using the latest version of our verifier.

## Installation

We assume `conda` is available on your system; if not, you should first install
[miniconda](https://docs.conda.io/en/latest/miniconda.html).

First, clone and install our α,β-CROWN verifier:

```bash
# Clone verifier code.
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name alpha-beta-crown
# Create conda environment with fixed package versions.
conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown
conda activate alpha-beta-crown

# Get python installation path. You will need to use this path in later steps.
export VNNCOMP_PYTHON_PATH=$(python -c 'import os; print(os.path.dirname(os.path.realpath("/proc/self/exe")))')
echo "Please run \"export VNNCOMP_PYTHON_PATH=${VNNCOMP_PYTHON_PATH}\" before you run vnncomp scripts."
```

### Activating Gurobi

Gurobi should be automatically installed via the command above, and you need to
activate it using the `grbgetkey` command. If you work in academia you can get an academic licenses
from [here](http://www.gurobi.com/academia/for-universities).  A Gurobi license is required for
`eran`, `mnistfc`, `marabou-cifar10` and `verivital` benchmarks; other
benchmarks do not depend on Gurobi.

### Installing CPLEX

CPLEX is needed for the `oval21` benchmark. It is free for students and
academics from
[here](https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students).

```bash
# Install IBM CPLEX >= 22.1.0
# Download from https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students
chmod +x cplex_studio2210.linux_x86_64.bin  # Any version >= 22.1.0 should work. Change executable name here.
# You can directly run the installer: ./cplex_studio2210.linux_x86_64.bin; the response.txt created below is for non-interactive installation.
cat > response.txt <<EOF
INSTALLER_UI=silent
LICENSE_ACCEPTED=true
EOF
sudo ./cplex_studio2210.linux_x86_64.bin -f response.txt
# Build the C++ code for CPLEX interface. Assuming we are still inside the alpha-beta-CROWN folder.
sudo apt install build-essential  # A modern g++ (>=8.0) is required to compile the code.
# Change CPX_PATH in complete_verifier/cuts/CPLEX_cuts/Makefile if you installed CPlex to a non-default location, like inside your home folder.
make -C complete_verifier/cuts/CPLEX_cuts/
```

## Prepare VNN-COMP benchmarks

Clone the VNN-COMP benchmarks and scripts:
```bash
git clone https://github.com/stanleybak/vnncomp2021.git

git clone https://github.com/ChristopherBrix/vnncomp2022_benchmarks.git
# Unzip and download necessary files
(cd vnncomp2022_benchmarks; ./setup.sh)

git clone https://github.com/ChristopherBrix/vnncomp2023_benchmarks.git
# Unzip and download necessary files
(cd vnncomp2023_benchmarks; ./setup.sh)

git clone https://github.com/ChristopherBrix/vnncomp2024_benchmarks.git
# Unzip and download necessary files
(cd vnncomp2024_benchmarks; ./setup.sh)
```

Then, change directory to the folder for the year you want to use. For example,
for VNN-COMP 2023, run:
```bash
cd vnncomp2023_benchmarks
```

## Run VNN-COMP benchmarks via VNN-COMP scripts

To run with the VNN-COMP scripts, provide the correct paths to the
verifier and conda environment via **setting the `VNNCOMP_PYTHON_PATH` environment
variable**. Here we assume the alpha-beta-crown conda environment is installed
to `${HOME}/miniconda3/envs/alpha-beta-crown` (the command in the section above
should print the right value for `VNNCOMP_PYTHON_PATH`), and the verifier code
is cloned to `${HOME}/alpha-beta-CROWN`:

```bash
# Please set this environment variable properly, see above instructions
export VNNCOMP_PYTHON_PATH=${HOME}/miniconda3/envs/alpha-beta-crown/bin
# Please check the path to the alpha-beta-CROWN repository and change it accordingly
./run_all_categories.sh v1 ${HOME}/alpha-beta-CROWN/vnncomp_scripts $(pwd) results_vit.csv counterexamples_vit "vit" all  # Example to run the vit benchmark
```

Then results csv for the `vit` benchmark will be saved in
`results_vit.csv`. Change the benchmark name to run other
benchmarks. Note that in the `vnncomp_scripts/prepare_instance.sh` script we
kill all python processes before each run, and you may want to comment out
these `killall` commands if you have other python processes running at the same
time.

We can also run multiple benchmarks at one time:
```bash
# For VNN-COMP 2021
./run_all_categories.sh v1 ${HOME}/alpha-beta-CROWN/vnncomp_scripts $(pwd) all_results.csv ./counterexamples "acasxu cifar10_resnet cifar2020 eran marabou-cifar10 mnistfc nn4sys oval21 verivital" all 2>&1 | tee stdout.log

# For VNN-COMP 2022
./run_all_categories.sh v1 ${HOME}/alpha-beta-CROWN/vnncomp_scripts $(pwd) all_results.csv ./counterexamples "oval21 tllverifybench carvana_unet_2022 cifar_biasfield collins_rul_cnn mnist_fc nn4sys reach_prob_density vggnet16_2022 rl_benchmarks sri_resnet_a sri_resnet_b cifar100_tinyimagenet_resnet acasxu cifar2020" all 2>&1 | tee stdout.log

# For VNN-COMP 2023
./run_all_categories.sh v1 ${HOME}/alpha-beta-CROWN/vnncomp_scripts $(pwd) all_results.csv ./counterexamples "acasxu cgan collins_yolo_robustness metaroom nn4sys tllverifybench vggnet16 yolo cctsdb_yolo collins_rul_cnn dist_shift ml4acopf test traffic_signs_recognition vit" all 2>&1 | tee stdout.log

# For VNN-COMP 2024
./run_all_categories.sh v1 ${HOME}/alpha-beta-CROWN/vnncomp_scripts $(pwd) all_results.csv ./counterexamples "acasxu_2023 cifar100 cora lsnc ml4acopf_2024 traffic_signs_recognition_2023 yolo_2023 cctsdb_yolo_2023 collins_aerospace_benchmark dist_shift_2023 metaroom_2023 nn4sys_2023 tinyimagenet vggnet16_2023 cgan_2023 collins_rul_cnn_2023 linearizenn ml4acopf_2023 safenlp tllverifybench_2023 vit_2023" all 2>&1 | tee stdout.log
```

## Run VNN-COMP benchmarks by config files

Alternatively, there is a more user-friendly way to run each benchmark by
loading a specific config file, for example to run all properties of `cifar100_small`
model in `cifar100_tinyimagenet_resnet` benchmark from VNN-COMP 2022:

```bash
cd alpha-beta-CROWN/complete_verifier
python abcrown.py --config exp_configs/vnncomp22/cifar100_small_2022.yaml
```

This command will run multiple properties by only calling our verifier once
to avoid import/compiling overhead and should produce same final verification results
with less running time compare to those with the `./run_all_categories.sh` scripts
(in the competitions, the overhead was measured and excluded from scoring).

All the config files are stored at `exp_configs/vnncomp21`, `exp_configs/vnncomp22`,
`exp_configs/vnncomp23` and `exp_configs/vnncomp24` for the four years of VNN-COMP respectively.

## Original code used in the competitions

The original code used in the competitions can be found at:
- [VNN-COMP 2021](https://github.com/Verified-Intelligence/alpha-beta-CROWN/tree/vnncomp2021)
- [VNN-COMP 2022](https://github.com/huanzhang12/alpha-beta-CROWN_vnncomp22)
- [VNN-COMP 2023](https://github.com/Verified-Intelligence/alpha-beta-CROWN_vnncomp23)
- [VNN-COMP 2024](https://github.com/Verified-Intelligence/alpha-beta-CROWN_vnncomp2024)

However, it is always recommended using the latest master version to
run the benchmarks, especially when compared to a newly developed verifier.
