Instructions for running the VNN-COMP 2021 and 2022 benchmarks
----------------

Here we provide general instructions for running our verifier on VNN-COMP 2021 and 2022
benchmarks using the latest version of our verifier.  

## Installation

We assume `conda` is
available on your system; if not, you should first install
[miniconda](https://docs.conda.io/en/latest/miniconda.html).

First, clone and install our α,β-CROWN verifier:

```bash
# Clone verifier code.
git clone https://github.com/huanzhang12/alpha-beta-CROWN.git
cd alpha-beta-CROWN
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name alpha-beta-crown
# Create conda environment with fixed package versions.
conda env create -f complete_verifier/environment.yml --name alpha-beta-crown
conda activate alpha-beta-crown

# Install IBM CPLEX >= 22.1.0
# Download from https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students
chmod +x cplex_studio2210.linux_x86_64.bin  # Any version >= 22.1.0 should work. Change executable name here.
# You can directly run the installer: ./cplex_studio2210.linux_x86_64.bin; the response.txt created below is for non-interactive installation.
cat > response.txt <<EOF
INSTALLER_UI=silent
LICENSE_ACCEPTED=true
EOF
sudo ./cplex_studio2210.linux_x86_64.bin -f response.txt
# Build the C++ code for CPLEX interface. Assumming we are still inside the alpha-beta-CROWN folder.
sudo apt install build-essential  # A modern g++ (>=8.0) is required to compile the code.
# Change CPX_PATH in complete_verifier/CPLEX_cuts/Makefile if you installed CPlex to a non-default location, like inside your home folder.
make -C complete_verifier/CPLEX_cuts/

# Get python installation path. You will need to use this path in later steps.
export VNNCOMP_PYTHON_PATH=$(python -c 'import os; print(os.path.dirname(os.path.realpath("/proc/self/exe")))')
echo "Please run \"export VNNCOMP_PYTHON_PATH=${VNNCOMP_PYTHON_PATH}\" before you run vnncomp scripts."
```

Gurobi should be automatically installed via the command above, and you need to
activate it using the `grbgetkey` command. If you work in academia you can get an academic licenses 
from [here](http://www.gurobi.com/academia/for-universities).  A Gurobi license is required for
`eran`, `mnistfc`, `marabou-cifar10` and `verivital` benchmarks; other
benchmarks do not depend on Gurobi.

CPLEX is needed for the `oval21` benchmark. It is free for students and
academics from
[here](https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students).

## Run VNN-COMP 2021 benchmarks.

Firstt, clone the VNN-COMP 2021 scripts and give it the correct paths to the
tool and conda environment via **setting the `VNNCOMP_PYTHON_PATH` environment
variable**. Here we assume the alpha-beta-crown conda environment is installed
to `${HOME}/miniconda3/envs/alpha-beta-crown` (the command in the section above
should print the right value for `VNNCOMP_PYTHON_PATH`), and the verifier code
is cloned to `${HOME}/alpha-beta-CROWN`:

```bash
git clone https://github.com/stanleybak/vnncomp2021.git
cd vnncomp2021
# Please set this environment variable properly, see above instructions
export VNNCOMP_PYTHON_PATH=${HOME}/miniconda3/envs/alpha-beta-crown/bin
# Please check the path to the alpha-beta-CROWN repository and change it accordingly
./run_all_categories.sh v1 ${HOME}/alpha-beta-CROWN/vnncomp_scripts $(pwd) results_cifar10_resnet.csv "cifar10_resnet" 0  # Example to run the CIFAR10_ResNet benchmark
```

Then results csv for the `cifar10_resnet` benchmark will be saved in
`results_cifar10_resnet.csv`. Change the benchmark name to run other
benchmarks. Note that in the `vnncomp_scripts/prepare_instance.sh` script we
kill all python processes before each run, and you may want to comment out
these `killall` commands if you have other python processes running at the same
time.

The original code used in the competition can be found in the `vnncomp2021`
branch. However, it is always recommended using the latest master version to
run the benchmarks, especially when compared to a newly developed verifier.

## Run VNN-COMP 2022 benchmarks.

The procedure of running α,β-CROWN on VNN-COMP 2022 is quite similar to the
above instructions for VNN-COMP 2021. After the installation of
α,β-CROWN (see above instruction), we just need to clone and setup
VNN-COMP 2022 repository.

```bash
git clone https://github.com/ChristopherBrix/vnncomp2022_benchmarks.git
cd vnncomp2022_benchmarks
# Unzip and download necessary files
./setup.sh
# install DNNV package to convert the vgg model in vggnet16_2022 benchmark
pip install git+https://github.com/dlshriver/DNNV.git@develop
# Please set this environment variable properly, see above instructions
export VNNCOMP_PYTHON_PATH=${HOME}/miniconda3/envs/alpha-beta-crown/bin
# Please check the path to the alpha-beta-CROWN repository and change it accordingly
./run_all_categories.sh v1 ${HOME}/alpha-beta-CROWN/vnncomp_scripts $(pwd) results_cifar100_tinyimagenet_resnet.csv ./counterexamples "cifar100_tinyimagenet_resnet" all  # Example to run the cifar100_tinyimagenet_resnet benchmark
```

The results will be saved in `results_cifar100_tinyimagenet_resnet.csv`. We can also
run multiple benchmarks at one time (may take around 48 hours):

```bash
./run_all_categories.sh v1 ${HOME}/alpha-beta-CROWN/vnncomp_scripts $(pwd) all_results.csv ./counterexamples "oval21 tllverifybench carvana_unet_2022 cifar_biasfield collins_rul_cnn mnist_fc nn4sys reach_prob_density vggnet16_2022 rl_benchmarks sri_resnet_a sri_resnet_b cifar100_tinyimagenet_resnet acasxu cifar2020" all 2>&1 | tee stdout.log
```

Alternatively, there is a more user-friendly way to run each benchmark by
loading a specific config file, for example to run all properties of `cifar100_small`
model in `cifar100_tinyimagenet_resnet` benchmark:

```bash
cd alpha-beta-CROWN/complete_verifier
python abcrown.py --config exp_configs/vnncomp22/cifar100_small_2022.yaml
```

This command will run multiple properties by only calling our verifier once
to avoid import/compiling overhead and should produce same final verification results 
with less running time compare to those with the `./run_all_categories.sh` scripts
(in VNN-COMP 2022 competition the overhead was measured and excluded from scoring).

