Run the VNN-COMP 21 benchmarks
----------------

Here we provide general instructions for running our verifier on VNN-COMP 2021
benchmarks using the latest version of our verifier.  We assume `conda` is
available on your system; if not, you should first install
[conda](https://docs.conda.io/en/latest/miniconda.html).

First, clone and install our α,β-CROWN verifier:

```bash
# Clone verifier code.
git clone https://github.com/huanzhang12/alpha-beta-CROWN.git
cd alpha-beta-CROWN
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name alpha-beta-crown
# Create conda environment with fixed package versions.
conda env create -f complete_verifier/environment.yml
conda activate alpha-beta-crown
# Get python installation path.
export VNNCOMP_PYTHON_PATH=$(python -c 'import os; print(os.path.dirname(os.path.realpath("/proc/self/exe")))')
echo "Please run \"export VNNCOMP_PYTHON_PATH=${VNNCOMP_PYTHON_PATH}\" before you run vnncomp2021 scripts"
```

Gurobi should be automatically installed via the command above, and you need to
activate it using the `grbgetkey` command.  A Gurobi license is required for
`eran`, `mnistfc`, `marabou-cifar10` and `verivital` benchmarks; other
benchmarks do not depend on Gurobi.

Then clone the VNN-COMP 2021 scripts and give it the correct paths to the tool
and conda environment via **setting the `VNNCOMP_PYTHON_PATH` environment variable**. Here I assume the conda environment is installed to
`${HOME}/miniconda3/envs/alpha-beta-crown` (the command above should print
the right value for `VNNCOMP_PYTHON_PATH`), and the verifier code is cloned to
`${HOME}/alpha-beta-CROWN`:

```bash
git clone https://github.com/stanleybak/vnncomp2021.git
cd vnncomp2021
# Please set this environment variable properly, see above instructions
export VNNCOMP_PYTHON_PATH=${HOME}/miniconda3/envs/alpha-beta-crown/bin
# Please check the path to the alpha-beta-CROWN repository and change it accordingly
./run_all_categories.sh v1 ${HOME}/alpha-beta-CROWN/vnncomp_scripts $(pwd) results_cifar10_resnet.csv cifar10_resnet 0  # Example to run the CIFAR10_ResNet benchmark
```

Then results csv for the `cifar10_resnet` benchmark will be saved in
`results_cifar10_resnet.csv`. Change the benchmark name to run other
benchmarks. Note that in the `vnncomp_scripts/prepare_instance.sh` script we
kill all python processes before each run, and you may want to comment out
these `killall` commands if you have other python processes running at the same
time.

The original code used in the competition can be found in the `vnncomp2021`
branch. However, it is always recommended to use the latest master version to
run the benchmarks, especially when compared to a newly developed verifier.

