α,β-CROWN (alpha-beta-CROWN): Complete and Incomplete Neural Network Verification with Efficient Bound Propagations
----------------------

α,β-CROWN (alpha-beta-CROWN) is the winning verifier in [VNN-COMP
2021](https://sites.google.com/view/vnn2021) with the highest total score
(details of competition results can be found
[in the slides here](https://docs.google.com/presentation/d/1oM3NqqU03EUqgQVc3bGK2ENgHa57u-W6Q63Vflkv000/edit#slide=id.ge4496ad360_14_21)).
α,β-CROWN combines our existing efforts on neural network verification:

* **CROWN** ([Zhang et al. NeurIPS 2018](https://arxiv.org/pdf/1811.00866.pdf)) is a very efficient bound propagation based verification algorithm. CROWN propagates a linear inequality backwards through the network and utilizes linear bounds to relax activation functions.

* **LiRPA** ([Xu et al., NeurIPS 2020](https://arxiv.org/pdf/2002.12920.pdf)) is a generalization of CROWN on general computational graphs and we also provide an efficient GPU implementation, the [auto\_LiRPA](https://github.com/KaidiXu/auto_LiRPA) library.

* The **"convex relaxation barrier"** ([Salman et al., NeurIPS 2019](https://arxiv.org/pdf/1902.08722)) paper concludes that optimizing the ReLU relaxation allows CROWN (referred to as a "greedy" primal space solver) to achieve the same solution as linear programming (LP) based verifiers.

* **α-CROWN** (sometimes referred to as optimized CROWN or optimized LiRPA) is used in the Fast-and-Complete verifier ([Xu et al., ICLR 2021](https://arxiv.org/pdf/2011.13824.pdf)), which jointly optimizes intermediate layer bounds and final layer bounds in CROWN via variable α. α-CROWN typically has greater power than LP since LP cannot cheaply tighten intermediate layer bounds.

* **β-CROWN** ([Wang et al. 2021](https://arxiv.org/pdf/2103.06624.pdf)) incorporates split constraints in branch and bound (BaB) into the CROWN bound propagation procedure via an additional optimizable parameter β. The combination of efficient and GPU accelerated bound propagation with branch and bound produces a powerful and scalable neural network verifier.

The original code used in the competition can be found in the `vnncomp2021` branch, and we are currently preparing a new release with code cleanups and better documentation.

Run α,β-CROWN verifier on VNN-COMP 2021 benchmarks
----------------------

Here we provide general instructions for running our verifier on VNN-COMP 2021
benchmarks using the latest version of our verifier.  We assume `conda` is
available on your system; if not, you should first install
[conda](https://docs.conda.io/en/latest/miniconda.html).

First, clone and install our α,β-CROWN verifier:

```bash
# Clone verifier code.
git clone https://github.com/huanzhang12/alpha-beta-CROWN.git
cd alpha-beta-CROWN
# Create conda environment with fixed package versions.
conda env create -f environment.yml
source activate alpha-beta-crown
# Install the auto_LiRPA library.
python setup.py develop
# Get python installation path.
export VNNCOMP_PYTHON_PATH=$(python -c 'import os; print(os.path.dirname(os.path.realpath("/proc/self/exe")))')
echo "Please run \"export VNNCOMP_PYTHON_PATH=${VNNCOMP_PYTHON_PATH}\" before you run vnncomp2021 scripts"
```

Gurobi should be automatically installed via the command above, and you need to
activate it using the `grbgetkey` command.  A Gurobi license is required for
`eran`, `mnistfc`, `marabou-cifar10` and `verivital` benchmarks; other
benchmarks do not depend on Gurobi. If you don't have access to a full Gurobi
license, by default the Gurobi installation includes a free and restricted
license, which is actually sufficient for many relatively small NNs.

Then clone the VNN-COMP 2021 scripts and give it the correct paths to the tool
and conda environment (here I assume the conda environment is installed to
`${HOME}/miniconda3/envs/alpha-beta-crown` (the command above should print
the right value for `VNNCOMP_PYTHON_PATH`), and the verifier code is cloned to
`${HOME}/alpha-beta-CROWN`:

```bash
git clone https://github.com/stanleybak/vnncomp2021.git
cd vnncomp2021
export VNNCOMP_PYTHON_PATH=${HOME}/miniconda3/envs/alpha-beta-crown/bin
./run_all_categories.sh v1 ${HOME}/alpha-beta-CROWN/vnncomp_scripts $(pwd) results_cifar10_resnet.csv cifar10_resnet 0
```

Then results csv for the `cifar10_resnet` benchmark will be saved in
`results_cifar10_resnet.csv`. Change the benchmark name to run other
benchmarks. Note that in the `vnncomp_scripts/prepare_instance.sh` script we
kill all python processes before each run, and you may want to comment out
these `killall` commands if you have other python processes running at the same
time.

## Publications

If you find our verifier useful, please kindly cite our papers:

```
@article{zhang2018efficient,
  title={Efficient Neural Network Robustness Certification with General Activation Functions},
  author={Zhang, Huan and Weng, Tsui-Wei and Chen, Pin-Yu and Hsieh, Cho-Jui and Daniel, Luca},
  journal={Advances in Neural Information Processing Systems},
  volume={31},
  pages={4939--4948},
  year={2018},
  url={https://arxiv.org/pdf/1811.00866.pdf}
}

@article{xu2020automatic,
  title={Automatic perturbation analysis for scalable certified robustness and beyond},
  author={Xu, Kaidi and Shi, Zhouxing and Zhang, Huan and Wang, Yihan and Chang, Kai-Wei and Huang, Minlie and Kailkhura, Bhavya and Lin, Xue and Hsieh, Cho-Jui},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

@article{salman2019convex,
  title={A Convex Relaxation Barrier to Tight Robustness Verification of Neural Networks},
  author={Salman, Hadi and Yang, Greg and Zhang, Huan and Hsieh, Cho-Jui and Zhang, Pengchuan},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  pages={9835--9846},
  year={2019}
}

@inproceedings{xu2021fast,
    title={{Fast and Complete}: Enabling Complete Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers},
    author={Kaidi Xu and Huan Zhang and Shiqi Wang and Yihan Wang and Suman Jana and Xue Lin and Cho-Jui Hsieh},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=nVZtXBI6LNn}
}

@article{wang2021beta,
  title={{Beta-CROWN}: Efficient bound propagation with per-neuron split constraints for complete and incomplete neural network verification},
  author={Wang, Shiqi and Zhang, Huan and Xu, Kaidi and Lin, Xue and Jana, Suman and Hsieh, Cho-Jui and Kolter, J Zico},
  journal={arXiv preprint arXiv:2103.06624},
  year={2021}
}
```

## Developers and Copyright

The α,β-CROWN verifier is developed by a team:
(\* indicates equal contribution)

* Huan Zhang\* (CMU), huan@huan-zhang.com (Team lead)
* Kaidi Xu\* (Northeastern University), xu.kaid@northeastern.edu
* Shiqi Wang\* (Columbia University), sw3215@columbia.edu
* Zhouxing Shi (UCLA) zshi@cs.ucla.edu
* Yihan Wang (UCLA) yihanwang@ucla.edu
* Xue Lin (Northeastern University), xue.lin@northeastern.edu (advisor)
* Suman Jana (Columbia University), suman@cs.columbia.edu (advisor)
* Cho-Jui Hsieh (UCLA), chohsieh@cs.ucla.edu (advisor)
* Zico Kolter (CMU), zkolter@cs.cmu.edu (advisor)

Our library is released under the BSD 3-Clause license.

