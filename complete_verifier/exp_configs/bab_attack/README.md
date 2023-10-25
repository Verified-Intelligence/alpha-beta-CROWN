# BaB-Attack: A Branch and Bound Framework for Stronger Adversarial Attacks of ReLU Networks

We proposed BaB-attack, a counter-example searching method inspired by branch
and bound based NN verifiers.  BaB-attack aims to solve *hard instances* that
no existing adversarial attacks can quickly solve. Instead of searching
adversarial examples relying on gradient guidance in input space, our bab
attack can search systematically in activation space using branch and bound.
More details can be found in [our
paper](https://proceedings.mlr.press/v162/zhang22ae/zhang22ae.pdf):

**A Branch and Bound Framework for Stronger Adversarial Attacks of ReLU Networks**
ICML 2022
Huan Zhang\*, Shiqi Wang\*, Kaidi Xu, Yihan Wang, Suman Jana, Cho-Jui Hsieh, and Zico Kolter (\*Equal contribution)

<p align="center">
<a href="https://proceedings.mlr.press/v162/zhang22ae/zhang22ae.pdf"><img src="https://www.huan-zhang.com/images/upload/bab-attack/bab_attack.png" width="90%"></a>
</p>

## Installation and setup

Our BaB attack is built based on the state-of-the-art [α,β-CROWN
(alpha-beta-CROWN)](https://github.com/huanzhang12/alpha-beta-CROWN) NN
verifier. We run our BaB attack on Python 3.7+ and PyTorch 1.11. It can be
installed easily into a conda environment. If you don't have conda, you can
install [miniconda](https://docs.conda.io/en/latest/miniconda.html). We
tested the setup and reproducing process in the September 2022 release of
the α,β-CROWN verifier.

```bash
# BaB-Attack is integrated into the alpha-beta-CROWN verifier.
git clone https://github.com/huanzhang12/alpha-beta-CROWN
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name bab-attack
conda env create -f complete_verifier/environment.yaml --name bab-attack  # install all dependents into the bab-attack environment
conda activate bab-attack  # activate the environment
```

## Reproducing results reported in paper

The configuration files of each experiment are located at the
[`complete_verifier/exp_configs/bab_attack`](https://github.com/huanzhang12/alpha-beta-CROWN/tree/main/complete_verifier/exp_configs/bab_attack)
folder. The hard instance indices that most SOTA gradient based adversarial
attacks fail are summarized in
[`complete_verifier/exp_configs/bab_attack/attack_idx/`](https://github.com/huanzhang12/alpha-beta-CROWN/tree/main/complete_verifier/exp_configs/bab_attack/attack_idx).
The hard instances that can be attacked directly using a MIP solver (can be very
slow) have their indices listed in `mip_unsafe_idx.txt`, and those that cannot be
attacked using a MIP solver are listed in `mip_unknown_idx.txt`. These `.txt`
indices are the "hard instances" reported in our paper.

To run each hard instance individually, you can directly specify the image
index (idx) in test set and then use the main script
[`complete_verifier/abcrown.py`](https://github.com/huanzhang12/alpha-beta-CROWN/blob/main/complete_verifier/abcrown.py)
with --start idx --end idx+1. Here are some examples:

```python
python abcrown.py --config exp_configs/bab_attack/mnist_MadryCNN_no_maxpool_tiny.yaml --start 758 --end 759  # MNIST Small-Adv image idx 758 is a hard instance
python abcrown.py --config exp_configs/bab_attack/cifar_cnn_a_mix.yaml.yaml --start 232 --end 233  # CIFAR CNN-A-Mix image idx 232 is a hard instance
```

To make the results in our paper easier to reproduce and more practical, we
limit the use of CPU cores to 8 (as well as baseline methods). To increase the
number of CPU cores used, you can modify the `solver: mip: parallel_solvers`
option (default to 8) in configuration files.

To run all examples in the hard instance indices file, you can use the
`--data_idx_file` parameter. We give the commands to run the models on
our paper below:

```python
# MNIST model A
python abcrown.py --config exp_configs/bab_attack/mnist_MadryCNN_no_maxpool_tiny.yaml --data_idx_file exp_configs/bab_attack/attack_idx/mnist_MadryCNN_no_maxpool_tiny/mip_unsafe_idx.txt --start 0 --end 20
python abcrown.py --config exp_configs/bab_attack/mnist_MadryCNN_no_maxpool_tiny.yaml --data_idx_file exp_configs/bab_attack/attack_idx/mnist_MadryCNN_no_maxpool_tiny/mip_unknown_idx.txt --start 0 --end 27
# MNIST model B
python abcrown.py --config exp_configs/bab_attack/mnist_cnn_a_adv.yaml --data_idx_file exp_configs/bab_attack/attack_idx/mnist_cnn_a_adv/mip_unsafe_idx.txt --start 0 --end 20
python abcrown.py --config exp_configs/bab_attack/mnist_cnn_a_adv.yaml --data_idx_file exp_configs/bab_attack/attack_idx/mnist_cnn_a_adv/mip_unknown_idx.txt --start 0 --end 50
# CIFAR model C
python abcrown.py --config exp_configs/bab_attack/cifar_cnn_a_adv.yaml --data_idx_file exp_configs/bab_attack/attack_idx/cifar_cnn_a_adv/mip_unsafe_idx.txt --start 0 --end 3
python abcrown.py --config exp_configs/bab_attack/cifar_cnn_a_adv.yaml --data_idx_file exp_configs/bab_attack/attack_idx/cifar_cnn_a_adv/mip_unknown_idx.txt --start 0 --end 50
# CIFAR model D
python abcrown.py --config exp_configs/bab_attack/cifar_cnn_a_adv_alt.yaml --data_idx_file exp_configs/bab_attack/attack_idx/cifar_cnn_a_adv_alt/mip_unsafe_idx.txt --start 0 --end 1
python abcrown.py --config exp_configs/bab_attack/cifar_cnn_a_adv_alt.yaml --data_idx_file exp_configs/bab_attack/attack_idx/cifar_cnn_a_adv_alt/mip_unknown_idx.txt --start 0 --end 50
# CIFAR model E
python abcrown.py --config exp_configs/bab_attack/cifar_cnn_a_mix.yaml --data_idx_file exp_configs/bab_attack/attack_idx/cifar_cnn_a_mix/mip_unsafe_idx.txt --start 0 --end 3
python abcrown.py --config exp_configs/bab_attack/cifar_cnn_a_mix.yaml --data_idx_file exp_configs/bab_attack/attack_idx/cifar_cnn_a_mix/mip_unknown_idx.txt --start 0 --end 150
# CIFAR model F
python abcrown.py --config exp_configs/bab_attack/cifar_marabou_small.yaml --data_idx_file exp_configs/bab_attack/attack_idx/cifar_marabou_small/mip_unsafe_idx.txt --start 0 --end 9
python abcrown.py --config exp_configs/bab_attack/cifar_marabou_small.yaml --data_idx_file exp_configs/bab_attack/attack_idx/cifar_marabou_small/mip_unknown_idx.txt --start 0 --end 50
```
