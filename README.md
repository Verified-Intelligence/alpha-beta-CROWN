α,β-CROWN (alpha-beta-CROWN): A Fast and Scalable Neural Network Verifier with Efficient Bound Propagation
======================

<p align="center">
<a href="https://arxiv.org/pdf/2103.06624.pdf"><img src="https://www.cs.columbia.edu/~tcwangshiqi/images/abcrown_logo.png" width="28%"></a>
</p>

α,β-CROWN (alpha-beta-CROWN) is a neural network verifier based on an efficient
bound propagation algorithm ([CROWN](https://arxiv.org/pdf/1811.00866.pdf)) and
branch and bound. It can be accelerated efficiently on **GPUs** and can scale
to relatively large convolutional networks. It also supports a wide range of
neural network architectures (e.g., **CNN**, **ResNet**, and various activation
functions), thanks to the versatile
[auto\_LiRPA](http://github.com/KaidiXu/auto_LiRPA) library developed by us.
α,β-CROWN can provide **provable robustness guarantees against adversarial
attacks** and can also verify other general properties of neural networks.

α,β-CROWN is the **winning verifier** in [VNN-COMP
2021](https://sites.google.com/view/vnn2021) (International Verification of
Neural Networks Competition) with the highest total score, outperforming 11
other neural network verifiers on a wide range of benchmarks.  Details of
competition results can be found [in the slides
here](https://docs.google.com/presentation/d/1oM3NqqU03EUqgQVc3bGK2ENgHa57u-W6Q63Vflkv000/edit#slide=id.ge4496ad360_14_21)
and [the report here](https://arxiv.org/abs/2109.00498).

<p align="center">
<a href="https://arxiv.org/pdf/2103.06624.pdf"><img src="https://www.huan-zhang.com/images/upload/alpha-beta-crown/banner.png" width="100%"></a>
</p>

Supported Features
----------------------

We support these verification algorithms:

* β-CROWN ([Wang et al. 2021](https://arxiv.org/pdf/2103.06624.pdf)): complete verification with CROWN ([Zhang et al. 2018](https://arxiv.org/pdf/1811.00866.pdf)) and branch and bound
* α-CROWN ([Xu et al., 2021](https://arxiv.org/pdf/2011.13824.pdf)): incomplete verification with optimized CROWN bound
* MIP ([Tjeng et al., 2017](https://arxiv.org/pdf/1711.07356.pdf)): mixed integer programming (slow but can be useful on small models).

We support these neural network architectures:

* Layers: fully connected (FC), convolutional (CNN), pooling (average pool and max pool)
* Activation functions: ReLU (incomplete/complete verification); sigmoid and tanh (incomplete verification)
* Residual connections

We support the following verification specifications:

* Lp norm perturbation (p=1,2,infinity, as often used in robustness verification)
* VNNLIB format input (at most two layers of AND/OR clause, as used in VNN-COMP 2021)
* Any linear specifications on neural network output (which can be added as a linear layer)

We also provide a few example configurations in
[`complete_verifier/exp_configs`](/complete_verifier/exp_configs) directory to
start with:

* MNIST: MLP and CNN models
* CIFAR: CNN and ResNet models
* ACASXu

Installation and Setup
----------------------

α,β-CROWN is based on Python 3.7+ and PyTorch 1.8.x LTS. It can be installed
easily into a conda environment. If you don't have conda, you can install
[miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name alpha-beta-crown
conda env create -f complete_verifier/environment.yml  # install all dependents into the alpha-beta-crown environment
conda activate alpha-beta-crown  # activate the environment
```

If you use the α-CROWN and/or β-CROWN verifiers (which covers the most use
cases), a Gurobi license is *not needed*.  If you want to use MIP based
verification algorithms (feasible only for small MLP models), you need to
install a Gurobi license with the `grbgetkey` command.  If you don't have
access to a license, by default the above installation procedure includes a
free and restricted license, which is actually sufficient for many relatively
small NNs.

If you prefer to install
packages manually, you can refer to this [installation
script](/vnncomp_scripts/install_tool_general.sh).

If you want to run α,β-CROWN verifier on the VNN-COMP 2021 benchmarks (e.g., to make a
comparison to a new verifier), you can follow [this guide](docs/vnn_comp.md).

Instructions
----------------------

We provide three frontends for the verifier, depending on different use cases:

* [`robustness_verifier.py`](/complete_verifier/robustness_verifier.py): for Lp norm robustness verification, often used to certify the robustness of a neural network
* [`bab_verification_general.py`](/complete_verifier/bab_verification_general.py): for verifying VNNLIB format specifications, as in the MNIST/CIFAR models in VNN-COMP 2021
* [`bab_verification_input_split.py`](/complete_verifier/bab_verification_input_split.py): for branch and bound on input space, such as the ACASXu model.

All parameters for the verifier are defined in a `yaml` config file. For example, to run robustness verification on a
CIFAR-10 ResNet network, you just run:

```bash
conda activate alpha-beta-crown  # activate the conda environment
cd complete_verifier
python robustness_verifier.py --config exp_configs/cifar_resnet_2b.yaml
```

You can find explanations for most useful parameters in [this example config
file](/complete_verifier/exp_configs/cifar_resnet_2b.yaml). For detailed usage please see the
[Usage Documentation](docs/usage.md).  We also provide a large range of examples in
the [`complete_verifier/exp_configs`](/complete_verifier/exp_configs) folder.

We have also provided [a tutorial example](https://papercode.cc/a-b-CROWN-Tutorial-MNIST) on colab for running α,β-CROWN.


Run α,β-CROWN (alpha-beta-CROWN) Verifier on Your Own Model
----------------------

It is easy to load and run your customized model using our verifier.

1. Put your PyTorch model definition in
   [`complete_verifier/model_defs.py`](/complete_verifier/model_defs.py#L192), and
   set the [`model/name` field](/complete_verifier/exp_configs/cifar_resnet_2b.yaml#L5) in your configuration file. You can copy [this
   configuration file](/complete_verifier/exp_configs/cifar_resnet_2b.yaml) as a template, or start with a config file in [`complete_verifier/exp_configs`](/complete_verifier/exp_configs) that is very close to your customized model. (This step is not needed
   if you use `bab_verification_general.py` with onnx input)
2. Add your dataloader into the `load_verification_dataset()` function in
   [`complete_verifier/utils.py`](/complete_verifier/utils.py#L463). If you just need
   the standard MNIST or CIFAR datasets, you can just set the [`dataset` field](https://github.com/KaidiXu/CROWN-GENERAL/blob/master/complete_verifier/exp_configs/cifar_resnet_2b.yaml#L8)
   in configuration to `CIFAR` or `MNIST` and use the existing dataloader. Be
   careful with dataset normalization. (This step is not needed if you use
   a `vnnlib` format input with `bab_verification_general.py`, which already contains the data)
3. Set verification specifications and hypereparameters in the config file. All parameters are
   documented [here](/docs/robustness_verifier_all_params.yaml). The default hyperparameters should work reasonably well.
4. Run the verifier with your configuration file, e.g. `python
   robustness_verifier.py --config your_model.yaml`.

For all supported options in config files, please see the [Usage Documentation](docs/usage.md).

We have also provided [tutorial examples](https://papercode.cc/a-b-CROWN-Tutorial-Custom) on colab for α,β-CROWN customization.

Publications
----------------------

If you use our verifier in your work, **please kindly cite our CROWN**([Zhang et
al., 2018](https://arxiv.org/pdf/1811.00866.pdf)),  **α-CROWN** ([Xu et al.,
2021](https://arxiv.org/pdf/2011.13824.pdf)) and **β-CROWN**([Wang et al.,
2021](https://arxiv.org/pdf/2103.06624.pdf)) **papers**. If your work involves the
convex relaxation of the NN verification please kindly cite [Salman et
al., 2019](https://arxiv.org/pdf/1902.08722).  If your work deals with ResNet/DenseNet,
LSTM (recurrent networks), Transformer or other complex architectures, or model weight perturbations
please kindly cite [Xu et al., 2020](https://arxiv.org/pdf/2002.12920.pdf).

α,β-CROWN combines our existing efforts on neural network verification:

* **CROWN** ([Zhang et al. NeurIPS 2018](https://arxiv.org/pdf/1811.00866.pdf)) is a very efficient bound propagation based verification algorithm. CROWN propagates a linear inequality backwards through the network and utilizes linear bounds to relax activation functions.

* **LiRPA** ([Xu et al., NeurIPS 2020](https://arxiv.org/pdf/2002.12920.pdf)) is a generalization of CROWN on general computational graphs and we also provide an efficient GPU implementation, the [auto\_LiRPA](https://github.com/KaidiXu/auto_LiRPA) library.

* The **"convex relaxation barrier"** ([Salman et al., NeurIPS 2019](https://arxiv.org/pdf/1902.08722)) paper concludes that optimizing the ReLU relaxation allows CROWN (referred to as a "greedy" primal space solver) to achieve the same solution as linear programming (LP) based verifiers.

* **α-CROWN** (sometimes referred to as optimized CROWN or optimized LiRPA) is used in the Fast-and-Complete verifier ([Xu et al., ICLR 2021](https://arxiv.org/pdf/2011.13824.pdf)), which jointly optimizes intermediate layer bounds and final layer bounds in CROWN via variable α. α-CROWN typically has greater power than LP since LP cannot cheaply tighten intermediate layer bounds.

* **β-CROWN** ([Wang et al. 2021](https://arxiv.org/pdf/2103.06624.pdf)) incorporates split constraints in branch and bound (BaB) into the CROWN bound propagation procedure via an additional optimizable parameter β. The combination of efficient and GPU accelerated bound propagation with branch and bound produces a powerful and scalable neural network verifier.


We provide bibtex entries below:

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
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

Developers and Copyright
----------------------

The α,β-CROWN verifier is developed by a team from CMU, Northeastern University, Columbia University and UCLA:
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

