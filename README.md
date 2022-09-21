α,β-CROWN (alpha-beta-CROWN): A Fast and Scalable Neural Network Verifier with Efficient Bound Propagation
======================

<p align="center">
<a href="https://arxiv.org/pdf/2103.06624.pdf"><img src="https://www.huan-zhang.com/images/upload/alpha-beta-crown/logo_2022.png" width="36%"></a>
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
2021](https://sites.google.com/view/vnn2021) and [VNN-COMP
2022](https://sites.google.com/view/vnn2022) (International Verification of
Neural Networks Competition) with the highest total score, outperforming many
other neural network verifiers on a wide range of benchmarks over 2 years.
Details of competition results can be found in [VNN-COMP 2021
slides](https://docs.google.com/presentation/d/1oM3NqqU03EUqgQVc3bGK2ENgHa57u-W6Q63Vflkv000/edit#slide=id.ge4496ad360_14_21),
[report](https://arxiv.org/abs/2109.00498) and [VNN-COMP 2022 slides (page
73)](https://drive.google.com/file/d/1nnRWSq3plsPvOT3V-drAF5D8zWGu02VF/view?usp=sharing).

Supported Features
----------------------

<p align="center">
<a href="https://arxiv.org/pdf/2103.06624.pdf"><img src="https://www.huan-zhang.com/images/upload/alpha-beta-crown/banner.png" width="100%"></a>
</p>

Our verifier consists of the following core algorithms:

* **β-CROWN** ([Wang et al. 2021](https://arxiv.org/pdf/2103.06624.pdf)): complete verification with **CROWN** ([Zhang et al. 2018](https://arxiv.org/pdf/1811.00866.pdf)) and branch and bound
* **α-CROWN** ([Xu et al., 2021](https://arxiv.org/pdf/2011.13824.pdf)): incomplete verification with optimized CROWN bound
* **GCP-CROWN** ([Zhang et al. 2021](https://arxiv.org/pdf/2208.05740.pdf)): CROWN-like bound propagation with general cutting plane constraints.
* **BaB-Attack** ([Zhang et al. 2021](https://proceedings.mlr.press/v162/zhang22ae/zhang22ae.pdf)): Branch and bound based adversarial attack for tackling hard instances.
* **MIP** ([Tjeng et al., 2017](https://arxiv.org/pdf/1711.07356.pdf)): mixed integer programming (slow but can be useful on small models).

We support these neural network architectures:

* Layers: fully connected (FC), convolutional (CNN), pooling (average pool and max pool), transposed convolution
* Activation functions: ReLU (incomplete/complete verification); sigmoid, tanh, arctan, sin, cos, tan (incomplete verification)
* Residual connections and other irregular graphs

We support the following verification specifications:

* Lp norm perturbation (p=1,2,infinity, as often used in robustness verification)
* VNNLIB format input (at most two layers of AND/OR clause, as used in VNN-COMP 2021 and 2022)
* Any linear specifications on neural network output (which can be added as a linear layer)

We provide many example configurations in
[`complete_verifier/exp_configs`](/complete_verifier/exp_configs) directory to
start with:

* MNIST: MLP and CNN models
* CIFAR-10, CIFAR-100, TinyImageNet: CNN and ResNet models
* ACASXu, NN4sys and other low input-dimension models

See the [Guide on Algorithm
Selection](/complete_verifier/docs/abcrown_usage.md#guide-on-algorithm-selection)
to find the most suitable example to get started.

Installation and Setup
----------------------

α,β-CROWN is tested on Python 3.7+ and PyTorch 1.11. It can be installed
easily into a conda environment. If you don't have conda, you can install
[miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name alpha-beta-crown
# install all dependents into the alpha-beta-crown environment
conda env create -f complete_verifier/environment.yml --name alpha-beta-crown
# activate the environment
conda activate alpha-beta-crown
```

If you use the α-CROWN and/or β-CROWN verifiers (which covers the most use
cases), a Gurobi license is *not needed*.  If you want to use MIP based
verification algorithms (feasible only for small MLP models), you need to
install a Gurobi license with the `grbgetkey` command.  If you don't have
access to a license, by default the above installation procedure includes a
free and restricted license, which is actually sufficient for many relatively
small NNs. If you use the GCP-CROWN verifier, an installation of IBM CPlex
solver is required. Instructions to install the CPlex solver can be found
in the [VNN-COMP benchmark instructions](/complete_verifier/docs/vnn_comp.md#installation)
or the [GCP-CROWN instructions](https://github.com/tcwangshiqi-columbia/GCP-CROWN).

If you prefer to install packages manually rather than using a prepared conda
environment, you can refer to this [installation
script](/vnncomp_scripts/install_tool_general.sh).

If you want to run α,β-CROWN verifier on the VNN-COMP 2021 and 2022 benchmarks
(e.g., to make a comparison to a new verifier), you can follow [this
guide](/complete_verifier/docs/vnn_comp.md).

Instructions
----------------------

We provide a unified front-end for the verifier, `abcrown.py`.  All parameters
for the verifier are defined in a `yaml` config file. For example, to run
robustness verification on a CIFAR-10 ResNet network, you just run:

```bash
conda activate alpha-beta-crown  # activate the conda environment
cd complete_verifier
python abcrown.py --config exp_configs/cifar_resnet_2b.yaml
```

You can find explanations for most useful parameters in [this example config
file](/complete_verifier/exp_configs/cifar_resnet_2b.yaml). For detailed usage
and tutorial examples please see the [Usage
Documentation](/complete_verifier/docs/abcrown_usage.md).  We also provide a
large range of examples in the
[`complete_verifier/exp_configs`](/complete_verifier/exp_configs) folder.


Publications
----------------------

If you use our verifier in your work, **please kindly cite our CROWN**([Zhang
et al., 2018](https://arxiv.org/pdf/1811.00866.pdf)),  **α-CROWN** ([Xu et al.,
2021](https://arxiv.org/pdf/2011.13824.pdf)), **β-CROWN**([Wang et al.,
2021](https://arxiv.org/pdf/2103.06624.pdf)) and **GCP-CROWN**([Zhang et al.,
2022](https://arxiv.org/pdf/2208.05740.pdf)) papers. If your work involves the
convex relaxation of the NN verification please kindly cite [Salman et al.,
2019](https://arxiv.org/pdf/1902.08722).  If your work deals with
ResNet/DenseNet, LSTM (recurrent networks), Transformer or other complex
architectures, or model weight perturbations please kindly cite [Xu et al.,
2020](https://arxiv.org/pdf/2002.12920.pdf). If you use our branch and bound
based adversarial attack (falsifier), please cite [Zhang et al.
2022](https://proceedings.mlr.press/v162/zhang22ae/zhang22ae.pdf).

α,β-CROWN combines our existing efforts on neural network verification:

* **CROWN** ([Zhang et al. NeurIPS 2018](https://arxiv.org/pdf/1811.00866.pdf)) is a very efficient bound propagation based verification algorithm. CROWN propagates a linear inequality backwards through the network and utilizes linear bounds to relax activation functions.

* The **"convex relaxation barrier"** ([Salman et al., NeurIPS 2019](https://arxiv.org/pdf/1902.08722)) paper concludes that optimizing the ReLU relaxation allows CROWN (referred to as a "greedy" primal space solver) to achieve the same solution as linear programming (LP) based verifiers.

* **LiRPA** ([Xu et al., NeurIPS 2020](https://arxiv.org/pdf/2002.12920.pdf)) is a generalization of CROWN on general computational graphs and we also provide an efficient GPU implementation, the [auto\_LiRPA](https://github.com/KaidiXu/auto_LiRPA) library.

* **α-CROWN** (sometimes referred to as optimized CROWN or optimized LiRPA) is used in the Fast-and-Complete verifier ([Xu et al., ICLR 2021](https://arxiv.org/pdf/2011.13824.pdf)), which jointly optimizes intermediate layer bounds and final layer bounds in CROWN via variable α. α-CROWN typically has greater power than LP since LP cannot cheaply tighten intermediate layer bounds.

* **β-CROWN** ([Wang et al., NeurIPS 2021](https://arxiv.org/pdf/2103.06624.pdf)) incorporates split constraints in branch and bound (BaB) into the CROWN bound propagation procedure via an additional optimizable parameter β. The combination of efficient and GPU accelerated bound propagation with branch and bound produces a powerful and scalable neural network verifier.

* **BaB-Attack** ([Zhang et al., ICML 2022](https://proceedings.mlr.press/v162/zhang22ae/zhang22ae.pdf)) is a strong falsifier (adversarial attack) based on branch and bound, which can find adversarial examples for hard instances where gradient or input-space-search based methods cannot succeed.

* **GCP-CROWN** ([Zhang et al., NeurIPS 2022](https://arxiv.org/pdf/2208.05740.pdf)) enables the use of general cutting planes methods for neural network verification in a GPU-accelerated and very efficient bound propagation framework.

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

@InProceedings{zhang22babattack,
  title = 	 {A Branch and Bound Framework for Stronger Adversarial Attacks of {R}e{LU} Networks},
  author =       {Zhang, Huan and Wang, Shiqi and Xu, Kaidi and Wang, Yihan and Jana, Suman and Hsieh, Cho-Jui and Kolter, Zico},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  volume = 	 {162},
  pages = 	 {26591--26604},
  year = 	 {2022},
}

@article{zhang2022general,
  title={General Cutting Planes for Bound-Propagation-Based Neural Network Verification},
  author={Zhang, Huan and Wang, Shiqi and Xu, Kaidi and Li, Linyi and Li, Bo and Jana, Suman and Hsieh, Cho-Jui and Kolter, J Zico},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}
```

Developers and Copyright
----------------------

The α,β-CROWN verifier is developed by a team from CMU, UCLA, Drexel University, Columbia University and UIUC:

Team lead:  
* Huan Zhang (huan@huan-zhang.com), CMU

Main developers:  
* Kaidi Xu (kx46@drexel.edu), Drexel University
* Zhouxing Shi (zshi@cs.ucla.edu), UCLA
* Shiqi Wang (sw3215@columbia.edu), Columbia University

Contributors:  
* Linyi Li (linyi2@illinois.edu), UIUC
* Jinqi (Kathryn) Chen (jinqic@cs.cmu.edu), CMU
* Zhuolin Yang (zhuolin5@illinois.edu), UIUC
* Yihan Wang (yihanwang@ucla.edu), UCLA

Advisors:  
* Zico Kolter (zkolter@cs.cmu.edu), CMU
* Cho-Jui Hsieh (chohsieh@cs.ucla.edu), UCLA
* Suman Jana (suman@cs.columbia.edu), Columbia University
* Bo Li (lbo@illinois.edu), UIUC
* Xue Lin (xue.lin@northeastern.edu), Northeastern University

Our library is released under the BSD 3-Clause license. A copy of the license is included [here](LICENSE).

