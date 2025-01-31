α,β-CROWN (alpha-beta-CROWN): A Fast and Scalable Neural Network Verifier with Efficient Bound Propagation
======================

<p align="center">
<a href="https://abcrown.org/"><img src="https://www.huan-zhang.com/images/upload/alpha-beta-crown/logo_2022.png" width="36%"></a>
</p>

α,β-CROWN (alpha-beta-CROWN) is a neural network verifier based on an efficient
linear bound propagation framework and branch and bound. It can be accelerated
efficiently on **GPUs** and can scale to relatively large convolutional
networks (e.g., millions of parameters). It also supports a wide range of
neural network architectures (e.g., **CNN**, **ResNet**, and various activation
functions), thanks to the versatile
**[auto\_LiRPA](http://github.com/Verified-Intelligence/auto_LiRPA) library developed by us**.
α,β-CROWN can provide provable robustness guarantees against adversarial
attacks and can also verify other general properties of neural networks,
such as [Lyapunov stability](https://arxiv.org/pdf/2404.07956) in control.

α,β-CROWN is the **winning verifier** in [VNN-COMP
2021](https://sites.google.com/view/vnn2021), [VNN-COMP
2022](https://sites.google.com/view/vnn2022), [VNN-COMP
2023](https://sites.google.com/view/vnn2023) and
[VNN-COMP 2024](https://sites.google.com/view/vnn2024) (International Verification of
Neural Networks Competition) with the highest total score, outperforming many
other neural network verifiers on a wide range of benchmarks over 4 years.
Details of competition results can be found in [VNN-COMP 2021
slides](https://docs.google.com/presentation/d/1oM3NqqU03EUqgQVc3bGK2ENgHa57u-W6Q63Vflkv000/edit#slide=id.ge4496ad360_14_21),
[report](https://arxiv.org/abs/2109.00498),
[VNN-COMP 2022 report](https://arxiv.org/pdf/2212.10376.pdf),
[VNN-COMP 2023 slides](https://github.com/ChristopherBrix/vnncomp2023_results/blob/main/SCORING/slides.pdf) and [report](https://arxiv.org/abs/2312.16760),
and [VNN-COMP 2024 slides](https://docs.google.com/presentation/d/1RvZWeAdTfRC3bNtCqt84O6IIPoJBnF4jnsEvhTTxsPE/edit) and [report](https://www.arxiv.org/pdf/2412.19985).

The α,β-CROWN team is created and led by Prof. [Huan Zhang](https://huan-zhang.com/) at UIUC with contributions from multiple institutions. See the **list of contributors** [below](#developers-and-copyright).
α,β-CROWN combines our efforts in neural network verification in a series of
papers building up the bound propagation framework since 2018. See [Publications](#publications) below.

News (2024 - )
----------------------

- α,β-CROWN is the winner of [VNN-COMP 2024](https://sites.google.com/view/vnn2024) and is **ranked top-1** in all benchmarks (including 12 [regular track](https://github.com/ChristopherBrix/vnncomp2024_results/blob/main/SCORING/latex/results_regular_track.pdf) and 9 [extended track](https://github.com/ChristopherBrix/vnncomp2024_results/blob/main/SCORING/latex/results_extended_track.pdf) benchmarks). (08/2024)
- BICCOS ([Zhou et al., NeurIPS 2024](https://openreview.net/pdf?id=FwhM1Zpyft)) is a new cutting plane generation method scalable to large networks and outperforms the MIP-based cuts in GCP-CROWN. (11/2024)

Supported Features
----------------------

<p align="center">
<a href="https://arxiv.org/pdf/2103.06624.pdf"><img src="https://www.huan-zhang.com/images/upload/alpha-beta-crown/banner.png" width="100%"></a>
</p>

Our verifier consists of the following core algorithms:

* **CROWN** ([Zhang et al., 2018](https://arxiv.org/pdf/1811.00866.pdf)): the basic linear bound propagation framework for neural networks.
* **auto_LiRPA** ([Xu et al. 2020](https://arxiv.org/pdf/2002.12920.pdf)): linear bound propagation for general computational graphs.
* **α-CROWN** ([Xu et al., 2021](https://arxiv.org/pdf/2011.13824.pdf)): incomplete verification with gradient optimized bound propagation.
* **β-CROWN** ([Wang et al., 2021](https://arxiv.org/pdf/2103.06624.pdf)): complete verification with bound propagation and branch and bound for ReLU networks.
* **GenBaB** ([Shi et al., 2024](https://arxiv.org/pdf/2405.21063.pdf)): Branch and bound for general nonlinear functions.
* **GCP-CROWN** ([Zhang et al., 2022](https://arxiv.org/pdf/2208.05740.pdf)): CROWN-like bound propagation with general cutting plane constraints.
* **BaB-Attack** ([Zhang et al., 2022](https://proceedings.mlr.press/v162/zhang22ae/zhang22ae.pdf)): Branch and bound based adversarial attack for tackling hard instances.
* **MIP** ([Tjeng et al., 2017](https://arxiv.org/pdf/1711.07356.pdf)): mixed integer programming (slow but can be useful on small models).
* **INVPROP** ([Kotha et al., 2023](https://arxiv.org/pdf/2302.01404.pdf)): tightens bounds with constraints on model outputs, and computes provable preimages for neural networks.
* **BICCOS** ([Zhou et al., 2024](https://openreview.net/pdf?id=FwhM1Zpyft)): an effective cutting plane generation method outperforming the MIP-based cuts in GCP-CROWN.

The bound propagation engine in α,β-CROWN is implemented as a separate library, **[auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) ([Xu et al. 2020](https://arxiv.org/pdf/2002.12920.pdf))**, for computing symbolic bounds for general computational graphs. We support these neural network architectures:

* Layers: fully connected (FC), convolutional (CNN), pooling (average pool and max pool), transposed convolution
* Activation functions or nonlinear functions: ReLU, sigmoid, tanh, arctan, sin, cos, tan, gelu, pow, multiplication and self-attention
* Residual connections, Transformers, LSTMs, and other irregular graphs

We support the following verification specifications:

* Lp norm perturbation (p=1,2,infinity, as often used in robustness verification)
* VNNLIB format input (at most two layers of AND/OR clause, as used in VNN-COMP)
* Any linear specifications on neural network output (which can be added as a linear layer)

We provide many example configurations in
[`complete_verifier/exp_configs`](/complete_verifier/exp_configs) directory to
start with:
* MNIST: MLP and CNN models (small models to help you get started)
* CIFAR-10, CIFAR-100, TinyImageNet: CNN and ResNet models with high dimensional inputs
* ACASXu, NN4sys, ML4ACOPF and other low input dimension models

And more examples in other repositories:
* Stability verification of NN controllers: [Verified-Intelligence/Lyapunov_Stable_NN_Controllers](https://github.com/Verified-Intelligence/Lyapunov_Stable_NN_Controllers)
* Branch-and-bound for models with non-ReLU nonlinearities and high dimensional inputs: [GenBaB](https://huggingface.co/datasets/zhouxingshi/GenBaB)

See the [Guide on Algorithm
Selection](/complete_verifier/docs/abcrown_usage.md#guide-on-algorithm-selection)
to find the most suitable example to get started.

Installation and Setup
----------------------

α,β-CROWN is tested on Python 3.11 and PyTorch 2.3.1 (recent versions may also work).
It can be installed
easily into a conda environment. If you don't have conda, you can install
[miniconda](https://docs.conda.io/en/latest/miniconda.html).

Clone our verifier including the [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) submodule:
```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
```

Setup the conda environment from [`environment.yaml`](complete_verifier/environment.yaml)
with pinned dependencies versions (CUDA>=12.1 is required):
```bash
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name alpha-beta-crown
# install all dependents into the alpha-beta-crown environment
conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown
# activate the environment
conda activate alpha-beta-crown
```

Alternatively, you may use `pip`
(if you want to add α,β-CROWN to your existing environment,
or if your system is not compatible with [`environment.yaml`](complete_verifier/environment.yaml)).
It is highly recommended to have a pre-installed PyTorch that matches your system and our version requirement
(see [PyTorch Get Started](https://pytorch.org/get-started)).
Then, you can run:
```bash
(cd auto_LiRPA; pip install -e .)
pip install -r complete_verifier/requirements.txt
```

Unless you use MIP-based verification algorithms, a Gurobi license is *not needed* (in most use cases).
If you want to use MIP-based verification algorithms (which are feasible only for small models), you need to
install a Gurobi license with the `grbgetkey` command.  If you don't have
access to a license, by default, the above installation procedure includes a
free and restricted license, which is actually sufficient for many relatively
small NNs. If you use the GCP-CROWN verifier, an installation of IBM CPlex
solver is required. Instructions to install the CPlex solver can be found
in the [VNN-COMP benchmark instructions](/complete_verifier/docs/vnn_comp.md#installation)
or the [GCP-CROWN instructions](https://github.com/tcwangshiqi-columbia/GCP-CROWN).

If you want to run α,β-CROWN verifier on the VNN-COMP benchmarks
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
python abcrown.py --config exp_configs/tutorial_examples/cifar_resnet_2b.yaml
```

You can find explanations for the most useful parameters in [this example config
file](/complete_verifier/exp_configs/tutorial_examples/cifar_resnet_2b.yaml). For detailed usage
and tutorial examples, please see the [Usage
Documentation](/complete_verifier/docs/abcrown_usage.md).  We also provide a
large range of examples in the
[`complete_verifier/exp_configs`](/complete_verifier/exp_configs) folder.


Publications
----------------------

If you use our verifier in your work, **please kindly cite our papers**:
- **CROWN** ([Zhang
et al., 2018](https://arxiv.org/pdf/1811.00866.pdf)),
**auto_LiRPA** ([Xu et al., 2020](https://arxiv.org/pdf/2002.12920.pdf)),
**α-CROWN** ([Xu et al., 2021](https://arxiv.org/pdf/2011.13824.pdf)),
**β-CROWN** ([Wang et al., 2021](https://arxiv.org/pdf/2103.06624.pdf)),
**GenBaB** ([Shi et al. 2024](https://arxiv.org/pdf/2405.21063.pdf)),
**GCP-CROWN** ([Zhang et al., 2022](https://arxiv.org/pdf/2208.05740.pdf)), and
**BICCOS** ([Zhou et al., NeurIPS 2024](https://openreview.net/pdf?id=FwhM1Zpyft)).
- **[Kotha et al., 2023](https://arxiv.org/pdf/2302.01404.pdf)** if you use constraints on the outputs of neural networks.
- **[Salman et al., 2019](https://arxiv.org/pdf/1902.08722)**,
if your work involves the convex relaxation of the NN verification.
- **[Zhang et al.
2022](https://proceedings.mlr.press/v162/zhang22ae/zhang22ae.pdf)**,
if you use our branch-and-bound based adversarial attack (falsifier).
We provide bibtex entries at the end of this section.

α,β-CROWN represents our continued efforts on neural network verification:

* **CROWN** ([Zhang et al. NeurIPS 2018](https://arxiv.org/pdf/1811.00866.pdf)) is a very efficient bound propagation based verification algorithm. CROWN propagates a linear inequality backward through the network and utilizes linear bounds to relax activation functions.

* The **"convex relaxation barrier"** ([Salman et al., NeurIPS 2019](https://arxiv.org/pdf/1902.08722)) paper concludes that optimizing the ReLU relaxation allows CROWN (referred to as a "greedy" primal space solver) to achieve the same solution as linear programming (LP) based verifiers.

* **auto_LiRPA** ([Xu et al., NeurIPS 2020](https://arxiv.org/pdf/2002.12920.pdf)) is a generalization of CROWN on general computational graphs and we also provide an efficient GPU implementation, the [auto\_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) library.

* **α-CROWN** (sometimes referred to as optimized CROWN or optimized LiRPA) is used in the Fast-and-Complete verifier ([Xu et al., ICLR 2021](https://arxiv.org/pdf/2011.13824.pdf)), which jointly optimizes intermediate layer bounds and final layer bounds in CROWN via variable α. α-CROWN typically has greater power than LP since LP cannot cheaply tighten intermediate layer bounds.

* **β-CROWN** ([Wang et al., NeurIPS 2021](https://arxiv.org/pdf/2103.06624.pdf)) incorporates ReLU split constraints in branch and bound (BaB) into the CROWN bound propagation procedure via an additional optimizable parameter β. The combination of efficient and GPU-accelerated bound propagation with branch and bound produces a powerful and scalable neural network verifier.

* **BaB-Attack** ([Zhang et al., ICML 2022](https://proceedings.mlr.press/v162/zhang22ae/zhang22ae.pdf)) is a strong falsifier (adversarial attack) based on branch and bound, which can find adversarial examples for hard instances where gradient or input-space-search based methods cannot succeed.

* **GCP-CROWN** ([Zhang et al., NeurIPS 2022](https://arxiv.org/pdf/2208.05740.pdf)) enables the use of general cutting planes methods for neural network verification in a GPU-accelerated and very efficient bound propagation framework. Cutting planes can significantly strengthen bound tightness.

* **INVPROP** ([Kotha et al., NeurIPS 2023](https://arxiv.org/pdf/2302.01404.pdf)) handles constraints on the outputs of neural networks which enables tight and provable bounds on the preimage of a neural network. We demonstrated several applications, including OOD detection, backward reachability analysis for NN-controlled systems, and tightening bounds for robustness verification.

* **BICCOS** ([Zhou et al., NeurIPS 2024](https://openreview.net/pdf?id=FwhM1Zpyft)) generates effective cutting planes during branch-and-bound to tighten verification bounds. The cutting plane generation process is efficient and scalable and does not require a MIP solver.

* **GenBaB** ([Shi et al., TACAS 2025](https://arxiv.org/pdf/2405.21063.pdf)) enables branch-and-bound based verification for general nonlinear functions, achieving significant improvements on verifying neural networks with non-ReLU nonlinearties (such as Transformers), and enabling new applications that contain complicated nonlinear functions on the output of neural networks, such as [ML for AC Optimal Power Flow](https://github.com/AI4OPT/ml4acopf_benchmark).

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

@inproceedings{kotha2023provably,
 author = {Kotha, Suhas and Brix, Christopher and Kolter, J. Zico and Dvijotham, Krishnamurthy and Zhang, Huan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {80270--80290},
 publisher = {Curran Associates, Inc.},
 title = {Provably Bounding Neural Network Preimages},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/fe061ec0ae03c5cf5b5323a2b9121bfd-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}

@inproceedings{zhou2024scalable,
  title={Scalable Neural Network Verification with Branch-and-bound Inferred Cutting Planes},
  author={Zhou, Duo and Brix, Christopher and Hanasusanto, Grani A and Zhang, Huan},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}

@inproceedings{shi2024genbab,
  title={Neural Network Verification with Branch-and-Bound for General Nonlinearities},
  author={Shi, Zhouxing and Jin, Qirui and Kolter, Zico and Jana, Suman and Hsieh, Cho-Jui and Zhang, Huan},
  booktitle={International Conference on Tools and Algorithms for the Construction and Analysis of Systems},
  year={2025}
}
```

Developers and Copyright
----------------------

The α,β-CROWN verifier is currently being developed by a multi-institutional team:

**Team lead**:
* Huan Zhang (huan@huan-zhang.com), UIUC

**Current developers**:
* Zhouxing Shi (zhouxingshichn@gmail.com), UCLA (Student Lead)
* Xiangru Zhong (xiangruzh0915@gmail.com), UIUC
* Jorge Chavez (jorgejc2@illinois.edu), UIUC
* Duo Zhou (duozhou2@illinois.edu), UIUC
* Christopher Brix (brix@cs.rwth-aachen.de), RWTH Aachen University
* Keyi Shen (keyis2@illinois.edu), UIUC
* Hongji Xu (hx84@duke.edu), Duke University (intern with Prof. Huan Zhang)
* Kaidi Xu (kx46@drexel.edu), Drexel University
* Hao Chen (haoc8@illinois.edu), UIUC
* Keyu Lu (keyulu2@illinois.edu), UIUC

Past developers:
* Sanil Chawla (schawla7@illinois.edu), UIUC
* Linyi Li (linyi2@illinois.edu), UIUC
* Zhuolin Yang (zhuolin5@illinois.edu), UIUC
* Zhuowen Yuan (realzhuowen@gmail.com), UIUC
* Qirui Jin (qiruijin@umich.edu), University of Michigan
* Shiqi Wang (sw3215@columbia.edu), Columbia University
* Yihan Wang (yihanwang@ucla.edu), UCLA
* Jinqi (Kathryn) Chen (jinqic@cs.cmu.edu), CMU

The team acknowledges the financial and advisory support from Prof. Zico Kolter (zkolter@cs.cmu.edu), Prof. Cho-Jui Hsieh (chohsieh@cs.ucla.edu), Prof. Suman Jana (suman@cs.columbia.edu), Prof. Bo Li (lbo@illinois.edu), and Prof. Xue Lin (xue.lin@northeastern.edu).

Our library is released under the BSD 3-Clause license. A copy of the license is included [here](LICENSE).
