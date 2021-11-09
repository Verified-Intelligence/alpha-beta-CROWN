A ResNet Benchmark on CIFAR-10 for Neural Network Verification
-----

We propose a new set of benchmarks of residual networks (ResNet) on CIFAR-10
for neural network verification in this repository.

Currently, most networks evaluated in the literature are feedforward NNs, and
many tools are hardcoded to handle feedforward networks only. To make neural
network verification more useful in practical scenarios, we advocate that tools
should handle more general architectures, and ResNet is the first step towards
this goal. We hope this can provide some incentives for the community to
develop better tools.

**Model details**: We provided two adversarially trained ResNet models on CIFAR-10 with the following structures:

- ResNet-2B with 2 residual blocks: 5 convolutional layers + 2 linear layers
- ResNet-4B with 4 residual blocks: 9 convolutional layers + 2 linear layers

The models can be found in this. PyTorch model definitions are available
[here](resnet.py). Networks are trained using adversarial training with L∞
perturbation epsilon (2/255). We report basic model performance numbers below:

| Model      | # ReLUs | Clean acc. |  PGD acc. <br> ε=2/255  |  PGD acc. <br> ε=1/255 | CROWN/DeepPoly <br> verified acc. <br> ε=2/255 | CROWN/DeepPoly <br> verified acc. <br> ε=1/255 |
|------------|---------|------------|-----------------|----------------|-----------------------------------|-----------------------------------|
| ResNet-2B  |   6244  |    69.25%  |      54.82%     |      62.24%    |   26.88%                          |   57.16%                          |
| ResNet-4B  |  14436  |    77.20%  |      61.41%     |      69.75%    |    0.24%                          |   23.28%                          |

**Data Format**: The input images should be normalized using mean and std
computed from CIFAR-10 training set. The perturbation budget is element-wise,
eps=2/255 on unnormalized images and clipped to the [0, 1] range. We provide
`cifar_eval.py` as a simple PyTorch example of loading data (e.g., data
preprocessing, channel ordering etc).

See the [VNN-COMP 2021
page](https://github.com/stanleybak/vnncomp2021/tree/main/benchmarks/cifar10_resnet)
for the usage of these models in VNN-COMP 2021.

**Citation:** If you use our ResNet benchmarks in your research, please kindly cite our paper:

```
@article{wang2021betacrown,
  title={Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Complete and Incomplete Neural Network Verification},
  author={Wang, Shiqi and Zhang, Huan and Xu, Kaidi and Lin, Xue and Jana, Suman and Hsieh, Cho-Jui and Kolter, Zico},
  journal={arXiv preprint arXiv:2103.06624},
  year={2021}
}
```
