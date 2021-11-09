Detailed Usage of the α,β-CROWN Verifier
====================

We provide three frontends for the α,β-CROWN verifier, depending on different use cases:

* [`robustness_verifier.py`](/complete_verifier/robustness_verifier.py): for Lp norm robustness verification, often used to certify the robustness of a neural network
* [`bab_verification_general.py`](/complete_verifier/bab_verification_general.py): for verifying VNNLIB format specifications, as in the MNIST/CIFAR models in VNN-COMP 2021
* [`bab_verification_input_split.py`](/complete_verifier/bab_verification_input_split.py): for branch and bound on input space, such as the ACASXu model.

We detailed the usage for each frontend below.

Robustness Verifier
--------------------

We provide a frontend for robustness verification, `robustness_verifier.py`,
for the popular use case of neural network verification to certify adversarial
robustness. 

All configuration parameters can be specified in a YAML format configuration file, and the verifier can be invoked by:

```bash
python robustness_verifier.py --config PATH_TO_CONFIG_FILE.yaml
```

All available configuration parameters are [listed
here](robustness_verifier_all_params.yaml).  We also provide a large range
of examples in the
[`complete_verifier/exp_configs`](/complete_verifier/exp_configs) folder. For example:

```bash
# Run robustness verification on the ResNet-2B model.
python robustness_verifier.py --config exp_configs/cifar_resnet_2b.yaml
```

In the example above, we run L inifity norm robustness verification with
eps=2/255 on the first 100 test images from CIFAR-10 dataset. You can change
verification specifications and verifier hyperparameters in
[`resnet_2b.yaml`](/complete_verifier/exp_configs/cifar_resnet_2b.yaml)

You may also use command line arguments for the verifier to override the
options specified in the configuration file. Run `python robustness_verifier.py
--help` to see all options.  Some common command line arguments for
`robustness_verifier.py` are listed below.

* `--load`: path to PyTorch model checkpoint.
* `--model`: model name. You need to add your own model to `util.py`.
* `--dataset`: dataset name. You need to add your own dataset to `util.py`, by changing the `load_verification_dataset()` function. Please be careful with data normalization settings.
* `--batch_size`: batch size used for running branch and bound. Typically you need a reasonably large batch size that fits onto your GPU.
* `--timeout`: timeout for branching and bound verifier.
* `--norm`: perturbation norm, e.g., 'inf' (default, Linf norm) or 2 (L2 norm).
* `--epsilon`: perturbation epsilon. Please be careful with data normalization settings.
* `--pgd_order`: can be 'before' (run PGD attack before incomplete verifier), 'after' (run PGD attack after incomplete verifier), or 'skip' (do not run PGD).
* `--lr_alpha`: learning rate for alpha-CROWN (default 0.01) in branch and bound.
* `--lr_beta`: learning rate for beta-CROWN (default 0.05) in branch and bound.
* `--iteration`: number of iterations for optimizing alpha and beta parameters in branch and bound (default 50; reduce it for larger models).
* `--lr_decay`: learning rate decay factor (default 0.98; you need to increase it if you run more iterations).
* `--lr_init_alpha`: learning rate for alpha-CROWN (default 0.1) in initial incomplete verification.
* `--init_iteration`: number of iterations for alpha-CROWN (default 100) in initial incomplete verification.
* `--start`: dataset example index to start verification.
* `--end`: dataset example index to end verification.
* `--device`: can be 'cuda' (default, run verifier on GPU) or 'cpu'.

For example, if you want to run robustness verification on the ResNet-2B model, changing epsilon from 2/255 (specified in config file) to 1/255:

```bash
python robustness_verifier.py --config exp_configs/cifar_resnet_2b.yaml --epsilon 0.00392156862745098
```

This allows you to quickly change some parameters without editing the config file.

General Property (VNNLIB) Verifier
-----------------------------------

The `bab_verification_general.py` frontend is for verifying a VNNLIB format
specifications with onnx files (used in VNN-COMP 2021).  Similar to
`robustness_verifier.py`, this frontend also takes a `yaml` format configration
file as its input.  We provide some example configuration files in the
[`exp_configs/vnncomp`](/complete_verifier/exp_configs/vnncomp) folder. Before
you run these models, you should first clone the `vnncomp21` repository which contains
data and models:

```bash
# Clone the VNN-COMP 2021 repository into the same parent folder as the alpha-beta-CROWN verifier
git clone https://github.com/stanleybak/vnncomp2021.git
# Run the verifier on the CIFAR10_ResNet benchmark
cd alpha-beta-CROWN/complete_verifier
python bab_verification_general.py --config exp_confis/vnncomp/cifar10_resnet.yaml
```
This frontend will go over a CSV file (like [this one](https://github.com/stanleybak/vnncomp2021/blob/main/benchmarks/cifar10_resnet/cifar10_resnet_instances.csv)) under the [`root_path`](https://github.com/KaidiXu/CROWN-GENERAL/blob/master/complete_verifier/exp_configs/vnncomp/cifar10-resnet.yaml#L3) (point it to the vnncomp2021 directory), which
contains the onnx model path, VNNLIB property path and timeout threshold (in second). Each line is a problem to verify.

All available configuration parameters are [listed
here](bab_verification_general_all_params.yaml).  We also provide many examples in the
[`exp_configs/vnncomp`](/complete_verifier/exp_configs/vnncomp) folder.

If you want to run α,β-CROWN verifier on the entire VNN-COMP 2021 benchmarks
(e.g., to make a comparison to a new verifier), you can follow [this
guide](vnn_comp.md).

Verifier with Branch and Bound of Input Space (ACASXu)
-----------------------------------

The `bab_verification_input_split.py` frontend is for verifying ACASXu dataset
using splits on input space.  (the `bab_verification_general.py` frontend uses
splits on ReLU neurons). To run the ACASXu dataset, first clone the VNN-COMP
2021 repository (as it contains the dataset) and then run
`bab_verification_input_split.py`:

```bash
# Clone the VNN-COMP 2021 repository into the same parent folder as the alpha-beta-CROWN verifier
git clone https://github.com/stanleybak/vnncomp2021.git
# Run the verifier
cd alpha-beta-CROWN/complete_verifier
python bab_verification_input_split.py --config exp_configs/vnncomp/acasxu.yaml
```
