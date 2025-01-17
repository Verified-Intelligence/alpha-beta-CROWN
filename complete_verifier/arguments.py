#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""
Arguments parser and config file loader.

When adding new commandline parameters, please make sure to provide a clear and descriptive help message and put it in under a related hierarchy.
"""

import re
import os
import sys
import yaml
import time
import argparse
from collections import defaultdict
import torch


class ConfigHandler:

    def __init__(self):
        self.config_file_hierarchies = {
            # Given a hierarchy for each commandline option. This hierarchy is used in yaml config.
            # For example: "batch_size": ["solver", "propagation", "batch_size"] will be an element in this dictionary.
            # The entries will be created in add_argument() method.
        }
        # Stores all arguments according to their hierarchy.
        self.all_args = {}
        # Parses all arguments with their defaults.
        self.defaults_parser = argparse.ArgumentParser()
        # Parses the specified arguments only. Not specified arguments will be ignored.
        self.no_defaults_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        # Help message for each configuration entry.
        self.help_messages = defaultdict(str)
        # Add all common arguments.
        self.add_common_options()
        # Path to the config file
        self.file = None
        # Parse an empty commandline to get all default arguments.
        self.default_args = vars(self.defaults_parser.parse_args([]))

    def add_common_options(self):
        """
        Add all parameters that are shared by different front-ends.
        """

        # We must set how each parameter will be presented in the config file, via the "hierarchy" parameter.
        # Global Configurations, not specific for a particular algorithm.

        # The "--config" option does not exist in our parameter dictionary.
        self.add_argument('--config', type=str, help='Path to YAML format config file.', hierarchy=None)

        h = ["general"]
        self.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                          help='Select device to run verifier, cpu or cuda (GPU).',
                          hierarchy=h + ["device"])
        self.add_argument("--seed", type=int, default=100, help='Random seed.',
                          hierarchy=h + ["seed"])
        self.add_argument("--conv_mode", default="patches", choices=["patches", "matrix"],
                          help='Convolution mode during bound propagation: "patches" mode (default) is very efficient, but may not support all architecture; "matrix" mode is slow but supports all architectures.',
                          hierarchy=h + ["conv_mode"])
        self.add_argument("--deterministic", action='store_true',
                          help='Run code in CUDA deterministic mode, which has slower performance but better reproducibility.',
                          hierarchy=h + ["deterministic"])
        self.add_argument("--double_fp", action='store_true',
                          help='Use double precision floating point. GPUs with good double precision support are preferable (NVIDIA P100, V100, A100, H100; AMD Radeon Instinct MI50, MI100).',
                          hierarchy=h + ["double_fp"])
        self.add_argument("--loss_reduction_func", default="sum",
                          help='When batch size is not 1, this reduction function is applied to reduce the bounds into a scalar (options are "sum" and "min").',
                          hierarchy=h + ["loss_reduction_func"])
        self.add_argument("--no_sparse_alpha", action='store_false',
                          help='Enable/disable sparse alpha.',
                          hierarchy=h + ["sparse_alpha"])
        self.add_argument("--no_sparse_interm", action='store_false',
                          help='Enable/disable sparse intermediate bounds.',
                          hierarchy=h + ["sparse_interm"])
        self.add_argument("--save_adv_example", action='store_true',
                          help='Save returned adversarial example in file.',
                          hierarchy=h + ["save_adv_example"])
        self.add_argument("--verify_onnxruntime_output", action='store_true',
                          help='Check if the inference results of counterexample are the same on both PyTorch and ONNXRuntime.',
                          hierarchy=h + ["verify_onnxruntime_output"])
        self.add_argument("--eval_adv_example", action='store_true',
                          help='Whether to validate the saved adversarial example.',
                          hierarchy=h + ["eval_adv_example"])
        self.add_argument("--show_adv_example", action='store_true',
                          help='Print the adversarial example.',
                          hierarchy=h + ["show_adv_example"])
        self.add_argument("--precompile_jit", action='store_true',
                          help='Precompile jit kernels to speed up after jit-wrapped functions, but will cost extra time at the beginning.',
                          hierarchy=h + ["precompile_jit"])
        self.add_argument("--prepare_only", action='store_true',
                          help='Prepare to run the instance (e.g., cache vnnlib and converted onnx files) without running the actual verification.',
                          hierarchy=h + ["prepare_only"])
        self.add_argument('--complete_verifier',
                          choices=["bab", "mip", "bab-refine", "skip", "auto", "Customized"], default="bab",
                          help='Complete verification verifier. "bab": branch and bound with beta-CROWN or GCP-CROWN; '
                               '"mip": mixed integer programming (MIP) formulation; '
                               '"bab-refine": branch and bound with intermediate layer bounds computed by MIP; '
                               '"Customized": customized verifier, need to specially def by user.',
                          hierarchy=h + ["complete_verifier"])
        self.add_argument('--no_incomplete', action='store_false', dest='incomplete',
                          help='Enable/Disable initial alpha-CROWN incomplete verification (disable this can save GPU memory).',
                          hierarchy=h + ["enable_incomplete_verification"])
        self.add_argument("--csv_name", type=str, default=None,
                          help='Name of .csv file containing a list of properties to verify (VNN-COMP specific).',
                          hierarchy=h + ["csv_name"])
        self.add_argument("--results_file", type=str, default='out.txt',
                          help='Path to results file.',
                          hierarchy=h + ["results_file"])
        self.add_argument("--root_path", type=str, default='',
                          help='Root path of the specification folder if using vnnlib.',
                          hierarchy=h + ["root_path"])
        self.add_argument("--deterministic_opt", action="store_true",
                          help="To ensure that the returned alphas match the returned bounds, so that we can reproduce the bounds using the returned alphas during debugging.",
                          hierarchy=h + ["deterministic_opt"])
        self.add_argument("--graph_optimizer", type=str, default='Customized("custom_graph_optimizer", "default_optimizer")',
                          help='BoundedModule model graph optimizer function name. For examples of customized graph optimizer, please see the config files for '
                               'the gtrsb benchmark in VNN-COMP 2023.',
                          hierarchy=h + ["graph_optimizer"])
        self.add_argument("--buffer_has_batchdim", action="store_true",
                          help='In most cases, the shape of buffers in an ONNX graph do not have a batch dimension. '
                               'Enabling this option will help load models with a buffer object that has a batch dimension. '
                               'In most case this can be inferred in the verifier automatically, and this option is not needed.',
                          hierarchy=h + ["buffer_has_batchdim"])
        self.add_argument("--save_output", action='store_true',
                          help='Save output for test.',
                          hierarchy=h + ["save_output"])
        self.add_argument("--output_file", type=str, default='out.pkl',
                          help="Path to the output file.",
                          hierarchy=h + ["output_file"])
        self.add_argument("--return_optimized_model", action='store_true',
                          help="Return the model with optimized bounds after incomplete verification is done.",
                          hierarchy=h + ["return_optimized_model"])


        h = ["model"]
        self.add_argument("--model", type=str, default=None, help='Model name. Will be evaluated as a python statement.',
                          hierarchy=h + ["name"])
        self.add_argument("--load_model", type=str, default=None,
                          help='Load pretrained model from this specified path.',
                          hierarchy=h + ["path"])
        self.add_argument("--onnx_path", type=str, default=None, help='Path to .onnx model file.',
                          hierarchy=h + ["onnx_path"])
        self.add_argument("--onnx_path_prefix", type=str, default='',
                          help='Add a prefix to .onnx model path to correct malformed csv files.',
                          hierarchy=h + ["onnx_path_prefix"])
        self.add_argument("--cache_onnx_conversion", action='store_true',
                          help='Cache the model converted from ONNX.',
                          hierarchy=h + ["cache_onnx_conversion"])
        self.add_argument("--debug_onnx", action='store_true',
                          help='Debug onnx conversion.',
                          hierarchy=h + ["debug_onnx"])
        self.add_argument('--onnx_quirks', type=str, default=None,
                          help="Load onnx model with quirks to workaround onnx model issue. "
                               "This string will be passed to onnx2pytorch as the 'quirks' argument, and it is typically a literal of a python dict, e.g., \"{'Reshape': {'fix_batch_size: True'}}\".",
                          hierarchy=h + ["onnx_quirks"])
        self.add_argument("--input_shape", nargs='+', type=int, default=None,
                          help='Specified input shape of the model.'
                               'Usually the shape can be automatically determined from dataset or onnx model, but some onnx models may have an incompatible shape (without batch dim). '
                               'You can specify shape explicitly here.'
                               'The shape should be (-1, input_shape) like (-1, 3, 32, 32) and -1 indicates the batch dim.',
                          hierarchy=h + ["input_shape"])
        self.add_argument("--onnx_loader", type=str, default='default_onnx_and_vnnlib_loader',
                          help='ONNX model loader function name. Can be the Customized() primitive; for examples of customized model loaders, please see the config files for '
                               'the marabou-cifar10 benchmark in VNN-COMP 2021 and the Carvana benchmark in VNN-COMP 2022.',
                          hierarchy=h + ["onnx_loader"])
        self.add_argument("--onnx_optimization_flags", nargs='+',
                          choices=["merge_bn", "merge_linear", "merge_gemm",
                                   "remove_ineffective_layers",
                                   "remove_relu_in_last_layer"],
                          default=[], help='Onnx graph optimization config.',
                          hierarchy=h + ["onnx_optimization_flags"])
        self.add_argument("--onnx_vnnlib_joint_optimization_flags", nargs='+',
                          choices=["peel_off_last_softmax_layer", "none"],
                          default="none", help='Joint optimization that changes both onnx model and vnnlib.',
                          hierarchy=h + ["onnx_vnnlib_joint_optimization_flags"])
        self.add_argument('--check_optimized', action='store_true',
                          help='Check the optimized onnx file instead the original one when converting to pytorch. This is used when input shape is changed during optimization.',
                          hierarchy=h + ["check_optimized"])
        self.add_argument("--flatten_final_output", action='store_true',
                          help='Manually add a flatten layer at the end of the model.',
                          hierarchy=h + ["flatten_final_output"])
        self.add_argument("--optimize_graph", type=str,
                          help="Specify a custom function for optimizing the graph on the BoundedModule.",
                          hierarchy=h + ["optimize_graph"])
        self.add_argument('--model_with_jacobian', action='store_true',
                          help='Indicate that the model contains JacobianOP.',
                          hierarchy=h + ['with_jacobian'])

        h = ["data"]
        self.add_argument("--start", type=int, default=0, help='Start from the i-th property in specified dataset.',
                          hierarchy=h + ["start"])
        self.add_argument("--end", type=int, default=10000, help='End with the (i-1)-th property in the dataset.',
                          hierarchy=h + ["end"])
        self.add_argument("--select_instance", type=int, nargs='+', default=None,
                          help='Select a list of instances to verify.',
                          hierarchy=h + ["select_instance"])
        # FIXME --num_outputs should be inferred automatically.
        self.add_argument('--num_outputs', type=int, default=10,
                          help="Number of classes for classification problem.",
                          hierarchy=h + ["num_outputs"])
        self.add_argument("--mean", nargs='+', type=float, default=0.0,
                          help='Mean vector used in data preprocessing.',
                          hierarchy=h + ["mean"])
        self.add_argument("--std", nargs='+', type=float, default=1.0,
                          help='Std vector used in data preprocessing.',
                          hierarchy=h + ["std"])
        self.add_argument('--pkl_path', type=str, default=None,
                          help="Load properties to verify from a .pkl file (only used for oval20 dataset).",
                          hierarchy=h + ["pkl_path"])
        self.add_argument("--dataset", type=str, default=None,
                          help="Dataset name (only if not using specifications from a .csv file). Dataset must be defined in utils.py. For customized data, checkout custom/custom_model_data.py.",
                          hierarchy=h + ["dataset"])
        self.add_argument("--filter_path", type=str, default=None,
                          help='A filter in pkl format contains examples that will be skipped (not used).',
                          hierarchy=h + ["data_filter_path"])
        self.add_argument("--data_idx_file", type=str, default=None,
                          help='A text file with a list of example IDs to run.',
                          hierarchy=h + ["data_idx_file"])

        h = ["specification"]
        self.add_argument("--spec_type", type=str, default='lp', choices=['lp', 'box', 'bound'],
                          help='Type of verification specification. "lp" = L_p norm, "box" = element-wise lower and upper bound provided by dataloader.',
                          hierarchy=h + ["type"])
        self.add_argument("--robustness_type", type=str, default="verified-acc",
                          choices=["verified-acc", "runnerup", "clean-acc", "specify-target", "all-positive"],
                          help='For robustness verification: verify against all labels ("verified-acc" mode), or just the runnerup labels ("runnerup" mode), '
                               'or using a specified label in dataset ("specify-target" mode, only used for oval20). Not used when a VNNLIB spec is used.',
                          hierarchy=h + ["robustness_type"])
        self.add_argument("--norm", type=float, default='inf',
                          help='Lp-norm for epsilon perturbation in robustness verification (1, 2, inf).',
                          hierarchy=h + ["norm"])
        self.add_argument("--epsilon", type=float, default=None,
                          help='Set perturbation size (Lp norm). If not set, a default value may be used based on dataset loader.',
                          hierarchy=h + ["epsilon"])
        self.add_argument("--epsilon_min", type=float, default=0.,
                          help='Set an optional minimum perturbation size (Lp norm).',
                          hierarchy=h + ["epsilon_min"])
        self.add_argument("--vnnlib_path", type=str, default=None,
                          help='Path to .vnnlib specification file. Will override any Lp/robustness verification arguments.', hierarchy=h + ["vnnlib_path"])
        self.add_argument("--vnnlib_path_prefix", type=str, default='',
                          help='Add a prefix to .vnnlib specs path to correct malformed csv files.',
                          hierarchy=h + ["vnnlib_path_prefix"])
        self.add_argument("--rhs_offset", type=float, default=None,
                          help='Adding an offset to RHS.',
                          hierarchy=h + ['rhs_offset'])

        h = ["solver"]
        self.add_argument("--batch_size", type=int, default=64,
                          help='Batch size in bound solver (number of parallel splits).',
                          hierarchy=h + ["batch_size"])
        self.add_argument("--auto_enlarge_batch_size", action='store_true',
                          help='Automatically increase batch size based on --batch_size in bab '
                               'if current VRAM usage < 45%%, only support input_split.',
                          hierarchy=h + ["auto_enlarge_batch_size"])
        self.add_argument('--min_batch_size_ratio', type=float, default=0.1,
                          help='The minimum batch size ratio in each iteration (splitting multiple layers if the number of domains is smaller than min_batch_size_ratio * batch_size).',
                          hierarchy=h + ["min_batch_size_ratio"])
        self.add_argument("--use_float64_in_last_iteration", action='store_true',
                          help='Use double fp (float64) at the last iteration in alpha/beta CROWN.',
                          hierarchy=h + ["use_float64_in_last_iteration"])
        self.add_argument('--early_stop_patience', type=int, default=10,
                          help='Number of iterations that we will start considering early stop if tracking no improvement.',
                          hierarchy=h + ["early_stop_patience"])
        self.add_argument('--start_save_best', type=float, default=0.5,
                          help='Start to save best optimized bounds when i > int(iteration*start_save_best). Early iterations are skipped for better efficiency.',
                          hierarchy=h + ["start_save_best"])
        self.add_argument('--bound_prop_method', default="alpha-crown",
                          choices=["alpha-crown", "crown", "forward", "forward+crown",
                                   "alpha-forward", "crown-ibp", "init-crown", "ibp",
                                   "dynamic-forward", "dynamic-forward+crown", "dynamic-forward+backward"],
                          help='Bound propagation method used for incomplete verification and input split based branch and bound.',
                          hierarchy=h + ["bound_prop_method"])
        self.add_argument('--init_bound_prop_method', default="same",
                          choices=["same", "alpha-crown", "crown", "forward",
                                   "forward+crown", "alpha-forward", "crown-ibp",
                                   "init-crown"],
                          help='Bound propagation method used for the initial bound in input split based branch and bound. '
                               'If "same" is specified, then it will use the same method as "bound_prop_method".',
                          hierarchy=h + ["init_bound_prop_method"])
        self.add_argument("--prune_after_crown", action='store_true',
                          help='After CROWN pass, prune verified labels before starting the alpha-CROWN pass.',
                          hierarchy=h + ["prune_after_crown"])
        self.add_argument("--optimize_disjuncts_separately", action='store_true',
                          help="If set, each neuron computes separate bounds for each disjunct. If set, do not set prune_after_crown=True.",
                          hierarchy=h + ["optimize_disjuncts_separately"])

        h = ["solver", "crown"]
        self.add_argument('--crown_batch_size', type=int, default=int(1e9),
                          help='Batch size in batched CROWN.',
                          hierarchy=h + ["batch_size"])
        self.add_argument('--max_crown_size', type=int, default=int(1e9),
                          help='Max output size in CROWN (when there are too many output neurons, only part of them will be bounded by CROWN).',
                          hierarchy=h + ["max_crown_size"])
        self.add_argument("--activation_bound_option", default='adaptive',
                          choices=["adaptive", "same-slope", "zero-lb", "one-lb"],
                          help='Options for specifying the the way to initialize CROWN bounds for acvitaions.',
                          hierarchy=h + ["activation_bound_option"])
        self.add_argument("--compare_crown_with_ibp", action="store_true",
                          help="Compare CROWN bounds with IBP bounds given existing intermediate bounds.",
                          hierarchy=h + ["compare_crown_with_ibp"])

        h = ["solver", "alpha-crown"]
        self.add_argument('--no_alpha', action='store_false', dest='alpha',
                          help='Disable/Enable alpha crown.',
                          hierarchy=h + ["alpha"])
        self.add_argument("--lr_init_alpha", type=float, default=0.1,
                          help='Learning rate for the optimizable parameter alpha in alpha-CROWN bound.',
                          hierarchy=h + ["lr_alpha"])
        self.add_argument('--init_iteration', type=int, default=100,
                          help='Number of iterations for alpha-CROWN incomplete verifier.',
                          hierarchy=h + ["iteration"])
        self.add_argument("--share_alphas", action='store_true',
                          help='Share some alpha variables to save memory at the cost of slightly looser bounds.',
                          hierarchy=h + ["share_alphas"])
        self.add_argument("--alpha_lr_decay", type=float, default=0.98,
                          help='Learning rate decay factor during alpha-CROWN optimization. Need to use a larger value like 0.99 or 0.995 when you increase the number of iterations.',
                          hierarchy=h + ["lr_decay"])
        self.add_argument("--no_full_conv_alpha", action='store_false', dest='full_conv_alpha',
                          help='Enable/disable the use of independent alpha for conv layers.',
                          hierarchy=h + ["full_conv_alpha"])
        self.add_argument("--max_coeff_mul", type=float, default=torch.inf,
                          help='Maximum coefficient value in the optimizable parameters for BoundMul.',
                          hierarchy=h + ["max_coeff_mul"])
        self.add_argument("--matmul_share_alphas", action='store_true',
                          help='Check alpha sharing for matmul.',
                          hierarchy=h + ['matmul_share_alphas'])
        self.add_argument("--disable_optimization", type=str, nargs='+', default=[],
                          help='A list of the names of operators which have bound optimization disabled.',
                          hierarchy=h + ['disable_optimization'])
        self.add_argument('--init_max_time', type=float, default=1.0,
                           help='Maximum time for the initial bound optimization (relative to the total timeout).',
                           hierarchy=h + ['max_time'])

        h = ["solver", "invprop"]
        self.add_argument("--apply_output_constraints_to", type=str, nargs='*', default=[],
                          help='Includes the output constraint in the optimization of linear layers. Can be a comma separated list of layer types (e.g. "BoundLinear"), layer names (e.g. "/input.7") or "all". This will disable patch mode for listed conv layers. When set, --optimize_disjuncts_separately must be set, too, if the safety property uses a disjunction.',
                          hierarchy=h + ['apply_output_constraints_to'])
        self.add_argument("--tighten_input_bounds", action='store_true',
                          help='Tighten input bounds using output constraints. If set, --apply_output_constraints_to should contain "BoundInput" or the corresponding layer name.',
                          hierarchy=h + ['tighten_input_bounds'])
        self.add_argument("--best_of_oc_and_no_oc", action='store_true',
                          help='Computes bounds of each layer both with and without output constraints. The best of both is used. Increases the runtime, but may improve the bounds.',
                          hierarchy=h + ['best_of_oc_and_no_oc'])
        self.add_argument("--directly_optimize", type=str, nargs='*', default=[],
                          help='A list of layer names whose bounds should be directly optimized for. IBP is disabled for these layers. Should only be needed for backward verification (approximation of input bounds) where the first linear layer defines the cs and should be optimized.',
                          hierarchy = h + ['directly_optimize'])
        self.add_argument("--oc_lr", type=float, default=0.1,
                          help='The learning rate for the dualized output constraints.',
                          hierarchy = h + ['oc_lr'])
        self.add_argument("--share_gammas", action='store_true',
                          help='Shares gammas across neurons in the optimized layer.',
                          hierarchy = h + ['share_gammas'])

        h = ["solver", "beta-crown"]
        self.add_argument("--lr_alpha", type=float, default=0.01,
                          help='Learning rate for optimizing alpha during branch and bound.',
                          hierarchy=h + ["lr_alpha"])
        self.add_argument("--lr_beta", type=float, default=0.05,
                          help='Learning rate for optimizing beta during branch and bound.',
                          hierarchy=h + ["lr_beta"])
        self.add_argument("--lr_decay", type=float, default=0.98,
                          help='Learning rate decay factor during beta-CROWN optimization. Need to use a larger value like 0.99 or 0.995 when you increase the number of iterations.',
                          hierarchy=h + ["lr_decay"])
        self.add_argument("--optimizer", default="adam", help='Optimizer used for alpha and beta optimization.',
                          hierarchy=h + ["optimizer"])
        self.add_argument("--iteration", type=int, default=50,
                          help='Number of iteration for optimizing alpha and beta during branch and bound.',
                          hierarchy=h + ["iteration"])
        self.add_argument('--no_beta', action='store_false', dest='beta',
                          help='Enable/disable beta split constraint (this option is for ablation study only and should not be used normally).',
                          hierarchy=h + ["beta"])
        self.add_argument('--no_beta_warmup', action='store_false', dest='beta_warmup',
                          help='Enable/disable beta warmup from branching history (this option is for ablation study only and should not be used normally).',
                          hierarchy=h + ["beta_warmup"])
        self.add_argument('--enable_opt_interm_bounds', action='store_true', default=False,
                          help='Enable optimizing intermediate bounds for beta-CROWN, only used when mip refine for now.',
                          hierarchy=h + ["enable_opt_interm_bounds"])
        self.add_argument('--enable_all_node_split_LP', action='store_true',
                          dest="all_node_split_LP", default=False,
                          help='When all nodes are split during Bab but not verified, using LP to check.',
                          hierarchy=h + ["all_node_split_LP"])

        h = ["solver", "forward"]
        self.add_argument('--forward_refine', action='store_true',
                          help='Refine forward bound with CROWN for unstable neurons.',
                          hierarchy=h + ["refine"])
        self.add_argument('--forward_max_dim', type=int, default=10000,
                          help='Maximum input dimension for forward bounds in a batch.',
                          hierarchy=h + ["max_dim"])
        self.add_argument('--reset_start_layer_threshold', type=float, default=1.0,
                          help='Reset the start layer if timeout neurons are above the threshold.',
                          hierarchy=h + ["reset_threshold"])


        h = ["solver", "multi_class"]
        self.add_argument('--label_batch_size', type=int,
                          help='Maximum target labels to handle in alpha-CROWN. Cannot be too large due to GPU memory limit.',
                          default=32, hierarchy=h + ["label_batch_size"])
        self.add_argument('--no_skip_with_refined_bound', action='store_false', dest='skip_with_refined_bound',
                          hierarchy=h + ['skip_with_refined_bound'],
                          help='By default we skip the second alpha-CROWN execution if all alphas are already initialized. Setting this to avoid this feature.')

        h = ["solver", "mip"]
        self.add_argument('--mip_unstable_neuron_threshold', type=int, default=0,
                          help='When complete_verifier=auto, enable MIP refinement when the number of unstable neurons exceeds this threshold.',
                          hierarchy=h + ["unstable_neuron_threshold"])
        self.add_argument('--mip_multi_proc', type=int, default=None,
                          help='Number of multi-processes for mip solver. Each process computes a mip bound for an intermediate neuron. Default (None) is to auto detect the number of CPU cores (note that each process may use multiple threads, see the next option).',
                          hierarchy=h + ["parallel_solvers"])
        self.add_argument('--mip_threads', type=int, default=1,
                          help='Number of threads for echo mip solver process (default is to use 1 thread for each solver process).',
                          hierarchy=h + ["solver_threads"])
        self.add_argument('--mip_perneuron_refine_timeout', type=float, default=15,
                          help='MIP timeout threshold for improving each intermediate layer bound (in seconds).',
                          hierarchy=h + ["refine_neuron_timeout"])
        self.add_argument('--mip_refine_timeout', type=float, default=0.8,
                          help='Percentage (x100%%) of time used for improving all intermediate layer bounds using mip. Default to be 0.8*timeout.',
                          hierarchy=h + ["refine_neuron_time_percentage"])
        self.add_argument('--no_mip_early_stop', action='store_false',
                          dest='mip_early_stop',
                          help='Enable/disable early stop when finding a positive lower bound or a adversarial example during MIP.',
                          hierarchy=h + ["early_stop"])
        self.add_argument('--disable_adv_warmup', action='store_false', dest='adv_warmup',
                          help='Disable using PGD adv as MIP refinement warmup starts.',
                          hierarchy=h + ["adv_warmup"])
        self.add_argument('--mip_solver', type=str, default='gurobi', help="MLP/LP solver package (SCIP support is experimental).",
                          choices=["gurobi", "scip"], hierarchy=h + ["mip_solver"])
        self.add_argument('--mip_skip_unsafe', action='store_true', dest='unsafe',
                          help='Do not use MIP solver to find counterexamples. This is useful in the case the MIP formulation is a relaxation and not complete.',
                          hierarchy=h + ["skip_unsafe"])
        self.add_argument('--topk_percentage', type=float, default=1., help="Refine top k neurons in mip solver.",
                          hierarchy=h + ["topk_filter"])
        self.add_argument('--mip_refine_sliding_window', type=int, default=-1,
                          help="Only when the neural network is very deep or too many timeout neurons that we use it to simplify the problem.",
                          hierarchy=h + ["sliding_window"])
        self.add_argument('--mip_refine_extra_input_constraints', action='store_true', dest='extra_constraint',
                          help="Add extra contraints to the MIP solver between the input and the starting layer, only activate when sliding window is enabled.",
                          hierarchy=h + ["extra_constraint"])
        self.add_argument('--mip_perneuron_refine_timeout_increasement', type=int, default=0,
                          help='MIP timeout threshold increasment for improving each intermediate layer bound (in seconds) if the remaining time is sufficient.',
                          hierarchy=h + ["refine_neuron_timeout_increasement"])
        self.add_argument('--mip_timeout_neuron', type=float, default=0.3,
                          help='Threshold for the timeout neurons to reset the perneuron timeout.',
                          hierarchy=h + ["timeout_neuron_percentage"])
        self.add_argument('--mip_remaining_timeout_coefficient_for_increasement', type=float, default=1,
                           help='MIP remaining timeout multiplier to determine whether following layer should increase the perneuron timeout.',
                           hierarchy=h + ["remaining_timeout_threshold"])
        self.add_argument('--mip_refine_remove_neurons', action='store_true', dest='remove_unstable_neurons',
                           help='Remove unstable neurons from MIP based on A_dict from compute_bounds.',
                           hierarchy=h + ['remove_unstable_neurons'])
        self.add_argument('--create_lp_model', action='store_true',
                          help="Create linear programming model instead of MIP.",
                          hierarchy=h + ["lp_solver"])

        h = ["bab"]
        self.add_argument("--initial_max_domains", type=int, default=1,
                          help='Number of domains we can add to domain list at the same time before bab. For multi-class problems this can be as large as the number of labels.',
                          hierarchy=h + ["initial_max_domains"])
        self.add_argument("--backing_up_max_domain", type=int, default=None,
                          help='Backup value for the number of domains. This will be set to the same value as initial_max_domains.',
                          hierarchy=h + ["backing_up_max_domain"])
        self.add_argument("--max_domains", type=int, default=float("inf"),
                          help='Max number of subproblems in branch and bound.',
                          hierarchy=h + ["max_domains"])
        self.add_argument("--decision_thresh", type=float, default=0,
                          help='Decision threshold of lower bounds. When lower bounds are greater than this value, verification is successful. Set to 0 for robustness verification.',
                          hierarchy=h + ["decision_thresh"])
        self.add_argument("--timeout", type=float, default=360,
                          help='Timeout (in second) for verifying one image/property.',
                          hierarchy=h + ["timeout"])
        self.add_argument("--timeout_scale", type=float, default=1, help='Scale the timeout for development purpose.',
                          hierarchy=h + ["timeout_scale"])
        self.add_argument("--max_iterations", type=int, default=-1,
                          help='Maximum number of BaB iterations.',
                          hierarchy=h + ["max_iterations"])
        self.add_argument("--override_timeout", type=float, default=None,
                          help='Override timeout.',
                          hierarchy=h + ["override_timeout"])
        self.add_argument("--get_upper_bound", action='store_true',
                          help='Update global upper bound during BaB (has extra overhead, typically the upper bound is not used).',
                          hierarchy=h + ["get_upper_bound"])
        self.add_argument("--disable_pruning_in_iteration", action='store_false',
                          dest='pruning_in_iteration',
                          help='Disable verified domain pruning within iteration.',
                          hierarchy=h + ["pruning_in_iteration"])
        self.add_argument("--pruning_in_iteration_ratio", type=float, default=0.2,
                          help='When ratio of positive domains >= this ratio, pruning in iteration optimization is open.',
                          hierarchy=h + ["pruning_in_iteration_ratio"])
        self.add_argument('--sort_targets', action='store_true',
                          help='Sort targets before BaB.',
                          hierarchy=h + ["sort_targets"])
        self.add_argument("--disable_batched_domain_list", action='store_false',
                          dest='batched_domain_list',
                          help='Disable batched domain list. Batched domain list is faster but picks domain to split in an unsorted way.',
                          hierarchy=h + ['batched_domain_list'])
        self.add_argument("--optimized_interm", type=str, default="",
                          help='A list of layer names that will be optimized during branch and bound, separated by comma.',
                          hierarchy=h + ['optimized_interm'])
        self.add_argument("--no_interm_transfer", action='store_false',
                          dest='interm_transfer',
                          help='Skip the intermediate bound transfer to save transfer-to-CPU time. Require intermediate bound does not change. Caution: cannot be used with cplex cut or intermediate beta refinement.',
                          hierarchy=h + ['interm_transfer'])
        self.add_argument("--recompute_interm", action="store_true",
                          dest="recompute_interm",
                          help="Recompute all the intermediate bounds during BaB.",
                          hierarchy=h + ['recompute_interm'])
        self.add_argument("--sort_domain_interval", type=int, default=-1,
                          help='If unsorted domains are used, sort the domains every sort_domain_interval iterations.',
                          hierarchy=h + ["sort_domain_interval"])
        self.add_argument("--vanilla_crown_bab", action="store_true",
                          dest="vanilla_crown_bab",
                          help='Use vanilla CROWN during BaB.',
                          hierarchy=h + ['vanilla_crown'])
        self.add_argument("--tree_traversal", default="depth_first",
                          choices=["depth_first", "breadth_first"],
                          help='During BaB, unkown domains can continue being split (deepening the tree, i.e. depth first traversal) or split only when all other unknown domains have the same number of splits (keeping the tree shallow for as long as possible, i.e. breadth first traversal). Depth first traversal minimizes memory access time, breadth first traversal is beneficial for BICCOS.',
                          hierarchy=h + ["tree_traversal"])

        # FIXME: cut should not be under bab. We don't have to use bab with cuts. It should be under "solver" category.
        h = ["bab", "cut"]
        self.add_argument('--enable_cut', action='store_true', dest='enable_cut',
                          help='Enable cutting planes using GCP-CROWN.',
                          hierarchy=h + ["enabled"])
        self.add_argument('--cuts_path', default="/tmp/abcrown_cuts",
                          help='For cuts from CPLEX, specify the path for saving intermediate files with generated cuts.',
                          hierarchy=h + ["cuts_path"])
        self.add_argument('--enable_implication', action='store_true', dest='enable_implication', help='Enable neuron implications.',
                          hierarchy=h + ["implication"])
        self.add_argument('--enable_bab_cut', action='store_true', dest='enable_bab_cut',
                          help='Enable cut constraints optimization during BaB.',
                          hierarchy=h + ["bab_cut"])
        self.add_argument('--enable_lp_cut', action='store_true', dest='enable_lp_cut',
                          help='enable lp with cut constraints to debug',
                          hierarchy=h + ["lp_cut"], private=True)
        self.add_argument('--cut_method', help='Cutting plane generation method (unused, for future extensions).', hierarchy=h + ["method"])
        self.add_argument("--lr_cuts", type=float, default=0.01, help='Learning rate for optimizing cuts.',
                          hierarchy=h + ["lr"])
        self.add_argument("--cut_lr_decay", type=float, default=1.0,
                          help='Learning rate decay for optimizing betas in GCP-CROWN.',
                          hierarchy=h + ["lr_decay"])
        self.add_argument("--cut_iteration", type=int, default=100,
                          help='Iterations for optimizing betas in GCP-CROWN.',
                          hierarchy=h + ["iteration"])
        self.add_argument("--cut_bab_iteration", type=int, default=-1,
                          help='Iterations for optimizing betas in GCP-CROWN during branch and bound. Set to -1 to use the same number of iterations without cuts.',
                          hierarchy=h + ["bab_iteration"])
        self.add_argument("--cut_early_stop_patience", type=int, default=-1,
                          help='Early stop patience for optimizing cuts. Set to -1 to use the same value when cuts are not used.',
                          hierarchy=h + ["early_stop_patience"])
        self.add_argument("--cut_lr_beta", type=float, default=0.02,
                          help='Learning rate for optimizing betas in GCP-CROWN.',
                          hierarchy=h + ["lr_beta"])
        self.add_argument("--number_cuts", type=int, default=50,
                          help='Maximum number of cuts that we want to add.',
                          hierarchy=h + ["number_cuts"])
        self.add_argument("--topk_implication", type=int, default=1000,
                          help='Only keep top K constraints when filtering cuts.',
                          hierarchy=h + ["topk_cuts_in_filter"])
        self.add_argument("--batch_size_primal", type=int, default=100,
                          help='Batch size when calculate primals, should be negative correlated to number of unstable neurons.',
                          hierarchy=h + ["batch_size_primal"])
        self.add_argument("--cut_max_num", type=int, default=int(1e9),
                          help='Maximum number of cuts.',
                          hierarchy=h + ["max_num"])
        self.add_argument("--enable_patches_cut", action='store_true',
                          help='Enable GCP-CROWN optimization for intermediate layer bounds in patches mode.',
                          hierarchy=h + ["patches_cut"])
        self.add_argument("--cplex_cuts", action='store_true',
                          help='Build and save mip mps models, let cplex find cuts, and use found cuts to improve lower bounds.',
                          hierarchy=h + ["cplex_cuts"])
        self.add_argument("--cplex_cuts_wait", type=float, default=0,
                          help='Wait a bit after cplex warmup in seconds, so that we tend to get some cuts at early stage of branch and bound.',
                          hierarchy=h + ["cplex_cuts_wait"])
        self.add_argument("--no_cplex_cuts_revpickup", action='store_false',
                          help='Enable/disable the inverse order domain pickout when cplex is enabled.',
                          hierarchy=h + ["cplex_cuts_revpickup"])
        self.add_argument("--no_cut_reference_bounds", action='store_false',
                          help='Enable/disable using reference bounds when GCP-CROWN cuts are used.',
                          hierarchy=h + ["cut_reference_bounds"])
        self.add_argument("--fix_cut_intermediate_bounds", action='store_true',
                          help='Fix intermediate bounds when GCP-CROWN cuts are used.',
                          hierarchy=h + ["fix_intermediate_bounds"])

        h = ["bab", "cut", "biccos"]
        self.add_argument("--biccos_cuts", action='store_true',
                          help='BICCOS from the the Branch and Bound process.',
                          hierarchy=h + ["enabled"])
        self.add_argument("--no_biccos_constraint_strengthening", action='store_false',
                          dest='constraint_strengthening',
                          help='Do not use constraint strengthening in BICCOS.',
                          hierarchy=h + ["constraint_strengthening"])
        self.add_argument("--biccos_recursively_strengthening", action='store_true',
                          help='Use recursive constraint strengthening.',
                          hierarchy=h + ["recursively_strengthening"])
        self.add_argument("--biccos_drop_ratio", type=float, default=0.5,
                          help='Neuron drop ratio parameter when using inferred cut and neuron influence score heuristic.',
                          hierarchy=h + ["drop_ratio"])
        self.add_argument("--biccos_verified_bonus", type=float, default=0.3,
                          help='When a neuron is verified, we add a bonus to the neuron influence score.',
                          hierarchy=h + ["verified_bonus"])
        self.add_argument("--biccos_max_infer_iter", type=int, default=20,
                          help='After max_infer_iter iterations, we will stop inferencing cuts.',
                          hierarchy=h + ["max_infer_iter"])
        self.add_argument("--biccos_dropping_heuristics", default="neuron_influence_score",
                          choices=["neuron_influence_score", "random_drop", "sparse_opt"],
                          help='Neuron dropping heuristic.',
                          hierarchy=h + ["heuristic"])
        self.add_argument("--biccos_save_biccos_cuts", action='store_true',
                          help='Save cuts to log/biccos.txt.',
                          hierarchy=h + ["save_biccos_cuts"])

        h = ["bab", "cut", "biccos", "multi_tree_branching"]
        self.add_argument("--multi_tree_branching", action="store_true",
                          help="Enables multi-tree branching. Instead of picking one new split per unkown domain, multiple splits are tested. In the end, the best (measured in bound tightness) regular BaB tree is selected.",
                          hierarchy=h + ["enabled"])
        self.add_argument("--multi_tree_branching_restore_best_tree", action="store_true",
                          help="Restore the best tree after the multi-tree search.",
                          hierarchy=h + ["restore_best_tree"])
        self.add_argument("--multi_tree_branching_keep_n_best_domains", type=int, default=1,
                          help="Number of best domains to keep in multi-tree branching.",
                          hierarchy=h + ["keep_n_best_domains"])
        self.add_argument("--multi_tree_branching_k_splits", type=int, default=1,
                          help="Number of splits for each neuron in multi-tree branching.",
                          hierarchy=h + ["k_splits"])
        self.add_argument("--multi_tree_branching_iterations", type=int, default=1,
                          help="Number of iterations for multi-tree branching.",
                          hierarchy=h + ["iterations"])

        h = ["bab", "branching"]
        self.add_argument("--branching_method", default="kfsb",
                          choices=["babsr", "fsb", "kfsb", "kfsb-intercept-only",
                                   "random", "intercept", "naive", "sb",
                                   "nonlinear", "brute-force"],
                          help='Branching heuristic. babsr is fast but less accurate; fsb is slow but most accurate; kfsb is usually a balance; kfsb-intercept-only is faster but may lead to worse branching; sb is fast smart branching which relies on the A matrix.',
                          hierarchy=h + ["method"])
        self.add_argument("--branching_candidates", type=int, default=3,
                          help='Number of candidates to consider when using fsb or kfsb. More candidates lead to slower but better branching.',
                          hierarchy=h + ["candidates"])
        self.add_argument("--branching_reduceop", choices=["min", "max", "mean", "auto"], default="min",
                          help='Reduction operation to compute branching scores from two sides of a branch (min or max). max can work better on some models.',
                          hierarchy=h + ["reduceop"])
        self.add_argument('--enable_intermediate_bound_opt', action='store_true',
                          default=False, help='Enable optimizing intermediate bounds for during bab.',
                          hierarchy=h + ["enable_intermediate_bound_opt"])
        self.add_argument("--branching_input_and_activation", action='store_true',
                          help="Branching input domains and relu domains (experimental).",
                          hierarchy=h + ["branching_input_and_activation"])
        self.add_argument("--branching_input_and_activation_order", nargs='+',
                          type=str, default=['input', 'relu'],
                          help="Order of branching input domains and relu domains (experimental).",
                          hierarchy=h + ["branching_input_and_activation_order"])
        self.add_argument("--branching_input_iterations", type=int, default=30,
                          help="Number of iterations to run input split before we run relu split.",
                          hierarchy=h + ["branching_input_iterations"])
        self.add_argument("--branching_relu_iterations", type=int, default=50,
                          help="Number of iterations to run relu split before we run input split.",
                          hierarchy=h + ["branching_relu_iterations"])

        h = ["bab", "branching", "nonlinear_split"]
        self.add_argument("--nonlinear_split_method", type=str, default='bbps',
                          choices=['bbps','babsr-like'],
                          help='Branching heuristic for the general nonlinear functions (either the default BBPS heuristic, or a BaBSR-like as a baseline).',
                          hierarchy=h + ['method'])
        self.add_argument("--disable_nonlinear_split", action='store_true',
                          help='Disable GenBaB even if there are non-ReLU functions to split.',
                          hierarchy=h + ['disable'])
        self.add_argument("--branching_point_method", type=str, default='uniform',
                          choices=['uniform', 'opt'],
                          help='For general nonlinear functions, the method for choosing the branching point ("uniform" stands for branching in the middle while "opt" stands for using pre-optimized branching points).',
                          hierarchy=h + ['branching_point_method'])
        self.add_argument("--nonlinear_branches", type=int, default=2,
                          help='Number of branches for nonlinear branching.',
                          hierarchy=h + ['num_branches'])
        self.add_argument("--nonlinear_split_filtering", action='store_true',
                          dest='nonlinear_split_filtering',
                          help='KFSB-like filtering in general nonlinear branching.',
                          hierarchy=h + ['filter'])
        self.add_argument("--nonlinear_split_filtering_beta", action='store_true',
                          dest='nonlinear_split_filtering_beta',
                          help='Use beta in the KFSB-like filtering.',
                          hierarchy=h + ['filter_beta'])
        self.add_argument("--nonlinear_split_filtering_clamp", action='store_true',
                          dest='nonlinear_split_filtering_clamp',
                          help='Clamping scores to 0 in BBPS filtering.',
                          hierarchy=h + ['filter_clamp'])
        self.add_argument("--filter_batch_size", type=int, default=10000,
                          help='Batch size for filtering.',
                          hierarchy=h + ['filter_batch_size'])
        self.add_argument("--filter_iterations", type=int, default=25,
                          help="Number of iterations for filtering.",
                          hierarchy=h + ['filter_iterations'])
        self.add_argument("--nonlinear_split_relu_only",
                          action="store_true",
                          help="When using BBPS, only consider branching ReLU instead of all the nonlinearities.",
                          hierarchy=h + ['relu_only'])
        self.add_argument("--loose_tanh_threshold", type=float, default=None,
                          help='Set a threshold for tanh/sigmoid to use a different relaxation when the pre-activation bounds are too loose.',
                          hierarchy=h + ['loose_tanh_threshold'])

        h = ["bab", "branching", "nonlinear_split", "branching_point"]
        self.add_argument("--branching_point_db", type=str, default='branching_points.pt',
                          help='Path to store pre-computed branching points.',
                          hierarchy=h + ['db_path'])
        self.add_argument('--branching_point_num_iterations', type=int, default=1000,
                          help='Number of iterations for the optimization.',
                          hierarchy=h + ['num_iterations'])
        self.add_argument('--branching_point_batch_size', type=int, default=1000000,
                          help='Batch size for the optimization.',
                          hierarchy=h + ['batch_size'])
        self.add_argument('--branching_point_range_l', type=float, default=-5.0,
                          help='Range (lower bound) for the optimization.',
                          hierarchy=h + ['range_l'])
        self.add_argument('--branching_point_range_u', type=float, default=5.0,
                          help='Range (upper bound) for the optimization.',
                          hierarchy=h + ['range_u'])
        self.add_argument('--branching_point_log_interval', type=int, default=100,
                          help='Log interval for the optimization.',
                          hierarchy=h + ['log_interval'])
        self.add_argument('--branching_point_step_size_1d', type=float, default=0.01,
                          help='Step size for 1d nonlinearities.',
                          hierarchy=h + ['step_size_1d'])
        self.add_argument('--branching_point_step_size', type=float, default=0.2,
                          help='Step size for 2d and above nonlinearities.',
                          hierarchy=h + ['step_size'])

        h = ["bab", "branching", "input_split"]
        self.add_argument("--enable_clip_domains", action='store_true',
                          help='Shrinks subdomains based on their specification.',
                          hierarchy=h + ["enable_clip_domains"])
        self.add_argument("--split_hint", type=float, nargs='+',
                          help="Specifies value to split at.",
                          hierarchy=h + ["split_hint"])
        self.add_argument("--reorder_bab", action="store_true",
                          help="Uses a reordered implementation of the input BaB procedure that bounds, splits and "
                               "clips domains rather than splitting, bounding, and clipping domains.",
                          hierarchy= h + ["reorder_bab"])
        # end of temporary arguments
        self.add_argument("--enable_input_split", action='store_true',
                          help='Branch on input domain rather than unstable neurons.',
                          hierarchy=h + ["enable"])
        self.add_argument('--enhanced_bound_prop_method', default="alpha-crown",
                          choices=["alpha-crown", "crown", "forward+crown", "crown-ibp"],
                          help='Specify a tighter bound propagation method if a problem cannot be verified after --input_split_enhanced_bound_patience.',
                          hierarchy=h + ["enhanced_bound_prop_method"])
        self.add_argument('--enhanced_branching_method', default="naive",
                          choices=["sb", "naive"],
                          help='Specify a branching method if a problem cannot be verified after --input_split_enhanced_bound_patience.',
                          hierarchy=h + ["enhanced_branching_method"])
        self.add_argument("--input_split_enhanced_bound_patience", type=int, default=1e8,
                          help='Time in seconds that will use an enhanced bound propagation method (e.g., alpha-CROWN) to bound input split sub domains.',
                          hierarchy=h + ["enhanced_bound_patience"])
        self.add_argument("--input_split_attack_patience", type=int, default=1e8,
                          help='Time in seconds that will start PGD attack to find adv examples during input split.',
                          hierarchy=h + ["attack_patience"])
        self.add_argument("--input_split_adv_check", type=int, default=0,
                          help='After the number of visited nodes, we will run adv_check in input split.',
                          hierarchy=h + ["adv_check"])
        self.add_argument("--input_split_partitions", type=int, default=2,
                          help='How many domains to split to for each dimension at each time. By default, it is 2. In very few limited experimental cases, can change to larger numbers.',
                          hierarchy=h + ["split_partitions"])
        self.add_argument("--sb_margin_weight", default=1.0, type=float,
                          help='Weight for the margin term in the sb heuristic.',
                          hierarchy=h + ["sb_margin_weight"])
        self.add_argument('--sb_sum', action='store_true',
                          help='Use sum for multiple specs in sb.',
                          hierarchy=h + ['sb_sum'])
        self.add_argument('--sb_primary_spec', type=int, default=None,
                          help='Focus on one particular spec for the SB score.',
                          hierarchy=h + ['sb_primary_spec'])
        self.add_argument('--bf_backup_thresh', type=float, default=-1,
                          help='Threshold for using the SB score as the backup when the brute force score is too bad.',
                          hierarchy=h + ['bf_backup_thresh'])
        self.add_argument('--bf_rhs_offset', type=float, default=0,
                          help='An offset on RHS used in computing the brute force heuristic.',
                          hierarchy=h + ['bf_rhs_offset'])
        self.add_argument('--bf_iters', type=int, default=1e9,
                          help='Number of iterations to use brute force.',
                          hierarchy=h + ['bf_iters'])
        self.add_argument('--bf_batch_size', type=int, default=100000,
                          help='A special batch size for brute force.',
                          hierarchy=h + ['bf_batch_size'])
        self.add_argument('--bf_zero_crossing_score', action='store_true',
                          help='A zero crossing score in BF.',
                          hierarchy=h + ['bf_zero_crossing_score'])
        self.add_argument('--touch_zero_score', type=float, default=0,
                          help='A touch-zero score in BF.',
                          hierarchy=h + ['touch_zero_score'])
        self.add_argument('--ibp_enhancement', action='store_true',
                          help='Use IBP bounds to enhance.',
                          hierarchy=h + ['ibp_enhancement'])
        self.add_argument('--compare_input_split_with_old_bounds',
                          action='store_true',
                          help='Compare bounds after an input split with bounds before the split and take the better one.',
                          hierarchy=h + ['compare_with_old_bounds'])
        self.add_argument('--input_split_update_rhs_with_attack',
                          action='store_true',
                          help='Run attack during input split and update RHS. BaB does not stop even if any counterexample is found.',
                          hierarchy=h + ['update_rhs_with_attack'])
        self.add_argument("--sb_coeff_thresh", default=1e-3, type=float,
                          help='Clamp values of coefficient matrix (A matrix) for sb branching heuristic.',
                          hierarchy=h + ["sb_coeff_thresh"])
        self.add_argument('--input_split_sort_index', type=int,
                          help='The output index to use for sorting domains in the input split.',
                          hierarchy=h + ["sort_index"])
        self.add_argument('--no_input_split_sort_descending', action='store_false',
                          dest="input_split_sort_descending",
                          help='Sort input split domains in an ascending/descending way.',
                          hierarchy=h + ["sort_descending"])
        self.add_argument('--input_split_show_progress', action='store_true',
                          help='Show progress during input split.',
                          hierarchy=h + ["show_progress"])
        self.add_argument('--input_split_presplit_domains', type=str, default=None,
                          help='Load pre-split domains from a file.',
                          hierarchy=h + ["presplit_domains"])
        self.add_argument('--input_split_skip_getting_worst_domain',
                          action='store_true',
                          help='Skip getting the worst domain at the end of each iteration, to save some time cost when the domain list is long.',
                          hierarchy=h + ["skip_getting_worst_domain"])

        h = ["bab", "attack"]  # BaB-Attack options.
        self.add_argument("--enable_bab_attack", action='store_true',
                          help='Enable beam search based BaB-attack.',
                          hierarchy=h + ["enabled"])
        self.add_argument("--beam_candidates_number", type=int, default=8,
                          help='Number of candidates in beam search.',
                          hierarchy=h + ["beam_candidates"])
        self.add_argument("--beam_split_depth", type=int, default=7,
                          help='Max additional level of splits to expand during beam search in BaB-Attack.',
                          hierarchy=h + ["beam_depth"])
        self.add_argument('--max_dive_fix_ratio', type=float, default=0.8,
                          help='Maximum ratio of fixed neurons during diving in BaB-Attack.',
                          hierarchy=h + ['max_dive_fix_ratio'])
        self.add_argument('--min_local_free_ratio', type=float, default=0.2,
                          help='Minimum ratio of free neurons during local search in BaB-Attack.',
                          hierarchy=h + ['min_local_free_ratio'])
        self.add_argument('--submip_start_iteration', type=int, default=5,
                          help='Iteration number to start sub-MIPs in BaB-Attack.',
                          hierarchy=h + ["mip_start_iteration"])
        self.add_argument('--submip_timeout', type=float, default=30.0,
                          help='Sub-MIP timeout threshold.',
                          hierarchy=h + ["mip_timeout"])
        self.add_argument('--adv_pool_threshold', type=float, default=None,
                          help='Minimum value of difference when adding to adv_pool; default `None` means auto select.',
                          hierarchy=h + ["adv_pool_threshold"])
        self.add_argument("--refined_mip_attacker", action='store_true',
                          help='Use full alpha crown bounds to refined intermediate bounds for sub-MIPs.',
                          hierarchy=h + ["refined_mip_attacker"])
        self.add_argument("--refined_batch_size", type=float, default=None,
                          help='Batch size for full alpha-CROWN to refined intermediate bounds for mip solver attack (to avoid OOM), default None to be the same as mip_multi_proc.',
                          hierarchy=h + ["refined_batch_size"])

        h = ["attack"]
        self.add_argument('--pgd_order', choices=["before", "after", "middle", "input_bab", "skip"], default="before",
                          help='Run PGD attack before/after/during incomplete verification, only during input bab, or skip it.', hierarchy=h + ["pgd_order"])
        self.add_argument('--pgd_steps', type=int, default=100,
                          help="Steps of PGD attack.",
                          hierarchy=h + ["pgd_steps"])
        self.add_argument('--pgd_restarts', type=int, default=30,
                          help="Number of random PGD restarts.",
                          hierarchy=h + ["pgd_restarts"])
        self.add_argument('--pgd_batch_size', type=int, default=int(1e8),
                          help='Batch size for number of restarts in PGD.',
                          hierarchy=h + ["pgd_batch_size"])
        self.add_argument('--no_pgd_early_stop', action='store_false',
                          dest='pgd_early_stop',
                          help="Enable/disable early stop PGD when an adversarial example is found.",
                          hierarchy=h + ["pgd_early_stop"])
        self.add_argument('--pgd_lr_decay', type=float, default=0.99,
                          help='Learning rate decay factor used in PGD attack.',
                          hierarchy=h + ["pgd_lr_decay"])
        self.add_argument('--pgd_alpha', type=str, default="auto",
                          help='Step size of PGD attack. Default (auto) is epsilon/4.',
                          hierarchy=h + ["pgd_alpha"])
        self.add_argument('--pgd_alpha_scale', action='store_true',
                          help='Scale PGD alpha according to data_max-data_min.',
                          hierarchy=h + ["pgd_alpha_scale"])
        self.add_argument('--pgd_loss_mode', type=str, choices=['hinge', 'sum'],
                          help='Loss mode for choosing the best delta.',
                          hierarchy=h + ["pgd_loss_mode"])
        self.add_argument('--pgd_restart_when_stuck', action='store_true',
                          help='Restart adversarial noise when they do not change over attack iterations.',
                          hierarchy=h + ["pgd_restart_when_stuck"])
        self.add_argument("--mip_attack", action='store_true',
                          help='Use MIP (Gurobi) based attack if PGD cannot find a successful adversarial example.',
                          hierarchy=h + ["enable_mip_attack"])
        self.add_argument("--adv_saver", type=str, default='default_adv_saver',
                          help='Customized saver of adverserial examples.',
                          hierarchy=h + ["adv_saver"])
        self.add_argument("--adv_verifier", type=str, default='default_adv_verifier',
                          help='Customized verifier of adverserial examples.',
                          hierarchy=h + ["adv_verifier"])
        self.add_argument("--early_stop_condition", type=str, default='default_early_stop_condition',
                          help='Customized early stop condition.',
                          hierarchy=h + ["early_stop_condition"])
        self.add_argument("--adv_example_finalizer", type=str, default='default_adv_example_finalizer',
                          help='Customized generation of adversarial examples, margins computation, etc.',
                          hierarchy=h + ["adv_example_finalizer"])
        self.add_argument("--pgd_loss", type=str, default="default_pgd_loss",
                          help='Customized pgd loss.',
                          hierarchy=h + ["pgd_loss"])
        self.add_argument('--cex_path', type=str, default='./test_cex.txt',
                          help='Save path for counter-examples.',
                          hierarchy=h + ["cex_path"])
        self.add_argument('--attack_mode', type=str, default='PGD',
                          help='Attack algorithm, including vanilla PGD and PGD with diversified output (Tashiro et al.), and GAMA loss (Sriramanan et al.).',
                          choices=['diversed_PGD', 'diversed_GAMA_PGD', 'PGD', 'boundary'],
                          hierarchy=h + ["attack_mode"])
        self.add_argument('--attack_tolerance', type=self.positive_float_checker, default=0.,
                          help='Tolerance of floating point error when checking whether attack is successful or not.',
                          hierarchy=h + ["attack_tolerance"])
        self.add_argument('--attack_func', type=str, default='attack_with_general_specs',
                          help='The specific customized attack.',
                          hierarchy=h + ["attack_func"])
        self.add_argument('--attack_gama_lambda', type=float, default=10.,
                          help='Regularization parameter in GAMA attack.',
                          hierarchy=h + ["gama_lambda"])
        self.add_argument('--attack_gama_decay', type=float, default=0.9,
                          help='Decay of regularization parameter in GAMA attack.',
                          hierarchy=h + ["gama_decay"])
        self.add_argument('--check_clean', action='store_true',
                          help='Check clean prediction before attack.',
                          hierarchy=h + ["check_clean"])

        h = ["attack", "input_split"]
        self.add_argument('--input_split_pgd_steps', type=int, default=100,
                          help="Steps of PGD attack in input split before branching starts.",
                          hierarchy=h + ["pgd_steps"])
        self.add_argument('--input_split_pgd_restarts', type=int, default=30,
                          help="Number of random PGD restarts in input split before branching starts.", hierarchy=h + ["pgd_restarts"])
        self.add_argument('--input_split_pgd_alpha', type=str, default="auto",
                          help="Step size (alpha) in input split before branching starts.",
                          hierarchy=h + ["pgd_alpha"])

        h = ["attack", "input_split_enhanced"]
        self.add_argument('--input_split_enhanced_pgd_steps', type=int, default=200,
                          help="Steps of PGD attack in enhanced pgd attack in input split.",
                          hierarchy=h + ["pgd_steps"])
        self.add_argument('--input_split_enhanced_pgd_restarts', type=int,
                          default=500000,
                          help="Number of random PGD restarts in enhanced pgd attack in input split.",
                          hierarchy=h + ["pgd_restarts"])
        self.add_argument('--input_split_enhanced_pgd_alpha', type=str,
                          default="auto",
                          help="Step size (alpha) in enhanced pgd attack in input split.", hierarchy=h + ["pgd_alpha"])

        h = ["attack", "input_split_check_adv"]
        self.add_argument('--enable_check_adv', choices=["auto", "true", "false"], default='auto',
                          help="Enable or disable counterexample checking during input-space branch-and-bound. Default is 'auto', which disables check_adv when pgd_order is skip.",
                          hierarchy=h + ["enabled"])
        self.add_argument('--input_split_check_adv_pgd_steps', type=int, default=5,
                          help="Steps of PGD attack in input split after each branching.",
                          hierarchy=h + ["pgd_steps"])
        self.add_argument('--input_split_check_adv_pgd_restarts', type=int, default=5,
                          help="Number of random PGD restarts in input split after each branching.",
                          hierarchy=h + ["pgd_restarts"])
        self.add_argument('--input_split_check_adv_pgd_alpha', type=str, default="auto",
                          help="Step size (alpha) in input split after each branching.",
                          hierarchy=h + ["pgd_alpha"])
        self.add_argument('--input_split_check_adv_max_num_domains', type=int, default=10,
                          help="Maximum number of domains for running attack during input split.",
                          hierarchy=h + ["max_num_domains"])

        h = ["debug"]
        self.add_argument('--view_model', action='store_true',
                          help='Print more detailed model information for analysis.',
                          hierarchy=h + ['view_model'])
        self.add_argument("--lp_test", type=str, default=None,
                          choices=["MIP", "LP", None],
                          help='Debugging option. Do not use.',
                          hierarchy=h + ['lp_test'], private=True)
        self.add_argument("--rescale_vnnlib_ptb", type=float, default=None,
                          help='Rescale the perturbation on X in vnnlib files to make them smaller. Used for debugging.',
                          hierarchy=h + ['rescale_vnnlib_ptb'], private=True)
        self.add_argument("--test_optimized_bounds", action='store_true', default=False,
                          help='Debugging option. Compares optimized bounds with those from Gurobi for the relaxed LP formulation',
                          hierarchy=h + ['test_optimized_bounds'], private=True)
        self.add_argument("--test_optimized_bounds_after_n_iterations", type=int, default=0,
                          help='Debugging option. Controls how many optimization steps for all layers are done before bounds are checked against the LP solver',
                          hierarchy=h + ['test_optimized_bounds_after_n_iterations'], private=True)
        self.add_argument("--print_verbose_decisions", action="store_true",
                          help="Print more detailed information about branching decisions",
                          hierarchy=h + ['print_verbose_decisions'], private=True)
        self.add_argument("--sanity_check", type=str, default=None,
                          choices=["Full", "Full+Graph", "Worst", None],
                          help="Using pgd upper bound as rhs-offset to check the feasibility of the method. Warning: If rhs_offset was specified "
                               "in the config or command line, it will get ignored.",
                          hierarchy=h + ['sanity_check'], private=True)
        self.add_argument("--save_minimal_config", type=str, default=None,
                          help="Path to save a minimal config file.",
                          hierarchy=h + ['save_minimal_config'])
        self.add_argument("--save_minimal_config_omit_keys", type=str, nargs="+", default=[],
                          help="Keys to omit from the minimal config file.",
                          hierarchy=h + ['save_minimal_config_omit_keys'])

    def update_arguments(self):
        """Adaptively tune arguments."""

        if (self["specification"]["norm"] != torch.inf
            and self["attack"]["pgd_order"] != "skip"):
            print('Only Linf-norm attack is supported, the pgd_order will be changed to skip')
            self["attack"]["pgd_order"] = "skip"

    def add_argument(self, *args, **kwargs):
        """Add a single parameter to the parser. We will check the 'hierarchy' specified and then pass the remaining arguments to argparse."""
        if 'hierarchy' not in kwargs:
            raise ValueError("please specify the 'hierarchy' parameter when using this function.")
        hierarchy = kwargs.pop('hierarchy')
        help = kwargs.get('help', '')
        private_option = kwargs.pop('private', False)
        # Make sure valid help is given
        if not private_option:
            if len(help.strip()) < 10:
                raise ValueError(
                    f'Help message must not be empty, and must be detailed enough. "{help}" is not good enough.')
            elif (not help[0].isupper()) or help[-1] != '.':
                raise ValueError(
                    f'Help message must start with an upper case letter and end with a dot (.); your message "{help}" is invalid.')
            elif help.count('%') != help.count('%%') * 2:
                raise ValueError(
                    f'Please escape "%" in help message with "%%"; your message "{help}" is invalid.')
        self.defaults_parser.add_argument(*args, **kwargs)
        # Build another parser without any defaults.
        if 'default' in kwargs:
            kwargs.pop('default')
        self.no_defaults_parser.add_argument(*args, **kwargs)
        # Determine the variable that will be used to save the argument by argparse.
        if 'dest' in kwargs:
            dest = kwargs['dest']
        else:
            dest = re.sub('^-*', '', args[-1]).replace('-', '_')
        # Also register this parameter to the hierarchy dictionary.
        self.config_file_hierarchies[dest] = hierarchy
        if hierarchy is not None and not private_option:
            self.help_messages[','.join(hierarchy)] = help

    def set_dict_by_hierarchy(self, args_dict, h, value, nonexist_ok=True):
        """Insert an argument into the dictionary of all parameters. The level in this dictionary is determined by list 'h'."""
        # Create all the levels if they do not exist.
        current_level = self.all_args
        assert len(h) != 0
        for config_name in h:
            if config_name not in current_level:
                if nonexist_ok:
                    current_level[config_name] = {}
                else:
                    raise ValueError(f"Config key {h} not found!")
            last_level = current_level
            current_level = current_level[config_name]
        # Add config value to leaf node.
        last_level[config_name] = value

    def construct_config_dict(self, args_dict, nonexist_ok=True):
        """Based on all arguments from argparse, construct the dictionary of all parameters in self.all_args."""
        for arg_name, arg_val in args_dict.items():
            h = self.config_file_hierarchies[arg_name]  # Get levels for this argument.
            if h is not None:
                assert len(h) != 0
                self.set_dict_by_hierarchy(self.all_args, h, arg_val,
                                           nonexist_ok=nonexist_ok)

    def update_config_dict(self, old_args_dict, new_args_dict, levels=None):
        """Recursively update the dictionary of all parameters based on the dict read from config file."""
        if levels is None:
            levels = []
        if isinstance(new_args_dict, dict):
            # Go to the next dict level.
            for k in new_args_dict:
                self.update_config_dict(old_args_dict, new_args_dict[k],
                                        levels=levels + [k])
        else:
            # Reached the leaf level. Set the corresponding key.
            self.set_dict_by_hierarchy(old_args_dict, levels, new_args_dict,
                                       nonexist_ok=False)

    def dump_config(self, args_dict, level=[], out_to_doc=False, show_help=False, omit_defaults=False, omit_keys=[]):
        """Generate a config file based on args_dict with help information."""
        ret_string = ''
        for key, val in args_dict.items():
            if isinstance(val, dict):
                ret = self.dump_config(val, level + [key], out_to_doc, show_help, omit_defaults, omit_keys)
                if len(ret) > 0:
                    # Next level is not empty, print it.
                    ret_string += ' ' * (len(level) * 2) + f'{key}:\n' + ret
            else:
                if omit_defaults:
                    default_value_key = list(self.config_file_hierarchies.keys())[
                        list(self.config_file_hierarchies.values()).index(level + [key])
                    ]
                    if default_value_key in omit_keys:
                        continue
                    if self.defaults_parser.get_default(default_value_key) == val:
                        continue
                if show_help:
                    h = self.help_messages[','.join(level + [key])]
                    if 'debug' in key or 'not use' in h or 'not be use' in h or 'debug' in h or len(h) == 0:
                        # Skip some debugging options.
                        continue
                    h = f'  # {h}'
                else:
                    h = ''
                yaml_line = yaml.safe_dump({key: val}, default_flow_style=None).strip().replace('{', '').replace('}',
                                                                                                                 '')
                ret_string += ' ' * (len(level) * 2) + f'{yaml_line}{h}\n'
        if len(level) > 0:
            return ret_string
        else:
            # Top level, output to file.
            if out_to_doc:
                output_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs',
                                           'abcrown_all_params.yaml')
                with open(output_name, 'w') as f:
                    f.write(ret_string)
            return ret_string

    def parse_config(self, args=None, verbose=True):
        """
        Main function to parse parameter configurations. The commandline arguments have the highest priority;
        then the parameters specified in yaml config file. If a parameter does not exist in either commandline
        or the yaml config file, we use the defaults defined in add_common_options() defined above.
        """
        if args is None:
            args = sys.argv[1:]
        # Create the dictionary of all parameters, all set to their default values.
        self.construct_config_dict(self.default_args)
        # These are arguments specified in command line.
        specified_args = vars(self.no_defaults_parser.parse_args(args))
        if 'root_path' in specified_args and 'config' not in specified_args:
            specified_args['config'] = os.path.join(specified_args['root_path'], 'config.yaml')
        # Read the yaml config files.
        if 'config' in specified_args:
            self.file = specified_args['config']
            with open(self.file, 'r') as file:
                loaded_args = yaml.safe_load(file)
                # Update the defaults with the parameters in the config file.
                self.update_config_dict(self.all_args, loaded_args)
        # Finally, override the parameters based on commandline arguments.
        self.construct_config_dict(specified_args, nonexist_ok=False)
        # For compatibility, we still return all the arguments from argparser.
        parsed_args = self.defaults_parser.parse_args(args)
        self.update_arguments()
        # Print all configuration.
        if verbose:
            print('Configurations:\n')
            print(self.dump_config(self.all_args))
        if self.all_args["debug"]["save_minimal_config"] is not None:
            with open(self.all_args["debug"]["save_minimal_config"], 'w') as f:
                f.write(self.dump_config(self.all_args, omit_defaults=True, omit_keys=self.all_args["debug"]["save_minimal_config_omit_keys"]))
        return parsed_args

    def positive_float_checker(self, x):
        tolerance = float(x)
        if tolerance < 0:
            raise argparse.ArgumentTypeError('invalid attack tolerance')
        return tolerance

    def keys(self):
        return self.all_args.keys()

    def items(self):
        return self.all_args.items()

    def __getitem__(self, key):
        """Read an item from the dictionary of parameters."""
        return self.all_args[key]

    def __setitem__(self, key, value):
        """Set an item from the dictionary of parameters."""
        self.all_args[key] = value


class ReadOnlyDict(dict):
    def __readonly__(self, *args, **kwargs):
        raise RuntimeError("You must register a global parameter in arguments.py.")

    def __setitem__(self, key, value):
        if key not in self:
            raise RuntimeError("You must register a global parameter in arguments.py.")
        else:
            super().__setitem__(key, value)

    __delitem__ = __readonly__
    pop = __readonly__
    popitem = __readonly__
    clear = __readonly__
    update = __readonly__
    setdefault = __readonly__
    del __readonly__


# Global configuration variable
Config = ConfigHandler()
# Global variables
Globals = ReadOnlyDict({
    "starting_timestamp": int(time.time()),
    "example_idx": -1,
    "out": {"idx": None, "pred": None, "attack_margin": None, "pred_adv": None,
            "init_crown_bounds": None, "init_alpha_crown": None,
            "refined_lb": None, "decisions": [], "results": 'timeout',
            "time": None, "neurons_visited": None}})


if __name__ == '__main__':
    Config.construct_config_dict(Config.default_args)
    Config.dump_config(Config.all_args, out_to_doc=True, show_help=True)
