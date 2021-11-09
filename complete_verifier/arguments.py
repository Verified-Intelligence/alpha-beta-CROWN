#########################################################################
##         This file is part of the alpha-beta-CROWN verifier          ##
##                                                                     ##
## Copyright (C) 2021, Huan Zhang <huan@huan-zhang.com>                ##
##                     Kaidi Xu <xu.kaid@northeastern.edu>             ##
##                     Shiqi Wang <sw3215@columbia.edu>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Yihan Wang <yihanwang@ucla.edu>                 ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
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
import argparse
from collections import defaultdict


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

    def add_common_options(self):
        """
        Add all parameters that are shared by different front-ends.
        """
        # We must set how each parameter will be presented in the config file, via the "hierarchy" parameter.
        # Global Configurations, not specific for a particular algorithm.

        # The "--config" option does not exist in our parameter dictionar.
        self.add_argument('--config', type=str, help='path to YAML format config file.', hierarchy=None)

        h = ["general"]
        self.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='Select device to run verifier, cpu or cuda (GPU).', hierarchy=h + ["device"])
        self.add_argument("--seed", type=int, default=100, help='Random seed.', hierarchy=h + ["seed"])
        self.add_argument("--conv_mode", default="patches", choices=["patches", "matrix"],
                help='Convolution mode during bound propagation: "patches" mode (default) is very efficient, but may not support all architecture; "matrix" mode is slow but supports all architectures.', hierarchy=h + ["conv_mode"])
        self.add_argument("--deterministic", action='store_true', help='Run code in CUDA deterministic mode, which has slower performance but better reproducibility.', hierarchy=h + ["deterministic"])
        self.add_argument("--double_fp", action='store_true',
                help='Use double precision floating point. GPUs with good double precision support are preferable (NVIDIA P100, V100, A100; AMD Radeon Instinc MI50, MI100).', hierarchy=h + ["double_fp"])
        self.add_argument("--loss_reduction_func", default="sum", help='When batch size is not 1, this reduction function is applied to reduce the bounds into a single number (options are "sum" and "min").', hierarchy=h + ["loss_reduction_func"])
        self.add_argument("--record_lb", action='store_true', help='Save lower bound during branch and bound. For debugging only.', hierarchy=h + ["record_bounds"])

        h = ["model"]
        self.add_argument("--load", type=str, default=None, help='Load pretrained model from this specified path.', hierarchy=h + ["path"])

        h = ["data"]
        self.add_argument("--start", type=int, default=0, help='Start from the i-th property in specified dataset.', hierarchy=h + ["start"])
        self.add_argument("--end", type=int, default=10000, help='End with the (i-1)-th property in the dataset.', hierarchy=h + ["end"])
        self.add_argument('--num_classes', type=int, default=10, help="Number of classes for classification problem.", hierarchy=h + ["num_classes"])
        self.add_argument("--mean", nargs='+', type=float, default=[0.0, 0.0, 0.0], help='Mean vector used in data preprocessing.', hierarchy=h + ["mean"])
        self.add_argument("--std", nargs='+', type=float, default=[1.0, 1.0, 1.0], help='Std vector used in data preprocessing.', hierarchy=h + ["std"])
        self.add_argument('--pkl_path', type=str, default=None, help="Load properties to verify from a .pkl file (only used for oval20 dataset).", hierarchy=h + ["pkl_path"])

        h = ["specification"]
        self.add_argument("--norm", type=float, default='inf', help='Lp-norm for epsilon perturbation in robustness verification (1, 2, inf).', hierarchy=h + ["norm"])
        self.add_argument("--epsilon", type=float, default=None, help='Set perturbation size (Lp norm). If not set, a default value may be used based on dataset loader.', hierarchy=h + ["epsilon"])

        h = ["solver", "alpha-crown"]
        self.add_argument("--lr_init_alpha", type=float, default=0.1, help='Learning rate for the optimizable parameter alpha in alpha-CROWN bound.', hierarchy=h + ["lr_alpha"])
        self.add_argument('--init_iteration', type=int, default=100, help='Number of iterations for alpha-CROWN incomplete verifier.', hierarchy=h + ["iteration"])
        self.add_argument("--share_slopes", action='store_true', help='Share some alpha variables to save memory at the cost of slightly looser bounds.', hierarchy=h + ["share_slopes"])
        self.add_argument("--no_joint_opt", action='store_true', help='alpha-CROWN bounds without joint optimization (only optimize alpha for the last layer bound).', hierarchy=h + ["no_joint_opt"])

        h = ["solver", "beta-crown"]
        self.add_argument("--batch_size", type=int, default=64, help='Batch size in beta-CROWN (number of parallel splits).', hierarchy=h + ["batch_size"])
        self.add_argument("--lr_alpha", type=float, default=0.01, help='Learning rate for optimizing alpha during branch and bound.', hierarchy=h + ["lr_alpha"])
        self.add_argument("--lr_beta", type=float, default=0.05, help='Learning rate for optimizing beta during branch and bound.', hierarchy=h + ["lr_beta"])
        self.add_argument("--lr_decay", type=float, default=0.98, help='Learning rate decay factor during optimization. Need to use a larger value like 0.99 or 0.995 when you increase the number of iterations.', hierarchy=h + ["lr_decay"])
        self.add_argument("--optimizer", default="adam", help='Optimizer used for alpha and beta optimization.', hierarchy=h + ["optimizer"])
        self.add_argument("--iteration", type=int, default=50, help='Number of iteration for optimizing alpha and beta during branch and bound.', hierarchy=h + ["iteration"])
        self.add_argument('--no_beta', action='store_false', dest='beta', help='Disable/Enable beta split constraint (this option is for ablation study only and should not be used normally).', hierarchy=h + ["beta"])
        self.add_argument('--no_beta_warmup', action='store_false', dest='beta_warmup', help='Do not use beta warmup from branching history (this option is for ablation study only and should not be used normally).', hierarchy=h + ["beta_warmup"])


        h = ["solver", "mip"]
        self.add_argument('--mip_multi_proc', type=int, default=None,
                help='Number of multi-processes for mip solver. Each process computes a mip bound for an intermediate neuron. Default (None) is to auto detect the number of CPU cores (note that each process may use multiple threads, see the next option).', hierarchy=h + ["parallel_solvers"])
        self.add_argument('--mip_threads', type=int, default=1,
                help='Number of threads for echo mip solver process (default is to use 1 thread for each solver process).', hierarchy=h + ["solver_threads"])
        self.add_argument('--mip_perneuron_refine_timeout', type=float, default=15, help='MIP timeout threshold for improving each intermediate layer bound (in seconds).', hierarchy=h + ["refine_neuron_timeout"])
        self.add_argument('--mip_refine_timeout', type=float, default=0.8, help='Percentage (x100%) of time used for improving all intermediate layer bounds using mip. Default to be 0.8*timeout.', hierarchy=h + ["refine_neuron_time_percentage"])

        h = ["bab"]
        self.add_argument("--max_domains", type=int, default=200000, help='Max number of subproblems in branch and bound.', hierarchy=h + ["max_domains"])
        self.add_argument("--decision_thresh", type=float, default=0, help='Decision threshold of lower bounds. When lower bounds are greater than this value, verification is successful. Set to 0 for robustness verification.', hierarchy=h + ["decision_thresh"])
        self.add_argument("--timeout", type=float, default=360, help='Timeout (in second) for verifying one image/property.', hierarchy=h + ["timeout"])
        self.add_argument("--get_upper_bound", action='store_true', help='Update global upper bound during BaB (has extra overhead, typically the upper bound is not used).', hierarchy=h + ["get_upper_bound"])
        self.add_argument("--DFS_percent", type=float, default=0., help='Percent of domains for depth first search (not used).', hierarchy=h + ["dfs_percent"])

        h = ["bab", "branching"]
        self.add_argument("--branching_method", default="kfsb", choices=["babsr", "fsb", "kfsb", "sb"], help='Branching heuristic. babsr is fast but less accurate; fsb is slow but most accurate; kfsb is usualy a balance.', hierarchy=h + ["method"])
        self.add_argument("--branching_candidates", type=int, default=3, help='Number of candidates to consider when using fsb or kfsb. More leads to slower but better branching.', hierarchy=h + ["candidates"])
        self.add_argument("--branching_reduceop", choices=["min", "max", "mean", "auto"], default="min", help='Reduction operation to compute branching scores from two sides of a branch (min or max). max can work better on some models.', hierarchy=h + ["reduceop"])


        h = ["attack"]
        self.add_argument('--pgd_order', choices=["before", "after", "skip"], default="before",  help='Run PGD before/after incomplete verification, or skip it.', hierarchy=h + ["pgd_order"])

    def add_argument(self, *args, **kwargs):
        """Add a single parameter to the parser. We will check the 'hierarchy' specified and then pass the remaining arguments to argparse."""
        if 'hierarchy' not in kwargs:
            raise ValueError("please specify the 'hierarchy' parameter when using this function.")
        hierarchy = kwargs.pop('hierarchy')
        help = kwargs.get('help', '')
        private_option = kwargs.pop('private', False)
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
                self.set_dict_by_hierarchy(self.all_args, h, arg_val, nonexist_ok=nonexist_ok)

    def update_config_dict(self, old_args_dict, new_args_dict, levels=None):
        """Recursively update the dictionary of all parameters based on the dict read from config file."""
        if levels is None:
            levels = []
        if isinstance(new_args_dict, dict):
            # Go to the next dict level.
            for k in new_args_dict:
                self.update_config_dict(old_args_dict, new_args_dict[k], levels=levels + [k])
        else:
            # Reached the leaf level. Set the corresponding key.
            self.set_dict_by_hierarchy(old_args_dict, levels, new_args_dict, nonexist_ok=False)

    def generate_template(self, args_dict, level=[]):
        """Generate a template config file with help information."""
        ret_string = ''
        for key, val in args_dict.items():
            if isinstance(val, dict):
                ret = self.generate_template(val, level + [key])
                if len(ret) > 0:
                    # Next level is not empty, print it.
                    ret_string += ' ' * (len(level) * 2) + f'{key}:\n' + ret
            else:
                h = self.help_messages[','.join(level + [key])]
                if 'debug' in key or 'not use' in h or 'not be use' in h or 'debug' in h or len(h) == 0:
                    # Skip some debugging options.
                    continue
                h = f'  # {h}'
                yaml_line = yaml.safe_dump({key: val}, default_flow_style=None).strip().replace('{', '').replace('}', '')
                ret_string += ' ' * (len(level) * 2) + f'{yaml_line}{h}\n'
        if len(level) > 0:
            return ret_string
        else:
            # Top level, output to file.
            output_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs',
                    os.path.splitext(os.path.basename(sys.argv[0]))[0] + '_all_params.yaml')
            with open(output_name, 'w') as f:
                f.write(ret_string)
            return ret_string

    def parse_config(self):
        """
        Main function to parse parameter configurations. The commandline arguments have the highest priority;
        then the parameters specified in yaml config file. If a parameter does not exist in either commandline
        or the yaml config file, we use the defaults defined in add_common_options() defined above.
        """
        # Parse an empty commandline to get all default arguments.
        default_args = vars(self.defaults_parser.parse_args([]))
        # Create the dictionary of all parameters, all set to their default values.
        self.construct_config_dict(default_args)
        # Update documents.
        # self.generate_template(self.all_args)
        # These are arguments specified in command line.
        specified_args = vars(self.no_defaults_parser.parse_args())
        # Read the yaml config files.
        if 'config' in specified_args:
            with open(specified_args['config'], 'r') as config_file:
                loaded_args = yaml.safe_load(config_file)
                # Update the defaults with the parameters in the config file.
                self.update_config_dict(self.all_args, loaded_args)
        # Finally, override the parameters based on commandline arguments.
        self.construct_config_dict(specified_args, nonexist_ok=False)
        # For compatibility, we still return all the arguments from argparser.
        parsed_args = self.defaults_parser.parse_args()
        return parsed_args

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


# Global configuration variable
Config = ConfigHandler()

