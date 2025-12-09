#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""α,β-CROWN (alpha-beta-CROWN) verifier main interface."""

import socket
import random
import os
import sys
import time
import torch
import numpy as np
import tempfile

import arguments
from jit_precompile import precompile_jit_kernels
from beta_CROWN_solver import LiRPANet
from lp_mip_solver import mip
from attack import attack, reset_attack_stats, get_attack_stats
from utils import Logger
from specifications import vnnlibHandler
from incomplete_verifier_func import SpecHandler
from loading import load_model_and_vnnlib, parse_run_mode, Customized  # pylint: disable=unused-import
from read_vnnlib import read_vnnlib
from cuts.cut_utils import terminate_mip_processes
from lp_test import compare_optimized_bounds_against_lp_bounds


class ABCROWN:
    def __init__(self, args=None, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, list):
                args.append(f'--{k}')
                args.extend(list(map(str, v)))
            elif isinstance(v, bool):
                if v:
                    args.append(f'--{k}')
                else:
                    args.append(f'--no_{k}')
            else:
                args.append(f'--{k}={v}')
        arguments.Config.parse_config(args)

    def attack(self, model_ori, verified_status, verified_success):
        if arguments.Config['model']['with_jacobian']:
            print('Using BoundedModule for attack for this model with JacobianOP')
            model = LiRPANet(model_ori, in_size=[1, *self.vnnlib_handler.input_shape[1:]]).net
        else:
            model = model_ori
        device = arguments.Config['general']['device']
        x, c, rhs, or_spec_size, _, _ = self.vnnlib_handler.all_specs.get(device)
        vnnlib = self.vnnlib_handler.vnnlib
        return attack(model, x, c, rhs, or_spec_size, vnnlib, verified_status, verified_success)

    def main(self, interm_bounds=None):
        print(f'Experiments at {time.ctime()} on {socket.gethostname()}')
        torch.manual_seed(arguments.Config['general']['seed'])
        random.seed(arguments.Config['general']['seed'])
        np.random.seed(arguments.Config['general']['seed'])
        torch.set_printoptions(precision=8)
        device = arguments.Config['general']['device']
        if device != 'cpu':
            torch.cuda.manual_seed_all(arguments.Config['general']['seed'])
            # Always disable TF32 (precision is too low for verification).
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if arguments.Config['general']['deterministic']:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.use_deterministic_algorithms(True)
        if arguments.Config['general']['double_fp']:
            torch.set_default_dtype(torch.float64)
        if arguments.Config['general']['precompile_jit']:
            precompile_jit_kernels()

        if arguments.Config['general']['reset_seed_after_precompile']:
            # Reset the seed after precompilation to ensure reproducibility.
            torch.manual_seed(arguments.Config['general']['seed'])
            random.seed(arguments.Config['general']['seed'])
            np.random.seed(arguments.Config['general']['seed'])
            if device != 'cpu':
                torch.cuda.manual_seed_all(arguments.Config['general']['seed'])

        bab_args = arguments.Config['bab']
        debug_args = arguments.Config['debug']
        timeout_threshold = bab_args['timeout']
        select_instance = arguments.Config['data']['select_instance']
        complete_verifier = arguments.Config['general']['complete_verifier']
        # FIXME: We always enable incomplete verification by default.
        # Maybe this flag can be removed in the future.
        enable_incomplete = arguments.Config['general']['enable_incomplete_verification']

        # FIXME: Remove this check when bab attack is fixed.
        bab_attack_enabled = arguments.Config['bab']['attack']['enabled']
        assert not bab_attack_enabled, 'Bab attack is not out of date and to be fixed.'

        cut_enabled = bab_args['cut']['enabled']

        (run_mode, save_path, file_root, example_idx_list, model_ori,
        vnnlib_all, shape) = parse_run_mode()
        self.logger = Logger(run_mode, save_path, timeout_threshold)

        if arguments.Config['general']['return_optimized_model']:
            assert len(example_idx_list) == 1, (
                'To return the optimized model, only one instance can be processed'
            )
        if debug_args['sanity_check']:
            print('Warning: Sanity Check Debugging is enabled.',
            'The PGD upper bound will be calculated and used as the RHS offset.')
            arguments.Config['attack']['pgd_order'] = 'before'

        self.vnnlib_handler = self.spec_handler_incomplete = None

        for new_idx, csv_item in enumerate(example_idx_list):
            # Reset attack statistics at the beginning of each test instance
            if arguments.Config['attack']['pgd_order'] != 'skip':
                reset_attack_stats()
            arguments.Globals['example_idx'] = new_idx

            vnnlib_id = new_idx + arguments.Config['data']['start']
            # Select some instances to verify
            if select_instance and not vnnlib_id in select_instance:
                continue
            self.logger.record_start_time()

            print(f'\n {"%"*35} idx: {new_idx}, vnnlib ID: {vnnlib_id} {"%"*35}')
            if arguments.Config['general']['save_output']:
                arguments.Globals['out']['idx'] = new_idx   # saved for test

            onnx_path = None
            if run_mode != 'customized_data':
                if len(csv_item) == 3:
                    # model, vnnlib, timeout
                    model_ori, shape, vnnlib, onnx_path = load_model_and_vnnlib(
                        file_root, csv_item)
                    arguments.Config['model']['onnx_path'] = os.path.join(file_root, csv_item[0])
                    arguments.Config['specification']['vnnlib_path'] = os.path.join(
                        file_root, csv_item[1])
                else:
                    # Each line contains only 1 item, which is the vnnlib spec.
                    vnnlib = read_vnnlib(os.path.join(file_root, csv_item[0]))
                    assert arguments.Config['model']['input_shape'] is not None, (
                        'vnnlib does not have shape information, '
                        'please specify by --input_shape')
                    shape = arguments.Config['model']['input_shape']
            else:
                vnnlib = vnnlib_all[new_idx]  # vnnlib_all is a list of all standard vnnlib

            # Skip running the actual verifier during preparation.
            if arguments.Config['general']['prepare_only']:
                continue

            # FIXME Don't write bab_args['timeout'] above.
            # Then these updates can be moved to arguments.update_arguments()
            bab_args['timeout'] = float(bab_args['timeout'])
            if bab_args['timeout_scale'] != 1:
                new_timeout = bab_args['timeout'] * bab_args['timeout_scale']
                print(f'Scaling timeout: {bab_args["timeout"]} -> {new_timeout}')
                bab_args['timeout'] = new_timeout
            if bab_args['override_timeout'] is not None:
                new_timeout = bab_args['override_timeout']
                print(f'Overriding timeout: {new_timeout}')
                bab_args['timeout'] = new_timeout
            timeout_threshold = bab_args['timeout']
            self.logger.update_timeout(timeout_threshold)

            if complete_verifier.startswith('Customized'):
                res = eval(  # pylint: disable=eval-used
                    complete_verifier
                )(model_ori, vnnlib, os.path.join(file_root, onnx_path))
                self.logger.summarize_results(res, new_idx)
                continue

            start_time = time.time()
            vnnlib_handler = vnnlibHandler(vnnlib, shape)
            print(f'Finished vnnlib processing in {time.time() - start_time:4f} seconds')

            self.vnnlib_handler = vnnlib_handler
            # FIXME: Remove adhoc_tuning()
            # [0:1] is an ad-hoc tmp operation to align with the API of other functions
            # eventually, other functions should be updated to use vnnlibHandler
            x = vnnlib_handler.x[0:1].to(device)
            data_min = vnnlib_handler.data_min[0:1].to(device)
            data_max = vnnlib_handler.data_max[0:1].to(device)
            model_ori = model_ori.eval().to(device)
            if arguments.Config['general']['adhoc_tuning']:
                eval(   # pylint: disable=eval-used
                    arguments.Config['general']['adhoc_tuning'])(model_ori, vnnlib_handler)

            # If complete_verifier is 'auto', the variable "complete_verifier" can be updated in
            # every iteration. So we need to check the global variable
            # arguments.Config['general']['complete_verifier'] to determine the method.
            if arguments.Config['general']['complete_verifier'] == 'auto':
                # Default threshold for input split is 20.
                use_input_split = (np.prod(np.array(vnnlib_handler.input_shape[1:]))
                                   <= bab_args['branching']['input_split']['input_dim_threshold'])
                if use_input_split:
                    complete_verifier = 'input_bab'
                else:
                    # We check if bab-refine can be used
                    # (i.e., if the model doesn't have Conv layers).
                    conv_keywords = ['Conv1d', 'Conv2d', 'ConvTranspose2d']
                    model_has_conv = any(type(m).__name__ in conv_keywords
                                         for m in model_ori.modules())
                    if not model_has_conv:
                        complete_verifier = 'bab-refine'
                    else:
                        complete_verifier = 'bab'

                bab_args['branching']['input_split']['enable'] = use_input_split
                bab_args['branching']['method'] = 'sb' if use_input_split else 'kfsb'
                # We always use crown bounds for input split.
                arguments.Config['solver']['bound_prop_method'] = (
                    'crown' if use_input_split else 'alpha-crown'
                )
                # We enable cut if complete_verifier is bab.
                bab_args['cut']['enabled'] = cut_enabled and complete_verifier == 'bab'
                # The default conv mode is 'patches',
                # but needs to be set to 'matrix' if cut is enabled.
                arguments.Config['general']['conv_mode'] = (
                    'matrix' if bab_args['cut']['enabled'] else 'patches'
                )

                if complete_verifier == 'bab-refine':
                    print('Disabling Clip-and-Verify because it does not support bab-refine!')
                    clip_cfg = arguments.Config['bab']['clip_n_verify']
                    clip_cfg['clip_input_domain']['enabled'] = False
                    clip_cfg['clip_interm_domain']['enabled'] = False

            # Create temporary directory for cplex_cuts if enabled and cuts_path is not set.
            use_temp_cuts_path = (
                arguments.Config['bab']['cut']['cplex_cuts']
                and bab_args['cut']['cuts_path'] is None
            )
            if use_temp_cuts_path:
                temp_cuts_folder = tempfile.TemporaryDirectory(prefix='abcrown_cuts_', dir='/tmp')
                bab_args['cut']['cuts_path'] = temp_cuts_folder.name
                print(f'Using temporary cuts path: {temp_cuts_folder}')

            rhs_offset_init = arguments.Config['specification']['rhs_offset']
            if rhs_offset_init is not None and not debug_args['sanity_check']:
                vnnlib_handler.add_rhs_offset(rhs_offset_init)

            verified_status, verified_success = 'unknown', False

            if arguments.Config['attack']['pgd_order'] == 'before':
                (verified_status, verified_success, attack_examples,
                 attack_margins, all_adv_candidates) = self.attack(
                    model_ori, verified_status, verified_success)
                # Record the PGD attack stats after the attack
                get_attack_stats(self.logger, new_idx)
                if debug_args['sanity_check']:
                    if debug_args['sanity_check'] == 'Full':
                        rhs_offset = attack_margins
                    else:
                        rhs_offset = attack_margins.min()
                    # changes the verification status back to unknown and the pgd_order is now skip
                    # so that now unsafe instances will also time out
                    print('Warning: Changing the RHS offset to the worst PGD '
                          'upper bound. If "rhs_offset" was set in the config/commandline, '
                          'it will be ignored.')
                    print(f'Using PGD upper bound:\n{rhs_offset}.')
                    print(f'Shape of attack_margins: {attack_margins.shape}')
                    print(f'Verified success: {verified_success} -> False')
                    print(f'Verified success: {verified_status} -> \'unknown\'')

                    vnnlib_handler.add_rhs_offset(rhs_offset)
                    arguments.Config['attack']['pgd_order'] = 'skip'
                    verified_status, verified_success = 'unknown', False
            else:
                attack_examples = attack_margins = all_adv_candidates = None

            model_incomplete = None
            ret = {}

            if debug_args['test_optimized_bounds']:
                compare_optimized_bounds_against_lp_bounds(
                    model_ori, x, data_ub=data_max, data_lb=data_min, vnnlib=vnnlib
                )

            # Incomplete verification is enabled by default. The intermediate lower
            # and upper bounds will be reused in bab and mip.
            if (not verified_success and enable_incomplete):
                verified_status, ret = self.incomplete_verifier(
                    model_ori, interm_bounds
                )
                # filter out the attack examples for verified OR specs.
                self.spec_handler_incomplete: SpecHandler
                attack_examples, attack_margins, all_adv_candidates = (
                    self.spec_handler_incomplete.prune_attack_ret(
                        attack_examples, attack_margins, all_adv_candidates
                    )
                )

                if arguments.Config['general']['return_optimized_model']:
                    # It is actually the LiRPANet model.
                    return verified_status

                verified_success = verified_status != 'unknown'
                model_incomplete = ret.get('model', None)

            if not verified_success and arguments.Config['attack']['pgd_order'] == 'after':
                (verified_status, verified_success, attack_examples,
                 attack_margins, all_adv_candidates) = self.attack(
                    model_ori, verified_status, verified_success)
                # Record the PGD attack stats after the attack
                get_attack_stats(self.logger, new_idx)
            # MIP or MIP refined bounds.
            if not verified_success and complete_verifier in ['bab-refine', 'mip']:
                # rhs = ? NEED TO SAVE TO LIRPA_MODULE
                mip_skip_unsafe = arguments.Config['solver']['mip']['skip_unsafe']

                # TODO: remove adhoc process for mip when mip is updated
                self.spec_handler_incomplete.adhoc_process_for_mip(ret)

                verified_status, ret_mip = mip(
                    model_incomplete, ret, vnnlib_handler, mip_skip_unsafe=mip_skip_unsafe,
                    pgd_attack_example=[attack_examples, attack_margins],
                    verifier=complete_verifier)
                verified_success = verified_status != 'unknown'

                ret.update(ret_mip)
                self.spec_handler_incomplete.adhoc_post_process_for_mip(ret)

            # extract the process pool for cut inquiry
            if bab_args['cut']['enabled'] and bab_args['cut']['cplex_cuts']:
                # use nullity of model_incomplete as an indicator of whether cut
                # processes are launched
                if model_incomplete is not None:
                    print('Cut inquiry processes are launched.')

            # BaB bounds. (not do bab if unknown by mip solver for now)
            if (not verified_success
                    and complete_verifier != 'skip'
                    and verified_status != 'unknown-mip'):
                # expand the interm bounds and alphas if needed.
                # NOTE: ideally, we can do it in incomplete_verifier().
                # But to be compatible with the MIP interface, we do it here.
                if enable_incomplete:
                    self.spec_handler_incomplete.expand_intermediate(ret)

                if bab_attack_enabled:
                    ret['attack_examples'] = all_adv_candidates
                    ret['attack_margins'] = attack_margins

                verified_status = self.complete_verifier(
                    model_ori,
                    new_idx, bab_ret=self.logger.bab_ret,
                    timeout_threshold=timeout_threshold - (time.time() - self.logger.start_time),
                    reference_dict=ret,
                )

            if (bab_args['cut']['enabled'] and bab_args['cut']['cplex_cuts']
                    and model_incomplete is not None):
                terminate_mip_processes(
                    model_incomplete.mip_building_proc, model_incomplete.processes
                )
                del model_incomplete.processes

            if use_temp_cuts_path:
                # Remove the temporary cuts directory.
                print(f'Removing temporary cuts path: {bab_args["cut"]["cuts_path"]}')
                temp_cuts_folder.cleanup()
                bab_args['cut']['cuts_path'] = None

            del ret

            if debug_args['sanity_check']:
                assert 'unknown' in verified_status, 'Sanity check failed. Something is wrong.'

            # Summarize results.
            self.logger.summarize_results(verified_status, new_idx)
            # At the end of each test instance, record attack statistics
            get_attack_stats(self.logger, new_idx)

        self.logger.finish()
        return self.logger.verification_summary

    from incomplete_verifier_func import incomplete_verifier
    from complete_verifier_func import complete_verifier, bab

if __name__ == '__main__':
    abcrown = ABCROWN(args=sys.argv[1:])
    abcrown.main()
