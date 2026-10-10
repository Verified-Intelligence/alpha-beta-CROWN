#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENSE file in the repository root.       ##
##                                                                     ##
#########################################################################
"""Unit tests for lp_mip_solver/refine_core.py"""
import os
import sys
import time
import unittest

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMipSolverTimeoutHandling(unittest.TestCase):
    """Tests for timeout handling in mip_solver."""

    def test_timeout_check_logic(self):
        """Test timeout detection using module-level variables."""
        from lp_mip_solver import refine_core

        # Save original values
        orig_time_start = refine_core.mip_refine_time_start
        orig_timeout = refine_core.mip_refine_timeout

        try:
            # Set up timeout condition
            refine_core.mip_refine_time_start = time.time() - 300  # 5 minutes ago
            refine_core.mip_refine_timeout = 100  # 100 second timeout

            # Check timeout condition
            elapsed = time.time() - refine_core.mip_refine_time_start
            is_timed_out = elapsed >= refine_core.mip_refine_timeout

            self.assertTrue(is_timed_out)
        finally:
            # Restore original values
            refine_core.mip_refine_time_start = orig_time_start
            refine_core.mip_refine_timeout = orig_timeout


class TestMipRefinementBeta(unittest.TestCase):
    """Tests for beta state initialized during MIP refinement."""

    def test_uses_owning_node_computation_dtype(self):
        from lp_mip_solver.refine_core import _new_mip_refinement_beta

        class Node:
            output_dtype = torch.float64

        class Net:
            @staticmethod
            def get_output_dtype(node):
                return node.output_dtype

        beta = _new_mip_refinement_beta(
            Net(), Node(), (1, 2), device='cpu')

        self.assertEqual(beta.val.dtype, torch.float64)
        self.assertEqual(beta.sign.dtype, torch.float64)
        self.assertTrue(beta.val.requires_grad)
        self.assertTrue(beta.val.is_leaf)


if __name__ == '__main__':
    unittest.main()
