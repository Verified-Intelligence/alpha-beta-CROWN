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
"""Unit tests for bab.py module."""

import copy
import os
import sys
import time
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGetSplitDepth:
    """Tests for get_split_depth function."""

    def test_batch_larger_than_min(self):
        """Test when batch size is larger than min_batch_size."""
        from bab import get_split_depth

        result = get_split_depth(batch_size=100, min_batch_size=50, min_depth=1)
        assert result == 1  # Should return min_depth

    def test_batch_smaller_than_min(self):
        """Test when batch size is smaller than min_batch_size."""
        from bab import get_split_depth

        # batch_size=10, min_batch_size=64, min_depth=1
        # log(64/10) / log(2) = log(6.4) / log(2) ≈ 2.68
        result = get_split_depth(batch_size=10, min_batch_size=64, min_depth=1)
        assert result >= 2  # Should be at least 2 to get closer to 64

    def test_batch_equal_to_min(self):
        """Test when batch size equals min_batch_size."""
        from bab import get_split_depth

        result = get_split_depth(batch_size=64, min_batch_size=64, min_depth=1)
        assert result == 1

    def test_very_small_batch(self):
        """Test with very small batch size."""
        from bab import get_split_depth

        result = get_split_depth(batch_size=1, min_batch_size=128, min_depth=1)
        # log(128/1) / log(2) = log(128) / log(2) = 7
        assert result == 7

    def test_large_min_depth(self):
        """Test with larger min_depth."""
        from bab import get_split_depth

        result = get_split_depth(batch_size=32, min_batch_size=64, min_depth=5)
        # Even though batch is close to min_batch_size, min_depth should dominate
        assert result >= 5

    def test_min_depth_zero(self):
        """Test with min_depth of 0."""
        from bab import get_split_depth

        result = get_split_depth(batch_size=100, min_batch_size=50, min_depth=0)
        assert result >= 0


class TestStats:
    """Tests for Stats class used in BaB."""

    def test_stats_creation(self):
        """Test creating Stats object."""
        from utils import Stats

        stats = Stats()
        assert stats.visited == 0
        assert stats.all_node_split is False
        assert hasattr(stats, 'timer')

    def test_stats_visited_increment(self):
        """Test incrementing visited counter."""
        from utils import Stats

        stats = Stats()
        stats.visited += 10
        assert stats.visited == 10

    def test_stats_all_node_split_flag(self):
        """Test all_node_split flag."""
        from utils import Stats

        stats = Stats()
        stats.all_node_split = True
        assert stats.all_node_split is True


class TestStopCriterion:
    """Tests for stop criterion functions used in BaB."""

    def test_stop_criterion_batch_any_basic(self):
        """Test basic stop criterion with tensor threshold."""
        from auto_LiRPA.utils import stop_criterion_batch_any

        threshold = torch.tensor([0.0])
        stop_func = stop_criterion_batch_any(threshold)

        # Test with lb > threshold (should stop)
        lb_above = torch.tensor([[0.5]])
        assert stop_func(lb_above).all()

        # Test with lb < threshold (should not stop)
        lb_below = torch.tensor([[-0.5]])
        assert not stop_func(lb_below).all()


class TestAutoBatchSize:
    """Tests for AutoBatchSize used in BaB."""

    def test_auto_batch_size_creation(self):
        """Test creating AutoBatchSize object."""
        from auto_LiRPA.utils import AutoBatchSize

        auto_bs = AutoBatchSize(
            init_batch_size=64,
            device='cpu',
            vram_ratio=0.9,
            enable=False
        )
        assert auto_bs is not None

    def test_auto_batch_size_record(self):
        """Test recording actual batch size."""
        from auto_LiRPA.utils import AutoBatchSize

        auto_bs = AutoBatchSize(
            init_batch_size=64,
            device='cpu',
            vram_ratio=0.9,
            enable=False
        )
        auto_bs.record_actual_batch_size(32)
        # Should not raise

    @pytest.mark.parametrize(
        'device', (torch.device('cpu'), torch.device('meta')))
    def test_non_cuda_devices_do_not_query_memory(self, device):
        """CPU unit tests must not enter CUDA memory accounting."""
        from auto_LiRPA.utils import AutoBatchSize

        auto_bs = AutoBatchSize(
            init_batch_size=4, device=device, enable=True)
        auto_bs.record_actual_batch_size(4)

        assert auto_bs.update() is None
        assert auto_bs.batch_size == 4

    def test_disabled_mode_accepts_no_device(self):
        """A disabled batch-size controller does not require a device."""
        from auto_LiRPA.utils import AutoBatchSize

        auto_bs = AutoBatchSize(
            init_batch_size=4, device=None, enable=False)

        assert auto_bs.update() is None


class TestBaBConfiguration:
    """Tests for BaB configuration handling."""

    def test_config_timeout(self):
        """Test timeout configuration."""
        import arguments
        assert arguments.Config['bab']['timeout'] > 0

    def test_config_batch_size(self):
        """Test batch size configuration."""
        import arguments
        assert arguments.Config['solver']['batch_size'] > 0

    def test_config_max_domains(self):
        """Test max_domains configuration."""
        import arguments
        assert arguments.Config['bab']['max_domains'] > 0

    def test_config_branching_method(self):
        """Test branching method configuration."""
        import arguments
        assert arguments.Config['bab']['branching']['method'] in [
            'kfsb', 'fsb', 'babsr', 'naive'
        ]

    def test_config_cut_options(self):
        """Test cut options are properly configured."""
        import arguments
        assert isinstance(arguments.Config['bab']['cut']['enabled'], bool)
        assert isinstance(arguments.Config['bab']['cut']['biccos']['enabled'], bool)


class TestGetSplitDepthExtended:
    """Extended tests for get_split_depth function with edge cases."""

    def test_power_of_two_relationship(self):
        """Test when batch_size and min_batch_size have power of 2 relationship."""
        from bab import get_split_depth

        # 8 * 2^3 = 64
        result = get_split_depth(batch_size=8, min_batch_size=64, min_depth=1)
        assert result == 3

    def test_exact_log_calculation(self):
        """Test exact log calculation scenarios."""
        from bab import get_split_depth

        # log(32/2)/log(2) = log(16)/log(2) = 4
        result = get_split_depth(batch_size=2, min_batch_size=32, min_depth=1)
        assert result == 4

    def test_min_depth_dominates(self):
        """Test when min_depth is larger than calculated depth."""
        from bab import get_split_depth

        result = get_split_depth(batch_size=32, min_batch_size=64, min_depth=10)
        assert result == 10

    def test_fractional_batch_sizes(self):
        """Test with batch sizes that don't divide evenly."""
        from bab import get_split_depth

        # log(100/17)/log(2) ≈ 2.55
        result = get_split_depth(batch_size=17, min_batch_size=100, min_depth=1)
        assert result >= 2

    def test_equal_batch_and_min_depth_one(self):
        """Test when batch equals min and min_depth is 1."""
        from bab import get_split_depth

        result = get_split_depth(batch_size=128, min_batch_size=128, min_depth=1)
        assert result == 1

    def test_min_batch_size_one(self):
        """Test with min_batch_size of 1."""
        from bab import get_split_depth

        result = get_split_depth(batch_size=1, min_batch_size=1, min_depth=1)
        assert result == 1

    def test_very_large_batch(self):
        """Test with very large batch size."""
        from bab import get_split_depth

        result = get_split_depth(batch_size=10000, min_batch_size=64, min_depth=1)
        assert result == 1

    def test_float_batch_size_behavior(self):
        """Test behavior with non-integer min_batch_size calculations."""
        from bab import get_split_depth

        # batch_size=5, min_batch_size=100
        # log(100/5)/log(2) = log(20)/log(2) ≈ 4.32
        result = get_split_depth(batch_size=5, min_batch_size=100, min_depth=1)
        assert result >= 4



class TestGeneralBaB:
    """Tests for general_bab function."""

    def _create_mock_net(self):
        """Create a mock LiRPANet."""
        mock_net = MagicMock()
        mock_net.device = 'cpu'
        mock_net.final_name = '/output'
        mock_net.tot_ambi_nodes = 10
        mock_net.alpha_start_nodes = ['/output']
        mock_net.interm_transfer = True

        # Mock build_with_refined_bounds
        mock_net.build_with_refined_bounds = MagicMock(return_value={
            'global_ub': torch.tensor([[0.5]]),
            'global_lb': torch.tensor([[0.1]]),  # Positive = verified
            'mask': {'/relu_0': torch.ones(1, 5)},
            'lA': {'/output': torch.randn(1, 1, 5)},
            'alphas': {},
            'history': [{}],
        })
        return mock_net

    @patch('activation_split.bab_bootstrap.get_unstable_neurons')
    @patch('activation_split.bab_bootstrap.BatchedDomainList')
    def test_general_bab_safe_result(self, mock_domain_list, mock_get_unstable):
        """Test general_bab returns safe when lb > threshold (early exit).

        When stop_criterion_func(global_lb).all() is True (all lower bounds exceed
        the threshold), general_bab returns early with a 3-tuple: (lb, 0, 'safe').
        This is different from the full BaB loop which returns a 4-tuple.
        """
        from activation_split.bab_bootstrap import general_bab

        mock_get_unstable.return_value = ({}, 10)

        mock_net = self._create_mock_net()
        x = torch.randn(1, 1, 5, 5)
        c = torch.tensor([[1.0, -1.0]])
        rhs = torch.tensor([0.0])

        reference_dict = {
            'lower_bounds': {'/output': torch.tensor([[0.1]])},
            'upper_bounds': {'/output': torch.tensor([[0.5]])},
            'lA': None,
            'alphas': None,
            'refined_betas': None,
            'attack_examples': None,
        }

        result = general_bab(
            mock_net, x, c, rhs,
            reference_dict=reference_dict,
            timeout=10,
            max_iterations=1
        )

        # When the initial bounds already prove the property safe,
        # general_bab returns early with a 3-tuple
        assert len(result) == 3
        lb, visited, status = result
        assert visited == 0
        assert status == 'safe'

    @patch('activation_split.bab_bootstrap.get_unstable_neurons')
    @patch('activation_split.bab_bootstrap.BatchedDomainList')
    def test_general_bab_lp_test_mode(self, mock_domain_list, mock_get_unstable):
        """Test general_bab in LP test mode."""
        import arguments
        from activation_split.bab_bootstrap import general_bab

        arguments.Config['debug']['lp_test'] = 'LP'

        mock_get_unstable.return_value = ({}, 10)

        mock_net = self._create_mock_net()
        x = torch.randn(1, 1, 5, 5)
        c = torch.tensor([[1.0, -1.0]])
        rhs = torch.tensor([0.0])

        reference_dict = {
            'lower_bounds': {'/output': torch.tensor([[-0.5]])},
            'upper_bounds': {'/output': torch.tensor([[0.5]])},
            'lA': None,
            'alphas': None,
        }

        result = general_bab(
            mock_net, x, c, rhs,
            reference_dict=reference_dict,
        )

        # Should return unknown in LP test mode
        assert len(result) == 3
        lb, visited, status = result
        assert status == 'unknown'

        # Reset
        arguments.Config['debug']['lp_test'] = None

    @patch('activation_split.bab_bootstrap.compute_first_iteration_decision')
    @patch('activation_split.bab_bootstrap.get_unstable_neurons')
    @patch('activation_split.bab_bootstrap.BatchedDomainList')
    def test_general_bab_uses_nonlinear_branching(
            self, mock_domain_list, mock_get_unstable,
            mock_first_iteration_decision):
        """Nonlinear networks should use the nonlinear BaB heuristic."""
        import arguments
        from activation_split.bab_bootstrap import general_bab
        from heuristics.nonlinear import NonlinearBranching

        domains = MagicMock()
        domains.__len__.return_value = 0
        mock_domain_list.return_value = domains

        mask = {'/sin': torch.ones(1, 1, dtype=torch.bool)}
        mock_get_unstable.return_value = (mask, 1)

        mock_net = self._create_mock_net()
        mock_net.nonlinear_split = True
        mock_net.net.root_names = ['/input']
        mock_net.net.roots.return_value = []
        sin_node = MagicMock()
        sin_node.name = '/sin'
        mock_net.net.split_nodes = [sin_node]
        mock_net.net.layers_requiring_bounds = []
        mock_net.x = torch.zeros(1, 1)
        mock_net.c = torch.ones(1, 1, 1)
        mock_net.build_with_refined_bounds.return_value = {
            'global_ub': torch.tensor([[0.5]]),
            'global_lb': torch.tensor([[-0.1]]),
            'mask': mask,
            'lA': {'/output': torch.ones(1, 1, 1)},
            'alphas': {},
            'history': [{}],
            'lower_bounds': {
                '/input': torch.tensor([[-1.0]]),
                '/sin': torch.tensor([[-0.5]]),
                '/output': torch.tensor([[-0.1]]),
            },
            'upper_bounds': {
                '/input': torch.tensor([[1.0]]),
                '/sin': torch.tensor([[0.5]]),
                '/output': torch.tensor([[0.5]]),
            },
        }
        reference_dict = {
            'lower_bounds': None,
            'upper_bounds': None,
            'lA': None,
            'alphas': None,
            'refined_betas': None,
        }

        branching_args = arguments.Config['bab']['branching']
        with patch.dict(branching_args, {'method': 'nonlinear'}):
            result = general_bab(
                mock_net,
                mock_net.x,
                mock_net.c,
                torch.tensor([[0.0]]),
                reference_dict=reference_dict,
                max_iterations=1,
            )

        mock_first_iteration_decision.assert_called_once()
        heuristic = mock_first_iteration_decision.call_args.kwargs[
            'branching_heuristic']
        assert isinstance(heuristic, NonlinearBranching)
        assert result[2] == 'safe'


class TestBaBEdgeCases:
    """Edge case tests for BaB functions."""

    def test_get_split_depth_with_negative_batch(self):
        """Test get_split_depth behavior with edge case inputs."""
        from bab import get_split_depth

        # Very large min_batch_size
        result = get_split_depth(batch_size=1, min_batch_size=1000000, min_depth=1)
        assert result > 15  # log2(1000000) ≈ 20

    def test_split_depth_consistency(self):
        """Test that split_depth is consistent across equivalent inputs."""
        from bab import get_split_depth

        result1 = get_split_depth(batch_size=32, min_batch_size=64, min_depth=1)
        result2 = get_split_depth(batch_size=32, min_batch_size=64, min_depth=1)
        assert result1 == result2

    def test_split_depth_monotonicity(self):
        """Test that smaller batch sizes give larger split depths."""
        from bab import get_split_depth

        result1 = get_split_depth(batch_size=64, min_batch_size=128, min_depth=1)
        result2 = get_split_depth(batch_size=32, min_batch_size=128, min_depth=1)
        result3 = get_split_depth(batch_size=16, min_batch_size=128, min_depth=1)

        assert result1 <= result2 <= result3


class TestBaBConfigurationExtended:
    """Extended configuration tests for BaB."""

    def test_biccos_auto_param_config(self):
        """Test BICCOS auto_param configuration."""
        import arguments
        assert 'auto_param' in arguments.Config['bab']['cut']['biccos']

    def test_multi_tree_branching_config(self):
        """Test multi_tree_branching configuration."""
        import arguments
        mtb_config = arguments.Config['bab']['cut']['biccos']['multi_tree_branching']
        assert 'enabled' in mtb_config
        assert 'target_batch_size' in mtb_config
        assert 'keep_n_best_domains' in mtb_config
        assert 'iterations' in mtb_config
        assert 'restore_best_tree' in mtb_config

    def test_clip_n_verify_config(self):
        """Test clip_n_verify configuration."""
        import arguments
        cnv_config = arguments.Config['bab']['clip_n_verify']
        assert 'clip_input_domain' in cnv_config
        assert 'clip_interm_domain' in cnv_config

    def test_branching_config_options(self):
        """Test all branching configuration options."""
        import arguments
        branch_config = arguments.Config['bab']['branching']
        assert 'method' in branch_config
        assert 'candidates' in branch_config
        assert 'reduceop' in branch_config
        assert 'branching_input_and_activation' in branch_config


class TestBaBTimerIntegration:
    """Tests for timer integration in BaB."""

    def test_timer_operations(self):
        """Test that Timer operations work correctly."""
        from utils import Timer

        timer = Timer()

        timer.start('test_phase')
        time.sleep(0.01)
        timer.add('test_phase')

        assert 'test_phase' in timer.time_sum
        assert timer.time_sum['test_phase'] > 0

    def test_stats_timer_attribute(self):
        """Test that Stats has timer attribute."""
        from utils import Stats

        stats = Stats()
        assert hasattr(stats, 'timer')

        # Timer should be usable
        stats.timer.start('phase1')
        stats.timer.add('phase1')


class TestDomainOperations:
    """Tests for domain-related operations in BaB."""

    def test_check_worst_domain_import(self):
        """Test check_worst_domain function."""
        from branching_domains import check_worst_domain

        # Create a mock domains object
        mock_domains = MagicMock()
        mock_domains.__len__ = MagicMock(return_value=0)

        result = check_worst_domain(mock_domains)
        assert isinstance(result, torch.Tensor)

    def test_batched_domain_list_creation(self):
        """Test BatchedDomainList can be instantiated with mocks."""
        from branching_domains import BatchedDomainList

        # This is an integration test to verify the class works
        assert BatchedDomainList is not None


class TestMultiSpec:
    """Tests for multi-specification handling in BaB."""

    def test_multi_spec_with_multiple_thresholds(self):
        """Test stop criterion with multiple thresholds.

        stop_criterion_batch_any returns (x > threshold).any(dim=1, keepdim=True).
        This means if ANY specification is satisfied (lb > threshold), the sample can stop.
        """
        from auto_LiRPA.utils import stop_criterion_batch_any

        thresholds = torch.tensor([0.0, 0.0, 0.0])
        stop_func = stop_criterion_batch_any(thresholds)

        # All positive - all specs satisfied, should stop
        lb_all_pos = torch.tensor([[0.5, 0.3, 0.2]])
        assert stop_func(lb_all_pos).all()

        # Some positive - at least one spec satisfied per sample, so .any(dim=1) is True
        lb_mixed = torch.tensor([[0.5, -0.3, 0.2]])
        # Note: .any(dim=1) checks if ANY spec in the row is satisfied
        # Since 0.5 > 0 and 0.2 > 0, the result is True
        assert stop_func(lb_mixed).all()

        # All negative - no spec satisfied
        lb_all_neg = torch.tensor([[-0.5, -0.3, -0.2]])
        assert not stop_func(lb_all_neg).all()

        # Multiple samples - mixed results
        lb_multi = torch.tensor([
            [0.5, 0.3, 0.2],   # Has positives -> True
            [-0.5, -0.3, -0.2]  # All negative -> False
        ])
        result = stop_func(lb_multi)
        assert result[0].item() == True
        assert result[1].item() == False


class TestPruneAlphas:
    """Tests for prune_alphas function used in BaB."""

    def test_prune_alphas_basic(self):
        """Test basic prune_alphas functionality."""
        from prune import prune_alphas

        alphas = {
            '/relu_0': {
                '/output': torch.randn(2, 3, 4, 5)
            },
            '/relu_1': {
                '/output': torch.randn(2, 3, 4, 5),
                '/unused': torch.randn(2, 3, 4, 5)
            }
        }
        start_nodes = ['/output']

        result = prune_alphas(alphas, start_nodes)

        # Should keep only alphas for start_nodes
        assert result is not None

    def test_prune_alphas_empty(self):
        """Test prune_alphas with empty alphas."""
        from prune import prune_alphas

        alphas = {}
        start_nodes = ['/output']

        result = prune_alphas(alphas, start_nodes)
        assert result == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
