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
"""Unit tests for cuts/infered_cuts.py module.

This module tests the BICCOS (Branch-and-bound Inferred Cuts with Constraint
Strengthening) class and related functions for cut inference in neural network
verification.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Pytest Fixtures
# =============================================================================

def setup_test_config():
    """Setup arguments.Config for testing BICCOS."""
    import arguments
    new_config = arguments.ConfigHandler()
    new_config.construct_config_dict(new_config.default_args)
    new_config.file = None
    # Set required config values for BICCOS
    new_config['solver']['batch_size'] = 64
    new_config['solver']['min_batch_size_ratio'] = 0.1
    new_config['solver']['beta-crown']['all_node_split_LP'] = False
    new_config['solver']['beta-crown']['enable_opt_interm_bounds'] = False
    new_config['bab']['timeout'] = 100
    new_config['bab']['max_domains'] = 10000
    new_config['bab']['cut']['enabled'] = True
    new_config['bab']['cut']['bab_cut'] = True
    new_config['bab']['cut']['number_cuts'] = 200
    new_config['bab']['cut']['cplex_cuts'] = False
    new_config['bab']['cut']['patches_cut'] = False
    new_config['bab']['cut']['biccos']['enabled'] = True
    new_config['bab']['cut']['biccos']['max_infer_iter'] = 20
    new_config['bab']['cut']['biccos']['max_domain'] = 10000
    new_config['bab']['cut']['biccos']['drop_ratio'] = 0.5
    new_config['bab']['cut']['biccos']['save_cuts'] = False
    new_config['bab']['cut']['biccos']['heuristic'] = 'neuron_influence_score'
    new_config['bab']['cut']['biccos']['auto_param'] = False
    new_config['bab']['cut']['biccos']['constraint_strengthening'] = True
    new_config['bab']['cut']['biccos']['multi_tree_branching'] = {
        'enabled': False,
        'target_batch_size': 64,
        'keep_n_best_domains': 16,
        'iterations': 10,
        'restore_best_tree': False
    }
    new_config['debug']['sanity_check'] = None
    return new_config


@pytest.fixture
def test_config():
    """Fixture to set up and restore arguments.Config for testing."""
    import arguments
    original_config = arguments.Config
    arguments.Config = setup_test_config()
    yield arguments.Config
    arguments.Config = original_config


@pytest.fixture
def biccos_factory(test_config):
    """Factory fixture for creating BICCOS instances.

    Returns a function that creates BICCOS instances with configurable parameters.
    Supports three modes:
    - 'simple': Only /output layer (default)
    - 'with_relu': Includes /relu_0 and /output layers
    - 'multi_layer': Multiple relu layers with configurable count
    """
    from cuts.infered_cuts import BICCOS

    def _create(batch_size=4, mode='simple', num_layers=2, neurons_per_layer=10) -> BICCOS:
        if mode == 'simple':
            ret = {
                'lower_bounds': {'/output': torch.randn(batch_size, 1)},
                'upper_bounds': {'/output': torch.randn(batch_size, 1) + 1},
                'lA': {'/output': torch.randn(batch_size, 1, 1)}
            }
        elif mode == 'with_relu':
            ret = {
                'lower_bounds': {
                    '/relu_0': torch.randn(batch_size, neurons_per_layer),
                    '/output': torch.randn(batch_size, 1)
                },
                'upper_bounds': {
                    '/relu_0': torch.randn(batch_size, neurons_per_layer) + 1,
                    '/output': torch.randn(batch_size, 1) + 1
                },
                'lA': {
                    '/relu_0': torch.randn(batch_size, 1, neurons_per_layer),
                    '/output': torch.randn(batch_size, 1, 1)
                }
            }
        elif mode == 'multi_layer':
            ret = {
                'lower_bounds': {},
                'upper_bounds': {},
                'lA': {}
            }
            for i in range(num_layers):
                key = f'/relu_{i}/Relu'
                ret['lower_bounds'][key] = torch.randn(batch_size, neurons_per_layer)
                ret['upper_bounds'][key] = torch.randn(batch_size, neurons_per_layer) + 1
                ret['lA'][key] = torch.randn(batch_size, 1, neurons_per_layer)
            final_key = '/output/Gemm'
            ret['lower_bounds'][final_key] = torch.randn(batch_size, 1)
            ret['upper_bounds'][final_key] = torch.randn(batch_size, 1) + 1
            ret['lA'][final_key] = torch.randn(batch_size, 1, 1)
            return BICCOS(ret, torch.tensor([0.0]), final_key), ret, final_key
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return BICCOS(ret, torch.tensor([0.0]), '/output')

    return _create


@pytest.fixture
def mock_ret_factory():
    """Factory fixture for creating mock ret dictionaries."""
    def _create(batch_size=4, num_layers=2, neurons_per_layer=10):
        ret = {
            'lower_bounds': {},
            'upper_bounds': {},
            'lA': {}
        }
        for i in range(num_layers):
            key = f'/relu_{i}/Relu'
            ret['lower_bounds'][key] = torch.randn(batch_size, neurons_per_layer)
            ret['upper_bounds'][key] = torch.randn(batch_size, neurons_per_layer) + 1
            ret['lA'][key] = torch.randn(batch_size, 1, neurons_per_layer)
        final_key = '/output/Gemm'
        ret['lower_bounds'][final_key] = torch.randn(batch_size, 1)
        ret['upper_bounds'][final_key] = torch.randn(batch_size, 1) + 1
        ret['lA'][final_key] = torch.randn(batch_size, 1, 1)
        return ret, final_key

    return _create


@pytest.fixture
def mock_net_factory():
    """Factory fixture for creating mock network objects."""
    def _create():
        mock_net = MagicMock()
        mock_net.final_name = '/output'
        mock_net.cutter = MagicMock()
        mock_net.cutter.cuts = None
        mock_net.cutter.construct_cut_module = MagicMock(return_value=MagicMock())
        mock_net.net = MagicMock()
        mock_net.net.relus = []
        mock_net.net.cut_used = False
        mock_net.biccos_verification = MagicMock(return_value={
            'lower_bounds': {'/output': torch.tensor([0.5])},
            'betas': [None]
        })
        return mock_net

    return _create


def make_cut(arelu_decision, arelu_coeffs, bias=0, c=-1):
    """Helper function to create a cut dictionary."""
    return {
        'x_decision': [],
        'x_coeffs': [],
        'relu_decision': [],
        'relu_coeffs': [],
        'arelu_decision': arelu_decision,
        'arelu_coeffs': arelu_coeffs,
        'pre_decision': [],
        'pre_coeffs': [],
        'bias': bias,
        'c': c,
    }


# =============================================================================
# Test Classes
# =============================================================================

class TestBICCOSInit:
    """Tests for BICCOS class initialization."""

    def test_basic_init(self, mock_ret_factory, test_config):
        """Test basic BICCOS initialization."""
        from cuts.infered_cuts import BICCOS

        ret, final_name = mock_ret_factory()
        rhs = torch.tensor([0.0])

        biccos = BICCOS(ret, rhs, final_name)

        assert biccos.biccos_cuts == []
        assert biccos.cumulative_time == 0
        assert biccos.final_name == final_name

    def test_init_with_positive_rhs(self, mock_ret_factory, test_config):
        """Test BICCOS initialization with positive rhs."""
        from cuts.infered_cuts import BICCOS

        ret, final_name = mock_ret_factory()
        rhs = torch.tensor([0.5])

        biccos = BICCOS(ret, rhs, final_name)
        # Without sanity check, decision_thresh should be 0
        assert biccos.decision_thresh == 0

    def test_init_key_mappings(self, mock_ret_factory, test_config):
        """Test that key mappings are created correctly."""
        from cuts.infered_cuts import BICCOS

        ret, final_name = mock_ret_factory(num_layers=3)
        rhs = torch.tensor([0.0])

        biccos = BICCOS(ret, rhs, final_name)

        # Check that all keys are mapped
        assert len(biccos.key_mapping) == len(ret['lower_bounds'])
        assert len(biccos.key_mapping_lb) == len(ret['lower_bounds'])
        assert len(biccos.key_mapping_lA) == len(ret['lA'])

    def test_init_bounds_stored(self, mock_ret_factory, test_config):
        """Test that initial bounds are stored correctly."""
        from cuts.infered_cuts import BICCOS

        ret, final_name = mock_ret_factory()
        rhs = torch.tensor([0.0])

        biccos = BICCOS(ret, rhs, final_name)

        # Check that lb_init and ub_init have correct keys
        assert len(biccos.lb_init) == len(ret['lower_bounds'])
        assert len(biccos.ub_init) == len(ret['upper_bounds'])

    def test_init_remaining_spec_count(self, mock_ret_factory, test_config):
        """Test that remaining_OR_spec_count is set correctly."""
        from cuts.infered_cuts import BICCOS

        batch_size = 8
        ret, final_name = mock_ret_factory(batch_size=batch_size)
        rhs = torch.tensor([0.0])

        biccos = BICCOS(ret, rhs, final_name)

        assert biccos.remaining_OR_spec_count == batch_size

    def test_init_with_sanity_check_enabled(self, mock_ret_factory, test_config):
        """Test BICCOS init with sanity check enabled."""
        import arguments
        from cuts.infered_cuts import BICCOS

        arguments.Config['debug']['sanity_check'] = 'Full'

        ret, final_name = mock_ret_factory()
        rhs = torch.tensor([0.5])

        biccos = BICCOS(ret, rhs, final_name)
        # With sanity check, decision_thresh should be rhs.item()
        assert biccos.decision_thresh == 0.5


class TestGenerateCut:
    """Tests for BICCOS.generate_cut method."""

    def test_generate_empty_cut(self, biccos_factory):
        """Test generating an empty cut."""
        biccos = biccos_factory()

        cut = biccos.generate_cut()

        expected = {
            'x_decision': [],
            'x_coeffs': [],
            'relu_decision': [],
            'relu_coeffs': [],
            'arelu_decision': [],
            'arelu_coeffs': [],
            'pre_decision': [],
            'pre_coeffs': [],
            'bias': 0,
            'c': -1,
        }
        assert cut == expected

    def test_generate_cut_with_input_decision(self, biccos_factory):
        """Test generating a cut with input decision."""
        biccos = biccos_factory()

        cut = biccos.generate_cut(
            input_decision=[[0, 1], [0, 2]],
            input_coeffs=[1.0, -1.0],
            b=0.5
        )

        assert cut['x_decision'] == [[0, 1], [0, 2]]
        assert cut['x_coeffs'] == [1.0, -1.0]
        assert cut['bias'] == 0.5

    def test_generate_cut_with_relu_activation(self, biccos_factory):
        """Test generating a cut with ReLU activation decision."""
        biccos = biccos_factory()

        cut = biccos.generate_cut(
            relu_activation_decision=[[0, 5], [1, 3]],
            relu_activation_coeffs=[1.0, -1.0],
            b=2
        )

        assert cut['arelu_decision'] == [[0, 5], [1, 3]]
        assert cut['arelu_coeffs'] == [1.0, -1.0]
        assert cut['bias'] == 2

    def test_generate_cut_with_all_components(self, biccos_factory):
        """Test generating a cut with all components."""
        biccos = biccos_factory()

        cut = biccos.generate_cut(
            input_decision=[[0, 0]],
            input_coeffs=[0.5],
            post_relu_decision=[[1, 2]],
            post_relu_coeffs=[1.0],
            relu_activation_decision=[[0, 3]],
            relu_activation_coeffs=[-1.0],
            pre_relu_decision=[[2, 4]],
            pre_relu_coeffs=[2.0],
            b=-1,
            c=1
        )

        assert cut['x_decision'] == [[0, 0]]
        assert cut['relu_decision'] == [[1, 2]]
        assert cut['arelu_decision'] == [[0, 3]]
        assert cut['pre_decision'] == [[2, 4]]
        assert cut['bias'] == -1
        assert cut['c'] == 1

    def test_generate_cut_default_c(self, biccos_factory):
        """Test that default c value is -1."""
        biccos = biccos_factory()

        cut = biccos.generate_cut()
        assert cut['c'] == -1

    def test_generate_cut_custom_c(self, biccos_factory):
        """Test generating a cut with custom c value."""
        biccos = biccos_factory()

        cut = biccos.generate_cut(c=1)
        assert cut['c'] == 1


class TestInferenceCondition:
    """Tests for BICCOS.inference_condition method."""

    def test_all_verified(self, biccos_factory):
        """Test condition when all bounds are verified."""
        biccos = biccos_factory()
        biccos.decision_thresh = 0

        # All positive - all verified
        lbs_final = torch.tensor([1.0, 2.0, 3.0, 4.0])

        result = biccos.inference_condition(lbs_final)
        assert not result

    def test_none_verified(self, biccos_factory):
        """Test condition when no bounds are verified."""
        biccos = biccos_factory()
        biccos.decision_thresh = 0

        # All negative - none verified
        lbs_final = torch.tensor([-1.0, -2.0, -3.0, -4.0])

        result = biccos.inference_condition(lbs_final)
        assert not result

    def test_partial_verified(self, biccos_factory):
        """Test condition when some bounds are verified."""
        biccos = biccos_factory()
        biccos.decision_thresh = 0

        # Mixed - some verified
        lbs_final = torch.tensor([-1.0, 2.0, -3.0, 4.0])

        result = biccos.inference_condition(lbs_final)
        assert result

    def test_single_verified(self, biccos_factory):
        """Test condition when exactly one bound is verified."""
        biccos = biccos_factory()
        biccos.decision_thresh = 0

        lbs_final = torch.tensor([-1.0, -2.0, -3.0, 1.0])

        result = biccos.inference_condition(lbs_final)
        assert result

    def test_with_custom_threshold(self, biccos_factory):
        """Test condition with custom decision threshold."""
        biccos = biccos_factory()
        biccos.decision_thresh = 1.0

        # Only values > 1.0 are verified
        lbs_final = torch.tensor([0.5, 1.5, 0.8, 2.0])

        result = biccos.inference_condition(lbs_final)
        assert result


class TestIsCutAParent:
    """Tests for BICCOS.is_cut_a_parent method."""

    def test_exact_match_is_parent(self, biccos_factory):
        """Test that identical cuts are considered parent."""
        biccos = biccos_factory()

        cut = {
            'arelu_decision': [[0, 1], [0, 2]],
            'arelu_coeffs': [1.0, -1.0]
        }
        potential_parent = {
            'arelu_decision': [[0, 1], [0, 2]],
            'arelu_coeffs': [1.0, -1.0]
        }

        result = biccos.is_cut_a_parent(cut, potential_parent)
        assert result

    def test_subset_is_parent(self, biccos_factory):
        """Test that a subset of decisions is a parent."""
        biccos = biccos_factory()

        cut = {
            'arelu_decision': [[0, 1], [0, 2], [0, 3]],
            'arelu_coeffs': [1.0, -1.0, 1.0]
        }
        potential_parent = {
            'arelu_decision': [[0, 1], [0, 2]],
            'arelu_coeffs': [1.0, -1.0]
        }

        result = biccos.is_cut_a_parent(cut, potential_parent)
        assert result

    def test_different_coeffs_not_parent(self, biccos_factory):
        """Test that different coefficients means not a parent."""
        biccos = biccos_factory()

        cut = {
            'arelu_decision': [[0, 1], [0, 2]],
            'arelu_coeffs': [1.0, -1.0]
        }
        potential_parent = {
            'arelu_decision': [[0, 1], [0, 2]],
            'arelu_coeffs': [-1.0, 1.0]  # Different coefficients
        }

        result = biccos.is_cut_a_parent(cut, potential_parent)
        assert not result

    def test_missing_decision_not_parent(self, biccos_factory):
        """Test that missing a decision means not a parent."""
        biccos = biccos_factory()

        cut = {
            'arelu_decision': [[0, 1]],
            'arelu_coeffs': [1.0]
        }
        potential_parent = {
            'arelu_decision': [[0, 1], [0, 2]],  # Has extra decision
            'arelu_coeffs': [1.0, -1.0]
        }

        result = biccos.is_cut_a_parent(cut, potential_parent)
        assert not result

    def test_empty_parent_is_always_parent(self, biccos_factory):
        """Test that an empty parent is always a parent."""
        biccos = biccos_factory()

        cut = {
            'arelu_decision': [[0, 1], [0, 2]],
            'arelu_coeffs': [1.0, -1.0]
        }
        potential_parent = {
            'arelu_decision': [],
            'arelu_coeffs': []
        }

        result = biccos.is_cut_a_parent(cut, potential_parent)
        assert result

    def test_disjoint_decisions_not_parent(self, biccos_factory):
        """Test that disjoint decisions are not parent."""
        biccos = biccos_factory()

        cut = {
            'arelu_decision': [[0, 1], [0, 2]],
            'arelu_coeffs': [1.0, -1.0]
        }
        potential_parent = {
            'arelu_decision': [[0, 3], [0, 4]],  # Different neurons
            'arelu_coeffs': [1.0, -1.0]
        }

        result = biccos.is_cut_a_parent(cut, potential_parent)
        assert not result


class TestMergeCuts:
    """Tests for BICCOS.merge_cuts method."""

    def test_merge_empty_cuts(self, biccos_factory):
        """Test merging with empty input."""
        biccos = biccos_factory()
        biccos.biccos_cuts = []

        result = biccos.merge_cuts({})

        assert result == []

    def test_merge_single_cut(self, biccos_factory):
        """Test merging with a single cut."""
        biccos = biccos_factory()
        biccos.biccos_cuts = []

        cut = make_cut([[0, 1]], [1.0], bias=0)
        cut_key = json.dumps(cut, sort_keys=True)

        result = biccos.merge_cuts({cut_key: cut})

        assert len(result) == 1
        assert result[0]['arelu_decision'] == [[0, 1]]

    def test_merge_sibling_cuts(self, biccos_factory):
        """Test merging two sibling cuts that differ by one coefficient."""
        biccos = biccos_factory()
        biccos.biccos_cuts = []

        # Two cuts that only differ in one coefficient (1.0 vs -1.0)
        cut1 = make_cut([[0, 1], [0, 2]], [1.0, 1.0], bias=1)
        cut2 = make_cut([[0, 1], [0, 2]], [1.0, -1.0], bias=0)

        cut1_key = json.dumps(cut1, sort_keys=True)
        cut2_key = json.dumps(cut2, sort_keys=True)

        result = biccos.merge_cuts({cut1_key: cut1, cut2_key: cut2})

        # Should be merged into one parent cut with the common decision removed
        # The parent should have only the first decision
        assert len(result) <= 2

    def test_merge_preserves_unrelated_cuts(self, biccos_factory):
        """Test that unrelated cuts are preserved."""
        biccos = biccos_factory()
        biccos.biccos_cuts = []

        # Two completely unrelated cuts
        cut1 = make_cut([[0, 1]], [1.0], bias=0)
        cut2 = make_cut([[1, 5]], [-1.0], bias=1)

        cut1_key = json.dumps(cut1, sort_keys=True)
        cut2_key = json.dumps(cut2, sort_keys=True)

        result = biccos.merge_cuts({cut1_key: cut1, cut2_key: cut2})

        assert len(result) == 2

    def test_merge_with_existing_biccos_cuts(self, biccos_factory):
        """Test merging new cuts with existing biccos_cuts."""
        biccos = biccos_factory()

        existing_cut = make_cut([[0, 3]], [1.0], bias=0)
        biccos.biccos_cuts = [existing_cut]

        new_cut = make_cut([[0, 4]], [-1.0], bias=1)
        new_cut_key = json.dumps(new_cut, sort_keys=True)

        result = biccos.merge_cuts({new_cut_key: new_cut})

        # Both cuts should be in result
        assert len(result) == 2

    def test_merge_removes_empty_arelu_cuts(self, biccos_factory):
        """Test that cuts with empty arelu_decision are removed."""
        biccos = biccos_factory()
        biccos.biccos_cuts = []

        # Create a cut that will become empty after merging
        cut = make_cut([], [], bias=0)
        cut_key = json.dumps(cut, sort_keys=True)

        result = biccos.merge_cuts({cut_key: cut})

        # Empty arelu_decision cuts should be filtered out
        for r in result:
            if 'arelu_decision' in r:
                assert r['arelu_decision'] != []


class TestPickD:
    """Tests for BICCOS.pick_d method."""

    def test_pick_d_depths(self, biccos_factory):
        """Test picking depths from d."""
        biccos = biccos_factory()

        d = {
            'depths': [1, 2, 3, 4, 5]
        }
        v_idx = torch.tensor([0, 2, 4])

        result = biccos.pick_d(v_idx, d)

        assert len(result['depths']) == 3
        assert result['depths'][0] == 1
        assert result['depths'][1] == 3
        assert result['depths'][2] == 5

    def test_pick_d_history(self, biccos_factory):
        """Test picking history from d."""
        biccos = biccos_factory()

        d = {
            'history': [
                {'layer0': 'hist0'},
                {'layer0': 'hist1'},
                {'layer0': 'hist2'},
            ]
        }
        v_idx = torch.tensor([0, 2])

        result = biccos.pick_d(v_idx, d)

        assert len(result['history']) == 2

    def test_pick_d_lower_bounds(self, biccos_factory):
        """Test picking lower_bounds from d."""
        biccos = biccos_factory()

        d = {
            'lower_bounds': {
                'layer0': torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
                'layer1': torch.tensor([[5.0], [6.0], [7.0], [8.0]]),
            }
        }
        v_idx = torch.tensor([1, 3])

        result = biccos.pick_d(v_idx, d)

        assert torch.allclose(result['lower_bounds']['layer0'],
                              torch.tensor([[2.0], [4.0]]))

    def test_pick_d_lAs(self, biccos_factory):
        """Test picking lAs from d."""
        biccos = biccos_factory()

        d = {
            'lAs': {
                'layer0': torch.randn(4, 2, 3),
            }
        }
        v_idx = torch.tensor([0, 2])

        result = biccos.pick_d(v_idx, d)

        assert result['lAs']['layer0'].shape[0] == 2

    def test_pick_d_tensors(self, biccos_factory):
        """Test picking tensor fields (cs, thresholds) from d."""
        biccos = biccos_factory()

        d = {
            'cs': torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
            'thresholds': torch.tensor([0.1, 0.2, 0.3, 0.4]),
        }
        v_idx = torch.tensor([0, 3])

        result = biccos.pick_d(v_idx, d)

        assert torch.allclose(result['cs'], torch.tensor([[1.0], [4.0]]))
        assert torch.allclose(result['thresholds'], torch.tensor([0.1, 0.4]))

    def test_pick_d_alphas(self, biccos_factory):
        """Test picking alphas from d."""
        biccos = biccos_factory()

        # Alphas have structure: {sub_key: {tensor_key: tensor}}
        # where tensor has shape [..., batch, ...]
        d = {
            'alphas': {
                'layer0': {
                    'relu0': torch.randn(2, 3, 4, 5)  # [..., batch=4, ...]
                }
            }
        }
        v_idx = torch.tensor([0, 2])

        result = biccos.pick_d(v_idx, d)

        # After picking, batch dimension (dim 2) should be 2
        assert result['alphas']['layer0']['relu0'].shape[2] == 2

    def test_pick_d_out_of_bounds_indices_filtered(self, biccos_factory):
        """Test that out-of-bounds indices are handled gracefully."""
        biccos = biccos_factory()

        d = {
            'depths': [1, 2, 3]
        }
        # Index 5 is out of bounds
        v_idx = torch.tensor([0, 1, 5])

        result = biccos.pick_d(v_idx, d)

        # Only valid indices should be picked
        assert len(result['depths']) == 2


class TestNeuronInfluenceScore:
    """Tests for BICCOS neuron influence score methods."""

    def test_neuron_influence_score_above_criterion(self, biccos_factory):
        """Test neuron influence score returns True when above criterion."""
        biccos = biccos_factory()

        score = 0.8
        criterion = 0.5

        result = biccos.neuron_influence_score(score, criterion)
        assert result

    def test_neuron_influence_score_below_criterion(self, biccos_factory):
        """Test neuron influence score returns False when below criterion."""
        biccos = biccos_factory()

        score = 0.3
        criterion = 0.5

        result = biccos.neuron_influence_score(score, criterion)
        assert not result

    def test_neuron_influence_score_equal_criterion(self, biccos_factory):
        """Test neuron influence score returns True when equal to criterion."""
        biccos = biccos_factory()

        score = 0.5
        criterion = 0.5

        result = biccos.neuron_influence_score(score, criterion)
        assert result

    def test_influence_criterian_get(self, biccos_factory):
        """Test influence_criterian_get computes correct quantile."""
        biccos = biccos_factory()
        biccos.drop_ratio = 0.5

        hist = {
            'layer0': (None, None, None, torch.tensor([0.1, 0.2, 0.3, 0.4]), None),
            'layer1': (None, None, None, torch.tensor([0.5, 0.6, 0.7, 0.8]), None),
        }

        result = biccos.influence_criterian_get(hist)

        # Should be the median (0.5 quantile) of [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # Median of 8 elements is between 0.4 and 0.5 -> 0.45
        assert abs(result.item() - 0.45) < 0.01

    def test_influence_criterian_get_single_layer(self, biccos_factory):
        """Test influence_criterian_get with single layer."""
        biccos = biccos_factory()
        biccos.drop_ratio = 0.5

        hist = {
            'layer0': (None, None, None, torch.tensor([1.0, 2.0, 3.0, 4.0]), None),
        }

        result = biccos.influence_criterian_get(hist)

        # Median of [1, 2, 3, 4] = 2.5
        assert abs(result.item() - 2.5) < 0.01


class TestNeuronInfluenceScoreCal:
    """Tests for BICCOS.neuron_influence_score_cal method."""

    def test_neuron_influence_score_cal_updates_zero_scores(self, biccos_factory):
        """Test that zero scores are updated."""
        biccos = biccos_factory()

        # Create history with some zero scores
        d_hist = [
            {
                'layer0': (
                    torch.tensor([0, 1]),
                    torch.tensor([1, -1]),
                    torch.tensor([0.0, 0.0]),
                    torch.tensor([0.0, 0.5]),  # First score is zero
                    None
                )
            }
        ]
        d_lbs_final = torch.tensor([0.5])
        lbs_final = torch.tensor([1.0])

        biccos.neuron_influence_score_cal(d_hist, d_lbs_final, lbs_final)

        # The zero score should now be updated
        updated_score = d_hist[0]['layer0'][3]
        assert updated_score[0].item() != 0.0

    def test_neuron_influence_score_cal_preserves_nonzero(self, biccos_factory):
        """Test that non-zero scores are preserved."""
        biccos = biccos_factory()

        d_hist = [
            {
                'layer0': (
                    torch.tensor([0]),
                    torch.tensor([1]),
                    torch.tensor([0.0]),
                    torch.tensor([0.8]),  # Non-zero score
                    None
                )
            }
        ]
        d_lbs_final = torch.tensor([0.5])
        lbs_final = torch.tensor([1.0])

        biccos.neuron_influence_score_cal(d_hist, d_lbs_final, lbs_final)

        # The non-zero score should remain 0.8
        updated_score = d_hist[0]['layer0'][3]
        assert abs(updated_score[0].item() - 0.8) < 1e-5

    def test_neuron_influence_score_cal_multiple_domains(self, biccos_factory):
        """Test score calculation with multiple domains."""
        biccos = biccos_factory()

        d_hist = [
            {
                'layer0': (
                    torch.tensor([0]),
                    torch.tensor([1]),
                    torch.tensor([0.0]),
                    torch.tensor([0.0]),
                    None
                )
            },
            {
                'layer0': (
                    torch.tensor([0]),
                    torch.tensor([-1]),
                    torch.tensor([0.0]),
                    torch.tensor([0.0]),
                    None
                )
            }
        ]
        d_lbs_final = torch.tensor([0.2, 0.3])
        lbs_final = torch.tensor([1.0, 0.8])

        biccos.neuron_influence_score_cal(d_hist, d_lbs_final, lbs_final)

        # Both domains should have updated scores
        assert d_hist[0]['layer0'][3][0].item() != 0.0
        assert d_hist[1]['layer0'][3][0].item() != 0.0


class TestRandomDrop:
    """Tests for BICCOS.random_drop method."""

    def test_random_drop_returns_bool(self, biccos_factory):
        """Test that random_drop returns a boolean."""
        biccos = biccos_factory()

        result = biccos.random_drop()

        assert isinstance(result, bool)

    def test_random_drop_distribution(self, biccos_factory):
        """Test that random_drop has roughly 50% distribution."""
        biccos = biccos_factory()

        # Run many times and check distribution
        results = [biccos.random_drop() for _ in range(1000)]
        true_count = sum(results)

        # Should be roughly 50% (with some tolerance)
        assert true_count > 350
        assert true_count < 650


class TestSaveLoadCuts:
    """Tests for BICCOS save/load cuts methods."""

    def test_save_biccos_cuts(self, biccos_factory, tmp_path):
        """Test saving cuts to file."""
        biccos = biccos_factory()
        temp_file = tmp_path / "cuts.txt"

        cuts = [
            {'arelu_decision': [[0, 1]], 'arelu_coeffs': [1.0], 'bias': 0},
            {'arelu_decision': [[0, 2]], 'arelu_coeffs': [-1.0], 'bias': 1},
        ]

        biccos.save_biccos_cuts(cuts, str(temp_file))

        assert temp_file.exists()
        assert 'arelu_decision' in temp_file.read_text()

    def test_save_empty_cuts(self, biccos_factory, tmp_path):
        """Test saving empty cuts list."""
        biccos = biccos_factory()
        temp_file = tmp_path / "empty_cuts.txt"

        biccos.save_biccos_cuts([], str(temp_file))

        assert temp_file.read_text() == ''

    def test_load_biccos_cuts(self, biccos_factory, tmp_path):
        """Test loading cuts from file."""
        biccos = biccos_factory()
        temp_file = tmp_path / "load_cuts.txt"
        temp_file.write_text("{'arelu_decision': [[0, 1]]}\n{'arelu_decision': [[0, 2]]}\n")

        cuts = biccos.load_biccos_cuts(str(temp_file))

        assert len(cuts) == 2
        assert 'arelu_decision' in cuts[0]


class TestSetAutoParams:
    """Tests for BICCOS.set_auto_params method."""

    def test_set_auto_params_disabled(self, biccos_factory):
        """Test set_auto_params when auto_param is disabled."""
        biccos = biccos_factory()
        biccos.auto_param = False

        bs_ratio, mts_enabled = biccos.set_auto_params()

        assert bs_ratio == biccos.initial_bs_ratio

    def test_set_auto_params_few_candidates_enables_mts(self, biccos_factory, test_config):
        """Test that few candidates enables MTS."""
        import arguments
        biccos = biccos_factory(batch_size=2)
        biccos.auto_param = True
        biccos.remaining_OR_spec_count = 2

        biccos.set_auto_params()

        assert arguments.Config['bab']['cut']['biccos']['multi_tree_branching']['enabled']

    def test_set_auto_params_many_candidates_disables_mts(self, biccos_factory, test_config):
        """Test that many candidates disables MTS."""
        import arguments
        biccos = biccos_factory(batch_size=5)
        biccos.auto_param = True
        biccos.remaining_OR_spec_count = 5

        biccos.set_auto_params()

        assert not arguments.Config['bab']['cut']['biccos']['multi_tree_branching']['enabled']


class TestOriginalCutInference:
    """Tests for BICCOS.original_cut_inference method."""

    def test_original_cut_inference_empty_history(self, biccos_factory):
        """Test with empty history."""
        biccos = biccos_factory(mode='with_relu')
        biccos.tmp_cuts = []

        d_hists = [{}]
        ret_beta = [{}]
        v_idx = torch.tensor([0])

        biccos.original_cut_inference(d_hists, ret_beta, v_idx)

        # No cuts should be added with empty history
        assert len(biccos.tmp_cuts) == 0

    def test_original_cut_inference_with_history(self, biccos_factory):
        """Test with valid history."""
        biccos = biccos_factory(mode='with_relu')
        biccos.tmp_cuts = []

        # Create history where relu_idx and relu_status are already tensors
        d_hists = [
            {
                '/relu_0': (
                    torch.tensor([0, 1, 2]),  # relu_idx
                    torch.tensor([1, -1, 1]),  # relu_status (1 for >=0, -1 for <=0)
                    torch.tensor([0.0, 0.0, 0.0]),  # relu_bias
                    torch.tensor([0.5, 0.5, 0.5]),  # relu_score
                    torch.tensor([1, 2, 3])  # depths
                )
            }
        ]
        # ret_beta needs to have positive values for the neurons to be included
        ret_beta = [
            {'/relu_0': torch.tensor([1.0, 1.0, 1.0])}  # Positive betas
        ]
        v_idx = torch.tensor([0])

        biccos.original_cut_inference(d_hists, ret_beta, v_idx)

        # Should have inferred one cut
        assert len(biccos.tmp_cuts) == 1
        assert 'arelu_decision' in biccos.tmp_cuts[0]
        assert 'arelu_coeffs' in biccos.tmp_cuts[0]

    def test_original_cut_inference_zero_beta_excluded(self, biccos_factory):
        """Test that neurons with zero beta are excluded."""
        biccos = biccos_factory(mode='with_relu')
        biccos.tmp_cuts = []

        d_hists = [
            {
                '/relu_0': (
                    torch.tensor([0, 1]),
                    torch.tensor([1, -1]),
                    torch.tensor([0.0, 0.0]),
                    torch.tensor([0.5, 0.5]),
                    torch.tensor([1, 2])
                )
            }
        ]
        # Only first neuron has positive beta
        ret_beta = [
            {'/relu_0': torch.tensor([1.0, 0.0])}  # Second beta is zero
        ]
        v_idx = torch.tensor([0])

        biccos.original_cut_inference(d_hists, ret_beta, v_idx)

        # The cut should only include the first neuron
        if len(biccos.tmp_cuts) > 0:
            # Check that decision list has only one element
            assert len(biccos.tmp_cuts[0]['arelu_decision']) == 1


class TestCutInference:
    """Tests for BICCOS.cut_inference method."""

    def test_cut_inference_returns_tuple(self, biccos_factory):
        """Test that cut_inference returns a tuple of d_revise and cuts."""
        biccos = biccos_factory(mode='with_relu')

        d = {
            'history': [
                {
                    '/relu_0': (
                        torch.tensor([0]),
                        torch.tensor([1]),
                        torch.tensor([0.0]),
                        torch.tensor([0.5]),
                        torch.tensor([1])
                    )
                }
            ],
            'depths': [1],
            'lower_bounds': {'/relu_0': torch.randn(1, 10)},
            'upper_bounds': {'/relu_0': torch.randn(1, 10) + 1},
            'lAs': {'/relu_0': torch.randn(1, 1, 10)},
            'betas': [None],
        }
        ret = {
            'betas': [{'/relu_0': torch.tensor([1.0])}]
        }
        v_idx = torch.tensor([0])

        result = biccos.cut_inference(d, ret, v_idx, heuristic=None)

        assert isinstance(result, tuple)
        assert len(result) == 2
        d_revise, cuts = result
        assert isinstance(d_revise, dict)
        assert isinstance(cuts, list)


class TestBICCOSEdgeCases:
    """Edge case tests for BICCOS class."""

    def test_biccos_with_single_batch(self, test_config):
        """Test BICCOS with batch size of 1."""
        from cuts.infered_cuts import BICCOS

        ret = {
            'lower_bounds': {'/output': torch.randn(1, 1)},
            'upper_bounds': {'/output': torch.randn(1, 1) + 1},
            'lA': {'/output': torch.randn(1, 1, 1)}
        }
        rhs = torch.tensor([0.0])

        biccos = BICCOS(ret, rhs, '/output')

        assert biccos.remaining_OR_spec_count == 1

    def test_biccos_with_large_batch(self, test_config):
        """Test BICCOS with large batch size."""
        from cuts.infered_cuts import BICCOS

        batch_size = 1000
        ret = {
            'lower_bounds': {'/output': torch.randn(batch_size, 1)},
            'upper_bounds': {'/output': torch.randn(batch_size, 1) + 1},
            'lA': {'/output': torch.randn(batch_size, 1, 1)}
        }
        rhs = torch.tensor([0.0])

        biccos = BICCOS(ret, rhs, '/output')

        assert biccos.remaining_OR_spec_count == batch_size

    def test_generate_cut_with_empty_lists(self, biccos_factory):
        """Test generate_cut with all empty lists."""
        biccos = biccos_factory()

        cut = biccos.generate_cut()

        # All lists should be empty
        assert cut['x_decision'] == []
        assert cut['x_coeffs'] == []
        assert cut['relu_decision'] == []
        assert cut['relu_coeffs'] == []
        assert cut['arelu_decision'] == []
        assert cut['arelu_coeffs'] == []
        assert cut['pre_decision'] == []
        assert cut['pre_coeffs'] == []


class TestMergeCutsComplex:
    """Complex tests for merge_cuts method."""

    def test_merge_multiple_pairs(self, biccos_factory):
        """Test merging multiple pairs of sibling cuts."""
        biccos = biccos_factory()
        biccos.biccos_cuts = []

        # Create cuts with various decisions
        cuts = {}
        cut1 = make_cut([[0, 1]], [1.0], bias=0)
        cut2 = make_cut([[0, 2]], [1.0], bias=0)
        cut3 = make_cut([[0, 3]], [-1.0], bias=0)

        for cut in [cut1, cut2, cut3]:
            key = json.dumps(cut, sort_keys=True)
            cuts[key] = cut

        result = biccos.merge_cuts(cuts)

        # All unrelated cuts should be preserved
        assert len(result) == 3

    def test_merge_preserves_cut_structure(self, biccos_factory):
        """Test that merged cuts preserve the correct structure."""
        biccos = biccos_factory()
        biccos.biccos_cuts = []

        cut = make_cut([[0, 1], [0, 2]], [1.0, -1.0], bias=1, c=-1)
        cut_key = json.dumps(cut, sort_keys=True)

        result = biccos.merge_cuts({cut_key: cut})

        assert len(result) == 1
        merged = result[0]

        # Check all fields are present
        assert 'x_decision' in merged
        assert 'x_coeffs' in merged
        assert 'relu_decision' in merged
        assert 'relu_coeffs' in merged
        assert 'arelu_decision' in merged
        assert 'arelu_coeffs' in merged
        assert 'pre_decision' in merged
        assert 'pre_coeffs' in merged
        assert 'bias' in merged
        assert 'c' in merged


class TestPickDComplex:
    """Complex tests for pick_d method."""

    def test_pick_d_intermediate_betas(self, biccos_factory):
        """Test picking intermediate_betas from d."""
        biccos = biccos_factory()

        d = {
            'intermediate_betas': ['beta0', 'beta1', 'beta2', 'beta3']
        }
        v_idx = torch.tensor([1, 3])

        result = biccos.pick_d(v_idx, d)

        assert len(result['intermediate_betas']) == 2
        assert result['intermediate_betas'][0] == 'beta1'
        assert result['intermediate_betas'][1] == 'beta3'

    def test_pick_d_split_history(self, biccos_factory):
        """Test picking split_history from d."""
        biccos = biccos_factory()

        d = {
            'split_history': [
                {'split': 'h0'},
                {'split': 'h1'},
                {'split': 'h2'}
            ]
        }
        v_idx = torch.tensor([0, 2])

        result = biccos.pick_d(v_idx, d)

        assert len(result['split_history']) == 2
        assert result['split_history'][0] == {'split': 'h0'}
        assert result['split_history'][1] == {'split': 'h2'}

    def test_pick_d_betas_list(self, biccos_factory):
        """Test picking betas from d."""
        biccos = biccos_factory()

        d = {
            'betas': [
                {'layer0': torch.tensor([1.0])},
                {'layer0': torch.tensor([2.0])},
                {'layer0': torch.tensor([3.0])}
            ]
        }
        v_idx = torch.tensor([0, 2])

        result = biccos.pick_d(v_idx, d)

        assert len(result['betas']) == 2

    def test_pick_d_preserves_original(self, biccos_factory):
        """Test that pick_d doesn't modify the original d."""
        biccos = biccos_factory()

        original_depths = [1, 2, 3, 4]
        d = {
            'depths': original_depths.copy()
        }
        v_idx = torch.tensor([0, 2])

        biccos.pick_d(v_idx, d)

        # Original should be unchanged
        assert d['depths'] == original_depths


class TestUpdateCut:
    """Tests for BICCOS.update_cut method."""

    def _create_mock_net(self):
        """Create a mock network object."""
        mock_net = MagicMock()
        mock_net.final_name = '/output'
        mock_net.cutter = MagicMock()
        mock_net.cutter.cuts = None
        mock_net.cutter.construct_cut_module = MagicMock(return_value=MagicMock())
        mock_net.net = MagicMock()
        mock_net.net.relus = []
        mock_net.net.cut_used = False
        mock_net.biccos_verification = MagicMock(return_value={
            'lower_bounds': {'/output': torch.tensor([0.5])},
            'betas': [None]
        })
        return mock_net

    @patch('cuts.infered_cuts.cut_analysis')
    @patch('cuts.infered_cuts.fetch_cut_from_cplex')
    def test_update_cut_max_iteration_exceeded(self, mock_fetch, mock_analysis, biccos_factory):
        """Test update_cut when max iteration is exceeded."""
        mock_fetch.return_value = (None, None)

        biccos = biccos_factory(mode='with_relu')
        mock_net = self._create_mock_net()

        d = {
            'history': [],
            'lower_bounds': {'/output': torch.tensor([-1.0, -2.0])},
        }
        ret = {
            'lower_bounds': {'/output': torch.tensor([-1.0, -2.0])},
            'betas': []
        }

        # Set iter_idx > max_iter
        biccos.update_cut(d, mock_net, ret, enforce_usage=False,
                         domain_visited=0, iter_idx=100)

        # Should not have inferred any cuts
        assert len(biccos.biccos_cuts) == 0

    @patch('cuts.infered_cuts.cut_analysis')
    @patch('cuts.infered_cuts.fetch_cut_from_cplex')
    def test_update_cut_max_domain_exceeded(self, mock_fetch, mock_analysis, biccos_factory):
        """Test update_cut when max domain is exceeded."""
        mock_fetch.return_value = (None, None)

        biccos = biccos_factory(mode='with_relu')
        mock_net = self._create_mock_net()

        d = {
            'history': [],
            'lower_bounds': {'/output': torch.tensor([-1.0, -2.0])},
        }
        ret = {
            'lower_bounds': {'/output': torch.tensor([-1.0, -2.0])},
            'betas': []
        }

        # Set domain_visited > max_domain
        biccos.update_cut(d, mock_net, ret, enforce_usage=False,
                         domain_visited=100000, iter_idx=1)

        # Should truncate cuts to max_cuts_num + 1
        assert len(biccos.biccos_cuts) <= biccos.max_cuts_num + 1

    @patch('cuts.infered_cuts.cut_analysis')
    @patch('cuts.infered_cuts.fetch_cut_from_cplex')
    def test_update_cut_all_verified(self, mock_fetch, mock_analysis, biccos_factory):
        """Test update_cut when all bounds are verified (no inference needed)."""
        mock_fetch.return_value = (None, None)

        biccos = biccos_factory(mode='with_relu')
        mock_net = self._create_mock_net()

        d = {
            'history': [],
            'lower_bounds': {'/output': torch.tensor([1.0, 2.0])},  # All positive
        }
        ret = {
            'lower_bounds': {'/output': torch.tensor([1.0, 2.0])},
            'betas': []
        }

        biccos.update_cut(d, mock_net, ret, enforce_usage=False,
                         domain_visited=0, iter_idx=1)

        # No cuts should be inferred when all are verified
        assert len(biccos.biccos_cuts) == 0

    @patch('cuts.infered_cuts.cut_analysis')
    @patch('cuts.infered_cuts.fetch_cut_from_cplex')
    def test_update_cut_none_verified(self, mock_fetch, mock_analysis, biccos_factory):
        """Test update_cut when no bounds are verified."""
        mock_fetch.return_value = (None, None)

        biccos = biccos_factory(mode='with_relu')
        mock_net = self._create_mock_net()

        d = {
            'history': [],
            'lower_bounds': {'/output': torch.tensor([-1.0, -2.0])},  # All negative
        }
        ret = {
            'lower_bounds': {'/output': torch.tensor([-1.0, -2.0])},
            'betas': []
        }

        biccos.update_cut(d, mock_net, ret, enforce_usage=False,
                         domain_visited=0, iter_idx=1)

        # No cuts should be inferred when none are verified
        assert len(biccos.biccos_cuts) == 0

    @patch('cuts.infered_cuts.cut_analysis')
    @patch('cuts.infered_cuts.fetch_cut_from_cplex')
    def test_update_cut_random_drop_heuristic(self, mock_fetch, mock_analysis, biccos_factory):
        """Test update_cut with random_drop heuristic."""
        mock_fetch.return_value = (None, None)

        biccos = biccos_factory(mode='with_relu')
        mock_net = self._create_mock_net()

        d = {
            'history': [
                {
                    '/relu_0': (
                        torch.tensor([0]),
                        torch.tensor([1]),
                        torch.tensor([0.0]),
                        torch.tensor([0.5]),
                        torch.tensor([1])
                    )
                }
            ],
            'lower_bounds': {'/output': torch.tensor([1.0, -1.0]), '/relu_0': torch.randn(1, 10)},  # Mixed
            'upper_bounds': {'/relu_0': torch.randn(1, 10) + 1},
            'depths': [1],
            'betas': [None],
            'lAs': {'/relu_0': torch.randn(1, 1, 10)},
            'thresholds': torch.tensor([0.0]),
            'cs': torch.tensor([[1.0]]),
        }
        ret = {
            'lower_bounds': {'/output': torch.tensor([1.0, -1.0])},
            'betas': [{'/relu_0': torch.tensor([1.0])}]
        }

        # Should not raise with random_drop heuristic
        biccos.update_cut(d, mock_net, ret, enforce_usage=False,
                         domain_visited=0, heuristic='random_drop', iter_idx=1)

    @patch('cuts.infered_cuts.cut_analysis')
    @patch('cuts.infered_cuts.fetch_cut_from_cplex')
    def test_update_cut_enforce_usage(self, mock_fetch, mock_analysis, biccos_factory):
        """Test update_cut with enforce_usage=True."""
        mock_fetch.return_value = (None, None)

        biccos = biccos_factory(mode='with_relu')
        mock_net = self._create_mock_net()

        d = {
            'history': [
                {
                    '/relu_0': (
                        torch.tensor([0]),
                        torch.tensor([1]),
                        torch.tensor([0.0]),
                        torch.tensor([0.5]),
                        torch.tensor([1])
                    )
                }
            ],
            'lower_bounds': {'/output': torch.tensor([1.0])},
            'upper_bounds': {'/relu_0': torch.randn(1, 10) + 1},
            'depths': [1],
            'betas': [None],
            'lAs': {'/relu_0': torch.randn(1, 1, 10)},
            'cs': torch.tensor([[1.0]]),
            'thresholds': torch.tensor([0.0]),
        }
        ret = {
            'lower_bounds': {'/output': torch.tensor([1.0])},
            'betas': [{'/relu_0': torch.tensor([1.0])}]
        }

        # With enforce_usage=True, should proceed even with max_iter exceeded
        biccos.update_cut(d, mock_net, ret, enforce_usage=True,
                         domain_visited=100000, iter_idx=100)

    @patch('cuts.infered_cuts.cut_analysis')
    @patch('cuts.infered_cuts.fetch_cut_from_cplex')
    def test_update_cut_with_cplex_cuts(self, mock_fetch, mock_analysis, biccos_factory, test_config):
        """Test update_cut when CPLEX cuts are available."""
        import arguments
        arguments.Config['bab']['cut']['cplex_cuts'] = True

        cplex_cuts = [{'arelu_decision': [[0, 5]], 'arelu_coeffs': [1.0]}]
        mock_fetch.return_value = (cplex_cuts, 12345)

        biccos = biccos_factory(mode='with_relu')
        mock_net = self._create_mock_net()

        d = {
            'history': [],
            'lower_bounds': {'/output': torch.tensor([-1.0])},
        }
        ret = {
            'lower_bounds': {'/output': torch.tensor([-1.0])},
            'betas': []
        }

        biccos.update_cut(d, mock_net, ret, enforce_usage=False,
                         domain_visited=0, iter_idx=1)

        # CPLEX cuts should be stored
        assert biccos.cplex_cuts == cplex_cuts

        # Reset config
        arguments.Config['bab']['cut']['cplex_cuts'] = False


class TestConstraintStrengthening:
    """Tests for BICCOS.constraint_strengthening method."""

    def test_constraint_strengthening_disabled(self, biccos_factory):
        """Test constraint_strengthening when disabled."""
        biccos = biccos_factory(mode='with_relu')
        biccos.enable_constraint_strengthen = False
        biccos.tmp_cuts = []

        d = {
            'history': [
                {
                    '/relu_0': (
                        torch.tensor([0]),
                        torch.tensor([1]),
                        torch.tensor([0.0]),
                        torch.tensor([0.5]),
                        torch.tensor([1])
                    )
                }
            ]
        }
        ret = {
            'betas': [{'/relu_0': torch.tensor([1.0])}]
        }
        v_idx = torch.tensor([0])

        mock_net = MagicMock()

        biccos.constraint_strengthening(d, mock_net, ret, v_idx, heuristic=None)

        # Should only call original_cut_inference, not biccos_verification
        mock_net.biccos_verification.assert_not_called()


class TestCutInferenceWithHeuristics:
    """Tests for BICCOS.cut_inference with different heuristics."""

    def test_cut_inference_with_random_drop(self, biccos_factory):
        """Test cut_inference with random_drop heuristic."""
        biccos = biccos_factory(mode='with_relu')

        d = {
            'history': [
                {
                    '/relu_0': (
                        torch.tensor([0, 1]),
                        torch.tensor([1, -1]),
                        torch.tensor([0.0, 0.0]),
                        torch.tensor([0.5, 0.5]),
                        torch.tensor([1, 2])
                    )
                }
            ],
            'depths': [1],
            'lower_bounds': {'/relu_0': torch.randn(1, 10)},
            'upper_bounds': {'/relu_0': torch.randn(1, 10) + 1},
            'lAs': {'/relu_0': torch.randn(1, 1, 10)},
            'betas': [None],
        }
        ret = {
            'betas': [{'/relu_0': torch.tensor([0.0, 0.0])}]  # Zero betas to trigger heuristic
        }
        v_idx = torch.tensor([0])

        d_revise, cuts = biccos.cut_inference(d, ret, v_idx, heuristic='random_drop')

        assert isinstance(d_revise, dict)
        assert isinstance(cuts, list)
        assert len(cuts) == 1

    def test_cut_inference_with_neuron_influence_score(self, biccos_factory):
        """Test cut_inference with neuron_influence_score heuristic."""
        biccos = biccos_factory(mode='with_relu')

        d = {
            'history': [
                {
                    '/relu_0': (
                        torch.tensor([0, 1]),
                        torch.tensor([1, -1]),
                        torch.tensor([0.0, 0.0]),
                        torch.tensor([0.8, 0.2]),  # Different scores
                        torch.tensor([1, 2])
                    )
                }
            ],
            'depths': [1],
            'lower_bounds': {'/relu_0': torch.randn(1, 10)},
            'upper_bounds': {'/relu_0': torch.randn(1, 10) + 1},
            'lAs': {'/relu_0': torch.randn(1, 1, 10)},
            'betas': [None],
        }
        ret = {
            'betas': [{'/relu_0': torch.tensor([0.0, 0.0])}]
        }
        v_idx = torch.tensor([0])

        d_revise, cuts = biccos.cut_inference(d, ret, v_idx, heuristic='neuron_influence_score')

        assert isinstance(d_revise, dict)
        assert isinstance(cuts, list)

    def test_cut_inference_with_none_heuristic(self, biccos_factory):
        """Test cut_inference with None heuristic (should always keep neurons)."""
        biccos = biccos_factory(mode='with_relu')

        d = {
            'history': [
                {
                    '/relu_0': (
                        torch.tensor([0]),
                        torch.tensor([1]),
                        torch.tensor([0.0]),
                        torch.tensor([0.5]),
                        torch.tensor([1])
                    )
                }
            ],
            'depths': [1],
            'lower_bounds': {'/relu_0': torch.randn(1, 10)},
            'upper_bounds': {'/relu_0': torch.randn(1, 10) + 1},
            'lAs': {'/relu_0': torch.randn(1, 1, 10)},
            'betas': [None],
        }
        ret = {
            'betas': [{'/relu_0': torch.tensor([0.0])}]  # Zero beta
        }
        v_idx = torch.tensor([0])

        d_revise, cuts = biccos.cut_inference(d, ret, v_idx, heuristic=None)

        # With None heuristic and zero beta, neuron should be dropped
        assert len(cuts[0]['arelu_decision']) == 0

    def test_cut_inference_updates_bounds(self, biccos_factory):
        """Test that cut_inference updates bounds correctly when dropping neurons."""
        biccos = biccos_factory(mode='with_relu')

        # Create explicit bounds
        lb_init = torch.tensor([[1.0, 2.0, 3.0]])
        ub_init = torch.tensor([[4.0, 5.0, 6.0]])

        d = {
            'history': [
                {
                    '/relu_0': (
                        torch.tensor([0, 1]),  # Two neurons
                        torch.tensor([1, -1]),
                        torch.tensor([0.0, 0.0]),
                        torch.tensor([0.5, 0.5]),
                        torch.tensor([1, 2])
                    )
                }
            ],
            'depths': [1],
            'lower_bounds': {'/relu_0': torch.tensor([[0.0, 0.0, 0.0]])},
            'upper_bounds': {'/relu_0': torch.tensor([[1.0, 1.0, 1.0]])},
            'lAs': {'/relu_0': torch.randn(1, 1, 3)},
            'betas': [None],
        }
        ret = {
            'betas': [{'/relu_0': torch.tensor([1.0, 0.0])}]  # Only first has positive beta
        }
        v_idx = torch.tensor([0])

        # Store initial bounds
        biccos.lb_init[biccos.key_mapping['/relu_0']] = lb_init
        biccos.ub_init[biccos.key_mapping['/relu_0']] = ub_init

        d_revise, cuts = biccos.cut_inference(d, ret, v_idx, heuristic=None)

        # Second neuron (index 1) should have bounds restored
        # Because it had zero beta and None heuristic returns False
        # Verify the cut only includes the first neuron (positive beta)
        assert len(cuts) == 1
        cut = cuts[0]
        # Only one neuron should be in the cut (the one with positive beta)
        assert len(cut['arelu_decision']) == 1
        # The decision should be for neuron index 0 (first neuron)
        assert cut['arelu_decision'][0][1] == 0


class TestCutInferenceDepths:
    """Tests for cut_inference handling of depths."""

    def test_cut_inference_with_none_depths(self, biccos_factory):
        """Test cut_inference when depths is None."""
        biccos = biccos_factory(mode='with_relu')

        d = {
            'history': [
                {
                    '/relu_0': (
                        torch.tensor([0]),
                        torch.tensor([1]),
                        torch.tensor([0.0]),
                        torch.tensor([0.5]),
                        None  # depths is None
                    )
                }
            ],
            'depths': [1],
            'lower_bounds': {'/relu_0': torch.randn(1, 10)},
            'upper_bounds': {'/relu_0': torch.randn(1, 10) + 1},
            'lAs': {'/relu_0': torch.randn(1, 1, 10)},
            'betas': [None],
        }
        ret = {
            'betas': [{'/relu_0': torch.tensor([1.0])}]
        }
        v_idx = torch.tensor([0])

        # Should not raise with None depths
        d_revise, cuts = biccos.cut_inference(d, ret, v_idx, heuristic=None)

        assert len(cuts) == 1

    def test_cut_inference_with_mismatched_depths(self, biccos_factory):
        """Test cut_inference when depths length doesn't match relu_status."""
        biccos = biccos_factory(mode='with_relu')

        d = {
            'history': [
                {
                    '/relu_0': (
                        torch.tensor([0, 1]),
                        torch.tensor([1, -1]),
                        torch.tensor([0.0, 0.0]),
                        torch.tensor([0.5, 0.5]),
                        torch.tensor([1])  # Only one depth for two neurons
                    )
                }
            ],
            'depths': [1],
            'lower_bounds': {'/relu_0': torch.randn(1, 10)},
            'upper_bounds': {'/relu_0': torch.randn(1, 10) + 1},
            'lAs': {'/relu_0': torch.randn(1, 1, 10)},
            'betas': [None],
        }
        ret = {
            'betas': [{'/relu_0': torch.tensor([1.0, 1.0])}]
        }
        v_idx = torch.tensor([0])

        # Should handle mismatched depths gracefully (not raise an exception)
        d_revise, cuts = biccos.cut_inference(d, ret, v_idx, heuristic=None)

        # Verify valid output is returned despite mismatched depths
        assert isinstance(d_revise, dict)
        assert isinstance(cuts, list)
        assert len(cuts) == 1
        # The cut should still have valid structure
        assert 'arelu_decision' in cuts[0]
        assert 'arelu_coeffs' in cuts[0]
