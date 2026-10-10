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
"""Unit tests for input_split/bounding.py"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Pytest Fixtures
# =============================================================================

def setup_module():
    """Setup arguments.Config for all tests."""
    import arguments
    global original_config
    original_config = arguments.Config
    new_config = arguments.ConfigHandler()
    new_config.construct_config_dict(new_config.default_args)
    new_config.file = None
    arguments.Config = new_config


def teardown_module():
    """Restore original config."""
    import arguments
    arguments.Config = original_config


@pytest.fixture
def mock_lirpa_net_factory():
    """Factory fixture for creating mock LiRPANet objects.

    Returns a function that creates mock LiRPANet instances with configurable parameters.
    Supports two modes:
    - 'basic': Without infeasible_bounds_constraints attribute
    - 'with_constraints': With infeasible_bounds_constraints = None (default)
    """
    def _create(mode='with_constraints'):
        mock_net = MagicMock()
        mock_net.net = MagicMock()
        mock_net.net.output_name = ['output']
        mock_net.net.input_name = ['input']
        if mode == 'with_constraints':
            mock_net.net.infeasible_bounds_constraints = None
        # 'basic' mode doesn't set infeasible_bounds_constraints
        return mock_net

    return _create


@pytest.fixture
def stats():
    """Fixture for creating a Stats object."""
    from utils import Stats
    return Stats()


# =============================================================================
# Test Classes
# =============================================================================

class TestGetLowerBoundNaiveAlphaForward:
    """Tests for alpha-forward method rejection."""

    def test_alpha_forward_raises_error(self, mock_lirpa_net_factory, stats):
        """Test that alpha-forward method raises ValueError."""
        from input_split.bounding import get_lower_bound_naive

        mock_self = mock_lirpa_net_factory(mode='basic')

        batch_size = 2
        input_dim = 4
        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        thresholds = torch.zeros(batch_size, 2)

        with pytest.raises(ValueError) as context:
            get_lower_bound_naive(
                mock_self,
                dm_lb=None,
                dm_l=dm_l,
                dm_u=dm_u,
                alphas={},
                bounding_method='alpha-forward',
                C=torch.randn(batch_size, 2, 5),
                stop_criterion=lambda t: lambda x: x > t,
                thresholds=thresholds,
                stats=stats
            )

        assert 'alpha-forward' in str(context.value)


class TestGetLowerBoundNaiveIBP:
    """Tests for IBP bounding method."""

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_ibp_method_returns_correct_shape(self, mock_config, mock_lirpa_net_factory, stats):
        """Test that IBP method returns correct output shape."""
        from input_split.bounding import get_lower_bound_naive

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            },
            'branching': {'input_split': {'ibp_enhancement': False}}
        }
        mock_config['solver'] = {
            'alpha-crown': {
                'input_split_alpha_iteration': 20,
                'input_split_lr_alpha': 0.01,
                'alpha_dtype': 'float32'
            }
        }

        mock_self = mock_lirpa_net_factory()

        batch_size = 3
        input_dim = 4
        num_specs = 2
        output_dim = 5

        lb_result = torch.randn(batch_size, num_specs)
        mock_self.net.compute_bounds = MagicMock(return_value=(lb_result,))

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        lb, ret_alphas, lA, lbias = get_lower_bound_naive(
            mock_self,
            dm_lb=None,
            dm_l=dm_l,
            dm_u=dm_u,
            alphas={},
            bounding_method='ibp',
            C=C,
            stop_criterion=lambda t: lambda x: x > t,
            thresholds=thresholds,
            stats=stats
        )

        assert lb.shape == (batch_size, num_specs)
        assert lA is None
        assert lbias is None

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_ibp_method_calls_compute_bounds(self, mock_config, mock_lirpa_net_factory, stats):
        """Test that IBP method calls compute_bounds with correct parameters."""
        from input_split.bounding import get_lower_bound_naive

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            },
            'branching': {'input_split': {'ibp_enhancement': False}}
        }
        mock_config['solver'] = {
            'alpha-crown': {
                'input_split_alpha_iteration': 20,
                'input_split_lr_alpha': 0.01,
                'alpha_dtype': 'float32'
            }
        }

        mock_self = mock_lirpa_net_factory()

        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4

        lb_result = torch.randn(batch_size, num_specs)
        mock_self.net.compute_bounds = MagicMock(return_value=(lb_result,))

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        lb, ret_alphas, lA, lbias = get_lower_bound_naive(
            mock_self,
            dm_lb=None,
            dm_l=dm_l,
            dm_u=dm_u,
            alphas={},
            bounding_method='ibp',
            C=C,
            stop_criterion=lambda t: lambda x: x > t,
            thresholds=thresholds,
            stats=stats
        )

        # Verify compute_bounds was called with correct method
        mock_self.net.compute_bounds.assert_called()
        call_kwargs = mock_self.net.compute_bounds.call_args[1]
        assert call_kwargs['method'] == 'ibp'

        # Verify return values
        assert torch.equal(lb, lb_result)
        assert lA is None  # return_A=False by default
        assert lbias is None  # return_b=False by default


class TestGetLowerBoundNaiveAlphaCrown:
    """Tests for alpha-crown bounding method."""

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_alpha_crown_method(self, mock_config, mock_lirpa_net_factory, stats):
        """Test that alpha-crown method uses CROWN-Optimized."""
        from input_split.bounding import get_lower_bound_naive

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            },
            'branching': {'input_split': {'ibp_enhancement': False}}
        }
        mock_config['solver'] = {
            'reproduce_alpha_bounds': False,
            'alpha-crown': {
                'input_split_alpha_iteration': 20,
                'input_split_lr_alpha': 0.01,
                'alpha_dtype': 'float32'
            }
        }

        mock_self = mock_lirpa_net_factory()
        

        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4

        lb_result = torch.randn(batch_size, num_specs)
        mock_self.net.compute_bounds = MagicMock(return_value=(lb_result,))
        mock_self.get_alpha = MagicMock(return_value={})

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        lb, ret_alphas, lA, lbias = get_lower_bound_naive(
            mock_self,
            dm_lb=None,
            dm_l=dm_l,
            dm_u=dm_u,
            alphas={},
            bounding_method='alpha-crown',
            C=C,
            stop_criterion=lambda t: lambda x: x > t,
            thresholds=thresholds,
            stats=stats
        )

        # Check that compute_bounds was called with CROWN-Optimized
        call_kwargs = mock_self.net.compute_bounds.call_args[1]
        assert call_kwargs['method'] == 'CROWN-Optimized'



class TestGetLowerBoundNaiveCrown:
    """Tests for CROWN bounding method."""

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_crown_method_with_ibp_enhancement_disabled(self, mock_config, mock_lirpa_net_factory, stats):
        """Test CROWN method without IBP enhancement."""
        from input_split.bounding import get_lower_bound_naive

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            },
            'branching': {'input_split': {'ibp_enhancement': False}}
        }
        mock_config['solver'] = {
            'alpha-crown': {
                'input_split_alpha_iteration': 20,
                'input_split_lr_alpha': 0.01,
                'alpha_dtype': 'float32'
            }
        }

        mock_self = mock_lirpa_net_factory()
        

        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4

        lb_result = torch.randn(batch_size, num_specs)
        mock_self.net.compute_bounds = MagicMock(return_value=(lb_result,))

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        dm_lb = torch.randn(batch_size, num_specs) - 1.0  # Lower bounds to compare
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        lb, ret_alphas, lA, lbias = get_lower_bound_naive(
            mock_self,
            dm_lb=dm_lb,
            dm_l=dm_l,
            dm_u=dm_u,
            alphas={},
            bounding_method='crown',
            C=C,
            stop_criterion=lambda t: lambda x: x > t,
            thresholds=thresholds,
            stats=stats
        )

        assert lb.shape == (batch_size, num_specs)

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_crown_method_compares_with_dm_lb(self, mock_config, mock_lirpa_net_factory, stats):
        """Test that CROWN method takes max of computed lb and dm_lb."""
        from input_split.bounding import get_lower_bound_naive

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            },
            'branching': {'input_split': {'ibp_enhancement': False}}
        }
        mock_config['solver'] = {
            'alpha-crown': {
                'input_split_alpha_iteration': 20,
                'input_split_lr_alpha': 0.01,
                'alpha_dtype': 'float32'
            }
        }

        mock_self = mock_lirpa_net_factory()
        

        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4

        # Set up computed lb to be lower than dm_lb in some places
        lb_result = torch.tensor([[-1.0, -2.0], [-0.5, -1.5]])
        dm_lb = torch.tensor([[0.0, -3.0], [0.5, -2.0]])  # Higher in some places

        mock_self.net.compute_bounds = MagicMock(return_value=(lb_result.clone(),))

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        lb, _, _, _ = get_lower_bound_naive(
            mock_self,
            dm_lb=dm_lb,
            dm_l=dm_l,
            dm_u=dm_u,
            alphas={},
            bounding_method='crown',
            C=C,
            stop_criterion=lambda t: lambda x: x > t,
            thresholds=thresholds,
            stats=stats
        )

        # lb should be max of lb_result and dm_lb
        expected_lb = torch.max(lb_result, dm_lb)
        assert torch.allclose(lb, expected_lb)


class TestGetLowerBoundNaiveReturnA:
    """Tests for return_A functionality."""

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_return_A_true_returns_lA(self, mock_config, mock_lirpa_net_factory, stats):
        """Test that return_A=True returns lA."""
        from input_split.bounding import get_lower_bound_naive

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            },
            'branching': {'input_split': {'ibp_enhancement': False}}
        }
        mock_config['solver'] = {
            'reproduce_alpha_bounds': False,
            'alpha-crown': {
                'input_split_alpha_iteration': 20,
                'input_split_lr_alpha': 0.01,
                'alpha_dtype': 'float32'
            }
        }

        mock_self = mock_lirpa_net_factory()
        

        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4

        lb_result = torch.randn(batch_size, num_specs)
        lA_result = torch.randn(batch_size, num_specs, input_dim)
        A_dict = {
            'output': {
                'input': {
                    'lA': lA_result,
                    'lbias': torch.randn(batch_size, num_specs)
                }
            }
        }
        mock_self.net.compute_bounds = MagicMock(return_value=(lb_result, A_dict))
        mock_self.get_alpha = MagicMock(return_value={})

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        lb, ret_alphas, lA, lbias = get_lower_bound_naive(
            mock_self,
            dm_lb=None,
            dm_l=dm_l,
            dm_u=dm_u,
            alphas={},
            bounding_method='alpha-crown',
            C=C,
            stop_criterion=lambda t: lambda x: x > t,
            thresholds=thresholds,
            return_A=True,
            stats=stats
        )

        assert torch.equal(lA, lA_result)

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_return_A_false_returns_none(self, mock_config, mock_lirpa_net_factory, stats):
        """Test that return_A=False returns None for lA."""
        from input_split.bounding import get_lower_bound_naive

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            },
            'branching': {'input_split': {'ibp_enhancement': False}}
        }
        mock_config['solver'] = {
            'alpha-crown': {
                'input_split_alpha_iteration': 20,
                'input_split_lr_alpha': 0.01,
                'alpha_dtype': 'float32'
            }
        }

        mock_self = mock_lirpa_net_factory()
        

        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4

        lb_result = torch.randn(batch_size, num_specs)
        mock_self.net.compute_bounds = MagicMock(return_value=(lb_result,))

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        lb, ret_alphas, lA, lbias = get_lower_bound_naive(
            mock_self,
            dm_lb=None,
            dm_l=dm_l,
            dm_u=dm_u,
            alphas={},
            bounding_method='ibp',
            C=C,
            stop_criterion=lambda t: lambda x: x > t,
            thresholds=thresholds,
            return_A=False,
            stats=stats
        )

        assert lA is None


class TestGetLowerBoundNaiveReturnB:
    """Tests for return_b functionality."""

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_return_b_true_returns_lbias(self, mock_config, mock_lirpa_net_factory, stats):
        """Test that return_b=True returns lbias."""
        from input_split.bounding import get_lower_bound_naive

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            },
            'branching': {'input_split': {'ibp_enhancement': False}}
        }
        mock_config['solver'] = {
            'reproduce_alpha_bounds': False,
            'alpha-crown': {
                'input_split_alpha_iteration': 20,
                'input_split_lr_alpha': 0.01,
                'alpha_dtype': 'float32'
            }
        }

        mock_self = mock_lirpa_net_factory()

        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4

        lb_result = torch.randn(batch_size, num_specs)
        lbias_result = torch.randn(batch_size, num_specs)
        A_dict = {
            'output': {
                'input': {
                    'lA': torch.randn(batch_size, num_specs, input_dim),
                    'lbias': lbias_result
                }
            }
        }
        mock_self.net.compute_bounds = MagicMock(return_value=(lb_result, A_dict))
        mock_self.get_alpha = MagicMock(return_value={})

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        lb, ret_alphas, lA, lbias = get_lower_bound_naive(
            mock_self,
            dm_lb=None,
            dm_l=dm_l,
            dm_u=dm_u,
            alphas={},
            bounding_method='alpha-crown',
            C=C,
            stop_criterion=lambda t: lambda x: x > t,
            thresholds=thresholds,
            return_A=True,
            return_b=True,
            stats=stats
        )

        assert torch.equal(lbias, lbias_result)


class TestGetLowerBoundNaiveInfeasibleBatch:
    """Tests for infeasible batch handling."""

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_infeasible_batch_sets_lb_to_inf(self, mock_config, mock_lirpa_net_factory, stats):
        """Test that infeasible batches get lb set to infinity."""
        from input_split.bounding import get_lower_bound_naive

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            },
            'branching': {'input_split': {'ibp_enhancement': False}}
        }
        mock_config['solver'] = {
            'alpha-crown': {
                'input_split_alpha_iteration': 20,
                'input_split_lr_alpha': 0.01,
                'alpha_dtype': 'float32'
            }
        }

        mock_self = mock_lirpa_net_factory()
        
        batch_size = 3
        input_dim = 3
        num_specs = 2
        output_dim = 4

        lb_result = torch.randn(batch_size, num_specs)
        # Mark batch 1 as infeasible
        mock_self.net.infeasible_bounds_constraints = torch.tensor([False, True, False])
        mock_self.net.compute_bounds = MagicMock(return_value=(lb_result,))

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        lb, _, _, _ = get_lower_bound_naive(
            mock_self,
            dm_lb=None,
            dm_l=dm_l,
            dm_u=dm_u,
            alphas={},
            bounding_method='crown',
            C=C,
            stop_criterion=lambda t: lambda x: x > t,
            thresholds=thresholds,
            stats=stats
        )

        # Batch 1 should have infinity
        assert torch.isinf(lb[1]).all()
        # Other batches should not be infinity
        assert not torch.isinf(lb[0]).any()
        assert not torch.isinf(lb[2]).any()


class TestGetLowerBoundNaiveConstrainedConcretize:
    """Tests for constrained concretize handling."""

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_constrained_concretize_enabled_calls_init(self, mock_config, mock_lirpa_net_factory, stats):
        """Test that constrained concretize calls init_infeasible_bounds_constraints."""
        from input_split.bounding import get_lower_bound_naive

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': True, 'clip_type': 'complete'},
                'rearrange_constraints': False
            },
            'branching': {'input_split': {'ibp_enhancement': False}}
        }
        mock_config['solver'] = {
            'alpha-crown': {
                'input_split_alpha_iteration': 20,
                'input_split_lr_alpha': 0.01,
                'alpha_dtype': 'float32'
            }
        }

        mock_self = mock_lirpa_net_factory()
        
        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4

        lb_result = torch.randn(batch_size, num_specs)
        mock_self.net.compute_bounds = MagicMock(return_value=(lb_result,))

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        get_lower_bound_naive(
            mock_self,
            dm_lb=None,
            dm_l=dm_l,
            dm_u=dm_u,
            alphas={},
            bounding_method='ibp',
            C=C,
            stop_criterion=lambda t: lambda x: x > t,
            thresholds=thresholds,
            stats=stats
        )

        mock_self.net.init_infeasible_bounds_constraints.assert_called_once()




class TestGetLowerBoundWithIbpEnhancementBasic:
    """Basic tests for get_lower_bound_with_ibp_enhancement."""

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_ibp_enhancement_basic(self, mock_config, mock_lirpa_net_factory):
        """Test basic IBP enhancement functionality."""
        from input_split.bounding import get_lower_bound_with_ibp_enhancement
        from auto_LiRPA import BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            }
        }

        mock_self = mock_lirpa_net_factory()

        batch_size = 3
        input_dim = 4
        num_specs = 2
        output_dim = 5

        # First call returns IBP bounds, second returns CROWN bounds
        lb_ibp = torch.tensor([[-1.0, -2.0], [-0.5, -1.5], [0.5, 0.5]])  # Third is verified
        lb_crown = torch.tensor([[-0.5, -1.0], [-0.3, -1.0]])  # Only for unverified

        call_count = [0]

        def mock_compute_bounds(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (lb_ibp,)
            else:
                return (lb_crown, {
                    'output': {
                        'input': {
                            'lA': torch.randn(2, num_specs, input_dim),
                            'lbias': torch.randn(2, num_specs)
                        }
                    }
                })

        mock_self.net.compute_bounds = mock_compute_bounds

        # Mock nodes for reference bounds
        mock_node = MagicMock()
        mock_node.perturbed = True
        mock_node.lower = torch.zeros(batch_size, 5)
        mock_node.upper = torch.ones(batch_size, 5)
        mock_node.name = 'node1'
        mock_self.net.nodes = MagicMock(return_value=[mock_node])

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        new_x = BoundedTensor(dm_l, ptb=PerturbationLpNorm(x_L=dm_l, x_U=dm_u))
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        # Stop criterion: verified if any lb > threshold
        def stop_criterion(t):
            def criterion(lb):
                return lb > t
            return criterion

        lb, lA, lbias = get_lower_bound_with_ibp_enhancement(
            mock_self, new_x, C, 'crown', thresholds,
            return_A=True, stop_criterion=stop_criterion
        )

        assert lb.shape == (batch_size, num_specs)


class TestGetLowerBoundWithIbpEnhancementAllVerified:
    """Tests for IBP enhancement when all domains are verified."""

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_all_verified_sets_first_unverified(self, mock_config, mock_lirpa_net_factory):
        """Test that when all verified, first domain is still processed."""
        from input_split.bounding import get_lower_bound_with_ibp_enhancement
        from auto_LiRPA import BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            }
        }

        mock_self = mock_lirpa_net_factory()

        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4

        # All domains verified (lb > thresholds)
        lb_ibp = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
        lb_crown = torch.randn(1, num_specs)  # Only first domain

        call_count = [0]

        def mock_compute_bounds(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (lb_ibp,)
            else:
                return (lb_crown, {
                    'output': {
                        'input': {
                            'lA': torch.randn(1, num_specs, input_dim),
                            'lbias': torch.randn(1, num_specs)
                        }
                    }
                })

        mock_self.net.compute_bounds = mock_compute_bounds

        mock_node = MagicMock()
        mock_node.perturbed = True
        mock_node.lower = torch.zeros(batch_size, 5)
        mock_node.upper = torch.ones(batch_size, 5)
        mock_node.name = 'node1'
        mock_self.net.nodes = MagicMock(return_value=[mock_node])

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        new_x = BoundedTensor(dm_l, ptb=PerturbationLpNorm(x_L=dm_l, x_U=dm_u))
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        def stop_criterion(t):
            def criterion(lb):
                return lb > t
            return criterion

        lb, lA, lbias = get_lower_bound_with_ibp_enhancement(
            mock_self, new_x, C, 'crown', thresholds,
            return_A=True, stop_criterion=stop_criterion
        )

        # CROWN should still be called (for at least first domain)
        assert call_count[0] == 2


class TestGetLowerBoundWithIbpEnhancementConstraints:
    """Tests for IBP enhancement with constraints."""

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_constraints_passed_correctly(self, mock_config, mock_lirpa_net_factory):
        """Test that constraints are filtered for unverified domains."""
        from input_split.bounding import get_lower_bound_with_ibp_enhancement
        from auto_LiRPA import BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            }
        }

        mock_self = mock_lirpa_net_factory()

        batch_size = 3
        input_dim = 4
        num_specs = 2
        output_dim = 5
        num_constraints = 3

        # First domain verified, others not
        lb_ibp = torch.tensor([[1.0, 2.0], [-0.5, -1.5], [-0.3, -1.0]])
        lb_crown = torch.randn(2, num_specs)  # For unverified domains

        call_count = [0]

        def mock_compute_bounds(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (lb_ibp,)
            else:
                return (lb_crown, {
                    'output': {
                        'input': {
                            'lA': torch.randn(2, num_specs, input_dim),
                            'lbias': torch.randn(2, num_specs)
                        }
                    }
                })

        mock_self.net.compute_bounds = mock_compute_bounds

        mock_node = MagicMock()
        mock_node.perturbed = True
        mock_node.lower = torch.zeros(batch_size, 5)
        mock_node.upper = torch.ones(batch_size, 5)
        mock_node.name = 'node1'
        mock_self.net.nodes = MagicMock(return_value=[mock_node])

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        new_x = BoundedTensor(dm_l, ptb=PerturbationLpNorm(x_L=dm_l, x_U=dm_u))
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)
        constraints_A = torch.randn(batch_size, num_constraints, input_dim)
        constraints_b = torch.randn(batch_size, num_constraints)
        constraints = (constraints_A, constraints_b)

        def stop_criterion(t):
            def criterion(lb):
                return lb > t
            return criterion

        lb, lA, lbias = get_lower_bound_with_ibp_enhancement(
            mock_self, new_x, C, 'crown', thresholds,
            return_A=True, stop_criterion=stop_criterion,
            constraints=constraints
        )

        assert lb.shape == (batch_size, num_specs)


class TestGetLowerBoundWithIbpEnhancementReturnAFalse:
    """Tests for IBP enhancement with return_A=False."""

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_return_A_false_returns_none(self, mock_config, mock_lirpa_net_factory):
        """Test that return_A=False returns None for lA and lbias."""
        from input_split.bounding import get_lower_bound_with_ibp_enhancement
        from auto_LiRPA import BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            }
        }

        mock_self = mock_lirpa_net_factory()

        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4

        lb_ibp = torch.tensor([[-1.0, -2.0], [-0.5, -1.5]])
        lb_crown = torch.randn(batch_size, num_specs)

        call_count = [0]

        def mock_compute_bounds(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (lb_ibp,)
            else:
                return (lb_crown, None)

        mock_self.net.compute_bounds = mock_compute_bounds

        mock_node = MagicMock()
        mock_node.perturbed = True
        mock_node.lower = torch.zeros(batch_size, 5)
        mock_node.upper = torch.ones(batch_size, 5)
        mock_node.name = 'node1'
        mock_self.net.nodes = MagicMock(return_value=[mock_node])

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        new_x = BoundedTensor(dm_l, ptb=PerturbationLpNorm(x_L=dm_l, x_U=dm_u))
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        def stop_criterion(t):
            def criterion(lb):
                return lb > t
            return criterion

        lb, lA, lbias = get_lower_bound_with_ibp_enhancement(
            mock_self, new_x, C, 'crown', thresholds,
            return_A=False, stop_criterion=stop_criterion
        )

        assert lA is None
        assert lbias is None


class TestGetLowerBoundWithIbpEnhancementInfeasible:
    """Tests for IBP enhancement infeasible batch handling."""

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_infeasible_sets_lb_to_inf(self, mock_config, mock_lirpa_net_factory):
        """Test that infeasible batches get lb set to infinity in CROWN step."""
        from input_split.bounding import get_lower_bound_with_ibp_enhancement
        from auto_LiRPA import BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            }
        }

        mock_self = mock_lirpa_net_factory()

        batch_size = 3
        input_dim = 4
        num_specs = 2
        output_dim = 5

        lb_ibp = torch.tensor([[-1.0, -2.0], [-0.5, -1.5], [-0.3, -1.0]])
        lb_crown = torch.randn(batch_size, num_specs)

        call_count = [0]

        def mock_compute_bounds(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Set infeasible to None for IBP
                mock_self.net.infeasible_bounds_constraints = None
                return (lb_ibp,)
            else:
                # Mark second unverified domain as infeasible
                mock_self.net.infeasible_bounds_constraints = torch.tensor([False, True, False])
                return (lb_crown, {
                    'output': {
                        'input': {
                            'lA': torch.randn(batch_size, num_specs, input_dim),
                            'lbias': torch.randn(batch_size, num_specs)
                        }
                    }
                })

        mock_self.net.compute_bounds = mock_compute_bounds

        mock_node = MagicMock()
        mock_node.perturbed = True
        mock_node.lower = torch.zeros(batch_size, 5)
        mock_node.upper = torch.ones(batch_size, 5)
        mock_node.name = 'node1'
        mock_self.net.nodes = MagicMock(return_value=[mock_node])

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        new_x = BoundedTensor(dm_l, ptb=PerturbationLpNorm(x_L=dm_l, x_U=dm_u))
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        def stop_criterion(t):
            def criterion(lb):
                return lb > t
            return criterion

        lb, lA, lbias = get_lower_bound_with_ibp_enhancement(
            mock_self, new_x, C, 'crown', thresholds,
            return_A=True, stop_criterion=stop_criterion
        )

        # Second domain should have infinity (infeasible)
        assert torch.isinf(lb[1]).all()


class TestGetLowerBoundNaiveWithIbpEnhancement:
    """Tests for get_lower_bound_naive when IBP enhancement is enabled."""

    @patch('input_split.bounding.get_lower_bound_with_ibp_enhancement')
    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_ibp_enhancement_enabled_calls_function(self, mock_config, mock_ibp_enhance, mock_lirpa_net_factory, stats):
        """Test that IBP enhancement calls the enhancement function."""
        from input_split.bounding import get_lower_bound_naive

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            },
            'branching': {'input_split': {'ibp_enhancement': True}}
        }
        mock_config['solver'] = {
            'alpha-crown': {
                'input_split_alpha_iteration': 20,
                'input_split_lr_alpha': 0.01,
                'alpha_dtype': 'float32'
            }
        }

        mock_self = mock_lirpa_net_factory()
        
        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4

        lb_result = torch.randn(batch_size, num_specs)
        mock_ibp_enhance.return_value = (lb_result, None, None)

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        get_lower_bound_naive(
            mock_self,
            dm_lb=None,
            dm_l=dm_l,
            dm_u=dm_u,
            alphas={},
            bounding_method='crown',
            C=C,
            stop_criterion=lambda t: lambda x: x > t,
            thresholds=thresholds,
            stats=stats
        )

        mock_ibp_enhance.assert_called_once()


class TestGetLowerBoundNaiveReferenceIntermBounds:
    """Tests for reference intermediate bounds."""

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_reference_bounds_passed_to_compute_bounds(self, mock_config, mock_lirpa_net_factory, stats):
        """Test that reference bounds are passed to compute_bounds."""
        from input_split.bounding import get_lower_bound_naive

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            },
            'branching': {'input_split': {'ibp_enhancement': False}}
        }
        mock_config['solver'] = {
            'alpha-crown': {
                'input_split_alpha_iteration': 20,
                'input_split_lr_alpha': 0.01,
                'alpha_dtype': 'float32'
            }
        }

        mock_self = mock_lirpa_net_factory()

        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4

        lb_result = torch.randn(batch_size, num_specs)
        mock_self.net.compute_bounds = MagicMock(return_value=(lb_result,))

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)
        reference_bounds = {'layer1': (torch.zeros(2, 5), torch.ones(2, 5))}

        get_lower_bound_naive(
            mock_self,
            dm_lb=None,
            dm_l=dm_l,
            dm_u=dm_u,
            alphas={},
            bounding_method='crown',
            C=C,
            stop_criterion=lambda t: lambda x: x > t,
            thresholds=thresholds,
            reference_interm_bounds=reference_bounds,
            stats=stats
        )

        call_kwargs = mock_self.net.compute_bounds.call_args[1]
        assert call_kwargs['reference_bounds'] == reference_bounds


class TestGetLowerBoundNaiveAlphaReuse:
    """Tests for alpha reuse behavior."""

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_reuse_alpha_true_when_crown_with_alphas(self, mock_config, mock_lirpa_net_factory, stats):
        """Test that reuse_alpha is True when method is crown and alphas exist."""
        from input_split.bounding import get_lower_bound_naive

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            },
            'branching': {'input_split': {'ibp_enhancement': False}}
        }
        mock_config['solver'] = {
            'alpha-crown': {
                'input_split_alpha_iteration': 20,
                'input_split_lr_alpha': 0.01,
                'alpha_dtype': 'float32'
            }
        }

        mock_self = mock_lirpa_net_factory()
        
        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4

        lb_result = torch.randn(batch_size, num_specs)
        mock_self.net.compute_bounds = MagicMock(return_value=(lb_result,))

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)
        # Non-empty alphas
        alphas = {'layer1': torch.randn(2, 3)}

        get_lower_bound_naive(
            mock_self,
            dm_lb=None,
            dm_l=dm_l,
            dm_u=dm_u,
            alphas=alphas,
            bounding_method='crown',
            C=C,
            stop_criterion=lambda t: lambda x: x > t,
            thresholds=thresholds,
            stats=stats
        )

        call_kwargs = mock_self.net.compute_bounds.call_args[1]
        assert call_kwargs['reuse_alpha']

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_reuse_alpha_false_when_empty_alphas(self, mock_config, mock_lirpa_net_factory, stats):
        """Test that reuse_alpha is False when alphas is empty."""
        from input_split.bounding import get_lower_bound_naive

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            },
            'branching': {'input_split': {'ibp_enhancement': False}}
        }
        mock_config['solver'] = {
            'alpha-crown': {
                'input_split_alpha_iteration': 20,
                'input_split_lr_alpha': 0.01,
                'alpha_dtype': 'float32'
            }
        }

        mock_self = mock_lirpa_net_factory()
        
        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4

        lb_result = torch.randn(batch_size, num_specs)
        mock_self.net.compute_bounds = MagicMock(return_value=(lb_result,))

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)
        # Empty alphas
        alphas = {}

        get_lower_bound_naive(
            mock_self,
            dm_lb=None,
            dm_l=dm_l,
            dm_u=dm_u,
            alphas=alphas,
            bounding_method='crown',
            C=C,
            stop_criterion=lambda t: lambda x: x > t,
            thresholds=thresholds,
            stats=stats
        )

        call_kwargs = mock_self.net.compute_bounds.call_args[1]
        assert not call_kwargs['reuse_alpha']


class TestGetLowerBoundNaiveBoundOpts:
    """Tests for bound options setting."""

    @patch('input_split.bounding.arguments.Config', new_callable=dict)
    def test_bound_opts_set_correctly(self, mock_config, mock_lirpa_net_factory, stats):
        """Test that bound options are set with correct values."""
        from input_split.bounding import get_lower_bound_naive

        mock_config['bab'] = {
            'clip_n_verify': {
                'clip_input_domain': {'enabled': False, 'clip_type': 'none'},
                'rearrange_constraints': False
            },
            'branching': {'input_split': {'ibp_enhancement': False}}
        }
        mock_config['solver'] = {
            'alpha-crown': {
                'input_split_alpha_iteration': 50,
                'input_split_lr_alpha': 0.05,
                'alpha_dtype': 'float32'
            }
        }

        mock_self = mock_lirpa_net_factory()
        
        batch_size = 2
        input_dim = 3
        num_specs = 2
        output_dim = 4

        lb_result = torch.randn(batch_size, num_specs)
        mock_self.net.compute_bounds = MagicMock(return_value=(lb_result,))

        dm_l = torch.zeros(batch_size, input_dim)
        dm_u = torch.ones(batch_size, input_dim)
        C = torch.randn(batch_size, num_specs, output_dim)
        thresholds = torch.zeros(batch_size, num_specs)

        get_lower_bound_naive(
            mock_self,
            dm_lb=None,
            dm_l=dm_l,
            dm_u=dm_u,
            alphas={},
            bounding_method='ibp',
            C=C,
            stop_criterion=lambda t: lambda x: x > t,
            thresholds=thresholds,
            stats=stats
        )

        # Verify set_bound_opts was called
        mock_self.net.set_bound_opts.assert_called_once()
        call_args = mock_self.net.set_bound_opts.call_args[0][0]
        assert not call_args['optimize_bound_args']['enable_beta_crown']
        assert call_args['optimize_bound_args']['fix_interm_bounds']
        assert call_args['optimize_bound_args']['iteration'] == 50
        assert call_args['optimize_bound_args']['lr_alpha'] == 0.05


if __name__ == '__main__':
    setup_module()
    unittest.main()
