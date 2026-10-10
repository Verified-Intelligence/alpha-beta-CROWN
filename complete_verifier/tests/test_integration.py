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
"""
Integration tests for α,β-CROWN verifier using real neural network models.

These tests use the simple MLP fixture (tests/fixtures/simple_mlp.onnx)
to test the full verification pipeline including:
- Model loading and ONNX optimization
- Bound computation
- Branch and bound verification
- API usage patterns
"""
import os
import sys
import unittest
import warnings

import numpy as np
import torch

from .conftest import requires_cuda

# Suppress onnx2pytorch warnings about non-writable numpy arrays
# This is a known issue in onnx2pytorch and doesn't affect functionality
warnings.filterwarnings(
    'ignore',
    message='The given NumPy array is not writable',
    category=UserWarning
)

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path to test fixtures
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
SIMPLE_MLP_PATH = os.path.join(FIXTURES_DIR, 'simple_mlp.onnx')

# Simple MLP structure: 4 inputs -> hidden -> 2 outputs
NUM_INPUTS = 4
NUM_OUTPUTS = 2


def skip_if_no_model(func):
    """Decorator to skip test if model is not available."""
    def wrapper(*args, **kwargs):
        if not os.path.exists(SIMPLE_MLP_PATH):
            raise unittest.SkipTest(f"Model not found at {SIMPLE_MLP_PATH}")
        return func(*args, **kwargs)
    return wrapper


class TestIntegrationModelLoading(unittest.TestCase):
    """Tests for model loading and ONNX processing."""

    @skip_if_no_model
    def test_load_model_returns_correct_structure(self):
        """Test that loaded model has expected structure."""
        from load_model import load_model_onnx
        model, input_shape = load_model_onnx(SIMPLE_MLP_PATH)

        # Model should be a torch.nn.Module
        self.assertIsInstance(model, torch.nn.Module)

        # Should be able to do forward pass
        test_input = torch.randn(1, NUM_INPUTS)
        with torch.no_grad():
            output = model(test_input)
        self.assertIsNotNone(output)

    @skip_if_no_model
    def test_model_forward_pass_shape(self):
        """Test that model produces correct output shape."""
        from load_model import load_model_onnx
        model, _ = load_model_onnx(SIMPLE_MLP_PATH)
        model.eval()

        batch_sizes = [1, 4, 16]
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, NUM_INPUTS)
            with torch.no_grad():
                output = model(test_input)
            # Output should have shape (batch_size, num_outputs)
            self.assertEqual(output.shape[0], batch_size)
            self.assertEqual(output.shape[1], NUM_OUTPUTS)

    @skip_if_no_model
    def test_model_deterministic_output(self):
        """Test that model produces deterministic outputs."""
        from load_model import load_model_onnx
        model, _ = load_model_onnx(SIMPLE_MLP_PATH)
        model.eval()

        test_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        with torch.no_grad():
            output1 = model(test_input).clone()
            output2 = model(test_input).clone()
        self.assertTrue(torch.allclose(output1, output2))


class TestIntegrationSpecificationBuilding(unittest.TestCase):
    """Tests for building verification specifications."""

    def test_build_spec_with_bounds(self):
        """Test building specification with explicit bounds."""
        from api import VerificationSpec, input_vars, output_vars

        x = input_vars(NUM_INPUTS)
        y = output_vars(NUM_OUTPUTS)

        # Simple robustness specification: output 0 > output 1
        lower = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        upper = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=(x >= lower) & (x <= upper),
            output_constraint=(y[0] > y[1]),
        )

        self.assertIsNotNone(spec)
        # input_shape should be (-1, NUM_INPUTS) or (1, NUM_INPUTS)
        self.assertEqual(spec.input_shape[-1], NUM_INPUTS)

    def test_build_spec_with_center_epsilon(self):
        """Test building specification with center and epsilon."""
        from api import VerificationSpec, input_vars, output_vars

        x = input_vars(NUM_INPUTS)
        y = output_vars(NUM_OUTPUTS)

        center = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        epsilon = 0.1

        spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=(x >= center - epsilon) & (x <= center + epsilon),
            output_constraint=(y[0] > y[1]),
        )

        self.assertIsNotNone(spec)

        # Verify input bounds are correctly computed from center ± epsilon
        expected_lower = center - epsilon
        expected_upper = center + epsilon
        self.assertTrue(torch.allclose(spec.lower, expected_lower))
        self.assertTrue(torch.allclose(spec.upper, expected_upper))

        # Verify input shape is correct
        self.assertEqual(spec.input_shape[-1], NUM_INPUTS)

        # Verify num_inputs matches
        self.assertEqual(spec.num_inputs, 1)

        # Verify output clauses exist and have correct structure
        self.assertIsNotNone(spec.clauses)
        self.assertEqual(len(spec.clauses), spec.num_inputs)
        # Each clause should be a list of (C, rhs) tuples
        for clause_list in spec.clauses:
            self.assertIsInstance(clause_list, list)
            for clause in clause_list:
                self.assertIsInstance(clause, tuple)
                self.assertEqual(len(clause), 2)
                C, rhs = clause
                # C should have shape (num_constraints, num_outputs)
                self.assertEqual(C.ndim, 2)
                self.assertEqual(C.shape[1], NUM_OUTPUTS)
                # rhs should have shape (num_constraints,)
                self.assertEqual(rhs.ndim, 1)
                self.assertEqual(rhs.shape[0], C.shape[0])

    def test_build_spec_disjunctive_constraint(self):
        """Test building specification with OR constraints."""
        from api import VerificationSpec, input_vars, output_vars

        x = input_vars(NUM_INPUTS)
        y = output_vars(NUM_OUTPUTS)

        lower = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        upper = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        # Either output 0 > output 1 OR output 1 > 0
        spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=(x >= lower) & (x <= upper),
            output_constraint=(y[0] > y[1]) | (y[1] > 0),
        )

        self.assertIsNotNone(spec)

        # Verify input bounds are correctly set
        self.assertTrue(torch.allclose(spec.lower, lower))
        self.assertTrue(torch.allclose(spec.upper, upper))

        # Verify input shape is correct
        self.assertEqual(spec.input_shape[-1], NUM_INPUTS)

        # Verify clauses structure for disjunctive constraint
        # The OR constraint (y[0] > y[1]) | (y[1] > 0) is negated to:
        # (y[0] <= y[1]) & (y[1] <= 0), which is a single conjunction
        # So we expect clauses to contain this conjunction
        self.assertIsNotNone(spec.clauses)
        self.assertGreater(len(spec.clauses), 0)

        # Each clause list should contain (C, rhs) tuples
        for clause_list in spec.clauses:
            self.assertIsInstance(clause_list, list)
            for clause in clause_list:
                self.assertIsInstance(clause, tuple)
                self.assertEqual(len(clause), 2)
                C, rhs = clause
                # C should have shape (num_constraints, num_outputs)
                self.assertEqual(C.ndim, 2)
                self.assertEqual(C.shape[1], NUM_OUTPUTS)
                # rhs should have shape (num_constraints,)
                self.assertEqual(rhs.ndim, 1)
                self.assertEqual(rhs.shape[0], C.shape[0])


class TestIntegrationBoundComputation(unittest.TestCase):
    """Tests for bound computation using auto_LiRPA."""

    @requires_cuda
    @skip_if_no_model
    def test_create_lirpa_net(self):
        """Test creating LiRPANet wrapper around model."""
        from load_model import load_model_onnx
        from beta_CROWN_solver import LiRPANet

        model, input_shape = load_model_onnx(SIMPLE_MLP_PATH)

        # Create LiRPANet wrapper
        lirpa_net = LiRPANet(model, in_size=input_shape)
        self.assertIsNotNone(lirpa_net)
        self.assertIsNotNone(lirpa_net.net)

    @skip_if_no_model
    def test_compute_bounds_ibp(self):
        """Test computing bounds using IBP (interval bound propagation)."""
        from load_model import load_model_onnx
        from auto_LiRPA import BoundedModule, BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        model, _ = load_model_onnx(SIMPLE_MLP_PATH)
        model.eval()

        # Create bounded input
        center = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        eps = 0.1

        bounded_model = BoundedModule(model, center)

        ptb = PerturbationLpNorm(norm=float('inf'), eps=eps)
        x = BoundedTensor(center, ptb)

        # Compute IBP bounds
        lb, ub = bounded_model.compute_bounds(x=(x,), method='IBP')

        self.assertEqual(lb.shape, (1, NUM_OUTPUTS))
        self.assertEqual(ub.shape, (1, NUM_OUTPUTS))
        # Lower bounds should be <= upper bounds
        self.assertTrue(torch.all(lb <= ub + 1e-5))

    @skip_if_no_model
    def test_compute_bounds_crown(self):
        """Test computing bounds using CROWN."""
        from load_model import load_model_onnx
        from auto_LiRPA import BoundedModule, BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        model, _ = load_model_onnx(SIMPLE_MLP_PATH)
        model.eval()

        center = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        eps = 0.1

        bounded_model = BoundedModule(model, center)
        ptb = PerturbationLpNorm(norm=float('inf'), eps=eps)
        x = BoundedTensor(center, ptb)

        # Compute CROWN bounds
        lb, ub = bounded_model.compute_bounds(x=(x,), method='CROWN')

        self.assertEqual(lb.shape, (1, NUM_OUTPUTS))
        self.assertEqual(ub.shape, (1, NUM_OUTPUTS))
        self.assertTrue(torch.all(lb <= ub + 1e-5))

    @skip_if_no_model
    def test_crown_tighter_than_ibp(self):
        """Test that CROWN bounds are at least as tight as IBP."""
        from load_model import load_model_onnx
        from auto_LiRPA import BoundedModule, BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        model, _ = load_model_onnx(SIMPLE_MLP_PATH)
        model.eval()

        center = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        eps = 0.1

        bounded_model = BoundedModule(model, center)
        ptb = PerturbationLpNorm(norm=float('inf'), eps=eps)
        x = BoundedTensor(center, ptb)

        # Compute both bounds
        lb_ibp, ub_ibp = bounded_model.compute_bounds(x=(x,), method='IBP')

        # Need to reset for CROWN
        bounded_model = BoundedModule(model, center)
        x = BoundedTensor(center, ptb)
        lb_crown, ub_crown = bounded_model.compute_bounds(x=(x,), method='CROWN')

        # CROWN lower bounds should be >= IBP lower bounds (tighter)
        # CROWN upper bounds should be <= IBP upper bounds (tighter)
        # Allow small tolerance for numerical issues
        self.assertTrue(torch.all(lb_crown >= lb_ibp - 1e-5))
        self.assertTrue(torch.all(ub_crown <= ub_ibp + 1e-5))


class TestIntegrationVerification(unittest.TestCase):
    """Full verification integration tests using ABCrownSolver."""

    @skip_if_no_model
    def test_solver_initialization(self):
        """Test that ABCrownSolver can be initialized."""
        from api import ABCrownSolver, VerificationSpec, input_vars, output_vars

        x = input_vars(NUM_INPUTS)
        y = output_vars(NUM_OUTPUTS)

        # Tight bounds
        lower = torch.tensor([[0.4, 0.4, 0.4, 0.4]])
        upper = torch.tensor([[0.6, 0.6, 0.6, 0.6]])

        spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=(x >= lower) & (x <= upper),
            output_constraint=(y[0] > y[1]),
        )

        solver = ABCrownSolver(spec, SIMPLE_MLP_PATH)
        self.assertIsNotNone(solver)
        self.assertEqual(solver.spec, spec)

    @skip_if_no_model
    def test_solver_with_config_builder(self):
        """Test ABCrownSolver with custom configuration."""
        from api import ABCrownSolver, VerificationSpec, ConfigBuilder, input_vars, output_vars

        x = input_vars(NUM_INPUTS)
        y = output_vars(NUM_OUTPUTS)

        lower = torch.tensor([[0.4, 0.4, 0.4, 0.4]])
        upper = torch.tensor([[0.6, 0.6, 0.6, 0.6]])

        spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=(x >= lower) & (x <= upper),
            output_constraint=(y[0] > y[1]),
        )

        # ConfigBuilder uses kwargs with double underscore for nested keys
        config = ConfigBuilder()
        config.set(general__device='cpu', bab__timeout=30)

        solver = ABCrownSolver(spec, SIMPLE_MLP_PATH, config=config)
        self.assertIsNotNone(solver)

        # Verify solver is using the custom config settings
        self.assertEqual(solver.config['general']['device'], 'cpu')
        self.assertEqual(solver.config['bab']['timeout'], 30)

        # Verify the config is a deep copy (modifying original doesn't affect solver)
        config.set(bab__timeout=999)
        self.assertEqual(solver.config['bab']['timeout'], 30)  # Still 30, not 999

    @skip_if_no_model
    def test_verification_runs(self):
        """Test that verification actually runs and returns a result."""
        from api import ABCrownSolver, VerificationSpec, ConfigBuilder, input_vars, output_vars

        x = input_vars(NUM_INPUTS)
        y = output_vars(NUM_OUTPUTS)

        # Very tight bounds for faster verification
        lower = torch.tensor([[0.45, 0.45, 0.45, 0.45]])
        upper = torch.tensor([[0.55, 0.55, 0.55, 0.55]])

        spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=(x >= lower) & (x <= upper),
            output_constraint=(y[0] > y[1]),
        )

        config = ConfigBuilder()
        config.set(general__device='cpu', bab__timeout=60, solver__batch_size=64)

        solver = ABCrownSolver(spec, SIMPLE_MLP_PATH, config=config)
        result = solver.solve()

        self.assertIsNotNone(result)
        # Result should have a status
        self.assertTrue(hasattr(result, 'status'))
        # Status should be one of the valid verification outcomes
        valid_statuses = [
            'verified', 'safe', 'holds', 'sat',
            'unsafe', 'unsafe-pgd', 'unsafe-bab', 'falsified',
            'unknown', 'timeout', 'unsat'
        ]
        self.assertIn(result.status, valid_statuses)

    @skip_if_no_model
    def test_verification_result_has_time(self):
        """Test that verification result includes timing information."""
        from api import ABCrownSolver, VerificationSpec, ConfigBuilder, input_vars, output_vars

        x = input_vars(NUM_INPUTS)
        y = output_vars(NUM_OUTPUTS)

        lower = torch.tensor([[0.45, 0.45, 0.45, 0.45]])
        upper = torch.tensor([[0.55, 0.55, 0.55, 0.55]])

        spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=(x >= lower) & (x <= upper),
            output_constraint=(y[0] > y[1]),
        )

        config = ConfigBuilder()
        config.set(general__device='cpu', bab__timeout=30)

        solver = ABCrownSolver(spec, SIMPLE_MLP_PATH, config=config)
        result = solver.solve()

        # Result should have timing info in stats['elapsed']
        self.assertTrue(hasattr(result, 'stats'))
        self.assertIn('elapsed', result.stats)
        elapsed = result.stats['elapsed']
        self.assertTrue(elapsed is None or isinstance(elapsed, (int, float)))


class TestIntegrationOnnxOptimization(unittest.TestCase):
    """Tests for ONNX graph optimization."""

    @skip_if_no_model
    def test_compress_onnx_runs(self):
        """Test that compress_onnx can process the model."""
        import onnx
        import tempfile
        from onnx_opt import compress_onnx

        onnx_model = onnx.load(SIMPLE_MLP_PATH)

        # compress_onnx needs paths
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            save_path = f.name

        try:
            optimized = compress_onnx(
                onnx_model, SIMPLE_MLP_PATH, save_path,
                onnx_optimization_flags=['none']
            )
            self.assertIsNotNone(optimized)
        finally:
            import os
            if os.path.exists(save_path):
                os.unlink(save_path)

    @skip_if_no_model
    def test_onnx_model_can_be_loaded_and_checked(self):
        """Test that ONNX model is valid."""
        import onnx

        onnx_model = onnx.load(SIMPLE_MLP_PATH)

        # Check model validity
        try:
            onnx.checker.check_model(onnx_model)
            valid = True
        except Exception:
            valid = False
        self.assertTrue(valid)


class TestIntegrationConfigBuilder(unittest.TestCase):
    """Tests for ConfigBuilder functionality."""

    def test_config_builder_creation(self):
        """Test creating a ConfigBuilder."""
        from api import ConfigBuilder
        config = ConfigBuilder()
        self.assertIsNotNone(config)

    def test_config_builder_set(self):
        """Test setting config values."""
        from api import ConfigBuilder
        config = ConfigBuilder()
        config.set(general__device='cpu')
        cfg_dict = config.to_dict()
        self.assertEqual(cfg_dict['general']['device'], 'cpu')

    def test_config_builder_chaining(self):
        """Test that config builder methods are chainable."""
        from api import ConfigBuilder
        config = (ConfigBuilder()
                  .set(general__device='cpu')
                  .set(bab__timeout=30))
        cfg_dict = config.to_dict()
        self.assertEqual(cfg_dict['general']['device'], 'cpu')
        self.assertEqual(cfg_dict['bab']['timeout'], 30)

    def test_config_builder_update(self):
        """Test update method with dict."""
        from api import ConfigBuilder
        config = ConfigBuilder()
        config.update({'general': {'device': 'cpu'}})
        cfg_dict = config.to_dict()
        self.assertEqual(cfg_dict['general']['device'], 'cpu')


if __name__ == '__main__':
    unittest.main()
