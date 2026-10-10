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
"""Unit tests for jit_precompile.py"""
import os
import sys
import unittest

import torch
import torch.nn as nn

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSimpleModelForJIT(unittest.TestCase):
    """Tests for SimpleModelForJIT class."""

    def test_model_creation(self):
        """Test that SimpleModelForJIT can be created."""
        from jit_precompile import SimpleModelForJIT
        model = SimpleModelForJIT()
        self.assertIsNotNone(model)

    def test_model_forward(self):
        """Test forward pass of SimpleModelForJIT."""
        from jit_precompile import SimpleModelForJIT
        model = SimpleModelForJIT()
        x = torch.randn(1, 3)
        output = model(x)
        self.assertEqual(output.shape, (1, 3))

    def test_model_batch_forward(self):
        """Test forward pass with batch input."""
        from jit_precompile import SimpleModelForJIT
        model = SimpleModelForJIT()
        x = torch.randn(10, 3)
        output = model(x)
        self.assertEqual(output.shape, (10, 3))

    def test_model_layers(self):
        """Test that model has expected layers."""
        from jit_precompile import SimpleModelForJIT
        model = SimpleModelForJIT()
        self.assertIsInstance(model.fc1, nn.Linear)
        self.assertIsInstance(model.relu1, nn.ReLU)
        self.assertIsInstance(model.fc2, nn.Linear)
        self.assertIsInstance(model.relu2, nn.ReLU)
        self.assertIsInstance(model.fc3, nn.Linear)

    def test_model_layer_dimensions(self):
        """Test model layer dimensions."""
        from jit_precompile import SimpleModelForJIT
        model = SimpleModelForJIT()
        self.assertEqual(model.fc1.in_features, 3)
        self.assertEqual(model.fc1.out_features, 32)
        self.assertEqual(model.fc2.in_features, 32)
        self.assertEqual(model.fc2.out_features, 32)
        self.assertEqual(model.fc3.in_features, 32)
        self.assertEqual(model.fc3.out_features, 3)


if __name__ == '__main__':
    unittest.main()
