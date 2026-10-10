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
"""Unit tests for data_utils.py"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile

import torch
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import make_eps_tensor, preprocess_cifar


class TestMakeEpsTensor(unittest.TestCase):
    """Tests for make_eps_tensor function."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        result = make_eps_tensor(None)
        self.assertIsNone(result)

    def test_float_to_tensor(self):
        """Test converting float to tensor."""
        result = make_eps_tensor(0.5)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.item(), 0.5)

    def test_int_to_tensor(self):
        """Test converting int to tensor."""
        result = make_eps_tensor(2)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.item(), 2)

    def test_list_to_tensor(self):
        """Test converting list to tensor."""
        result = make_eps_tensor([0.1, 0.2, 0.3])
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(torch.allclose(result, torch.tensor([0.1, 0.2, 0.3])))

    def test_numpy_array_to_tensor(self):
        """Test converting numpy array to tensor."""
        arr = np.array([0.1, 0.2])
        result = make_eps_tensor(arr)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (2,))

    def test_tensor_returns_tensor(self):
        """Test that tensor input returns tensor."""
        t = torch.tensor([0.5, 0.6])
        result = make_eps_tensor(t)
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.equal(result, t))


class TestPreprocessCifar(unittest.TestCase):
    """Tests for preprocess_cifar function."""

    def test_basic_preprocessing(self):
        """Test basic CIFAR preprocessing with default settings."""
        # Create a sample image (values 0-1)
        image = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        result = preprocess_cifar(image)
        # Result should be normalized
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (32, 32, 3))

    def test_preprocessing_with_zeros(self):
        """Test preprocessing with zero input."""
        image = np.zeros((32, 32, 3), dtype=np.float32)
        result = preprocess_cifar(image)
        # Should be (0 - means) / stds
        self.assertIsInstance(result, np.ndarray)
        # All values should be negative (since we subtract positive means)
        self.assertTrue(np.all(result < 0))

    def test_preprocessing_with_ones(self):
        """Test preprocessing with ones input."""
        image = np.ones((32, 32, 3), dtype=np.float32)
        result = preprocess_cifar(image)
        # Should be (1 - means) / stds
        self.assertIsInstance(result, np.ndarray)

    def test_inception_preprocessing(self):
        """Test preprocessing with inception mode."""
        image = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        result = preprocess_cifar(image, inception_preprocess=True)
        # In inception mode, uses 0.5 for both mean and std
        # (0.5 - 0.5) / 0.5 = 0
        expected = np.zeros_like(image)
        self.assertTrue(np.allclose(result, expected))

    def test_perturbation_mode(self):
        """Test preprocessing in perturbation mode."""
        eps = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        result = preprocess_cifar(eps, perturbation=True)
        # In perturbation mode, only divides by std (no mean subtraction)
        self.assertIsInstance(result, np.ndarray)
        # All values should be positive since we're just scaling
        self.assertTrue(np.all(result > 0))

    def test_perturbation_with_inception(self):
        """Test perturbation mode with inception preprocessing."""
        eps = 0.1
        result = preprocess_cifar(eps, inception_preprocess=True, perturbation=True)
        # In inception perturbation mode: eps / 0.5 = 0.2
        self.assertAlmostEqual(result, 0.2)

    def test_scalar_perturbation(self):
        """Test perturbation preprocessing with scalar input."""
        eps = 2./255.
        result = preprocess_cifar(eps, perturbation=True)
        # Should be eps / STD for each channel
        self.assertIsInstance(result, np.ndarray)

    def test_batch_preprocessing(self):
        """Test preprocessing with batch of images."""
        images = np.random.rand(10, 32, 32, 3).astype(np.float32)
        result = preprocess_cifar(images)
        self.assertEqual(result.shape, (10, 32, 32, 3))

    def test_mean_std_values(self):
        """Test that correct mean and std are used."""
        # Test that the function uses the expected mean/std values
        MEANS = np.array([125.3, 123.0, 113.9], dtype=np.float32)/255
        STD = np.array([63.0, 62.1, 66.7], dtype=np.float32)/255

        # Create test with known values
        image = MEANS.reshape(1, 1, 3)  # Image equal to means
        result = preprocess_cifar(image)
        # (means - means) / std = 0
        expected = np.zeros((1, 1, 3), dtype=np.float32)
        self.assertTrue(np.allclose(result, expected, atol=1e-6))


class TestPreprocessCifarEdgeCases(unittest.TestCase):
    """Edge case tests for preprocess_cifar."""

    def test_single_pixel(self):
        """Test preprocessing a single pixel."""
        image = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        result = preprocess_cifar(image)
        self.assertEqual(result.shape, (1, 1, 3))

    def test_different_dtypes(self):
        """Test preprocessing with different input dtypes."""
        image_float32 = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        image_float64 = np.ones((4, 4, 3), dtype=np.float64) * 0.5

        result32 = preprocess_cifar(image_float32)
        result64 = preprocess_cifar(image_float64)

        self.assertTrue(np.allclose(result32, result64, atol=1e-5))


class TestMakeEpsTensorEdgeCases(unittest.TestCase):
    """Additional edge case tests for make_eps_tensor."""

    def test_tensor_is_cloned(self):
        """Test that tensor input returns a cloned tensor."""
        t = torch.tensor([0.5, 0.6])
        result = make_eps_tensor(t)
        # Modify original - should not affect result
        t[0] = 1.0
        self.assertNotEqual(result[0].item(), 1.0)
        self.assertEqual(result[0].item(), 0.5)

    def test_tensor_is_detached(self):
        """Test that tensor with grad is detached."""
        t = torch.tensor([0.5, 0.6], requires_grad=True)
        result = make_eps_tensor(t)
        self.assertFalse(result.requires_grad)

    def test_zero_eps(self):
        """Test with zero epsilon value."""
        result = make_eps_tensor(0.0)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.item(), 0.0)

    def test_negative_eps(self):
        """Test with negative epsilon value."""
        result = make_eps_tensor(-0.1)
        self.assertIsInstance(result, torch.Tensor)
        self.assertAlmostEqual(result.item(), -0.1, places=6)

    def test_2d_array(self):
        """Test with 2D numpy array."""
        arr = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = make_eps_tensor(arr)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (2, 2))

    def test_very_small_eps(self):
        """Test with very small epsilon value."""
        result = make_eps_tensor(1e-10)
        self.assertIsInstance(result, torch.Tensor)
        self.assertAlmostEqual(result.item(), 1e-10)

    def test_empty_list(self):
        """Test with empty list."""
        result = make_eps_tensor([])
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (0,))


class TestLoadCifarSampleData(unittest.TestCase):
    """Tests for load_cifar_sample_data function."""

    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_load_cifar_sample_data_normalized(self, mock_print, mock_np_load):
        """Test loading CIFAR sample data with normalization."""
        from data_utils import load_cifar_sample_data

        # Create mock data
        mock_X = np.random.rand(10, 32, 32, 3).astype(np.float32)
        mock_y = np.arange(10).astype(np.int64)
        mock_runnerup = np.arange(10).astype(np.int64)

        mock_np_load.side_effect = [mock_X, mock_y, mock_runnerup]

        X, y, runnerup = load_cifar_sample_data(normalized=True, MODEL="test_model")

        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertIsInstance(runnerup, torch.Tensor)
        self.assertEqual(X.shape, (10, 3, 32, 32))  # Transposed
        self.assertEqual(y.shape, (10,))
        self.assertEqual(runnerup.shape, (10,))

    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_load_cifar_sample_data_unnormalized(self, mock_print, mock_np_load):
        """Test loading CIFAR sample data without normalization."""
        from data_utils import load_cifar_sample_data

        mock_X = np.random.rand(5, 32, 32, 3).astype(np.float32)
        mock_y = np.arange(5).astype(np.int64)
        mock_runnerup = np.arange(5).astype(np.int64)

        mock_np_load.side_effect = [mock_X, mock_y, mock_runnerup]

        X, y, runnerup = load_cifar_sample_data(normalized=False, MODEL="test_model")

        self.assertIsInstance(X, torch.Tensor)
        # Shape should be transposed
        self.assertEqual(X.shape, (5, 3, 32, 32))

    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_load_cifar_sample_data_print_messages(self, mock_print, mock_np_load):
        """Test that load_cifar_sample_data prints status messages."""
        from data_utils import load_cifar_sample_data

        mock_X = np.random.rand(2, 32, 32, 3).astype(np.float32)
        mock_y = np.array([0, 1]).astype(np.int64)
        mock_runnerup = np.array([1, 0]).astype(np.int64)

        mock_np_load.side_effect = [mock_X, mock_y, mock_runnerup]

        load_cifar_sample_data(normalized=True, MODEL="test")

        # Check print was called
        self.assertTrue(mock_print.called)


class TestLoadMnistSampleData(unittest.TestCase):
    """Tests for load_mnist_sample_data function."""

    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_load_mnist_sample_data_basic(self, mock_print, mock_np_load):
        """Test loading MNIST sample data."""
        from data_utils import load_mnist_sample_data

        mock_X = np.random.rand(10, 28, 28, 1).astype(np.float32)
        mock_y = np.arange(10).astype(np.int64)
        mock_runnerup = np.arange(10).astype(np.int64)

        mock_np_load.side_effect = [mock_X, mock_y, mock_runnerup]

        X, y, runnerup = load_mnist_sample_data(MODEL="test_model")

        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertIsInstance(runnerup, torch.Tensor)
        self.assertEqual(X.shape, (10, 1, 28, 28))  # Transposed
        self.assertEqual(y.shape, (10,))

    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_load_mnist_sample_data_with_custom_model(self, mock_print, mock_np_load):
        """Test loading MNIST sample data with custom model name."""
        from data_utils import load_mnist_sample_data

        mock_X = np.random.rand(5, 28, 28, 1).astype(np.float32)
        mock_y = np.arange(5).astype(np.int64)
        mock_runnerup = np.arange(5).astype(np.int64)

        mock_np_load.side_effect = [mock_X, mock_y, mock_runnerup]

        X, y, runnerup = load_mnist_sample_data(MODEL="custom_mnist")

        self.assertEqual(X.shape[0], 5)


class TestLoadDataset(unittest.TestCase):
    """Tests for load_dataset function."""

    @patch('data_utils.arguments')
    @patch('data_utils.datasets.MNIST')
    def test_load_dataset_mnist(self, mock_mnist_loader, mock_arguments):
        """Test loading MNIST dataset."""
        from data_utils import load_dataset

        mock_arguments.Config = {
            'data': {
                'dataset': 'MNIST',
                'mean': [0.1307],
                'std': [0.3081],
            }
        }

        mock_test_data = MagicMock()
        mock_mnist_loader.return_value = mock_test_data

        test_data, data_max, data_min = load_dataset()

        mock_mnist_loader.assert_called_once()
        self.assertIsNotNone(data_max)
        self.assertIsNotNone(data_min)

    @patch('data_utils.arguments')
    @patch('data_utils.datasets.CIFAR10')
    def test_load_dataset_cifar(self, mock_cifar_loader, mock_arguments):
        """Test loading CIFAR10 dataset."""
        from data_utils import load_dataset

        mock_arguments.Config = {
            'data': {
                'dataset': 'CIFAR',
                'mean': [0.4914, 0.4822, 0.4465],
                'std': [0.2023, 0.1994, 0.2010],
            }
        }

        mock_test_data = MagicMock()
        mock_cifar_loader.return_value = mock_test_data

        test_data, data_max, data_min = load_dataset()

        mock_cifar_loader.assert_called_once()
        # data_max and data_min should be tensors
        self.assertIsInstance(data_max, torch.Tensor)
        self.assertIsInstance(data_min, torch.Tensor)

    @patch('data_utils.arguments')
    @patch('data_utils.datasets.CIFAR100')
    def test_load_dataset_cifar100(self, mock_cifar100_loader, mock_arguments):
        """Test loading CIFAR100 dataset."""
        from data_utils import load_dataset

        mock_arguments.Config = {
            'data': {
                'dataset': 'CIFAR100',
                'mean': [0.5071, 0.4867, 0.4408],
                'std': [0.2675, 0.2565, 0.2761],
            }
        }

        mock_test_data = MagicMock()
        mock_cifar100_loader.return_value = mock_test_data

        test_data, data_max, data_min = load_dataset()

        mock_cifar100_loader.assert_called_once()

    @patch('data_utils.arguments')
    def test_load_dataset_unsupported(self, mock_arguments):
        """Test loading unsupported dataset raises ValueError."""
        from data_utils import load_dataset

        mock_arguments.Config = {
            'data': {
                'dataset': 'UNSUPPORTED_DATASET',
                'mean': [0.5],
                'std': [0.5],
            }
        }

        with self.assertRaises(ValueError) as context:
            load_dataset()

        self.assertIn('not supported', str(context.exception))

    @patch('data_utils.arguments')
    @patch('data_utils.datasets.MNIST')
    def test_load_dataset_sets_mean_std(self, mock_mnist_loader, mock_arguments):
        """Test that load_dataset sets mean and std on test_data."""
        from data_utils import load_dataset

        mock_arguments.Config = {
            'data': {
                'dataset': 'MNIST',
                'mean': [0.1307],
                'std': [0.3081],
            }
        }

        mock_test_data = MagicMock()
        mock_mnist_loader.return_value = mock_test_data

        test_data, _, _ = load_dataset()

        # Verify mean and std were set
        self.assertTrue(hasattr(test_data, 'mean'))
        self.assertTrue(hasattr(test_data, 'std'))


class TestLoadSampledDataset(unittest.TestCase):
    """Tests for load_sampled_dataset function."""

    @patch('data_utils.arguments')
    @patch('data_utils.load_cifar_sample_data')
    def test_load_sampled_dataset_cifar(self, mock_load_cifar, mock_arguments):
        """Test loading CIFAR sampled dataset."""
        from data_utils import load_sampled_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'CIFAR_SAMPLE'},
            'model': {'name': 'test_model'},
        }

        mock_X = torch.randn(10, 3, 32, 32)
        mock_labels = torch.arange(10)
        mock_runnerup = torch.arange(10)
        mock_load_cifar.return_value = (mock_X, mock_labels, mock_runnerup)

        spec = {'epsilon': None}
        X, labels, data_max, data_min, eps_temp, runnerup = load_sampled_dataset(spec)

        mock_load_cifar.assert_called_once_with(normalized=True, MODEL='test_model')
        self.assertEqual(X.shape, (10, 3, 32, 32))
        self.assertIsInstance(data_max, torch.Tensor)
        self.assertIsInstance(data_min, torch.Tensor)
        self.assertIsInstance(eps_temp, torch.Tensor)

    @patch('data_utils.arguments')
    @patch('data_utils.load_mnist_sample_data')
    def test_load_sampled_dataset_mnist(self, mock_load_mnist, mock_arguments):
        """Test loading MNIST sampled dataset."""
        from data_utils import load_sampled_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'MNIST_SAMPLE'},
            'model': {'name': 'mnist_model'},
        }

        mock_X = torch.randn(10, 1, 28, 28)
        mock_labels = torch.arange(10)
        mock_runnerup = torch.arange(10)
        mock_load_mnist.return_value = (mock_X, mock_labels, mock_runnerup)

        spec = {'epsilon': None}
        X, labels, data_max, data_min, eps_temp, runnerup = load_sampled_dataset(spec)

        mock_load_mnist.assert_called_once_with(MODEL='mnist_model')
        self.assertEqual(data_max.item(), 1.0)
        self.assertEqual(data_min.item(), 0.0)
        self.assertAlmostEqual(eps_temp.item(), 0.3)


class TestLoadSdpDataset(unittest.TestCase):
    """Tests for load_sdp_dataset function."""

    @patch('data_utils.arguments')
    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_load_sdp_dataset_cifar(self, mock_print, mock_np_load, mock_arguments):
        """Test loading CIFAR SDP dataset."""
        from data_utils import load_sdp_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'CIFAR_SDP'},
        }

        mock_X = np.random.rand(10, 32, 32, 3).astype(np.float32)
        mock_y = np.arange(10).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_y]

        spec = {'epsilon': None}
        X, y, data_max, data_min, eps_temp, runnerup = load_sdp_dataset(spec)

        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(X.shape, (10, 3, 32, 32))  # Transposed
        self.assertIsInstance(eps_temp, torch.Tensor)

    @patch('data_utils.arguments')
    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_load_sdp_dataset_mnist(self, mock_print, mock_np_load, mock_arguments):
        """Test loading MNIST SDP dataset."""
        from data_utils import load_sdp_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'MNIST_SDP'},
        }

        mock_X = np.random.rand(10, 28, 28, 1).astype(np.float32)
        mock_y = np.arange(10).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_y]

        spec = {'epsilon': None}
        X, y, data_max, data_min, eps_temp, runnerup = load_sdp_dataset(spec)

        self.assertIsInstance(X, torch.Tensor)
        self.assertEqual(X.shape, (10, 1, 28, 28))  # Transposed
        self.assertEqual(data_max.item(), 1.0)
        self.assertEqual(data_min.item(), 0.0)

    @patch('data_utils.arguments')
    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_load_sdp_dataset_cifar_with_custom_epsilon(self, mock_print, mock_np_load, mock_arguments):
        """Test loading CIFAR SDP dataset with custom epsilon."""
        from data_utils import load_sdp_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'CIFAR_SDP'},
        }

        mock_X = np.random.rand(5, 32, 32, 3).astype(np.float32)
        mock_y = np.arange(5).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_y]

        spec = {'epsilon': 0.01}
        X, y, data_max, data_min, eps_temp, runnerup = load_sdp_dataset(spec)

        # eps_temp should be reshaped to (1, -1, 1, 1)
        self.assertEqual(eps_temp.shape, (1, 3, 1, 1))

    @patch('data_utils.arguments')
    def test_load_sdp_dataset_unsupported(self, mock_arguments):
        """Test loading unsupported SDP dataset exits."""
        from data_utils import load_sdp_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'UNSUPPORTED_SDP'},
        }

        spec = {'epsilon': None}
        with self.assertRaises(SystemExit):
            load_sdp_dataset(spec)


class TestLoadGenericDataset(unittest.TestCase):
    """Tests for load_generic_dataset function."""

    @patch('data_utils.arguments')
    @patch('data_utils.load_dataset')
    @patch('builtins.print')
    def test_load_generic_dataset_basic(self, mock_print, mock_load_dataset, mock_arguments):
        """Test loading generic dataset."""
        from data_utils import load_generic_dataset

        mock_arguments.Config = {
            'data': {
                'std': [0.3081],
            }
        }

        mock_test_data = MagicMock()
        mock_test_data.__iter__ = MagicMock(return_value=iter([
            (torch.randn(100, 1, 28, 28), torch.arange(100))
        ]))
        mock_data_max = torch.tensor([[[[2.8]]]])
        mock_data_min = torch.tensor([[[[-0.4]]]])
        mock_load_dataset.return_value = (mock_test_data, mock_data_max, mock_data_min)

        # Create a mock DataLoader
        with patch('data_utils.torch.utils.data.DataLoader') as mock_dataloader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.__iter__ = MagicMock(return_value=iter([
                (torch.randn(100, 1, 28, 28), torch.arange(100))
            ]))
            mock_dataloader.return_value = mock_loader_instance

            spec = {'epsilon': 0.1}
            X, labels, data_max, data_min, eps_temp, runnerup = load_generic_dataset(spec)

            self.assertIsInstance(X, torch.Tensor)
            self.assertIsInstance(labels, torch.Tensor)
            self.assertIsNone(runnerup)

    @patch('data_utils.arguments')
    @patch('data_utils.load_dataset')
    @patch('builtins.print')
    def test_load_generic_dataset_none_epsilon_raises(self, mock_print, mock_load_dataset, mock_arguments):
        """Test that load_generic_dataset raises ValueError when epsilon is None."""
        from data_utils import load_generic_dataset

        mock_arguments.Config = {
            'data': {
                'std': [0.3081],
            }
        }

        mock_test_data = MagicMock()
        mock_data_max = torch.tensor([[[[2.8]]]])
        mock_data_min = torch.tensor([[[[-0.4]]]])
        mock_load_dataset.return_value = (mock_test_data, mock_data_max, mock_data_min)

        spec = {'epsilon': None}
        with self.assertRaises(ValueError) as context:
            load_generic_dataset(spec)

        self.assertIn('epsilon', str(context.exception))


class TestLoadEranDataset(unittest.TestCase):
    """Tests for load_eran_dataset function."""

    @patch('data_utils.arguments')
    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_load_eran_dataset_cifar(self, mock_print, mock_np_load, mock_arguments):
        """Test loading CIFAR ERAN dataset."""
        from data_utils import load_eran_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'CIFAR_ERAN'},
        }

        mock_X = np.random.rand(10, 3, 32, 32).astype(np.float32)
        mock_labels = np.arange(10).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_labels]

        spec = {'epsilon': None}
        X, labels, data_max, data_min, eps_temp, runnerup = load_eran_dataset(spec)

        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertIsInstance(data_max, torch.Tensor)
        self.assertIsInstance(data_min, torch.Tensor)
        self.assertIsInstance(eps_temp, torch.Tensor)
        # Default eps for CIFAR_ERAN is 2/255
        self.assertEqual(eps_temp.shape, (1, 3, 1, 1))

    @patch('data_utils.arguments')
    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_load_eran_dataset_mnist(self, mock_print, mock_np_load, mock_arguments):
        """Test loading MNIST ERAN dataset."""
        from data_utils import load_eran_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'MNIST_ERAN'},
        }

        mock_X = np.random.rand(10, 1, 28, 28).astype(np.float32)
        mock_labels = np.arange(10).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_labels]

        spec = {'epsilon': None}
        X, labels, data_max, data_min, eps_temp, runnerup = load_eran_dataset(spec)

        self.assertIsInstance(X, torch.Tensor)
        # Default eps for MNIST_ERAN is 0.3
        self.assertEqual(eps_temp.shape, (1, 1, 1, 1))

    @patch('data_utils.arguments')
    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_load_eran_dataset_mnist_unnormalized(self, mock_print, mock_np_load, mock_arguments):
        """Test loading MNIST ERAN unnormalized dataset."""
        from data_utils import load_eran_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'MNIST_ERAN_UN'},
        }

        mock_X = np.random.rand(10, 1, 28, 28).astype(np.float32)
        mock_labels = np.arange(10).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_labels]

        spec = {'epsilon': None}
        X, labels, data_max, data_min, eps_temp, runnerup = load_eran_dataset(spec)

        self.assertIsInstance(X, torch.Tensor)
        self.assertEqual(data_max.item(), 1.0)
        self.assertEqual(data_min.item(), 0.0)

    @patch('data_utils.arguments')
    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_load_eran_dataset_mnist_madry(self, mock_print, mock_np_load, mock_arguments):
        """Test loading MNIST MADRY unnormalized dataset."""
        from data_utils import load_eran_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'MNIST_MADRY_UN'},
        }

        # Note: MNIST_MADRY_UN expects flat images that get reshaped
        mock_X = np.random.rand(10, 784).astype(np.float32)
        mock_labels = np.arange(10).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_labels]

        spec = {'epsilon': None}
        X, labels, data_max, data_min, eps_temp, runnerup = load_eran_dataset(spec)

        self.assertIsInstance(X, torch.Tensor)
        self.assertEqual(X.shape, (10, 1, 28, 28))

    @patch('data_utils.arguments')
    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_load_eran_dataset_with_custom_epsilon(self, mock_print, mock_np_load, mock_arguments):
        """Test loading ERAN dataset with custom epsilon."""
        from data_utils import load_eran_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'MNIST_ERAN'},
        }

        mock_X = np.random.rand(5, 1, 28, 28).astype(np.float32)
        mock_labels = np.arange(5).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_labels]

        spec = {'epsilon': 0.1}
        X, labels, data_max, data_min, eps_temp, runnerup = load_eran_dataset(spec)

        # eps_temp should be scaled by std
        self.assertIsInstance(eps_temp, torch.Tensor)

    @patch('data_utils.arguments')
    def test_load_eran_dataset_unsupported(self, mock_arguments):
        """Test loading unsupported ERAN dataset raises."""
        from data_utils import load_eran_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'UNSUPPORTED_ERAN'},
        }

        spec = {'epsilon': None}
        with self.assertRaises(Exception):
            load_eran_dataset(spec)


class TestLoadPklDataset(unittest.TestCase):
    """Tests for load_pkl_dataset function."""

    @patch('data_utils.arguments')
    @patch('data_utils.pd.read_pickle')
    @patch('data_utils.load_dataset')
    def test_load_pkl_dataset_basic(self, mock_load_dataset, mock_read_pickle, mock_arguments):
        """Test loading PKL dataset."""
        from data_utils import load_pkl_dataset

        mock_arguments.Config = {
            'specification': {'epsilon': None},
            'data': {'pkl_path': '/path/to/data.pkl'},
        }

        # Mock PKL data
        mock_gt_results = MagicMock()
        mock_gt_results.__getitem__ = MagicMock(side_effect=lambda x: {
            'Idx': MagicMock(to_list=MagicMock(return_value=[0, 1, 2])),
            'prop': MagicMock(to_list=MagicMock(return_value=[1, 2, 3])),
            'Eps': MagicMock(to_list=MagicMock(return_value=[0.01, 0.02, 0.03])),
        }[x])
        mock_read_pickle.return_value = mock_gt_results

        # Mock test_data
        mock_test_data = [
            (torch.randn(1, 28, 28), torch.tensor(0)),
            (torch.randn(1, 28, 28), torch.tensor(1)),
            (torch.randn(1, 28, 28), torch.tensor(2)),
            (torch.randn(1, 28, 28), torch.tensor(3)),
            (torch.randn(1, 28, 28), torch.tensor(4)),
        ]
        mock_data_max = torch.tensor([[[[2.8]]]])
        mock_data_min = torch.tensor([[[[-0.4]]]])
        mock_load_dataset.return_value = (mock_test_data, mock_data_max, mock_data_min)

        spec = {}  # epsilon should come from pkl
        with patch('builtins.print'):
            result = load_pkl_dataset(spec)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 7)  # X, labels, data_max, data_min, eps_new, runnerup, target_label

    @patch('data_utils.arguments')
    def test_load_pkl_dataset_raises_with_epsilon(self, mock_arguments):
        """Test load_pkl_dataset raises assertion when epsilon is provided."""
        from data_utils import load_pkl_dataset

        mock_arguments.Config = {
            'specification': {'epsilon': 0.1},  # Non-None epsilon
            'data': {'pkl_path': '/path/to/data.pkl'},
        }

        spec = {}
        with self.assertRaises(AssertionError):
            load_pkl_dataset(spec)


class TestPreprocessCifarMathematical(unittest.TestCase):
    """Mathematical correctness tests for preprocess_cifar."""

    def test_preprocessing_formula_default(self):
        """Test that default preprocessing follows (image - means) / stds."""
        MEANS = np.array([125.3, 123.0, 113.9], dtype=np.float32) / 255
        STD = np.array([63.0, 62.1, 66.7], dtype=np.float32) / 255

        image = np.array([[[0.5, 0.6, 0.7]]], dtype=np.float32)
        result = preprocess_cifar(image)

        expected = (image - MEANS) / STD
        self.assertTrue(np.allclose(result, expected, atol=1e-6))

    def test_preprocessing_formula_inception(self):
        """Test that inception preprocessing follows (image - 0.5) / 0.5."""
        image = np.array([[[0.3, 0.4, 0.5]]], dtype=np.float32)
        result = preprocess_cifar(image, inception_preprocess=True)

        expected = (image - 0.5) / 0.5
        self.assertTrue(np.allclose(result, expected, atol=1e-6))

    def test_perturbation_formula_default(self):
        """Test that default perturbation follows image / stds."""
        STD = np.array([63.0, 62.1, 66.7], dtype=np.float32) / 255

        eps = np.array([0.01, 0.01, 0.01], dtype=np.float32)
        result = preprocess_cifar(eps, perturbation=True)

        expected = eps / STD
        self.assertTrue(np.allclose(result, expected, atol=1e-6))

    def test_perturbation_formula_inception(self):
        """Test that inception perturbation follows image / 0.5."""
        eps = 0.1
        result = preprocess_cifar(eps, inception_preprocess=True, perturbation=True)

        expected = eps / 0.5
        self.assertAlmostEqual(result, expected, places=6)


class TestDataMaxDataMinCalculation(unittest.TestCase):
    """Tests for data_max and data_min calculation in load_dataset."""

    @patch('data_utils.arguments')
    @patch('data_utils.datasets.MNIST')
    def test_data_max_data_min_shape(self, mock_mnist_loader, mock_arguments):
        """Test data_max and data_min have correct shape."""
        from data_utils import load_dataset

        mock_arguments.Config = {
            'data': {
                'dataset': 'MNIST',
                'mean': [0.1307],
                'std': [0.3081],
            }
        }

        mock_test_data = MagicMock()
        mock_mnist_loader.return_value = mock_test_data

        _, data_max, data_min = load_dataset()

        # Shape should be (1, -1, 1, 1) where -1 is number of channels
        self.assertEqual(data_max.shape, (1, 1, 1, 1))
        self.assertEqual(data_min.shape, (1, 1, 1, 1))

    @patch('data_utils.arguments')
    @patch('data_utils.datasets.CIFAR10')
    def test_data_max_data_min_shape_cifar(self, mock_cifar_loader, mock_arguments):
        """Test data_max and data_min have correct shape for CIFAR."""
        from data_utils import load_dataset

        mock_arguments.Config = {
            'data': {
                'dataset': 'CIFAR',
                'mean': [0.4914, 0.4822, 0.4465],
                'std': [0.2023, 0.1994, 0.2010],
            }
        }

        mock_test_data = MagicMock()
        mock_cifar_loader.return_value = mock_test_data

        _, data_max, data_min = load_dataset()

        # Shape should be (1, 3, 1, 1) for CIFAR with 3 channels
        self.assertEqual(data_max.shape, (1, 3, 1, 1))
        self.assertEqual(data_min.shape, (1, 3, 1, 1))

    @patch('data_utils.arguments')
    @patch('data_utils.datasets.MNIST')
    def test_data_max_formula(self, mock_mnist_loader, mock_arguments):
        """Test data_max is calculated as (1 - mean) / std."""
        from data_utils import load_dataset

        mean = 0.1307
        std = 0.3081
        mock_arguments.Config = {
            'data': {
                'dataset': 'MNIST',
                'mean': [mean],
                'std': [std],
            }
        }

        mock_test_data = MagicMock()
        mock_mnist_loader.return_value = mock_test_data

        _, data_max, _ = load_dataset()

        expected = (1.0 - mean) / std
        self.assertAlmostEqual(data_max.item(), expected, places=5)

    @patch('data_utils.arguments')
    @patch('data_utils.datasets.MNIST')
    def test_data_min_formula(self, mock_mnist_loader, mock_arguments):
        """Test data_min is calculated as (0 - mean) / std."""
        from data_utils import load_dataset

        mean = 0.1307
        std = 0.3081
        mock_arguments.Config = {
            'data': {
                'dataset': 'MNIST',
                'mean': [mean],
                'std': [std],
            }
        }

        mock_test_data = MagicMock()
        mock_mnist_loader.return_value = mock_test_data

        _, _, data_min = load_dataset()

        expected = (0.0 - mean) / std
        self.assertAlmostEqual(data_min.item(), expected, places=5)


class TestLoadSampledDatasetEpsilon(unittest.TestCase):
    """Tests for epsilon handling in load_sampled_dataset."""

    @patch('data_utils.arguments')
    @patch('data_utils.load_cifar_sample_data')
    def test_cifar_sample_epsilon_shape(self, mock_load_cifar, mock_arguments):
        """Test CIFAR sample epsilon has correct shape."""
        from data_utils import load_sampled_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'CIFAR_SAMPLE'},
            'model': {'name': 'test_model'},
        }

        mock_X = torch.randn(10, 3, 32, 32)
        mock_labels = torch.arange(10)
        mock_runnerup = torch.arange(10)
        mock_load_cifar.return_value = (mock_X, mock_labels, mock_runnerup)

        spec = {'epsilon': None}
        _, _, _, _, eps_temp, _ = load_sampled_dataset(spec)

        # eps_temp for CIFAR_SAMPLE should be (1, 3, 1, 1)
        self.assertEqual(eps_temp.shape, (1, 3, 1, 1))

    @patch('data_utils.arguments')
    @patch('data_utils.load_mnist_sample_data')
    def test_mnist_sample_epsilon_shape(self, mock_load_mnist, mock_arguments):
        """Test MNIST sample epsilon has correct shape."""
        from data_utils import load_sampled_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'MNIST_SAMPLE'},
            'model': {'name': 'mnist_model'},
        }

        mock_X = torch.randn(10, 1, 28, 28)
        mock_labels = torch.arange(10)
        mock_runnerup = torch.arange(10)
        mock_load_mnist.return_value = (mock_X, mock_labels, mock_runnerup)

        spec = {'epsilon': None}
        _, _, _, _, eps_temp, _ = load_sampled_dataset(spec)

        # eps_temp for MNIST_SAMPLE should be (1, 1, 1, 1)
        self.assertEqual(eps_temp.shape, (1, 1, 1, 1))


class TestLoadSdpDatasetEpsilon(unittest.TestCase):
    """Tests for epsilon handling in load_sdp_dataset."""

    @patch('data_utils.arguments')
    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_sdp_cifar_default_epsilon(self, mock_print, mock_np_load, mock_arguments):
        """Test CIFAR SDP default epsilon is 2/255."""
        from data_utils import load_sdp_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'CIFAR_SDP'},
        }

        mock_X = np.random.rand(5, 32, 32, 3).astype(np.float32)
        mock_y = np.arange(5).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_y]

        spec = {'epsilon': None}
        _, _, _, _, eps_temp, _ = load_sdp_dataset(spec)

        # Default epsilon for CIFAR_SDP is 2/255 then preprocessed
        self.assertIsInstance(eps_temp, torch.Tensor)
        self.assertEqual(eps_temp.shape, (1, 3, 1, 1))

    @patch('data_utils.arguments')
    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_sdp_mnist_default_epsilon(self, mock_print, mock_np_load, mock_arguments):
        """Test MNIST SDP default epsilon is 0.3."""
        from data_utils import load_sdp_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'MNIST_SDP'},
        }

        mock_X = np.random.rand(5, 28, 28, 1).astype(np.float32)
        mock_y = np.arange(5).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_y]

        spec = {'epsilon': None}
        _, _, _, _, eps_temp, _ = load_sdp_dataset(spec)

        self.assertAlmostEqual(eps_temp.item(), 0.3)


class TestLoadEranDatasetNormalization(unittest.TestCase):
    """Tests for normalization in load_eran_dataset."""

    @patch('data_utils.arguments')
    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_cifar_eran_normalization_values(self, mock_print, mock_np_load, mock_arguments):
        """Test CIFAR ERAN uses correct mean/std values."""
        from data_utils import load_eran_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'CIFAR_ERAN'},
        }

        # Create known input to verify normalization
        mock_X = np.ones((1, 3, 32, 32)).astype(np.float32) * 0.5
        mock_labels = np.array([0]).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_labels]

        spec = {'epsilon': 0.01}
        X, _, data_max, data_min, eps_temp, _ = load_eran_dataset(spec)

        # CIFAR_ERAN mean: [0.4914, 0.4822, 0.4465]
        # CIFAR_ERAN std: [0.2023, 0.1994, 0.201]
        # Verify data is normalized
        self.assertIsInstance(X, torch.Tensor)

    @patch('data_utils.arguments')
    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_mnist_eran_normalization_values(self, mock_print, mock_np_load, mock_arguments):
        """Test MNIST ERAN uses correct mean/std values."""
        from data_utils import load_eran_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'MNIST_ERAN'},
        }

        mock_X = np.ones((1, 1, 28, 28)).astype(np.float32) * 0.5
        mock_labels = np.array([0]).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_labels]

        spec = {'epsilon': 0.1}
        X, _, data_max, data_min, eps_temp, _ = load_eran_dataset(spec)

        # MNIST_ERAN mean: 0.1307, std: 0.3081
        # eps scaled by std: 0.1 / 0.3081
        expected_eps = 0.1 / 0.3081
        self.assertAlmostEqual(eps_temp.item(), expected_eps, places=4)


class TestRunnerupHandling(unittest.TestCase):
    """Tests for runnerup handling across different load functions."""

    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_cifar_sample_runnerup(self, mock_print, mock_np_load):
        """Test runnerup is loaded correctly for CIFAR sample data."""
        from data_utils import load_cifar_sample_data

        mock_X = np.random.rand(5, 32, 32, 3).astype(np.float32)
        mock_y = np.array([0, 1, 2, 3, 4]).astype(np.int64)
        mock_runnerup = np.array([1, 2, 3, 4, 0]).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_y, mock_runnerup]

        X, y, runnerup = load_cifar_sample_data(normalized=False, MODEL="test")

        self.assertTrue(torch.equal(runnerup, torch.tensor([1, 2, 3, 4, 0])))

    @patch('data_utils.arguments')
    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_sdp_runnerup_is_copy_of_y(self, mock_print, mock_np_load, mock_arguments):
        """Test SDP datasets set runnerup as copy of y."""
        from data_utils import load_sdp_dataset

        mock_arguments.Config = {
            'data': {'dataset': 'MNIST_SDP'},
        }

        mock_X = np.random.rand(3, 28, 28, 1).astype(np.float32)
        mock_y = np.array([0, 5, 9]).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_y]

        spec = {'epsilon': 0.1}
        _, y, _, _, _, runnerup = load_sdp_dataset(spec)

        # runnerup should equal y for SDP datasets
        self.assertTrue(torch.equal(y, runnerup))


class TestTensorTypes(unittest.TestCase):
    """Tests for correct tensor types in output."""

    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_cifar_sample_data_types(self, mock_print, mock_np_load):
        """Test CIFAR sample data returns correct tensor types."""
        from data_utils import load_cifar_sample_data

        mock_X = np.random.rand(3, 32, 32, 3).astype(np.float32)
        mock_y = np.array([0, 1, 2]).astype(np.int64)
        mock_runnerup = np.array([1, 2, 0]).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_y, mock_runnerup]

        X, y, runnerup = load_cifar_sample_data(normalized=False, MODEL="test")

        self.assertEqual(X.dtype, torch.float32)
        # y and runnerup should be int tensors
        self.assertTrue(y.dtype in [torch.int32, torch.int64])
        self.assertTrue(runnerup.dtype in [torch.int32, torch.int64])

    @patch('data_utils.np.load')
    @patch('builtins.print')
    def test_mnist_sample_data_types(self, mock_print, mock_np_load):
        """Test MNIST sample data returns correct tensor types."""
        from data_utils import load_mnist_sample_data

        mock_X = np.random.rand(3, 28, 28, 1).astype(np.float32)
        mock_y = np.array([0, 1, 2]).astype(np.int64)
        mock_runnerup = np.array([1, 2, 0]).astype(np.int64)
        mock_np_load.side_effect = [mock_X, mock_y, mock_runnerup]

        X, y, runnerup = load_mnist_sample_data(MODEL="test")

        self.assertEqual(X.dtype, torch.float32)
        self.assertTrue(y.dtype in [torch.int32, torch.int64])


if __name__ == '__main__':
    unittest.main()
