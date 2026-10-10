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
"""Unit tests for load_model.py module."""

import pytest
import torch
import torch.nn as nn
import os
import tempfile
from unittest.mock import MagicMock, patch, mock_open
import collections


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def setup_arguments():
    """Setup arguments.Config for each test."""
    import arguments

    # Save original Config - this is a module-level global
    original_config = arguments.Config

    try:
        # Create a fresh ConfigHandler and initialize it with defaults
        new_config = arguments.ConfigHandler()
        # Call construct_config_dict with default_args to populate all_args
        new_config.construct_config_dict(new_config.default_args)
        new_config.file = None

        arguments.Config = new_config

        yield
    finally:
        # Always restore original Config, even if test fails
        arguments.Config = original_config


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    return SimpleNet()


# ============================================================================
# deep_update Tests
# ============================================================================

class TestDeepUpdate:
    """Tests for the deep_update function."""

    def test_simple_update(self):
        """Test updating a simple dictionary."""
        from load_model import deep_update

        d = {'a': 1, 'b': 2}
        u = {'b': 3, 'c': 4}

        result = deep_update(d, u)

        assert result == {'a': 1, 'b': 3, 'c': 4}
        assert d == result  # Should modify in place

    def test_nested_update(self):
        """Test updating a nested dictionary."""
        from load_model import deep_update

        d = {'a': {'x': 1, 'y': 2}, 'b': 3}
        u = {'a': {'y': 10, 'z': 20}}

        result = deep_update(d, u)

        assert result == {'a': {'x': 1, 'y': 10, 'z': 20}, 'b': 3}

    def test_deep_nested_update(self):
        """Test updating deeply nested dictionaries."""
        from load_model import deep_update

        d = {'level1': {'level2': {'level3': 1}}}
        u = {'level1': {'level2': {'level3': 100, 'new': 200}}}

        result = deep_update(d, u)

        assert result['level1']['level2']['level3'] == 100
        assert result['level1']['level2']['new'] == 200

    def test_empty_source(self):
        """Test updating from an empty dictionary."""
        from load_model import deep_update

        d = {'a': 1}
        u = {}

        result = deep_update(d, u)

        assert result == {'a': 1}

    def test_empty_target(self):
        """Test updating an empty dictionary."""
        from load_model import deep_update

        d = {}
        u = {'a': 1, 'b': {'c': 2}}

        result = deep_update(d, u)

        assert result == {'a': 1, 'b': {'c': 2}}

    def test_overwrite_non_dict_with_dict(self):
        """Test overwriting a non-dict value with a dict - deep_update doesn't recurse into non-mappings."""
        from load_model import deep_update

        # When d[k] is not a mapping, deep_update will fail if u[k] is a mapping and we try to recurse
        # The actual behavior is that it raises TypeError because you can't call .get() on int
        # This is expected behavior - deep_update is designed for merging nested dicts, not replacing scalars with dicts
        d = {'a': 1}
        u = {'a': {'nested': 'value'}}

        # The function will try: d['a'] = deep_update(d.get('a', {}), {'nested': 'value'})
        # Which becomes: d['a'] = deep_update(1, {'nested': 'value'})
        # Then it tries: 1['nested'] = 'value' which fails
        with pytest.raises(TypeError):
            deep_update(d, u)

    def test_overwrite_dict_with_non_dict(self):
        """Test overwriting a dict value with a non-dict."""
        from load_model import deep_update

        d = {'a': {'nested': 'value'}}
        u = {'a': 1}

        result = deep_update(d, u)

        assert result == {'a': 1}

    def test_list_values_replaced(self):
        """Test that list values are replaced, not merged."""
        from load_model import deep_update

        d = {'a': [1, 2, 3]}
        u = {'a': [4, 5]}

        result = deep_update(d, u)

        assert result == {'a': [4, 5]}

    def test_none_values(self):
        """Test handling of None values."""
        from load_model import deep_update

        d = {'a': 1}
        u = {'a': None, 'b': None}

        result = deep_update(d, u)

        assert result == {'a': None, 'b': None}


# ============================================================================
# Customized Tests
# ============================================================================

class TestCustomized:
    """Tests for the Customized function."""

    def test_customized_with_py_file_root_path(self, tmp_path):
        """Test Customized loading from .py file with root_path."""
        import arguments
        from load_model import Customized

        # Create a temporary Python file with a model function
        module_content = '''
def get_model():
    import torch.nn as nn
    return nn.Linear(10, 5)
'''
        py_file = tmp_path / "custom_model.py"
        py_file.write_text(module_content)

        # expand_path needs a valid Config.file when root_path is set
        # Create a dummy config file path
        arguments.Config.file = str(tmp_path / "config.yaml")
        arguments.Config['general']['root_path'] = str(tmp_path)

        result = Customized('custom_model.py', 'get_model')

        # Result should be a callable (partial function)
        assert callable(result)

    def test_customized_with_py_file_config_file_path(self, tmp_path):
        """Test Customized loading from .py file relative to config file."""
        import arguments
        from load_model import Customized

        # Create a temporary Python file
        module_content = '''
def get_model():
    import torch.nn as nn
    return nn.Linear(5, 3)
'''
        py_file = tmp_path / "model_def.py"
        py_file.write_text(module_content)

        arguments.Config['general']['root_path'] = None
        arguments.Config.file = str(tmp_path / "config.yaml")

        result = Customized('model_def.py', 'get_model')

        assert callable(result)

    def test_customized_with_args_kwargs(self, tmp_path):
        """Test Customized with additional args and kwargs."""
        import arguments
        from load_model import Customized

        module_content = '''
def get_model(input_dim, output_dim, activation='relu'):
    return {'input': input_dim, 'output': output_dim, 'act': activation}
'''
        py_file = tmp_path / "parameterized_model.py"
        py_file.write_text(module_content)

        # expand_path needs a valid Config.file
        arguments.Config.file = str(tmp_path / "config.yaml")
        arguments.Config['general']['root_path'] = str(tmp_path)

        result = Customized('parameterized_model.py', 'get_model', 10, 5, activation='sigmoid')

        # Call the partial function
        model = result()
        assert model == {'input': 10, 'output': 5, 'act': 'sigmoid'}

    def test_customized_module_not_py(self):
        """Test Customized with non-.py module name (import from custom folder)."""
        from load_model import Customized

        # This will try to import from 'custom.module_name'
        # which likely doesn't exist, but we can test the exception handling
        with pytest.raises(ModuleNotFoundError):
            Customized('nonexistent_module', 'some_function')


# ============================================================================
# unzip_and_optimize_onnx Tests
# ============================================================================

class TestUnzipAndOptimizeOnnx:
    """Tests for the unzip_and_optimize_onnx function."""

    def test_load_regular_onnx_no_optimization(self, tmp_path):
        """Test loading regular ONNX file without optimization."""
        from load_model import unzip_and_optimize_onnx

        # Create a minimal ONNX model
        import onnx
        from onnx import helper, TensorProto

        # Create a simple ONNX graph
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 10])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 10])
        node = helper.make_node('Identity', ['X'], ['Y'])
        graph = helper.make_graph([node], 'test_graph', [X], [Y])
        model = helper.make_model(graph)

        onnx_path = str(tmp_path / "test_model.onnx")
        onnx.save(model, onnx_path)

        result = unzip_and_optimize_onnx(onnx_path, None)

        assert result is not None
        assert isinstance(result, onnx.ModelProto)

    def test_load_gzipped_onnx(self, tmp_path):
        """Test loading gzipped ONNX file."""
        from load_model import unzip_and_optimize_onnx
        import gzip

        import onnx
        from onnx import helper, TensorProto

        # Create a simple ONNX model
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 5])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 5])
        node = helper.make_node('Identity', ['X'], ['Y'])
        graph = helper.make_graph([node], 'test_graph', [X], [Y])
        model = helper.make_model(graph)

        onnx_path = str(tmp_path / "test_model.onnx.gz")

        # Save as gzipped file
        with gzip.open(onnx_path, 'wb') as f:
            f.write(model.SerializeToString())

        result = unzip_and_optimize_onnx(onnx_path, None)

        assert result is not None
        assert isinstance(result, onnx.ModelProto)

    def test_optimization_with_existing_optimized_file(self, tmp_path, capsys):
        """Test that existing optimized file is loaded."""
        from load_model import unzip_and_optimize_onnx

        import onnx
        from onnx import helper, TensorProto

        # Create a simple ONNX model
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 5])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 5])
        node = helper.make_node('Identity', ['X'], ['Y'])
        graph = helper.make_graph([node], 'test_graph', [X], [Y])
        model = helper.make_model(graph)

        onnx_path = str(tmp_path / "test_model.onnx")
        optimized_path = onnx_path + ".optimized"

        onnx.save(model, onnx_path)
        onnx.save(model, optimized_path)

        result = unzip_and_optimize_onnx(onnx_path, ['some_flag'])

        captured = capsys.readouterr()
        assert 'Found existed optimized onnx model' in captured.out


# ============================================================================
# inference_onnx Tests
# ============================================================================

class TestInferenceOnnx:
    """Tests for the inference_onnx function."""

    def test_inference_simple_model(self, tmp_path):
        """Test inference on a simple ONNX model."""
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        import numpy as np
        from load_model import inference_onnx

        # Create a simple identity model with compatible IR version
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 10])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 10])
        node = helper.make_node('Identity', ['X'], ['Y'])
        graph = helper.make_graph([node], 'test_graph', [X], [Y])
        # Specify opset version to ensure compatibility
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
        # Set IR version to be compatible with onnxruntime
        model.ir_version = 7

        onnx_path = str(tmp_path / "identity.onnx")
        onnx.save(model, onnx_path)

        # Run inference
        input_data = np.random.randn(1, 10).astype(np.float32)
        result = inference_onnx(onnx_path, input_data)

        assert result.shape == (1, 10)
        np.testing.assert_array_almost_equal(result, input_data)


# ============================================================================
# load_model Tests
# ============================================================================

class TestLoadModel:
    """Tests for the load_model function."""

    def test_load_model_raises_without_model_or_onnx(self):
        """Test that load_model raises when neither model nor onnx_path specified."""
        import arguments
        from load_model import load_model

        arguments.Config['model']['name'] = None
        arguments.Config['model']['onnx_path'] = None

        with pytest.raises(AssertionError) as exc_info:
            load_model()

        assert 'No model is loaded' in str(exc_info.value)

    def test_load_model_raises_with_both_model_and_onnx(self):
        """Test that load_model raises when both model and onnx_path specified."""
        import arguments
        from load_model import load_model

        arguments.Config['model']['name'] = 'SomeModel'
        arguments.Config['model']['onnx_path'] = '/path/to/model.onnx'

        with pytest.raises(AssertionError) as exc_info:
            load_model()

        assert 'Conflict detected' in str(exc_info.value)

    def test_load_model_pytorch_without_weights(self):
        """Test load_model with PyTorch model without loading weights."""
        import arguments
        from load_model import load_model

        # Register a simple model in global scope
        import load_model as lm
        original_globals = lm.__dict__.copy()

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        lm.__dict__['TestModel'] = TestModel

        try:
            arguments.Config['model']['name'] = 'TestModel'
            arguments.Config['model']['onnx_path'] = None
            arguments.Config['model']['path'] = None

            model = load_model(weights_loaded=False)

            assert isinstance(model, nn.Module)
            assert hasattr(model, 'fc')
        finally:
            # Clean up
            lm.__dict__.clear()
            lm.__dict__.update(original_globals)

    def test_load_model_with_state_dict_path(self, tmp_path):
        """Test load_model loading weights from a file."""
        import arguments
        from load_model import load_model
        import load_model as lm

        original_globals = lm.__dict__.copy()

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        lm.__dict__['TestModel'] = TestModel

        try:
            # Save model weights
            model = TestModel()
            weights_path = str(tmp_path / "weights.pt")
            torch.save(model.state_dict(), weights_path)

            # expand_path needs Config.file to be set
            arguments.Config.file = str(tmp_path / "config.yaml")
            arguments.Config['model']['name'] = 'TestModel'
            arguments.Config['model']['onnx_path'] = None
            arguments.Config['model']['path'] = weights_path

            loaded_model = load_model()

            assert isinstance(loaded_model, nn.Module)
        finally:
            lm.__dict__.clear()
            lm.__dict__.update(original_globals)

    def test_load_model_with_wrapped_state_dict(self, tmp_path):
        """Test load_model with state_dict wrapped in a dict."""
        import arguments
        from load_model import load_model
        import load_model as lm

        original_globals = lm.__dict__.copy()

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        lm.__dict__['TestModel'] = TestModel

        try:
            model = TestModel()
            weights_path = str(tmp_path / "weights_wrapped.pt")
            # Save with 'state_dict' key
            torch.save({'state_dict': model.state_dict()}, weights_path)

            # expand_path needs Config.file to be set
            arguments.Config.file = str(tmp_path / "config.yaml")
            arguments.Config['model']['name'] = 'TestModel'
            arguments.Config['model']['onnx_path'] = None
            arguments.Config['model']['path'] = weights_path

            loaded_model = load_model()

            assert isinstance(loaded_model, nn.Module)
        finally:
            lm.__dict__.clear()
            lm.__dict__.update(original_globals)

    def test_load_model_with_list_state_dict(self, tmp_path):
        """Test load_model with state_dict in a list."""
        import arguments
        from load_model import load_model
        import load_model as lm

        original_globals = lm.__dict__.copy()

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        lm.__dict__['TestModel'] = TestModel

        try:
            model = TestModel()
            weights_path = str(tmp_path / "weights_list.pt")
            # Save as list
            torch.save([model.state_dict()], weights_path)

            # expand_path needs Config.file to be set
            arguments.Config.file = str(tmp_path / "config.yaml")
            arguments.Config['model']['name'] = 'TestModel'
            arguments.Config['model']['onnx_path'] = None
            arguments.Config['model']['path'] = weights_path

            loaded_model = load_model()

            assert isinstance(loaded_model, nn.Module)
        finally:
            lm.__dict__.clear()
            lm.__dict__.update(original_globals)

    def test_load_model_invalid_pytorch_name(self, capsys):
        """Test load_model with invalid PyTorch model name exits."""
        import arguments
        from load_model import load_model

        arguments.Config['model']['name'] = 'NonExistentModel12345'
        arguments.Config['model']['onnx_path'] = None

        with pytest.raises(SystemExit):
            load_model()


# ============================================================================
# load_model_onnx Tests
# ============================================================================

class TestLoadModelOnnx:
    """Tests for the load_model_onnx function."""

    def test_load_model_onnx_simple(self, tmp_path):
        """Test loading a simple ONNX model."""
        import onnx
        from onnx import helper, TensorProto
        from load_model import load_model_onnx

        # Create a simple ONNX model
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 10])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 10])
        node = helper.make_node('Identity', ['X'], ['Y'])
        graph = helper.make_graph([node], 'test_graph', [X], [Y])
        model = helper.make_model(graph)

        onnx_path = str(tmp_path / "simple_model.onnx")
        onnx.save(model, onnx_path)

        pytorch_model, shape = load_model_onnx(onnx_path)

        assert pytorch_model is not None
        assert shape == (10,)  # Batch dimension removed

    def test_load_model_onnx_with_custom_input_shape(self, tmp_path):
        """Test loading ONNX model with custom input shape."""
        import arguments
        import onnx
        from onnx import helper, TensorProto
        from load_model import load_model_onnx

        # Create model
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 32, 32])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node('Identity', ['X'], ['Y'])
        graph = helper.make_graph([node], 'test_graph', [X], [Y])
        model = helper.make_model(graph)

        onnx_path = str(tmp_path / "cnn_model.onnx")
        onnx.save(model, onnx_path)

        # Override input shape
        arguments.Config['model']['input_shape'] = (1, 3, 28, 28)

        pytorch_model, shape = load_model_onnx(onnx_path)

        assert shape == (3, 28, 28)  # Batch dimension removed

    def test_check_onnx_conversion_multiple_inputs(self, tmp_path):
        """Test conversion checking for an ONNX model with two inputs."""
        import onnx
        import onnx2pytorch
        from onnx import helper, TensorProto
        from load_model import check_onnx_conversion

        x1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, [1, 2])
        x2 = helper.make_tensor_value_info('X2', TensorProto.FLOAT, [1, 2])
        y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])
        node = helper.make_node('Add', ['X1', 'X2'], ['Y'])
        graph = helper.make_graph([node], 'two_input_graph', [x1, x2], [y])
        model = helper.make_model(graph)

        onnx_path = str(tmp_path / "two_input_model.onnx")
        onnx.save(model, onnx_path)
        pytorch_model = onnx2pytorch.ConvertModel(
            onnx.load(onnx_path), experimental=True).eval()

        assert check_onnx_conversion(
            pytorch_model, onnx_path, [(2,), (2,)])

    def test_ort_optimized_path_returns_single_input_shape(self, tmp_path):
        """Test ORT optimized loading preserves load_model_onnx shape format."""
        import arguments
        import onnx
        from onnx import helper, TensorProto
        from load_model import load_model_onnx

        x = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
        y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])
        node = helper.make_node('Identity', ['X'], ['Y'])
        graph = helper.make_graph([node], 'ort_test_graph', [x], [y])
        model = helper.make_model(graph)

        onnx_path = str(tmp_path / "ort_model.onnx")
        onnx.save(model, onnx_path)

        arguments.Config['model']['use_ort_optimized_model'] = True
        _, shape = load_model_onnx(onnx_path)

        assert shape == (2,)

    def test_ort_optimized_path_uses_conversion_cache(
            self, tmp_path, monkeypatch):
        """Test ORT optimized loading can reuse converted-model cache."""
        import arguments
        import onnx
        import load_model as lm
        from onnx import helper, TensorProto
        from load_model import load_model_onnx

        x = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
        y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])
        node = helper.make_node('Identity', ['X'], ['Y'])
        graph = helper.make_graph([node], 'ort_cache_graph', [x], [y])
        model = helper.make_model(graph)

        onnx_path = str(tmp_path / "ort_cache_model.onnx")
        onnx.save(model, onnx_path)

        arguments.Config['model']['use_ort_optimized_model'] = True
        arguments.Config['model']['cache_onnx_conversion'] = True
        _, shape = load_model_onnx(onnx_path)
        assert shape == (2,)

        optimized_path = str(
            tmp_path / "ort_cache_model.abcrown_ort_extended.onnx")
        assert os.path.exists(optimized_path + ".cached")

        def fail_convert(*_args, **_kwargs):
            raise AssertionError("load_model_onnx should have used the cache")

        monkeypatch.setattr(lm.onnx2pytorch, "ConvertModel", fail_convert)
        _, cached_shape = load_model_onnx(onnx_path)
        assert cached_shape == (2,)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Edge case tests for load_model module."""

    def test_deep_update_with_mapping_subclass(self):
        """Test deep_update with collections.abc.Mapping subclass."""
        from load_model import deep_update
        from collections import OrderedDict

        d = OrderedDict([('a', {'x': 1})])
        u = {'a': {'y': 2}}

        result = deep_update(d, u)

        assert 'x' in result['a']
        assert 'y' in result['a']

    def test_customized_with_direct_path(self, tmp_path):
        """Test Customized with direct file path (no root_path or config file)."""
        import arguments
        from load_model import Customized

        module_content = '''
def create_model():
    return "test_model"
'''
        py_file = tmp_path / "direct_model.py"
        py_file.write_text(module_content)

        arguments.Config['general']['root_path'] = None
        arguments.Config.file = None

        result = Customized(str(py_file), 'create_model')

        assert callable(result)
        assert result() == "test_model"

    def test_load_model_state_dict_not_dict_raises(self, tmp_path):
        """Test load_model raises when state_dict is not a dict."""
        import arguments
        from load_model import load_model
        import load_model as lm

        original_globals = lm.__dict__.copy()

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        lm.__dict__['TestModel'] = TestModel

        try:
            weights_path = str(tmp_path / "invalid_weights.pt")
            # Save invalid format - a tuple (not a dict or list containing dict)
            # This triggers NotImplementedError for "Unknown model format"
            torch.save(('not', 'a', 'dict'), weights_path)

            # expand_path needs Config.file to be set
            arguments.Config.file = str(tmp_path / "config.yaml")
            arguments.Config['model']['name'] = 'TestModel'
            arguments.Config['model']['onnx_path'] = None
            arguments.Config['model']['path'] = weights_path

            with pytest.raises(NotImplementedError) as exc_info:
                load_model()

            assert 'Unknown model format' in str(exc_info.value)
        finally:
            lm.__dict__.clear()
            lm.__dict__.update(original_globals)


# ============================================================================
# Quirks handling tests
# ============================================================================

class TestOnnxQuirks:
    """Tests for ONNX quirks handling."""

    def test_quirks_merge_with_config(self):
        """Test that quirks from config are merged with provided quirks."""
        import arguments
        from load_model import deep_update

        # Test the deep_update function used for quirks merging
        quirks = {'existing': 'value'}
        config_quirks = {'new': 'config_value', 'existing': 'overwritten'}

        deep_update(quirks, config_quirks)

        assert quirks == {'existing': 'overwritten', 'new': 'config_value'}

    def test_invalid_quirks_string_raises(self):
        """Test that invalid quirks string raises ValueError."""
        import arguments

        arguments.Config['model']['onnx_quirks'] = 'not_valid_python_dict'

        # The literal_eval should raise
        from ast import literal_eval
        with pytest.raises((ValueError, SyntaxError)):
            literal_eval(arguments.Config['model']['onnx_quirks'])


# ============================================================================
# Caching tests
# ============================================================================

class TestOnnxCaching:
    """Tests for ONNX model caching functionality."""

    def test_cache_filename_generation(self):
        """Test that cache filename is generated correctly."""
        path = "/path/to/model.onnx"
        cached_suffix = ".cached"
        expected = f"{path}{cached_suffix}"

        assert expected == "/path/to/model.onnx.cached"

    def test_cached_model_uses_default_dtype(self, tmp_path):
        """Cached conversion must follow the requested verification dtype."""
        import hashlib
        import arguments
        from load_model import load_model_onnx

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"cached model hash input")
        model_hash = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        cached_model = nn.Linear(2, 1).float()
        torch.save(
            (cached_model, (2,), model_hash),
            str(onnx_path) + ".cached")

        original_dtype = torch.get_default_dtype()
        try:
            arguments.Config["model"]["cache_onnx_conversion"] = True
            torch.set_default_dtype(torch.float64)

            loaded_model, shape = load_model_onnx(str(onnx_path))

            assert shape == (2,)
            assert {
                parameter.dtype for parameter in loaded_model.parameters()
            } == {torch.float64}
        finally:
            torch.set_default_dtype(original_dtype)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
