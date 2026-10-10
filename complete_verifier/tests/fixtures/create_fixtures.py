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
"""Script to create test fixtures (ONNX models and VNNLIB files)."""
import os
import torch
import torch.nn as nn
import numpy as np

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


def create_simple_mlp(input_dim=4, hidden_dim=8, output_dim=2):
    """Create a simple 2-layer MLP model."""
    class SimpleMLP(nn.Module):
        def __init__(self, in_dim, hid_dim, out_dim):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hid_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hid_dim, out_dim)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model = SimpleMLP(input_dim, hidden_dim, output_dim)
    # Initialize with deterministic weights for reproducibility
    torch.manual_seed(42)
    nn.init.xavier_uniform_(model.fc1.weight)
    nn.init.zeros_(model.fc1.bias)
    nn.init.xavier_uniform_(model.fc2.weight)
    nn.init.zeros_(model.fc2.bias)
    return model


def create_simple_cnn():
    """Create a simple CNN model."""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d((2, 2))
            self.fc = nn.Linear(16, 2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = SimpleCNN()
    torch.manual_seed(42)
    return model


def export_to_onnx(model, input_shape, filename):
    """Export PyTorch model to ONNX."""
    model.eval()
    dummy_input = torch.randn(1, *input_shape)
    filepath = os.path.join(FIXTURES_DIR, filename)
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported {filename}")
    return filepath


def create_vnnlib_robustness(input_dim, output_dim, eps, filename):
    """Create a simple robustness VNNLIB specification."""
    filepath = os.path.join(FIXTURES_DIR, filename)

    with open(filepath, 'w') as f:
        # Declare input variables
        for i in range(input_dim):
            f.write(f"(declare-const X_{i} Real)\n")

        # Declare output variables
        for i in range(output_dim):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write("\n; Input constraints (box around origin)\n")
        for i in range(input_dim):
            f.write(f"(assert (>= X_{i} {-eps}))\n")
            f.write(f"(assert (<= X_{i} {eps}))\n")

        f.write("\n; Output constraint (Y_0 <= Y_1)\n")
        f.write("(assert (<= Y_0 Y_1))\n")

    print(f"Created {filename}")
    return filepath


def create_vnnlib_targeted(input_dim, output_dim, eps, target_class, filename):
    """Create a targeted attack VNNLIB specification."""
    filepath = os.path.join(FIXTURES_DIR, filename)

    with open(filepath, 'w') as f:
        # Declare input variables
        for i in range(input_dim):
            f.write(f"(declare-const X_{i} Real)\n")

        # Declare output variables
        for i in range(output_dim):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write("\n; Input constraints\n")
        for i in range(input_dim):
            f.write(f"(assert (>= X_{i} {-eps}))\n")
            f.write(f"(assert (<= X_{i} {eps}))\n")

        f.write(f"\n; Output constraint (target class {target_class} is largest)\n")
        for i in range(output_dim):
            if i != target_class:
                f.write(f"(assert (>= Y_{target_class} Y_{i}))\n")

    print(f"Created {filename}")
    return filepath


def create_vnnlib_disjunctive(input_dim, output_dim, eps, filename):
    """Create a VNNLIB specification with disjunctive constraints."""
    filepath = os.path.join(FIXTURES_DIR, filename)

    with open(filepath, 'w') as f:
        # Declare input variables
        for i in range(input_dim):
            f.write(f"(declare-const X_{i} Real)\n")

        # Declare output variables
        for i in range(output_dim):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write("\n; Input constraints\n")
        for i in range(input_dim):
            f.write(f"(assert (>= X_{i} {-eps}))\n")
            f.write(f"(assert (<= X_{i} {eps}))\n")

        f.write("\n; Disjunctive output constraint: (Y_0 <= Y_1) OR (Y_1 <= 0)\n")
        f.write("(assert (or\n")
        f.write("  (and (<= Y_0 Y_1))\n")
        f.write("  (and (<= Y_1 0.0))\n")
        f.write("))\n")

    print(f"Created {filename}")
    return filepath


def main():
    """Generate all test fixtures."""
    print("Creating test fixtures...")

    # Create simple MLP model
    mlp = create_simple_mlp(input_dim=4, hidden_dim=8, output_dim=2)
    export_to_onnx(mlp, (4,), "simple_mlp.onnx")

    # Create simple CNN model
    cnn = create_simple_cnn()
    export_to_onnx(cnn, (1, 8, 8), "simple_cnn.onnx")

    # Create VNNLIB specifications
    create_vnnlib_robustness(
        input_dim=4, output_dim=2, eps=0.1,
        filename="robustness_mlp.vnnlib")

    create_vnnlib_targeted(
        input_dim=4, output_dim=2, eps=0.1, target_class=1,
        filename="targeted_mlp.vnnlib")

    create_vnnlib_disjunctive(
        input_dim=4, output_dim=2, eps=0.1,
        filename="disjunctive_mlp.vnnlib")

    # CNN specifications
    create_vnnlib_robustness(
        input_dim=64, output_dim=2, eps=0.05,
        filename="robustness_cnn.vnnlib")

    print("Done!")


if __name__ == "__main__":
    main()
