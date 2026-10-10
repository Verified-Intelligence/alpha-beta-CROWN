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
"""Minimal image-classification verification demo."""

import os
import torch

from abcrown import (
    ABCrownSolver,
    IOConstraints,
    input_vars,
    output_vars,
)

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    'image_classification_dependency/image_safe_eps002.pt',
)

IMG_H, IMG_W = 384, 384
NUM_CLASSES = 10


class SimpleConvClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(16 * IMG_H * IMG_W, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(x.shape[0], 3, IMG_H, IMG_W)
        feats = self.conv(x)
        flat = feats.view(feats.shape[0], -1)
        return self.head(flat)


def create_safe_checkpoint(path: str, label: int = 0, margin: float = 1.0) -> None:
    model = SimpleConvClassifier()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.head.bias.zero_()
        model.head.bias[label] = margin

    torch.save(model.state_dict(), path)


def main() -> None:
    torch.manual_seed(42)
    base_image = torch.rand(1, 3, IMG_H, IMG_W)
    eps = 0.02
    label = 0

    x = input_vars((3, IMG_H, IMG_W))
    y = output_vars(NUM_CLASSES)
    input_constraint = (x >= (base_image - eps)) & (x <= (base_image + eps))
    output_constraint = None
    for i in range(NUM_CLASSES):
        if i == label:
            continue
        pred = y[label] > y[i]
        output_constraint = pred if output_constraint is None else (output_constraint & pred)
    constraints = IOConstraints(
        input_vars=x,
        output_vars=y,
        input_constraint=input_constraint,
        output_constraint=output_constraint,
    )

    model = SimpleConvClassifier()
    if not os.path.exists(CHECKPOINT_PATH):
        create_safe_checkpoint(CHECKPOINT_PATH, label=label)
        print(f'[info] created checkpoint: {CHECKPOINT_PATH}')
    else:
        print(f'[info] using checkpoint: {CHECKPOINT_PATH}')

    checkpoint = torch.load(CHECKPOINT_PATH)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    solver = ABCrownSolver(model, x, y)
    result = solver.verify(constraints=constraints)

    print(f'[info] verifying epsilon={eps:.4f} around a random base image')
    print(f'status={result.status}, success={result.success}')


if __name__ == "__main__":
    main()
