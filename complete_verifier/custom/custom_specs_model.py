#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""An simple example to run custom model and specs."""

import torch
import torch.nn as nn

class SimpleFeedForward(nn.Module):
    """A very simple model, just for demonstration."""
    def __init__(self, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            # Input dimension is 2, should match vnnlib file.
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # Output dimension is 1, should match vnnlib file.
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs):
        return self.model(inputs)

if __name__ == "__main__":
    # Save a random model for testing.
    model = SimpleFeedForward(hidden_size=32)
    torch.save(model.state_dict(), 'models/custom_specs/custom_specs.pth')

