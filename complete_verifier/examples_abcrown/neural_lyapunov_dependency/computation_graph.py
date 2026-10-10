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
import torch
import torch.nn as nn

from auto_LiRPA.jacobian import JacobianOP


class VanDerPolDynamics:
    """Van der Pol oscillator dynamics with a control input."""

    def __init__(self, mu: float = 1.0):
        self.nx = 2
        self.nu = 1
        self.ny = 1
        self.mu = mu

    def f_torch(self, x, u):
        x1 = x[:, 0].unsqueeze(-1)
        x2 = x[:, 1].unsqueeze(-1)
        d_x1 = x2
        d_x2 = -x1 + self.mu * (1 - x1**2) * x2 + u
        return torch.cat((d_x1, d_x2), dim=1)

    @property
    def x_equilibrium(self):
        return torch.zeros((2,))

    @property
    def u_equilibrium(self):
        return torch.zeros((1,))


class Controller(nn.Module):
    def __init__(self, dims, x_equilibrium, u_equilibrium, scale=1.0):
        super().__init__()
        self.dims = dims
        self.scale = scale
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.layers = nn.Sequential(*layers)
        self.register_buffer("x_equilibrium", x_equilibrium)
        self.register_buffer("u_equilibrium", u_equilibrium)
        self.register_buffer("b", torch.atanh(self.u_equilibrium / self.scale))

    def forward(self, x):
        u_x = self.layers(x)
        x_eq = self.x_equilibrium.unsqueeze(0)
        u_x_star = self.layers(x_eq)[0]
        u_diff = u_x - u_x_star
        return self.scale * torch.tanh(u_diff + self.b)


class Lyapunov(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ClosedLoopComputationGraph(nn.Module):
    def __init__(self, dynamics: VanDerPolDynamics, controller: Controller):
        super().__init__()
        self.dynamics = dynamics
        self.controller = controller

    def forward(self, x):
        u = self.controller(x)
        return self.dynamics.f_torch(x, u)


class LyapunovComputationGraph(nn.Module):
    def __init__(
        self,
        dynamics: VanDerPolDynamics,
        controller: Controller,
        lyapunov: Lyapunov,
    ):
        super().__init__()
        self.dynamics = dynamics
        self.controller = controller
        self.lyapunov = lyapunov

    def forward(self, x):
        x = x.clone().requires_grad_(True)
        v_x = self.lyapunov(x)
        u = self.controller(x)
        x_dot = self.dynamics.f_torch(x, u)
        d_v_dx = JacobianOP.apply(self.lyapunov(x), x).squeeze(1)
        v_dot = torch.sum(d_v_dx * x_dot, dim=1, keepdim=True)
        return torch.cat((v_x, v_dot), dim=1)
