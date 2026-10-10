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
"""Unit tests for heuristics/compute_bounds_chunks.py.

Covers:
  * Functional correctness (CPU): chunked results agree with update_bounds reference.
  * Memory reduction (CUDA only): sub-batching lowers peak GPU memory,
    mirroring the OOM scenario documented in demo_oom.py.
"""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setUpModule():
    import arguments
    global _original_config
    _original_config = arguments.Config
    new_config = arguments.ConfigHandler()
    new_config.construct_config_dict(new_config.default_args)
    new_config.file = None
    arguments.Config = new_config
    arguments.Config['general']['sparse_alpha'] = False
    arguments.Config['solver']['alpha-crown']['full_conv_alpha'] = False
    arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds'] = False


setup_module = setUpModule


def tearDownModule():
    import arguments
    arguments.Config = _original_config


teardown_module = tearDownModule


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_net_and_domain(device, batch, in_dim, hidden, out_dim, num_spec):
    """Build a LiRPANet and a domain dict replicated to `batch` entries.

    Runs a brief alpha-CROWN pass on a single example to initialise bounds
    and alphas (shape (2, 1, 1, neurons) keyed on net.final_name), then
    replicates to `batch` domains, matching the BaB domain format used by
    compute_lower_bounds_chunks.

    Returns (net, domain, alphas_ref).
    """
    from auto_LiRPA import BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from beta_CROWN_solver import LiRPANet

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )
        def forward(self, x):
            return self.net(x)

    torch.manual_seed(0)
    model = _MLP().to(device)
    net = LiRPANet(model, in_size=(1, in_dim), device=device)

    x_data = torch.zeros(1, in_dim, device=device)
    eps = 0.01
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_data - eps, x_U=x_data + eps)
    x_bt = BoundedTensor(x_data, ptb)

    c = torch.zeros(1, num_spec, out_dim, device=device)
    for i in range(num_spec):
        c[0, i, 0] = 1.0
        c[0, i, i + 1] = -1.0

    net.x = x_bt
    net.c = c

    net.net.set_bound_opts({
        'optimize_bound_args': {
            'enable_beta_crown': False,
            'fix_interm_bounds': False,
            'iteration': 3,
        },
        'enable_opt_interm_bounds': False,
    })
    net.set_crown_bound_opts('beta')
    with torch.enable_grad():
        lb, _ = net.net.compute_bounds(
            x=(x_bt,), C=c, method='CROWN-optimized', bound_upper=False,
        )

    lb_dict, ub_dict = {}, {}
    for node in net.net.nodes():
        if (hasattr(node, 'lower') and node.lower is not None
                and hasattr(node, 'upper') and node.upper is not None
                and node.lower.dim() >= 2 and node.lower.shape[0] == 1):
            lb_dict[node.name] = node.lower.detach()
            ub_dict[node.name] = node.upper.detach()
    lb_dict[net.final_name] = lb.detach()
    ub_dict[net.final_name] = (lb + 1.0).detach()

    alphas_ref = {}
    for node in net.net.get_enabled_opt_act():
        if net.final_name in node.alpha:
            n = node.alpha[net.final_name].shape[-1]
        elif node.alpha:
            n = next(iter(node.alpha.values())).shape[-1]
        elif node.lower is not None:
            n = node.lower.shape[-1]
        else:
            continue
        a = torch.full((2, 1, 1, n), 0.5, device=device)
        node.alpha = {net.final_name: a.clone().detach().requires_grad_(True)}
        alphas_ref[node.name] = {net.final_name: a.detach().clone()}

    domain = {
        'lower_bounds': {k: v.repeat(batch, *([1] * (v.dim() - 1)))
                         for k, v in lb_dict.items()},
        'upper_bounds': {k: v.repeat(batch, *([1] * (v.dim() - 1)))
                         for k, v in ub_dict.items()},
        'cs': c.repeat(batch, 1, 1),
        'alphas': {
            node_name: {sn: a.repeat(1, 1, batch, 1)
                        for sn, a in per_node.items()}
            for node_name, per_node in alphas_ref.items()
        },
    }
    return net, domain, alphas_ref


def _reset_alpha(net, alphas_ref, batch):
    """Re-attach fresh batch-replicated alpha tensors to the net nodes."""
    for node in net.net.get_enabled_opt_act():
        if node.name in alphas_ref:
            node.alpha = {
                sn: a.repeat(1, 1, batch, 1).detach().requires_grad_(True)
                for sn, a in alphas_ref[node.name].items()
            }


def _clone_domain(domain):
    return {
        'lower_bounds': {k: v.clone() for k, v in domain['lower_bounds'].items()},
        'upper_bounds': {k: v.clone() for k, v in domain['upper_bounds'].items()},
        'cs': domain['cs'].clone(),
        'alphas': {
            node_name: {sn: a.clone() for sn, a in per_node.items()}
            for node_name, per_node in domain['alphas'].items()
        },
    }


# ---------------------------------------------------------------------------
# Functional correctness tests (CPU, no GPU required)
# ---------------------------------------------------------------------------

class TestComputeLowerBoundsChunksFunctional(unittest.TestCase):
    """Chunked and non-chunked lower bounds agree on a small CPU model."""

    IN_DIM = 16
    HIDDEN = 64
    OUT_DIM = 5
    NUM_SPEC = OUT_DIM - 1   # 4 specs
    BATCH = 8
    CHUNK = 4

    @classmethod
    def setUpClass(cls):
        cls.device = 'cpu'
        cls.net, cls.domain, cls.alphas_ref = _build_net_and_domain(
            cls.device, cls.BATCH,
            cls.IN_DIM, cls.HIDDEN, cls.OUT_DIM, cls.NUM_SPEC,
        )

    def _run(self, sub_batch_size):
        from heuristics.compute_bounds_chunks import compute_lower_bounds_chunks
        _reset_alpha(self.net, self.alphas_ref, self.BATCH)
        return compute_lower_bounds_chunks(
            self.net, _clone_domain(self.domain),
            sub_batch_size=sub_batch_size,
        )

    def _run_update_bounds_reference(self):
        _reset_alpha(self.net, self.alphas_ref, self.BATCH)
        return self.net.update_bounds(
            _clone_domain(self.domain),
            beta=False, fix_interm_bounds=True, shortcut=True,
        )

    def test_unchunked_output_shape(self):
        """sub_batch_size=None returns (batch, num_spec) tensor."""
        lb = self._run(sub_batch_size=None)
        self.assertEqual(lb.shape, (self.BATCH, self.NUM_SPEC))

    def test_chunked_output_shape(self):
        """Chunked path concatenates chunks and returns (batch, num_spec) tensor."""
        lb = self._run(sub_batch_size=self.CHUNK)
        self.assertEqual(lb.shape, (self.BATCH, self.NUM_SPEC))

    def test_chunked_matches_unchunked(self):
        """Chunked lower bounds agree with the original update_bounds reference."""
        lb_ref = self._run_update_bounds_reference()
        lb_chunked = self._run(sub_batch_size=self.CHUNK)
        torch.testing.assert_close(lb_chunked, lb_ref, rtol=1e-5, atol=1e-5)

    def test_sub_batch_larger_than_batch_matches_unchunked(self):
        """sub_batch_size >= batch takes the single-pass fast path."""
        lb_full = self._run(sub_batch_size=None)
        lb_oversized = self._run(sub_batch_size=self.BATCH * 2)
        torch.testing.assert_close(lb_oversized, lb_full, rtol=1e-5, atol=1e-5)

    def test_sub_batch_equal_to_batch_matches_unchunked(self):
        """sub_batch_size == batch is a degenerate one-chunk case."""
        lb_full = self._run(sub_batch_size=None)
        lb_exact = self._run(sub_batch_size=self.BATCH)
        torch.testing.assert_close(lb_exact, lb_full, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Memory reduction tests (CUDA only)
# ---------------------------------------------------------------------------

@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class TestComputeLowerBoundsChunksMemory(unittest.TestCase):
    """Sub-batching lowers peak GPU memory — mirrors the demo_oom.py scenario.

    Setup (scaled down from demo_oom.py):
      hidden=1024, OUT_DIM=20, NUM_SPEC=19, batch=256, sub_batch=32 (8 chunks).
    A-matrix per FC layer: 256x19x1024x4 B ~20 MB full vs ~2.5 MB per chunk.
    """

    IN_DIM = 32
    HIDDEN = 1024
    OUT_DIM = 20
    NUM_SPEC = OUT_DIM - 1   # 19 specs
    BATCH = 256
    CHUNK = 32               # 8 chunks

    @classmethod
    def setUpClass(cls):
        cls.device = 'cuda'
        cls.net, cls.domain, cls.alphas_ref = _build_net_and_domain(
            cls.device, cls.BATCH,
            cls.IN_DIM, cls.HIDDEN, cls.OUT_DIM, cls.NUM_SPEC,
        )

    def _peak_delta(self, sub_batch_size):
        """Return (peak - base) GPU memory during one bound computation pass."""
        from heuristics.compute_bounds_chunks import compute_lower_bounds_chunks
        _reset_alpha(self.net, self.alphas_ref, self.BATCH)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        compute_lower_bounds_chunks(
            self.net, _clone_domain(self.domain),
            sub_batch_size=sub_batch_size,
        )
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    def test_chunking_reduces_peak_memory(self):
        """Peak GPU memory delta with sub_batch_size is less than full-batch delta."""
        delta_full = self._peak_delta(sub_batch_size=None)
        delta_chunked = self._peak_delta(sub_batch_size=self.CHUNK)
        self.assertLess(
            delta_chunked, delta_full,
            msg=(f"Chunked peak delta ({delta_chunked/1e6:.1f} MB) should be less than "
                 f"full-batch peak delta ({delta_full/1e6:.1f} MB)"),
        )

if __name__ == '__main__':
    unittest.main()
