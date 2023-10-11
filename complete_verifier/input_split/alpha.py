import torch


def set_alpha_input_split(self, alpha, set_all=False, double=False, split_partitions=2):
    if len(alpha) == 0:
        return
    for m in self.net.perturbed_optimizable_activations:
        for spec_name in list(m.alpha.keys()):
            if spec_name in alpha[0][m.name]:
                # Only setup the last layer alphas if no refinement is done.
                if spec_name in self.alpha_start_nodes or set_all:
                    # Merge all alpha vectors together in this batch. Size is (2, spec, batch, *shape).
                    m.alpha[spec_name] = torch.cat(
                        [alpha[i][m.name][spec_name]
                         for i in range(len(alpha))], dim=2)
                    if double:
                        # Duplicate for the second half of the batch.
                        # (Supporting input split which doesn't branch alphas
                        # by itself.)
                        m.alpha[spec_name] = m.alpha[spec_name].repeat(
                            1, 1, split_partitions, *([1] * (m.alpha[spec_name].ndim - 3)))
                    m.alpha[spec_name] = m.alpha[spec_name].detach().requires_grad_()
            else:
                # This layer's alpha is not used. For example, we can drop all intermediate layer alphas.
                del m.alpha[spec_name]
