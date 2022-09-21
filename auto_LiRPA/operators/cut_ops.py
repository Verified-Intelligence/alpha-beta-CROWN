""" Cut operators"""
from .base import *
from .clampmult import multiply_by_A_signs


class CutModule():
    # store under BoundedModule
    def __init__(self, relu_nodes=[], general_beta=None, x_coeffs=None,
                 active_cuts=None, cut_bias=None):
        # all dict, storing cut parameters for each start node
        # {start node name: (2 (lA, uA), spec (out_c, out_h, out_w), batch, num_cuts)}
        self.general_beta = general_beta
        # {start node name: (# active cut constraints,)}
        self.active_cuts = active_cuts

        # all dict with tensor, storing coeffs for each relu layer, no grad
        # coeffs: {relu layername: (num_cuts, flattened_nodes)}
        self.relu_coeffs, self.arelu_coeffs, self.pre_coeffs = {}, {}, {}
        for m in relu_nodes:
            self.relu_coeffs[m.name] = self.arelu_coeffs[m.name] = self.pre_coeffs[m.name] = None

        # single tensor, always the same, no grad
        # bias: (num_cuts,)
        self.cut_bias = cut_bias
        # x_coeffs: (num_cuts, flattened input dims)
        self.x_coeffs = x_coeffs

    def use_patches(self, start_node):
        # check if we are using patches mode for the start node
        A = start_node.lA if start_node.lA is not None else start_node.uA
        return type(A) is Patches

    def select_active_general_beta(self, start_node, unstable_idx=None):
        # if one constraint have nodes deeper than start node, we do not count its effect for now
        # self.general_beta[start_node.name]: (2(0 lower, 1 upper), spec (out_c, out_h, out_w/# fc nodes), batch, num_constrs)
        # self.active_cuts[start_node.name]: a long() tensor with constraint index that
        # should be index on current layer with current start node
        if self.general_beta[start_node.name].ndim == 4:
            general_beta = self.general_beta[start_node.name][:, :, :, self.active_cuts[start_node.name]]
        elif self.general_beta[start_node.name].ndim == 6:
            general_beta = self.general_beta[start_node.name][:, :, :, :, :, self.active_cuts[start_node.name]]
        else:
            print("general beta shape not supported!")
            exit()
        if unstable_idx is not None:
            if self.use_patches(start_node):
                general_beta = general_beta[:, unstable_idx[0], unstable_idx[1], unstable_idx[2], :, :]
            else:
                # matrix mode
                if general_beta.ndim == 6:
                    # conv layers general_beta: (2(0 lower, 1 upper), spec (out_c, out_h, out_w), batch, num_constrs)
                    _, out_c, out_h, out_w, batch, num_constrs = general_beta.shape
                    general_beta = general_beta.view(2, -1, batch, num_constrs)
                else:
                    # dense layers general_beta: (2(0 lower, 1 upper), spec, batch, num_constrs)
                    pass
                general_beta = general_beta[:, unstable_idx]
        else:
            # unstable_idx is None
            if general_beta.ndim == 6:
                # flatten spec layer shape
                _, out_c, out_h, out_w, batch, num_constrs = general_beta.shape
                general_beta = general_beta.view(2, -1, batch, num_constrs)
        return general_beta

    def general_beta_coeffs_mm(self, unstable_spec_beta, coeffs, A, current_layer_shape):
        if type(A) is Patches:
            # lA, uA are patches, we have to unfold beta and coeffs to match lA and uA
            # coeffs: (num_constrs, current_c, current_h, current_w)
            # coeffs_unfolded: (num_constrs, out_h, out_w, in_c, H, W)
            # current_layer_shape = x.lower.size()[1:]
            coeffs_unfolded = inplace_unfold(coeffs.view(-1, *current_layer_shape), \
                                             kernel_size=A.patches.shape[-2:], padding=A.padding, stride=A.stride)
            # unstable_coeffs_unfolded: (num_constrs, unstable, in_c, H, W)
            # A.unstable_idx is the unstable idx for spec layer
            unstable_coeffs_unfolded = coeffs_unfolded[:, A.unstable_idx[1], A.unstable_idx[2], :, :, :]
            # A.unstable_idx: unstable index on out_c, out_h and out_w
            # general_beta: (2(0 lower, 1 upper), spec (out_c, out_h, out_w), batch, num_constrs)
            # unstable_spec_beta: (2(0 lower, 1 upper), unstable, batch, num_constrs)
            # unstable_spec_beta = general_beta[:, A.unstable_idx[0],\
            #             A.unstable_idx[1], A.unstable_idx[2], :, :]
            # beta_mm_coeffs_unfolded: (2(0 lower, 1 upper), unstable, batch, in_c, H, W)
            beta_mm_coeffs = torch.einsum('sihj,jiabc->sihabc', unstable_spec_beta, unstable_coeffs_unfolded)
        else:
            # unstable_spec_beta: (2(0 lower, 1 upper), unstable, batch, num_constrs)
            # coeffs: (num_constrs, current flattened layer nodes)
            # beta_mm_coeffs: (2(0 lower, 1 upper), unstable, batch, current flattened layer nodes)
            beta_mm_coeffs = torch.einsum('sihj,ja->siha', unstable_spec_beta, coeffs)
            assert beta_mm_coeffs[0].numel() == A.numel(), f"the shape of beta is not initialized correctly! {beta_mm_coeffs[0].shape} v.s. {A.shape}"
        return beta_mm_coeffs.reshape(2, *A.shape)

    def general_beta_coeffs_addmm_to_A(self, lA, uA, general_beta, coeffs, current_layer_shape):
        A = lA if lA is not None else uA
        # general_beta: (2(0 lower, 1 upper), spec (out_c, out_h, out_w), batch, num_constrs)
        # coeffs: (num_constrs, current_c, current_h, current_w)
        # beta_mm_coeffs[0] shape is the same as A
        # patches mode: (2(0 lower, 1 upper), unstable, batch, in_c, H, W)
        # not patches: (2(0 lower, 1 upper), unstable, batch, current flattened layer nodes)
        beta_mm_coeffs = self.general_beta_coeffs_mm(general_beta, coeffs, A, current_layer_shape)
        assert beta_mm_coeffs[0].shape == A.shape
        if type(A) is Patches:
            # lA, uA are patches, we have to unfold beta and coeffs to match lA and uA
            # lA_patches: (unstable, batch, in_c, H, W)
            if lA is not None:
                lA = Patches(lA.patches - beta_mm_coeffs[0], A.stride, A.padding, \
                             A.patches.shape, unstable_idx=A.unstable_idx, output_shape=A.output_shape)
            if uA is not None:
                uA = Patches(uA.patches + beta_mm_coeffs[1], A.stride, A.padding, \
                             A.patches.shape, unstable_idx=A.unstable_idx, output_shape=A.output_shape)
        else:
            # dense layers
            if lA is not None:
                lA = lA - beta_mm_coeffs[0]
            if uA is not None:
                uA = uA + beta_mm_coeffs[1]
        return lA, uA

    def patch_trick(self, start_node, layer_name, A, current_layer_shape):
        ######## A problem with patches mode for cut constraint start ##########
        # There are cases that the node that is in the constraint but not selected by the patches for the output node
        # trick: only count the small patches that have all the split node coeffs[ci].sum() equal to coeffs_unfolded[ci][out_h, out_w, -1].sum()
        # we should force these beta to be 0 to disable the effect of these constraints
        # this only apply if current layer uses patches mode; if the target layer is patches but current layer not, we should not use it!
        assert type(A) is Patches, "this trick fix only works for patches mode"
        # unstable_spec_beta stores the current propagation, self.general_beta[start_node.name] selected with active_cuts, spec unstable
        coeffs = 0
        if layer_name != "input":
            if self.relu_coeffs[layer_name] is not None:
                coeffs = coeffs + self.relu_coeffs[layer_name]
            if self.arelu_coeffs[layer_name] is not None:
                coeffs = coeffs + self.arelu_coeffs[layer_name]
            if self.pre_coeffs[layer_name] is not None:
                coeffs = coeffs + self.pre_coeffs[layer_name]
        else:
            if self.x_coeffs is not None:
                coeffs = coeffs + self.x_coeffs
        coeffs_unfolded = inplace_unfold(coeffs.view(-1, *current_layer_shape), \
                                         kernel_size=A.patches.shape[-2:], padding=A.padding, stride=A.stride)
        num_constrs, out_h, out_w, in_c, H, W = coeffs_unfolded.shape
        # make sure the small patch selected include all the nonzero coeffs
        ####### NOTE: This check could be costly #######
        patch_mask_on_beta = (coeffs_unfolded.reshape(num_constrs, out_h, out_w, -1).abs().sum(-1) == \
                              coeffs.reshape(num_constrs, -1).abs().sum(-1).reshape(num_constrs, 1, 1))
        # patch_mask_on_beta: (out_h, out_w, num_constrs)
        patch_mask_on_beta = patch_mask_on_beta.permute(1, 2, 0)
        # 2(lower, upper), out_c, out_h, out_w, batch, num_constrs
        patch_mask_on_beta = patch_mask_on_beta.reshape(1, 1, out_h, out_w, 1, num_constrs)
        self.general_beta[start_node.name].data = self.general_beta[start_node.name].data * patch_mask_on_beta

    def relu_cut(self, start_node, layer_name, last_lA, last_uA, current_layer_shape, unstable_idx=None, batch_mask=None):
        # propagate relu neuron in cut constraints through relu layer
        # start_node.name in self.general_beta means there are intermediate betas that can optimize this start node separately
        relu_coeffs = self.relu_coeffs[layer_name]
        active_cuts = self.active_cuts[start_node.name]
        # active_cuts.size(0) == 0 means all constraints containing this layer have deep layer nodes
        if relu_coeffs is None or active_cuts.size(0) == 0:
            # do nothing
            return last_lA, last_uA
        assert start_node.name in self.general_beta
        # select current relu layer general beta
        general_beta = self.select_active_general_beta(start_node, unstable_idx)
        relu_coeffs = relu_coeffs[active_cuts]
        if batch_mask is not None:
            general_beta = general_beta[:, :, batch_mask]
        last_lA, last_uA = self.general_beta_coeffs_addmm_to_A(last_lA, last_uA, general_beta,
                                                               relu_coeffs, current_layer_shape)
        return last_lA, last_uA

    def pre_cut(self, start_node, layer_name, lA, uA, current_layer_shape, unstable_idx=None, batch_mask=None):
        # propagate prerelu neuron in cut constraints through relu layer
        # start_node.name in self.general_beta means there are intermediate betas that can optimize this start node separately
        pre_coeffs = self.pre_coeffs[layer_name]
        active_cuts = self.active_cuts[start_node.name]
        # active_cuts.size(0) == 0 means all constraints containing this layer have deep layer nodes
        if pre_coeffs is None or active_cuts.size(0) == 0:
            # do nothing
            return lA, uA
        general_beta = self.select_active_general_beta(start_node, unstable_idx)
        pre_coeffs = pre_coeffs[active_cuts]
        if batch_mask is not None:
            general_beta = general_beta[:, :, batch_mask]
        lA, uA = self.general_beta_coeffs_addmm_to_A(lA, uA, general_beta, pre_coeffs, current_layer_shape)
        return lA, uA


    @staticmethod
    @torch.jit.script
    def jit_arelu_lA(last_lA, lower, upper, beta_mm_coeffs, unstable_or_cut_index, upper_d):
        nu_hat_pos = last_lA.clamp(max=0.).abs()
        tao = (-lower.unsqueeze(0) * nu_hat_pos - beta_mm_coeffs[0]) / (upper.unsqueeze(0) - lower.unsqueeze(0) + 1e-10)
        pi = (upper.unsqueeze(0) * nu_hat_pos + beta_mm_coeffs[0]) / (upper.unsqueeze(0) - lower.unsqueeze(0) + 1e-10)
        tao, pi = tao.clamp(min=0.), pi.clamp(min=0.)
        tao, pi = torch.min(tao, nu_hat_pos), torch.min(pi, nu_hat_pos)
        new_upper_d = pi / (pi + tao + 1e-10)
        # need to customize the upper bound slope and lbias for (1) unstable relus and 
        # (2) relus that are used with upper boundary relaxation
        # original upper bound slope is u/(u-l) also equal to pi/(pi+tao) if no beta_mm_coeffs[0]
        # now the upper bound slope should be pi/(p+tao) updated with beta_mm_coeffs[0]
        unstable_upper_bound_index = unstable_or_cut_index.unsqueeze(0).logical_and(last_lA < 0)
        # conv layer:
        # upper_d: 1, batch, current_c, current_w, current_h
        # unstable_upper_bound_index, new_upper_d: spec unstable, batch, current_c, current_w, current_h
        # dense layer:
        # upper_d: 1, batch, current flattened nodes
        # unstable_upper_bound_index, new_upper_d: spec unstable, batch, current flattened nodes
        new_upper_d = new_upper_d * unstable_upper_bound_index.float() + \
                      upper_d * (1. - unstable_upper_bound_index.float())
        return nu_hat_pos, tao, pi, new_upper_d, unstable_upper_bound_index

    @staticmethod
    @torch.jit.script
    def jit_arelu_lbias(unstable_or_cut_index, nu_hat_pos, beta_mm_coeffs, lower, upper, lbias, pi, tao):
        # if no unstable, following bias should always be 0
        if unstable_or_cut_index.sum() > 0:
            # update lbias with new form, only contribued by unstable relus
            uC = -upper.unsqueeze(0) * nu_hat_pos
            lC = -lower.unsqueeze(0) * nu_hat_pos
            # lbias: (spec unstable, batch, current flattened nodes) same as lA
            lbias = (pi * lower.unsqueeze(0))

            uC_mask = (beta_mm_coeffs[0] <= uC).to(lbias)
            lC_mask = (beta_mm_coeffs[0] >= lC).to(lbias)
            default_mask = ((1-uC_mask) * (1-lC_mask)).to(lbias)
            lbias = - beta_mm_coeffs[0].to(lbias) * lC_mask + lbias * default_mask

            # lbias[beta_mm_coeffs[0] <= uC] = 0.
            # lbias[beta_mm_coeffs[0] >= lC] = -beta_mm_coeffs[0][beta_mm_coeffs[0] >= lC].to(lbias)

            # final lbias: (spec unstable, batch)
            lbias = (lbias * unstable_or_cut_index.unsqueeze(0).float()).view(lbias.shape[0], lbias.shape[1], -1).sum(-1)
        return lbias

    @staticmethod
    @torch.jit.script
    def jit_arelu_uA(last_uA, lower, upper, beta_mm_coeffs, unstable_or_cut_index, upper_d):
        nu_hat_pos = (-last_uA).clamp(max=0.).abs()
        tao = (- lower.unsqueeze(0) * nu_hat_pos - beta_mm_coeffs[1]) / (upper.unsqueeze(0) - lower.unsqueeze(0) + 1e-10)
        pi = (upper.unsqueeze(0) * nu_hat_pos + beta_mm_coeffs[1]) / (upper.unsqueeze(0) - lower.unsqueeze(0) + 1e-10)
        tao, pi = tao.clamp(min=0.), pi.clamp(min=0.)
        tao, pi = torch.min(tao, nu_hat_pos), torch.min(pi, nu_hat_pos)
        new_upper_d = pi / (pi + tao + 1e-10)

        # assert ((tao + pi - nu_hat_pos).abs()*unstable_or_cut_index).max() <= 1e-5, "pi+tao should always be the same as nu_hat_pos"

        # unstable_or_cut_index = self.I.logical_or(self.arelu_coeffs.sum(0).view(self.I.shape) != 0)
        unstable_upper_bound_index = unstable_or_cut_index.unsqueeze(0).logical_and(-last_uA < 0)
        new_upper_d = new_upper_d * unstable_upper_bound_index.float() + \
                      upper_d * (1. - unstable_upper_bound_index.float())
        return nu_hat_pos, tao, pi, new_upper_d, unstable_upper_bound_index

    @staticmethod
    @torch.jit.script
    def jit_arelu_ubias(unstable_or_cut_index, nu_hat_pos, beta_mm_coeffs, lower, upper, ubias, pi, tao):
        if unstable_or_cut_index.sum() > 0:
            uC = -upper.unsqueeze(0) * nu_hat_pos
            lC = -lower.unsqueeze(0) * nu_hat_pos
            ubias = -(pi * lower.unsqueeze(0))

            uC_mask = (beta_mm_coeffs[1] <= uC).to(ubias)
            lC_mask = (beta_mm_coeffs[1] >= lC).to(ubias)
            default_mask = ((1-uC_mask) * (1-lC_mask)).to(ubias)
            ubias = beta_mm_coeffs[1].to(ubias) * lC_mask + ubias * default_mask

            # ubias[beta_mm_coeffs[1] <= uC] = 0.
            # ubias[beta_mm_coeffs[1] >= lC] = beta_mm_coeffs[1][beta_mm_coeffs[1] >= lC].to(ubias)

            ubias = (ubias * unstable_or_cut_index.unsqueeze(0).float()).view(ubias.shape[0], ubias.shape[1], -1).sum(-1)
        return ubias


    def arelu_cut(self, start_node, layer_name, last_lA, last_uA, lower_d, upper_d,
                  lower_b, upper_b, lb_lower_d, ub_lower_d, I, x, patch_size,
                  current_layer_shape, unstable_idx=None, batch_mask=None):
        # propagate integer var of relu neuron (arelu) in cut constraints through relu layer
        arelu_coeffs = self.arelu_coeffs[layer_name]
        active_cuts = self.active_cuts[start_node.name]
        # active_cuts.size(0) == 0 means all constraints containing this layer have deep layer nodes
        if arelu_coeffs is None or active_cuts.size(0) == 0:
            # do regular propagation without cut
            uA, ubias = _bound_oneside(last_uA, upper_d, ub_lower_d if lower_d is None else lower_d, upper_b, lower_b, start_node, patch_size)
            lA, lbias = _bound_oneside(last_lA, lb_lower_d if lower_d is None else lower_d, upper_d, lower_b, upper_b, start_node, patch_size)
            return lA, uA, lbias, ubias

        # general_beta: (2(0 lower, 1 upper), spec (out_c, out_h, out_w), batch, num_constrs)
        general_beta = self.select_active_general_beta(start_node, unstable_idx)
        # arelu_coeffs: (num_constrs, flattened current layer nodes)
        arelu_coeffs = arelu_coeffs[active_cuts]
        if batch_mask is not None:
            general_beta = general_beta[:, :, batch_mask]
        A = last_lA if last_lA is not None else last_uA
        # beta_mm_coeffs[0] shape is the same as A
        # patches mode: (2(0 lower, 1 upper), unstable, batch, in_c, H, W)
        # not patches: (2(0 lower, 1 upper), unstable, batch, current flattened layer nodes)
        beta_mm_coeffs = self.general_beta_coeffs_mm(general_beta, arelu_coeffs, A, current_layer_shape)
        # unstable_this_layer = torch.logical_and(x.lower < 0, x.upper > 0).unsqueeze(0)
        # I is the unstable index in this relu layer: (batch, *layer shape)
        # if there is node in cut constraint that is stable, also need to count its effect 
        # self.arelu_coeffs: (num_constrs, flattened current layer)
        unstable_or_cut_index = I.logical_or(arelu_coeffs.sum(0).view(I[0:1].shape) != 0)

        if type(A) is Patches:
            # patches mode, conv layer only
            # x.lower (always regular shape): batch, current_c, current_h, current_w
            # x_lower_unfold: unstable, batch, in_C, H, W (same as patches last_lA)
            x_lower_unfold = _maybe_unfold(x.lower.unsqueeze(0), A)
            x_upper_unfold = _maybe_unfold(x.upper.unsqueeze(0), A)
            # first minus upper and lower and then unfold to patch size will save memory
            x_upper_minus_lower_unfold = _maybe_unfold((x.upper - x.lower).unsqueeze(0), A)
            ####### be careful with the unstable_this_layer and unstable_idx #######
            # unstable_this_layer is the unstable index in this layer
            # unstable_idx is the unstable index in spec layer
            # unstable_this_layer: spec unstable, batch, in_C, H, W (same as patches last_lA)
            # unstable_this_layer = torch.logical_and(x_lower_unfold < 0, x_upper_unfold > 0)
            # unstable_this_layer = _maybe_unfold(self.I.unsqueeze(0), A)
            unstable_or_cut_index = _maybe_unfold(unstable_or_cut_index.unsqueeze(0), A)
            if last_lA is not None:
                assert beta_mm_coeffs[0].shape == last_lA.shape, f"{beta_mm_coeffs[0].shape} != {last_lA.shape}"
                # last_lA.patches, nu_hat_pos, tao, pi: (unstable, batch, in_c, H, W)
                nu_hat_pos = last_lA.patches.clamp(max=0.).abs()
                tao = (-x_lower_unfold * nu_hat_pos - beta_mm_coeffs[0]) / (x_upper_minus_lower_unfold.clamp(min=1e-10))
                pi = (x_upper_unfold * nu_hat_pos + beta_mm_coeffs[0]) / (x_upper_minus_lower_unfold.clamp(min=1e-10))
                tao, pi = tao.clamp(min=0.), pi.clamp(min=0.)
                tao, pi = torch.min(tao, nu_hat_pos), torch.min(pi, nu_hat_pos)
                new_upper_d = pi / (pi + tao + 1e-10)

                assert ((tao + pi - nu_hat_pos).abs()*unstable_or_cut_index).max() <= 1e-5, "pi+tao should always be the same as nu_hat_pos"

                # unstable_upper_bound_index: spec unstable, batch, in_C, H, W (same as patches last_lA)
                unstable_upper_bound_index = unstable_or_cut_index.logical_and(last_lA.patches < 0)
                # upper_d: (spec unstable, 1, in_C, H, W) (unfolded shape, same as patches last_lA)
                new_upper_d = new_upper_d * unstable_upper_bound_index.float() + \
                              upper_d * (1. - unstable_upper_bound_index.float())

                if last_uA is None: uA, ubias = None, 0.
                # lbias: unstable, batch
                # lA: unstable, batch, in_C, H, W (same as patches last_lA)
                lA, lbias = _bound_oneside(last_lA, lb_lower_d if lower_d is None else lower_d, new_upper_d, lower_b, upper_b, start_node, patch_size)

                # if general_beta[0].sum()!=0: import pdb; pdb.set_trace()
                # there is any unstable relus in this layer
                if unstable_or_cut_index.sum() > 0:
                    uC = -x_upper_unfold * nu_hat_pos
                    lC = -x_lower_unfold * nu_hat_pos
                    lbias = (pi * x_lower_unfold)
                    lbias[beta_mm_coeffs[0] <= uC] = 0.
                    lbias[beta_mm_coeffs[0] >= lC] = -beta_mm_coeffs[0][beta_mm_coeffs[0] >= lC].to(lbias)
                    # lbias: unstable, batch, in_C, H, W (same as patches last_lA) => lbias: (unstable, batch)
                    lbias = (lbias * unstable_or_cut_index.float()).view(lbias.shape[0], lbias.shape[1], -1).sum(-1)

            if last_uA is not None:
                # get the upper bound
                nu_hat_pos = (-last_uA.patches).clamp(max=0.).abs()
                tao = (-x_lower_unfold * nu_hat_pos - beta_mm_coeffs[1]) / (x_upper_minus_lower_unfold + 1e-10)
                pi = (x_upper_unfold * nu_hat_pos + beta_mm_coeffs[1]) / (x_upper_minus_lower_unfold + 1e-10)
                tao, pi = tao.clamp(min=0.), pi.clamp(min=0.)
                tao, pi = torch.min(tao, nu_hat_pos), torch.min(pi, nu_hat_pos)
                new_upper_d = pi / (pi + tao + 1e-10)

                assert ((tao + pi - nu_hat_pos).abs()*unstable_or_cut_index).max() <= 1e-5, "pi+tao should always be the same as nu_hat_pos"

                unstable_upper_bound_index = unstable_or_cut_index.logical_and((-last_uA.patches) < 0)
                new_upper_d = new_upper_d * unstable_upper_bound_index.float() + \
                              upper_d * (1. - unstable_upper_bound_index.float())

                uA, ubias = _bound_oneside(last_uA, new_upper_d, ub_lower_d if lower_d is None else lower_d, upper_b, lower_b, start_node, patch_size)
                if last_lA is None: lA, lbias = None, 0.

                if unstable_or_cut_index.sum() > 0:
                    uC = -x_upper_unfold * nu_hat_pos
                    lC = -x_lower_unfold * nu_hat_pos
                    ubias = -(pi * x_lower_unfold)
                    ubias[beta_mm_coeffs[1] <= uC] = 0.
                    ubias[beta_mm_coeffs[1] >= lC] = beta_mm_coeffs[1][beta_mm_coeffs[1] >= lC].to(ubias)
                    # ubias: unstable, batch, in_C, H, W (same as patches last_uA) => ubias: (unstable, batch)
                    ubias = (ubias * unstable_or_cut_index.float()).view(ubias.shape[0], ubias.shape[1], -1).sum(-1)
        else:
            # dense
            if last_lA is not None:
                # #####################
                # # C is nu_hat_pos
                # # last_lA: (spec unstable, batch, current flattened nodes (current_c*current_h*current_w))
                # nu_hat_pos = last_lA.clamp(max=0.).abs()
                # # pi, tao: spec_unstable, batch, current layer shape (same as last_lA)
                # tao = (-x.lower.unsqueeze(0) * nu_hat_pos - beta_mm_coeffs[0]) / (x.upper.unsqueeze(0) - x.lower.unsqueeze(0) + 1e-10)
                # pi = (x.upper.unsqueeze(0) * nu_hat_pos + beta_mm_coeffs[0]) / (x.upper.unsqueeze(0) - x.lower.unsqueeze(0) + 1e-10)
                # tao, pi = tao.clamp(min=0.), pi.clamp(min=0.)
                # tao, pi = torch.min(tao, nu_hat_pos), torch.min(pi, nu_hat_pos)
                # new_upper_d = pi / (pi + tao + 1e-10)

                # assert ((tao + pi - nu_hat_pos).abs()*unstable_or_cut_index).max() <= 1e-5, "pi+tao should always be the same as nu_hat_pos"

                # # need to customize the upper bound slope and lbias for (1) unstable relus and 
                # # (2) relus that are used with upper boundary relaxation
                # # original upper bound slope is u/(u-l) also equal to pi/(pi+tao) if no beta_mm_coeffs[0]
                # # now the upper bound slope should be pi/(p+tao) updated with beta_mm_coeffs[0]
                # unstable_upper_bound_index = unstable_or_cut_index.unsqueeze(0).logical_and(last_lA < 0)
                # # conv layer:
                # # upper_d: 1, batch, current_c, current_w, current_h
                # # unstable_upper_bound_index, new_upper_d: spec unstable, batch, current_c, current_w, current_h
                # # dense layer:
                # # upper_d: 1, batch, current flattened nodes
                # # unstable_upper_bound_index, new_upper_d: spec unstable, batch, current flattened nodes
                # new_upper_d = new_upper_d * unstable_upper_bound_index.float() +\
                #             upper_d * (1. - unstable_upper_bound_index.float())
                # #####################

                nu_hat_pos, tao, pi, new_upper_d, unstable_upper_bound_index = self.jit_arelu_lA(last_lA, x.lower, x.upper, beta_mm_coeffs, unstable_or_cut_index, upper_d)

                if last_uA is None: uA, ubias = None, 0.
                lA, lbias = _bound_oneside(last_lA, lb_lower_d if lower_d is None else lower_d, new_upper_d, lower_b, upper_b, start_node, patch_size)

                # if unstable_or_cut_index.sum() == 0: assert (lbias == 0).all(), "lbias should be 0 if no unstable relus"

                # #####################
                # # if no unstable, following bias should always be 0
                # if unstable_or_cut_index.sum() > 0:
                #     # update lbias with new form, only contribued by unstable relus
                #     uC = -x.upper.unsqueeze(0) * nu_hat_pos
                #     lC = -x.lower.unsqueeze(0) * nu_hat_pos
                #     # lbias: (spec unstable, batch, current flattened nodes) same as lA
                #     lbias = (pi * x.lower.unsqueeze(0))
                #     lbias[beta_mm_coeffs[0] <= uC] = 0.
                #     lbias[beta_mm_coeffs[0] >= lC] = -beta_mm_coeffs[0][beta_mm_coeffs[0] >= lC].to(lbias)
                #     # final lbias: (spec unstable, batch)
                #     lbias = (lbias * unstable_or_cut_index.unsqueeze(0).float()).view(lbias.shape[0], lbias.shape[1], -1).sum(-1)
                # #####################
                lbias = self.jit_arelu_lbias(unstable_or_cut_index, nu_hat_pos, beta_mm_coeffs, x.lower, x.upper, lbias, pi, tao)

            if last_uA is not None:
                # # C is nu_hat_pos
                # nu_hat_pos = (-last_uA).clamp(max=0.).abs()
                # tao = (- x.lower.unsqueeze(0) * nu_hat_pos - beta_mm_coeffs[1]) / (x.upper.unsqueeze(0) - x.lower.unsqueeze(0) + 1e-10)
                # pi = (x.upper.unsqueeze(0) * nu_hat_pos + beta_mm_coeffs[1]) / (x.upper.unsqueeze(0) - x.lower.unsqueeze(0) + 1e-10)
                # tao, pi = tao.clamp(min=0.), pi.clamp(min=0.)
                # tao, pi = torch.min(tao, nu_hat_pos), torch.min(pi, nu_hat_pos)
                # new_upper_d = pi / (pi + tao + 1e-10)

                # assert ((tao + pi - nu_hat_pos).abs()*unstable_or_cut_index).max() <= 1e-5, "pi+tao should always be the same as nu_hat_pos"

                # # unstable_or_cut_index = self.I.logical_or(self.arelu_coeffs.sum(0).view(self.I.shape) != 0)
                # unstable_upper_bound_index = unstable_or_cut_index.unsqueeze(0).logical_and(-last_uA < 0)
                # new_upper_d = new_upper_d * unstable_upper_bound_index.float() +\
                #             upper_d * (1. - unstable_upper_bound_index.float())
                nu_hat_pos, tao, pi, new_upper_d, unstable_upper_bound_index = self.jit_arelu_uA(last_uA, x.lower, x.upper, beta_mm_coeffs, unstable_or_cut_index, upper_d)

                # one can test uA by optimize -obj which should have the same obj value
                uA, ubias = _bound_oneside(last_uA, new_upper_d, ub_lower_d if lower_d is None else lower_d, upper_b, lower_b, start_node, patch_size)
                if last_lA is None: lA, lbias = None, 0.

                # if unstable_or_cut_index.sum() == 0: assert ubias == 0, "ubias should be 0 if no unstable relus"

                # if unstable_or_cut_index.sum() > 0:
                #     uC = -x.upper.unsqueeze(0) * nu_hat_pos
                #     lC = -x.lower.unsqueeze(0) * nu_hat_pos
                #     ubias = -(pi * x.lower.unsqueeze(0))
                #     ubias[beta_mm_coeffs[1] <= uC] = 0.
                #     ubias[beta_mm_coeffs[1] >= lC] = beta_mm_coeffs[1][beta_mm_coeffs[1] >= lC].to(ubias)
                #     ubias = (ubias * unstable_or_cut_index.unsqueeze(0).float()).view(ubias.shape[0], ubias.shape[1], -1).sum(-1)

                ubias = self.jit_arelu_ubias(unstable_or_cut_index, nu_hat_pos, beta_mm_coeffs, x.lower, x.upper, ubias, pi, tao)

        return lA, uA, lbias, ubias

    def input_cut(self, start_node, lA, uA, current_layer_shape, unstable_idx=None, batch_mask=None):
        # propagate input neuron in cut constraints through relu layer
        active_cuts = self.active_cuts[start_node.name]
        if self.x_coeffs is None or active_cuts.size(0) == 0:
            return lA, uA

        if type(lA) is Patches:
            A = lA if lA is not None else uA
            self.patch_trick(start_node, "input", A, current_layer_shape)

        general_beta = self.select_active_general_beta(start_node, unstable_idx)
        x_coeffs = self.x_coeffs[active_cuts]
        if batch_mask is not None:
            general_beta = general_beta[:, :, batch_mask]
        # general_beta: (2(0 lower, 1 upper), spec, batch, num_constrs)
        # x_coeffs: (num_constrs, flattened input dims)
        # beta_bias: (2(0 lower, 1 upper), batch, spec)
        lA, uA = self.general_beta_coeffs_addmm_to_A(lA, uA, general_beta, x_coeffs, current_layer_shape)
        return lA, uA

    def bias_cut(self, start_node, lb, ub, unstable_idx=None, batch_mask=None):
        active_cuts = self.active_cuts[start_node.name]
        if self.cut_bias is None or active_cuts.size(0) == 0:
            return lb, ub
        bias_coeffs = self.cut_bias[active_cuts]
        general_beta = self.select_active_general_beta(start_node, unstable_idx)
        if batch_mask is not None:
            general_beta = general_beta[:, :, batch_mask]
        # add bias for the bias term of general cut
        # general_beta: (2(0 lower, 1 upper), spec, batch, num_constrs)
        # bias_coeffs: (num_constrs,)
        # beta_bias: (2(0 lower, 1 upper), batch, spec)
        beta_bias = torch.einsum('sihj,j->shi', general_beta, bias_coeffs)
        lb = lb + beta_bias[0] if lb is not None else None
        ub = ub - beta_bias[1] if ub is not None else None
        return lb, ub


# Choose upper or lower bounds based on the sign of last_A
# this is a copy from activation.py
def _bound_oneside(last_A, d_pos, d_neg, b_pos, b_neg, start_node, patch_size):
    if last_A is None:
        return None, 0
    if type(last_A) == Tensor:
        A, bias = multiply_by_A_signs(last_A, d_pos, d_neg, b_pos, b_neg, contiguous=True)
        return A, bias
    elif type(last_A) == Patches:
        # if last_A is not an identity matrix
        assert last_A.identity == 0
        if last_A.identity == 0:
            # last_A shape: [out_c, batch_size, out_h, out_w, in_c, H, W]. Here out_c is the spec dimension.
            # or (unstable_size, batch_size, in_c, H, W) when it is sparse.
            patches = last_A.patches
            patches_shape = patches.shape
            if len(patches_shape) == 6:
                patches = patches.view(*patches_shape[:2], -1, *patches_shape[-2:])
                if d_pos is not None:
                    d_pos = d_pos.view(*patches_shape[:2], -1, *patches_shape[-2:])
                if d_neg is not None:
                    d_neg = d_neg.view(*patches_shape[:2], -1, *patches_shape[-2:])
                if b_pos is not None:
                    b_pos = b_pos.view(*patches_shape[:2], -1, *patches_shape[-2:])
                if b_neg is not None:
                    b_neg = b_neg.view(*patches_shape[:2], -1, *patches_shape[-2:])
            A_prod, bias = multiply_by_A_signs(patches, d_pos, d_neg, b_pos, b_neg)
            # prod has shape [out_c, batch_size, out_h, out_w, in_c, H, W] or (unstable_size, batch_size, in_c, H, W) when it is sparse.
            # For sparse patches the return bias size is (unstable_size, batch).
            # For regular patches the return bias size is (spec, batch, out_h, out_w).
            if len(patches_shape) == 6:
                A_prod = A_prod.view(*patches_shape)
            # Save the patch size, which will be used in init_slope() to determine the number of optimizable parameters.
            if start_node is not None:
                if last_A.unstable_idx is not None:
                    # Sparse patches, we need to construct the full patch size: (out_c, batch, out_h, out_w, c, h, w).
                    patch_size[start_node.name] = [last_A.output_shape[1], A_prod.size(1), last_A.output_shape[2], last_A.output_shape[3], A_prod.size(-3), A_prod.size(-2), A_prod.size(-1)]
                else:
                    # Regular patches.
                    patch_size[start_node.name] = A_prod.size()
            return Patches(A_prod, last_A.stride, last_A.padding, A_prod.shape, unstable_idx=last_A.unstable_idx, output_shape=last_A.output_shape), bias


# In patches mode, we need to unfold lower and upper slopes. In matrix mode we simply return.
# this is a copy from activation.py
def _maybe_unfold(d_tensor, last_A):
    # d_tensor (out_c, current_c, current_h, current_w): out_c shared the same alpha for spec layer
    if d_tensor is None:
        return None
    # if mode == "matrix" or d_tensor is None or last_A is None:
    if type(last_A) is not Patches or d_tensor is None or last_A is None:
        return d_tensor
    # Input are slopes with shape (spec, batch, input_c, input_h, input_w)
    # Here spec is the same as out_c.
    # assert d_tensor.ndim == 5
    origin_d_shape = d_tensor.shape
    if d_tensor.ndim == 6:
        d_tensor = d_tensor.view(*origin_d_shape[:2], -1, *origin_d_shape[-2:])
    d_shape = d_tensor.size()
    # Reshape to 4-D tensor to unfold.
    d_tensor = d_tensor.view(-1, *d_tensor.shape[-3:])
    # unfold the slope matrix as patches. Patch shape is [spec * batch, out_h, out_w, in_c, H, W).
    d_unfolded = inplace_unfold(d_tensor, kernel_size=last_A.patches.shape[-2:], stride=last_A.stride, padding=last_A.padding)
    # Reshape to (spec, batch, out_h, out_w, in_c, H, W); here spec_size is out_c.
    d_unfolded_r = d_unfolded.view(*d_shape[:-3], *d_unfolded.shape[1:])
    if last_A.unstable_idx is not None:
        if d_unfolded_r.size(0) == 1:
            if len(last_A.unstable_idx) == 3:
                # Broadcast the spec shape, so only need to select the reset dimensions.
                # Change shape to (out_h, out_w, batch, in_c, H, W) or (out_h, out_w, in_c, H, W).
                d_unfolded_r = d_unfolded_r.squeeze(0).permute(1, 2, 0, 3, 4, 5)
                d_unfolded_r = d_unfolded_r[last_A.unstable_idx[1], last_A.unstable_idx[2]]
            elif len(last_A.unstable_idx) == 4:
                # [spec, batch, output_h, output_w, input_c, H, W]
                # to [output_h, output_w, batch, in_c, H, W]
                d_unfolded_r = d_unfolded_r.squeeze(0).permute(1, 2, 0, 3, 4, 5)
                d_unfolded_r = d_unfolded_r[last_A.unstable_idx[2], last_A.unstable_idx[3]]
            else:
                raise NotImplementedError()
            # output shape: (unstable_size, batch, in_c, H, W).
        else:
            d_unfolded_r = d_unfolded_r[last_A.unstable_idx[0], :, last_A.unstable_idx[1], last_A.unstable_idx[2]]
        # For sparse patches, the shape after unfold is (unstable_size, batch_size, in_c, H, W).
    # For regular patches, the shape after unfold is (spec, batch, out_h, out_w, in_c, H, W).
    if d_unfolded_r.ndim != last_A.patches.ndim:
        d_unfolded_r = d_unfolded_r.unsqueeze(2).unsqueeze(-4)
    return d_unfolded_r
