#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import time
import torch
import arguments

from utils import Timer, convert_history_from_list
from tensor_storage import TensorStorage
from auto_LiRPA.patches import Patches
from auto_LiRPA.concretize_func import constraints_solving, sort_out_constr_batches, construct_constraints

class DomainClipper:
    """
    This class is used to clip the domain of the input to the network
    based on the CROWN-IBP bounds, then update the intermediate bounds.

    We save the unstable index for the final layer and update the
    intermediate bounds based on the unstable index.
    """
    def __init__(
            self, A, x, final_name, input_name,
            lower_bounds=None, upper_bounds=None,
            mask=None, check_intersection=False
        ):
        """
        Initialize the domain clipper object with the following parameters.

        @params:    A (dict):   
                        key: Each layer's output name
                        value (dict):
                            key: Each layer's input name
                            value (dict):
                                lA (Tensor): CROWN lA matrix
                                uA (Tensor): CROWN uA matrix or None
                                lbias (Tensor): CROWN lbias matrix
                                ubias (Tensor): CROWN ubias matrix or None
                                unstable_idx (Tensor): unstable index or None
                    x: The model input (BoundedTensor)
                    output_name (list): The name of the output layers
                    input_name (list): The name of the input layers

        @init:      self.lA (dict): The lower bound A matrix for each layer
                        key: Each layer's output name
                        value (Tensor): The lower bound ceof A
                    self.uA (dict): The upper bound A matrix for each layer
                        key: Each layer's output name
                        value (Tensor): The upper bound ceof A
                    self.lbias (dict): The lower bound bias matrix for each layer
                        key: Each layer's output name
                        value (Tensor): The lower bound bias
                    self.ubias (dict): The upper bound bias matrix for each layer
                        key: Each layer's output name
                        value (Tensor): The upper bound bias
                    self.unstable_idx (dict): The unstable index for each layer
                        key: Each layer's output name
                        value (Tensor): The unstable neuron index
        """
        x_shape = x.shape
        self.x_L = x.ptb.x_L
        self.x_U = x.ptb.x_U
        self.final_name = final_name
        self.input_name = input_name
        self.biccos_cuts_num = 0

        self.lA = {} # save the lower bound coefficient A for each unstable neuron per layer
        self.uA = {} # save the upper bound coefficient A for each unstable neuron per layer
        self.lbias = {} # save the lower bound bias for each unstable neuron per layer
        self.ubias = {} # save the upper bound bias for each unstable neuron per layer
        self.unstable_idx = {} # save the unstable index for each unstable neuron per layer
        self.mask = {} # save the mask for each unstable neuron per layer
        self.mapping = {} # save the mapping from the original index to the unstable index
        self.true_indices = {} # save the true indices for each unstable neuron per layer
        self.stop_func = None

        self.timer = Timer()
        self.sol_timer = Timer()

        clip_config = arguments.Config['bab']['clip_n_verify']
        self.using_final_layer = clip_config['final_layer']
        self.prune = clip_config['prune']
        self.rearrage = clip_config['rearrange_constraints']

        self.clip_interm_domain = clip_config['clip_interm_domain']['enabled']
        self.clip_input_domain = clip_config['clip_interm_domain']['with_input']
        self.topk_objective = clip_config['clip_interm_domain']['topk_objective']

        self.max_iter = 2

        # Define a mapping from the keys in A to the corresponding attributes in self.
        keys_map = {
            'lA': self.lA,
            'uA': self.uA,
            'lbias': self.lbias,
            'ubias': self.ubias,
            'unstable_idx': self.unstable_idx
        }

        print('Initializing Domain Clipper...')
        now_time = time.time()
        for k1 in A:
            val = A[k1][input_name[0]]
            print(f"Processing layer: {k1}, input name: {input_name[0]}")
            if k1 == self.final_name:
                continue
            print(f"Layer: {k1}")
            # Assign each key's value to the appropriate dictionary in a single loop
            for key_name, attr_dict in keys_map.items():
                if type(val[key_name]) is Patches:
                    print(f"    {key_name} is a Patch object with shape {val[key_name].shape}, transferring to matrix.")
                    attr_dict[k1] = val[key_name].to_matrix(x_shape)
                    if val[key_name] is not None and key_name != 'unstable_idx':
                        print(f"    {key_name} transfered from Patch with shape: {attr_dict[k1].shape}")
                else:
                    attr_dict[k1] = val[key_name]
                    if val[key_name] is not None and key_name != 'unstable_idx':
                        print(f"    {key_name} with shape: {attr_dict[k1].shape}")

        if check_intersection:
            print('check the intersection of the input domain')
            for k in self.lA.keys():
                print(f"Layer: {k}")

                #broadcast x_L/x_U along the batch dimension so that we can try out each constraint individually
                exp_x_L = x.ptb.x_L.repeat(self.lA[k].shape[1], *[1]*(x.ptb.x_L.ndim - 1))
                exp_x_U = x.ptb.x_U.repeat(self.lA[k].shape[1], *[1]*(x.ptb.x_U.ndim - 1))
                test_lA = self.lA[k].transpose(0, 1)
                test_lbias = self.lbias[k].transpose(0, 1)
                test_uA = self.uA[k].transpose(0, 1)
                test_ubias = self.ubias[k].transpose(0, 1)

                # tests using one constraint at a time (like a split depth of 1)
                print('lower')
                self.intersection_check(x, self.lA[k], self.lbias[k])
                self.clip_domains(x_L=exp_x_L, x_U=exp_x_U,
                                    lA=test_lA, lbias=test_lbias, interm_bounds=None, is_lower=True)
                print('upper')
                self.intersection_check(x, -self.uA[k], -self.ubias[k])
                self.clip_domains(x_L=exp_x_L, x_U=exp_x_U,
                                    lA=test_uA, lbias=test_ubias, interm_bounds=None, is_lower=False)

        print(f'Domain Clipper Initialized in {time.time() - now_time} s.')

    def intersection_check(self, x, lA, lbias):
        """
        Check if the input box x is completely inside, completely outside, or intersects
        with the halfspace defined by lA x + lbias <= 0.
        
        @params:    x (Tensor): The input box, represented as a tensor with its lower
                                and upper bounds stored in x.ptb.x_L and x.ptb.x_U
                    lA (Tensor): The coefficient matrix for the halfspace constraints
                    lbias (Tensor): The bias vector for the halfspace constraints

        @return:    tuple: (intersect_mask, redundant_mask, infeasible_mask)
                        - intersect_mask: Boolean tensor indicating which constraints intersect with the box
                        - redundant_mask: Boolean tensor indicating which constraints are redundant (all points satisfy them)
                        - infeasible_mask: Boolean tensor indicating which constraints are infeasible (no point satisfies them)
        """
        x_L = x.ptb.x_L.flatten(1)
        x_U = x.ptb.x_U.flatten(1)
        flat_lA = lA.flatten(2)

        # For each component i, the max of lA[i] * x[i] is achieved at x_U[i] if lA[i] > 0, and at x_L[i] if lA[i] < 0
        pos_mask = flat_lA > 0
        neg_mask = flat_lA < 0

        # Compute max_val for each constraint, i.e., max(lA x + lbias) over the box
        max_term_pos = pos_mask * flat_lA * x_U.unsqueeze(1)
        max_term_neg = neg_mask * flat_lA * x_L.unsqueeze(1)
        max_term = max_term_pos + max_term_neg
        max_val = torch.sum(max_term, dim=2) + lbias

        # Compute min_val for each constraint, i.e., min(lA x + lbias) over the box
        min_term_pos = pos_mask * flat_lA * x_L.unsqueeze(1)
        min_term_neg = neg_mask * flat_lA * x_U.unsqueeze(1)
        min_term = min_term_pos + min_term_neg
        min_val = torch.sum(min_term, dim=2) + lbias

        # A constraint is redundant if the maximum value is <= 0 (all points satisfy)
        redundant_mask = max_val <= 0
        # A constraint is infeasible if the minimum value is > 0 (no point satisfies)
        infeasible_mask = min_val > 0
        # A constraint intersects the box if it's neither redundant nor infeasible (some points satisfy, some don't)
        intersect_mask = ~(redundant_mask | infeasible_mask)

        # Print summary
        intersect_count = intersect_mask.sum().item()
        redundant_count = redundant_mask.sum().item()
        infeasible_count = infeasible_mask.sum().item()
        total_constraints = lA.shape[0] * lA.shape[1]

        # Original approach (for backward compatibility)
        center = (x_L + x_U) / 2
        distances = _all_dist(center, flat_lA, lbias)
        orig_intersect_mask = (distances > torch.zeros_like(center).unsqueeze(1)).any(2)
        orig_intersect = orig_intersect_mask.sum().item()

        print(f"    {orig_intersect}/{total_constraints} intersecting lower bound constraints (original)")
        print(f"    {intersect_count}/{total_constraints} intersecting constraints (some points satisfy, some don't)")
        print(f"    {redundant_count}/{total_constraints} redundant constraints (all points satisfy)")
        print(f"    {infeasible_count}/{total_constraints} infeasible constraints (no point satisfies)")

    def update_unstable_idx(self, updated_mask, net):
        """
        Update the unstable index by mapping masks to the correct input nodes.

        @params:
            updated_mask (dict): Dict where keys are operation node names and
                                values are lists of unstable masks.
            net: LiRPANet.

        @init:
            self.mask (dict): The mask for each unstable neuron per layer.
            self.mapping (dict): Mapping from original to unstable index.
        """
        # Iterate over each operation node and its corresponding list of masks.
        for op_node_name, mask_list in updated_mask.items():
            # Find the operation node in the network graph.
            node = net.net[op_node_name]
            # Get the names of the input nodes for this operation.
            input_node_names = [inp.name for inp in node.inputs]

            # Zip the input names with their corresponding masks.
            for input_name, mask in zip(input_node_names, mask_list):
                # Skip any masks that are empty.
                if mask is None or not mask.any():
                    continue

                # The rest of the logic is the same, but uses the correct
                # `input_name` as the key `k`.
                val = mask.to('cpu')
                self.mask[input_name] = val

                true_indices = val.view(-1).nonzero(as_tuple=True)[0]
                self.true_indices[input_name] = true_indices
                self.mapping[input_name] = {idx.item(): i for i, idx in enumerate(true_indices)}

                # Check if remasking of other attributes is needed.
                if self.lA[input_name].shape[1] > len(true_indices) + 1e2:
                    print(f"Layer: {input_name}, unstable: {len(true_indices)}, lA shape: {self.lA[input_name].shape[1]}")
                    print(f"Remasking lA, lbias, uA, ubias for layer {input_name}")
                    self.lA[input_name] = self.lA[input_name][:, true_indices]
                    self.uA[input_name] = self.uA[input_name][:, true_indices]
                    self.lbias[input_name] = self.lbias[input_name][:, true_indices]
                    self.ubias[input_name] = self.ubias[input_name][:, true_indices]

        # Rebuild the key-to-index mappings at the end with all updated keys.
        self.key_to_layer = {key: index for index, key in enumerate(self.mapping.keys())}
        self.layer_to_key = {index: key for index, key in enumerate(self.mapping.keys())}

        print('Unstable Index Updated for Domain Clipper')

    def build_final_lA_lbias(self, histories):
        """
        Build a single lA and lbias for each history using index references.

        @param histories: A list of 'history' dictionaries.
        @return: final_lA:    [batch, 1, x, x, x].
                final_lbias: [batch, 1, ...].
        """
        # Store just the indices and metadata
        tensor_refs = []  # List of (key, unstable_idx, is_upper_bound)

        for history in histories:
            for key, value in history.items():
                history_tuple = convert_history_from_list(value)
                idx, status = history_tuple[0], history_tuple[1]

                if idx.size(0) == 0:
                    continue

                last_idx = idx[-1].item()
                last_status = status[-1].item()
                unstable_idx = self.mapping[key][last_idx]
                is_upper = (last_status > 0)

                # Just store the key, index and whether to use upper or lower bound
                tensor_refs.append((key, unstable_idx, is_upper))
                break

        # Now gather the actual tensors only once at the end
        lA_list = []
        lbias_list = []

        for key, unstable_idx, is_upper in tensor_refs:
            if is_upper:
                # Use negated upper bound
                lA_list.append(-self.uA[key][0][unstable_idx])
                lbias_list.append(-self.ubias[key][0][unstable_idx])
            else:
                # Use lower bound directly
                lA_list.append(self.lA[key][0][unstable_idx])
                lbias_list.append(self.lbias[key][0][unstable_idx])

        # Stack only at the end
        final_lA = torch.stack(lA_list, dim=0).unsqueeze(1)
        final_lbias = torch.stack(lbias_list, dim=0).unsqueeze(1)

        return final_lA, final_lbias
    
    def build_final_lA_lbias_all(self, histories):
        """
        Build all lA and lbias for each history using index references.

        @params: histories: A list of 'history' dictionaries.
        @return: final_lA:    [batch, max_len, x, x, x].
                final_lbias: [batch, max_len, ...].
        """
        # For each history, store a list of (key, unstable_idx, is_upper) tuples
        history_refs = []

        for history in histories:
            local_refs = []

            for key in history.keys():
                history[key] = convert_history_from_list(history[key])

            for key, (idx, status, _, _, _) in history.items():
                if idx.size(0) == 0:
                    continue

                # Get all indices and statuses for this key
                idx_list = idx.tolist() if idx.ndim > 0 else [idx.item()]
                status_list = status.tolist() if status.ndim > 0 else [status.item()]

                # Store references as tuples: (key, unstable_idx, is_upper)
                for i, s in zip(idx_list, status_list):
                    unstable_idx = self.mapping[key][i]
                    is_upper = (s == 1.0)
                    local_refs.append((key, unstable_idx, is_upper))

            history_refs.append(local_refs)

        # Find max length across histories
        max_len = max([len(refs) for refs in history_refs]) if history_refs else 0

        # Now materialize tensors for each history
        padded_lA_list = []
        padded_lbias_list = []

        for refs in history_refs:
            if not refs:
                # Empty history case
                device = next(iter(self.lA.values()))[0].device
                cat_lA = torch.empty(0, 1, 1, 1, device=device)
                cat_lbias = torch.empty(0, 1, device=device)
            else:
                # Gather tensors
                lA_list = []
                lbias_list = []

                for key, unstable_idx, is_upper in refs:
                    if is_upper:
                        lA_list.append(-self.uA[key][0][unstable_idx])
                        lbias_list.append(-self.ubias[key][0][unstable_idx])
                    else:
                        lA_list.append(self.lA[key][0][unstable_idx])
                        lbias_list.append(self.lbias[key][0][unstable_idx])

                # Combine for this history
                cat_lA = torch.stack(lA_list, dim=0).unsqueeze(1)
                cat_lbias = torch.stack(lbias_list, dim=0).unsqueeze(1)

            # Pad if needed
            if cat_lA.size(0) < max_len:
                lA_padded = pad_first_dim(cat_lA, max_len)
                lbias_padded = pad_first_dim(cat_lbias, max_len)
            else:
                lA_padded = cat_lA
                lbias_padded = cat_lbias

            padded_lA_list.append(lA_padded)
            padded_lbias_list.append(lbias_padded)

        # Stack across histories
        final_lA = torch.stack(padded_lA_list, dim=0).squeeze(2)
        final_lbias = torch.stack(padded_lbias_list, dim=0).squeeze(2)

        return final_lA, final_lbias

    def get_constraints(self, histories):
        if self.iter_idx <= self.max_iter:
            print('Use all histories')
            A, bias = self.build_final_lA_lbias_all(histories)
        else:
            A, bias = self.build_final_lA_lbias(histories)
        return A, bias

    def optimize_interm_bounds(self, domains, x_L, x_U, interm_bounds, split_activations, mask=None, constraints=None):
        self.timer.start('optimize_interm_bounds')

        self.timer.start('get_constraints')
        if domains is not None:
            if self.topk_objective > 0:
                objective_indices = self.get_branching_scores(domains, split_activations, self.topk_objective)
                print(f"Objective masks generated with topk: {self.topk_objective} objectives.")
            else:
                print("No objective masks generated.")
                return interm_bounds
            try:
                constraints = self.get_constraints(domains['history'])
            except Exception as e:
                print(f"Error occurred while getting constraints: {e}")
                return interm_bounds
        else:
            assert constraints is not None, "If domains are None, constraints must be provided."
            objective_indices = {}  # No objectives if no domains provided
        if constraints is not None:
            constraints = constraints = construct_constraints(constraints[0], constraints[1],
                                                    torch.zeros_like(constraints[1]),
                                                    x_L.shape[0], x_L.flatten(1).shape[1])
            constraints, sorted_out_batches = sort_out_constr_batches(x_L, x_U, constraints, rearrange_constraints=self.rearrage)
        self.timer.add('get_constraints')

        self.timer.start('concretize_interm_bounds')
        unstable_interm_bounds = self.concretize_interm_bounds(
            x_L, x_U, constraints, sorted_out_batches, objective_indices=objective_indices
        )
        self.timer.add('concretize_interm_bounds')

        self.timer.start('update_interm_bounds')
        mask = mask if mask is not None else self.mask
        interm_bounds = update_interm_bounds(interm_bounds, unstable_interm_bounds, self.final_name, mask, verbose=True)
        self.timer.add('update_interm_bounds')

        self.timer.add('optimize_interm_bounds')
        self.sol_timer.print()
        self.timer.print()
        return interm_bounds

    def concretize_interm_bounds(self, x_L, x_U, constraints=None, sorted_out_batches=None, objective_indices=None):
        new_interm_bounds = {}
        if objective_indices is None:
            objective_indices = {} # Ensure it's a dict

        for keys in self.uA.keys():
            if self.lA[keys] is not None and self.uA[keys] is not None:
                # Get the specific mask for this layer, default to None if not present
                current_objective_indices = objective_indices.get(keys, None)
                lA, uA = self.lA[keys][-1:].flatten(2), self.uA[keys][-1:].flatten(2)
                lbias, ubias = self.lbias[keys][-1:], self.ubias[keys][-1:]
                try:
                    lower_bound = constraints_solving(x_L, x_U, lA, constraints, -1,
                        sorted_out_batches, timer=self.sol_timer, objective_indices=current_objective_indices
                    ).squeeze(-1)
                    upper_bound = constraints_solving(x_L, x_U, uA, constraints, 1,
                        sorted_out_batches, timer=self.sol_timer, objective_indices=current_objective_indices
                    ).squeeze(-1)
                    # Add bias in-place if these tensors are not needed elsewhere unmodified
                    new_interm_bounds[keys] = [
                        lower_bound.add_(lbias),
                        upper_bound.add_(ubias)
                    ]
                except:
                    # Naive concretization (doesn't use mask)
                    new_interm_bounds[keys] = [
                        concretize_bounds(x_L.flatten(1), x_U.flatten(1), lA, lbias, True),
                        concretize_bounds(x_L.flatten(1), x_U.flatten(1), uA, ubias, False)
                    ]

        return new_interm_bounds

    def domain_clip_ReLU(self, d, x, interm_bounds):
        """
        Main function to clip the input domain based on the CROWN bounds
        and update the intermediate bounds.
        """
        self.timer.start('domain_clip_ReLU')

        print('\n####### Updating Interm Bound #######')
        self.timer.start('get_constraints')
        constraints = self.get_constraints(d['history'])
        A, bias = constraints
        self.intersection_check(x, A, bias)
        x_L, x_U = x.ptb.x_L, x.ptb.x_U
        self.timer.add('get_constraints')

        self.timer.start('clip_domains')
        x_L, x_U, prune_mask, interm_bounds = self.clip_domains(
                                    x_L, x_U, A, bias, interm_bounds,
                                    is_lower=True
                                )
        self.timer.add('clip_domains')

        if self.prune and prune_mask is not None:
            self.timer.start('prune')
            d = prune_d(prune_mask, d)
            self.timer.add('prune')

        self.timer.add('domain_clip_ReLU')
        self.timer.print()
        print('#######################################\n')
        return x_L, x_U, interm_bounds, d, prune_mask

    def domain_clip_outputs(self, A, x, interm_bounds):
        """
        Function to clip the output domain based on the provided constraints
        and update the intermediate bounds.
        """
        print('\n####### Updating Output Bound #######')
        lA = A[self.final_name][self.input_name[0]]['lA']
        lbias = A[self.final_name][self.input_name[0]]['lbias']

        x_L, x_U, _, interm_bounds = self.clip_domains(x_L=x.ptb.x_L, x_U=x.ptb.x_U,
                                                            lA=lA, lbias=lbias, interm_bounds=interm_bounds, is_lower=True)

        print('#######################################\n')
        return x_L, x_U, interm_bounds

    def clip_domains(self, x_L, x_U, lA, lbias, interm_bounds, is_lower=True):
        """
        Takes subdomains (or original domain) and shrinks along dimensions
        to remove verified portions of the input domain.

        Note: We will always deal with the constraints as >= form.

        @param          x_L:    The lower bound on the inputs of the subdomains
                                    shape: (batch, input_dim) or possibly (batch, c, h, w)
                        x_U:    The upper bound on the inputs of the subdomains
                                    shape: (batch, input_dim) or possibly (batch, c, h, w)
                        lA:     CROWN lA for subdomains
                                    shape: (batch, num_constr, input_dim)
                        lbias:  CROWN lbias for subdomains.
                                    shape: (batch, num_constr)
                        interm_bounds: (dict) intermediate bounds to be updated
                        is_lower: (bool) whether we are clipping a lower- or upper-bound problem

        @return:    
            - x_L_new: pruned lower bounds with shape [feasible_batch, ...]
            - x_U_new: pruned upper bounds with shape [feasible_batch, ...]
            - feasible_mask: Boolean tensor indicating which rows of the original batch are feasible
        """
        # Flatten lA's last two dims if needed
        lA = lA.flatten(2)     # shape: [batch, num_constr, input_dim]
        batches, num_constr, input_dim = lA.shape
        
        # Capture the original shape of x_L, x_U
        #   e.g. x_shape might be [batch, c, h, w] or [batch, input_dim].
        x_shape = x_L.shape

        # Flatten x_L and x_U to shape [batch, input_dim] for the core math
        x_L = x_L.clone().reshape(x_shape[0], input_dim)
        x_U = x_U.clone().reshape(x_shape[0], input_dim)

        ######## Main procedure to solve for x_L_new, x_U_new ########
        x_L_new, x_U_new = parallel_clipping(x_L, x_U, lA, lbias, num_constr, batches, is_lower)

        # Identify infeasible (where L >= U in ALL dims)
        infeasible_mask = (x_L_new.flatten(1) > x_U_new.flatten(1)).any(dim=1)
        if self.prune and not infeasible_mask.sum() == 0:
            feasible_mask = ~infeasible_mask
        else: 
            feasible_mask = None
        x_L_new = torch.clamp(x_L_new, min=x_L, max=x_U)
        x_U_new = torch.clamp(x_U_new, min=x_L, max=x_U)

        dimensionwise_shrinkage_stats(x_L, x_U, x_L_new, x_U_new)

        if self.prune and feasible_mask is not None:
            print(f'Pruning #{infeasible_mask.sum().item()} infeasible subdomains')
            x_L = x_L[feasible_mask]
            x_U = x_U[feasible_mask]
            x_L_new = x_L_new[feasible_mask]
            x_U_new = x_U_new[feasible_mask]

        # Update intermediate bounds in dictionary
        # (pass feasible_mask to prune their batch dimension as well)
        if interm_bounds is not None:
            # Recompute intermediate bounds for the feasible subset
            new_interm_bounds = self.concretize_interm_bounds(x_L_new, x_U_new)
            interm_bounds = update_interm_bounds(interm_bounds, new_interm_bounds, self.final_name, self.mask, feasible_mask)

        feasible_batch = x_L_new.shape[0]
        # Rebuild the shape as (feasible_batch, x_shape[1], x_shape[2], ...)
        # so that the only dimension changed is the 0th dimension.
        new_x_shape = list(x_shape)
        new_x_shape[0] = feasible_batch

        return (
            x_L_new.view(*new_x_shape),
            x_U_new.view(*new_x_shape),
            feasible_mask,
            interm_bounds
        )

    @torch.no_grad()
    def get_branching_scores(self, domains, split_activations, topk=50):
        lbs, ubs, lAs = domains['lower_bounds'], domains['upper_bounds'], domains['lAs']
        batch = lbs[self.final_name].shape[0]
        # --- Generate Masks (Top-K per layer per batch) ---
        objective_indices = {}
        for layer_name in lbs.keys():
            if layer_name == self.final_name:
                continue
            if layer_name not in self.true_indices.keys():
                continue
            A_key = split_activations[layer_name][0][0].name
            ratio = ((-lbs[layer_name]).clamp(0, None) * ubs[layer_name].clamp(0, None)) / (ubs[layer_name] - lbs[layer_name])
            ratio *= (-lAs[A_key].mean(dim=1)).clamp(0, None)
            layer_scores = ratio.reshape(batch, -1)

            _, num_neurons = layer_scores.shape
            # Ensure k is not larger than the number of neurons
            actual_k = min(topk, num_neurons, len(self.true_indices[layer_name]))
            if actual_k > 0:
                # Get indices of top-k scores for each batch item
                layer_scores = layer_scores[:, self.true_indices[layer_name]]
                _, topk_indices = torch.topk(layer_scores, k=actual_k, dim=1) # Shape (batch, k)
                objective_indices[layer_name] = topk_indices
            else:
                # Handle case with 0 neurons or k=0
                mask = torch.zeros_like(layer_scores, dtype=torch.bool)
                objective_indices[layer_name] = mask[:, self.true_indices[layer_name]]
        return objective_indices

    def get_stop_criterion_and_iter(self, stop_func, iter_idx):
        self.stop_func = stop_func
        self.iter_idx = iter_idx

def prune_d(mask, d):
    # Convert the boolean mask to indices once.
    mask_idx = torch.nonzero(mask, as_tuple=False).view(-1)
    if len(mask_idx) == 0:
        return
    mask_list = mask_idx.tolist()  # For iterating in list comprehensions.
    max_idx = mask_idx.max().item() + 1

    # Predefine key sets for faster membership checking.
    list_keys   = {'history', 'betas', 'intermediate_betas', 'split_history'}
    dict_keys   = {'lower_bounds', 'upper_bounds', 'lAs', 'mask'}
    tensor_keys = {'cs', 'thresholds', 'x_Ls', 'x_Us'}

    for key, value in d.items():
        if key in list_keys:
            # Process list values; ensure index is within range.
            d[key] = [value[i] for i in mask_list if i < len(value)]
        elif key in dict_keys:
            # Process dict values; use index_select on dimension 0.
            d[key] = {k: v.index_select(0, mask_idx) for k, v in value.items() if v.size(0) >= max_idx}
        elif key == 'alphas':
            # Process nested dicts: index_select on the 3rd dimension.
            d[key] = {
                sub_key: {
                    tensor_key: tensor.index_select(2, mask_idx)
                    for tensor_key, tensor in sub_nested_dict.items()
                    if tensor.size(2) >= max_idx
                }
                for sub_key, sub_nested_dict in value.items()
            }
        elif key in tensor_keys:
            # Process tensor values; index_select on dimension 0.
            d[key] = value.index_select(0, mask_idx)
    return d

def parallel_clipping(x_L, x_U, lA, lbias, num_constr, batches, is_lower=True, num_iters=1):
    # Center + radius
    xhat = (x_U + x_L) / 2
    eps  = (x_U - x_L) / 2
    sign = 1 if is_lower else -1

    concretized = concretize_bounds(x_L, x_U, lA, lbias, is_lower)
    eop = 'bsn,bn->bsn'  # for torch.einsum broadcasting

    num_iters = min(num_iters, num_constr, x_L.shape[1])
    if num_iters > 1:
        print(f"Parallel clipping will be executed for {num_iters} iterations.")
    for i in range(num_iters):

        if i > 0:
            concretized = concretize_bounds(x_L, x_U, lA, lbias, is_lower)

        # (concretized) - (lA*xhat) + sign*(|lA|*eps)
        concrete_minus_one = (concretized.reshape(batches, num_constr, 1)
                                - torch.einsum(eop, lA, xhat)
                                + sign * torch.einsum(eop, lA.abs(), eps))

        candidates = - concrete_minus_one / lA   # shape: [batch, num_constr, input_dim]

        # For the lower bound, keep only solutions where lA < 0, because sign*lA < 0 => x >= that solution
        # For the upper bound, keep only solutions where lA > 0
        torch_inf = torch.full_like(candidates, float('inf'))
        lower_candidates = torch.where(
                    sign * lA < 0,
                    candidates,
                    -torch_inf
                )
        upper_candidates = torch.where(
                    sign * lA > 0,
                    candidates,
                    torch_inf
                )

        # Combine with original x_L/x_U
        x_L_new = torch.max(lower_candidates.max(dim=1)[0], x_L)
        x_U_new = torch.min(upper_candidates.min(dim=1)[0], x_U)
    return x_L_new, x_U_new

def _all_dist(pts, lA, lbias):
    numerator = torch.einsum('bmn,bn->bm', lA, pts) + lbias
    denominator = torch.norm(lA, dim=2)
    return (numerator / (denominator + 1e-10)).unsqueeze(-1)
    
def dimensionwise_shrinkage_stats(x_L, x_U, x_L_new, x_U_new, eps=1e-12):
    """
    Compare old and new bounding boxes in high-dim by computing dimension-wise
    side-length ratios, then aggregating them (e.g. via mean) per sample.
    Also reports how many dimensions have shrunk for each sample, on average.
    
    @params:    x_L, x_U:        (batch, input_dim) lower and upper bounds of original domain.
                x_L_new, x_U_new:(batch, input_dim) lower and upper bounds of new (shrunk) domain.
                eps:             small constant to avoid divide-by-zero.
    """
    print("input domain shrinkage stats")
    # 1) Compute side lengths (clamp negative intervals to 0)
    old_side_lengths = (x_U - x_L).clamp(min=0.)
    new_side_lengths = (x_U_new - x_L_new).clamp(min=0.)

    # 2) Compute dimension-wise ratios:
    #       ratio_{i,j} = new_side_lengths[i,j] / max(old_side_lengths[i,j], eps)
    #    If old_side_lengths[i,j] ~ 0, ratio could be large, 
    #    but we clamp denominator with eps to avoid inf/nan.
    dim_ratios = new_side_lengths / (old_side_lengths + eps)  # (batch, input_dim)
    best_shrinkages = dim_ratios.amin(dim=1)
    mean_best_shrinkage_per_batch = best_shrinkages.mean().item()
    
    # 3) Aggregate each sample’s ratios by taking the mean across all dimensions
    mean_ratios_per_batch = dim_ratios.mean(dim=1)  # shape: (batch,)

    # 4) Summary stats of mean ratio
    min_ratio = mean_ratios_per_batch.min().item()
    max_ratio = mean_ratios_per_batch.max().item()
    mean_ratio = mean_ratios_per_batch.mean().item()

    # Print all shrinkages < 1
    print(f" Per-sample mean side-length ratio stats across the batch:")
    print(f"  min:  {min_ratio:.6%},  max:  {max_ratio:.6%},  mean: {mean_ratio:.6%}")
    print(f"    best shrinkage {dim_ratios.min().item():.3%}, average of best shrinkages: {mean_best_shrinkage_per_batch:.3%}")

    # Still keep the topk display for reference
    best_k_shrinkages = min(20, x_L.size(0))
    print(f"    top {best_k_shrinkages}: {(best_shrinkages.topk(best_k_shrinkages, largest=False).values).tolist()}")
    return min_ratio

def pad_first_dim(tensor, max_len):
    """
    Pad 'tensor' along the first dimension (dim=0) up to 'max_len'.
    For example, if tensor.shape == [n, 1, x, x, x],
    the result will be [max_len, 1, x, x, x], with tensor[0] duplicated in
    the extra (max_len-n) rows.

    If n == 0, we return a zero-filled tensor of shape [max_len, *tensor.shape[1:]].
    """
    n = tensor.size(0)
    if n == max_len:
        # Already at desired size
        return tensor

    if n == 0:
        # If no data, fill with zeros
        shape = (max_len,) + tensor.shape[1:]
        return torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)

    # Build an index list: [0, 1, 2, ..., n-1, 0, 0, 0, ...] or simply replicate index 0
    # But here we only replicate row 0 for the extra positions:
    idx = list(range(n))
    while len(idx) < max_len:
        idx.append(0)  # Duplicate the first row

    return tensor[idx, ...]

def concretize_bounds(x_L, x_U, lA, lbias, is_lower=True):
    xhat, eps = (x_L + x_U)/2, (x_U - x_L)/2
    if isinstance(is_lower, bool):
        status = -1 if is_lower is True else 1
    # use lbias to concretize dm_lb for the subdomains
    eop = 'bsn,bn->bs'  # specify batch matrix multiplication
    # b = batch, s = num_constr, n = input_dim
    return status * torch.einsum(eop, lA.abs(), eps) + torch.einsum(eop, lA, xhat) + lbias

def expand_x_batch(x_L, x_U, x_shape, batches):
    """
    Expands the input domain to match the batch size by repeating the tensors.

    This function automatically handles tensors of any dimension.
    """
    print(f'Expanding the input domain of shape {x_shape[0]} to match the batch size: {batches}')

    if x_shape[0] == 0:
        raise ValueError("The first dimension of the input tensor cannot be zero.")

    num_verifiable = batches // x_shape[0]

    # Create a list of repeat factors.
    # The first dimension (batch) is repeated 'num_verifiable' times.
    # All other dimensions are not repeated (multiplied by 1).
    repeats = [1] * x_L.dim()  # Get the number of dimensions
    repeats[0] = num_verifiable

    # Use the splat (*) operator to pass the list as arguments
    x_L = x_L.repeat(*repeats)
    x_U = x_U.repeat(*repeats)
    x_shape = x_L.shape  # Update shape, now shape[0] == batches

    return x_L, x_U, x_shape

def update_interm_bounds(interm_bounds,
                        new_interm_bounds,
                        final_name,
                        unstable_mask,
                        prune_mask=None,
                        verbose=False):
    """
    Update each lb tensor in lb_dict based on some new_interm_bounds,
    optionally pruning along the batch dimension via prune_mask.

    @params:
        interm_bounds : dict
            key: Each layer's name
            value: a list
                v[0]: lb with shape [batch, input_dim]
                v[1]: ub with shape [batch, input_dim]

        new_interm_bounds : dict
            key: same layer names as above
            value: a list
                v[0]: lb with shape [batch, num_unstable_neurons]
                v[1]: ub with shape [batch, num_unstable_neurons]

        prune_mask : torch.BoolTensor or None
            A boolean mask of shape [batch]. If provided, we only keep 
            the rows where prune_mask == True.

    @return:
        updated_interm_bounds: A dict with same structure as interm_bounds
                            but updated (and optionally pruned).
    """
    updated_interm_bounds = {}
    for key in interm_bounds.keys():
        # 1) If it's the final layer, or if new_interm_bounds doesn't have it, just copy over
        if key == final_name or key not in new_interm_bounds.keys():
            updated_interm_bounds[key] = interm_bounds[key]
            continue

        # 2) Extract lb/ub from both dictionaries
        lb, ub = interm_bounds[key]
        some_lb, some_ub = new_interm_bounds[key]

        # ---------------------------------------------------
        # 3) PRUNE BATCH if a prune_mask is provided
        # ---------------------------------------------------
        if prune_mask is not None:
            lb = lb[prune_mask]           # shape: [new_batch, input_dim]
            ub = ub[prune_mask]           # shape: [new_batch, input_dim]

        # 4) The "mask" in self.mask[key] is usually about neuron indices (columns), not the batch.
        #    e.g. mask might indicate which neurons are "unstable" to be updated.
        mask = unstable_mask[key]  # Typically a boolean or index list for columns

        # Safety check
        if mask.sum().item() != some_lb.size(1):
            print(f'Warning: Key {key} has mismatch: mask size={mask.sum().item()}, some_lb size={some_lb.size()}')
            # If mismatch, just keep original (already pruned) lb/ub
            updated_interm_bounds[key] = [lb, ub]
            continue

        # 5) Slice the columns (neurons) out from lb and ub
        #    Here mask is presumably shape [num_neurons] or something like that
        lb_masked = lb[:, mask[0]]  # shape: [new_batch, N]
        ub_masked = ub[:, mask[0]]

        if isinstance(lb_masked, TensorStorage):
            lb_masked = lb_masked.tensor()
            ub_masked = ub_masked.tensor()

        if isinstance(some_lb, TensorStorage):
            some_lb = some_lb.tensor()
            some_ub = some_ub.tensor()

        some_lb = some_lb.to(lb_masked.device)
        some_ub = some_ub.to(ub_masked.device)

        # 6) Compute better bounds
        lb_best = torch.maximum(lb_masked, some_lb)  # shape: [new_batch, N]
        ub_best = torch.minimum(ub_masked, some_ub)

        # 7) Write back the improved lb/ub into the original (pruned) lb/ub
        lb[:, mask[0]] = lb_best
        ub[:, mask[0]] = ub_best

        if verbose:
            # 8) Collect some stats
            lb_diff = lb_best - lb_masked
            ub_diff = ub_best - ub_masked
            #  new batch size after pruning
            new_batch_size = lb.size(0)
            if new_batch_size > 0:

                lb_num_improved = (lb_diff > 0).sum().item() / new_batch_size
                ub_num_improved = (ub_diff < 0).sum().item() / new_batch_size

                lb_improved = lb_diff.sum().item() / new_batch_size
                ub_improved = ub_diff.sum().item() / new_batch_size

                print(f' layer: {key}')
                print(f'    lower bounds improved: average # {lb_num_improved}, value {lb_improved}')
                print(f'    upper bounds improved: average # {ub_num_improved}, value {ub_improved}')
        # 9) Save updated (and pruned) lb, ub
        updated_interm_bounds[key] = [lb, ub]

    return updated_interm_bounds
