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
import torch
import numpy as np

def build_bound_implication_graph(cut_info):
    """
    Build the bound implication graph based on the extracted neuron implications.
    Graph node X corresponds to the actual neuron X // 2 in model. The parity of X indicates the neuron's inactive domain (even) or active domain (odd).

    dependency_edge_list: X => Y indicates the neuron X // 2's bound implicates neuron Y // 2's bound based on X or Y's parity.
    optimized_bound_indices: X => (Y, z) indicates with neuron X // 2's active/inactive bound, we can imply neuron Y's bound change to z.
    components[0]: X => {} stores all successor node in dependency graph given neuron X is assumed to be inactive (ub=0).
    components[1]: X => {} stores all successor node in dependency graph given neuron X is assumed to be active (lb=0).
    components[2]: X => {} stores all successor node in dependency graph based neurons X's split (i.e. union of components[0] and components[1]).
    """

    all_cuts = cut_info["all_cuts"]
    unstable_idx_list = cut_info["unstable_idx_list"]
    ref_idx = cut_info["ref_idx"]

    idx_mapping = {}
    all_unstable_neurons = 0
    for layers in range(len(unstable_idx_list)):
        for idx in (unstable_idx_list[layers]):
            idx_mapping[(ref_idx[layers], idx.item())] = all_unstable_neurons
            all_unstable_neurons += 1

    unstable_idx = []
    for layers in range(len(unstable_idx_list)):
        for idx in unstable_idx_list[layers]:
            unstable_idx.append([ref_idx[layers], idx.item()])

    total_layers = max(ref_idx) + 1
    optimized_bound_indices = [[] for i in range(all_unstable_neurons * 2 * total_layers * 2)]
    optimized_bound_values = [[] for i in range(all_unstable_neurons * 2 * total_layers * 2)]

    dependency_edge_list = [[] for i in range(all_unstable_neurons * 2)]

    all_improved = 0
    all_flipped = 0
    for key, _ in all_cuts.items():
        values = _["improved"]
        for i in range(values.shape[1]):
            nd, nidx = ref_idx[values[0][i].to(torch.long).item()], values[1][i].to(torch.long).item()
            base_idx = idx_mapping[(ref_idx[values[2][i].to(torch.long).item()], values[3][i].to(torch.long).item())]
            if key[-3] == "<": # base_idx's inactive domain
                base_idx = base_idx * 2
            else: # base_idx's active domain
                base_idx = base_idx * 2 + 1

            if key[0] == 'l': # target idx's lower bound
                optimized_bound_indices[base_idx * total_layers * 2 + nd * 2 + 0].append(nidx)
                optimized_bound_values[base_idx * total_layers * 2 + nd * 2 + 0].append(values[4][i].item())
            else:
                optimized_bound_indices[base_idx * total_layers * 2 + nd * 2 + 1].append(nidx)
                optimized_bound_values[base_idx * total_layers * 2 + nd * 2 + 1].append(values[4][i].item())

        all_improved += values.shape[1]

        values = _["flipped"]
        for i in range(values.shape[1]):
            improved_idx = idx_mapping[(ref_idx[values[0][i].to(torch.long).item()], values[1][i].to(torch.long).item())]
            base_idx = idx_mapping[(ref_idx[values[2][i].to(torch.long).item()], values[3][i].to(torch.long).item())]
            if key[-3] == "<": # base_idx's inactive domain
                base_idx = base_idx * 2
            else: # base_idx's active domain
                base_idx = base_idx * 2 + 1

            if key[0] == 'l': # target idx's lower bound > 0 => active domain
                improved_idx = improved_idx * 2 + 1
            else: # inactive domain otherwise
                improved_idx = improved_idx * 2
            dependency_edge_list[base_idx].append(improved_idx)
        all_flipped += values.shape[1]

    for i in range(all_unstable_neurons * 2 * total_layers * 2):
        optimized_bound_values[i] = torch.FloatTensor(optimized_bound_values[i]).cuda()
        optimized_bound_indices[i] = torch.LongTensor(optimized_bound_indices[i]).cuda()

    total_relus = total_layers

    dependency_statistics = np.zeros(total_relus).astype(np.int64) # the number of implications comes from each layer.
    indegree = np.zeros(all_unstable_neurons)
    dependency_statistics_all = np.zeros((total_relus, total_relus)) # the number of implications comes from layer to layer.
    for i in range(all_unstable_neurons * 2):
        dependencies = len(dependency_edge_list[i])
        dd = np.zeros(total_relus)
        for j in dependency_edge_list[i]:
            layer, idx = unstable_idx[j // 2]
            dd[layer] += 1
            indegree[j // 2] += 1
        layer, idx = unstable_idx[i // 2]
        dependency_statistics[layer] += dependencies
        for l in range(total_relus):
            dependency_statistics_all[layer][l] += dd[l]

    for _, i in enumerate(ref_idx):
        print(f"[Layer {i}]: All implications: {dependency_statistics[i]}")

    print("Implication matrix:")
    print(dependency_statistics_all)
    components = [[] for i in range(all_unstable_neurons)]

    max_comp_size, avg_comp_size = 0, 0

    def traverse(start_idx):
        node_queue = []
        node_queue.append(start_idx)
        visited = np.zeros(all_unstable_neurons * 2)
        visited[start_idx] = True
        head, tail = 0, 1
        while (head < tail):
            cur_node = node_queue[head]
            head += 1
            for neighbour in dependency_edge_list[cur_node]:
                if visited[neighbour] == False:
                    visited[neighbour] = True
                    node_queue.append(neighbour)
                    tail += 1
        return node_queue

    for neuron_idx in range(all_unstable_neurons):
        F_queue = traverse(neuron_idx * 2)
        T_queue = traverse(neuron_idx * 2 + 1)

        all_queue = []
        for i in range(len(F_queue)): all_queue.append(F_queue[i] // 2)
        for i in range(len(T_queue)): all_queue.append(T_queue[i] // 2)

        all_queue = np.unique(np.array(all_queue)).tolist()
        components[neuron_idx].append(F_queue)
        components[neuron_idx].append(T_queue)
        components[neuron_idx].append(all_queue)

        max_comp_size = max(max_comp_size, len(all_queue))
        avg_comp_size += len(all_queue)

    print("Total improved dependencies: %d, flipped dependencies: %d" % (all_improved, all_flipped))
    print("Maximum Comp size: %d, Average Comp size: %.4f" % (max_comp_size, avg_comp_size / all_unstable_neurons))

    return components, idx_mapping, optimized_bound_indices, optimized_bound_values, unstable_idx