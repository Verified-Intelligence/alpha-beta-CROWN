#########################################################################
##         This file is part of the alpha-beta-CROWN verifier          ##
##                                                                     ##
## Copyright (C) 2021, Huan Zhang <huan@huan-zhang.com>                ##
##                     Kaidi Xu <xu.kaid@northeastern.edu>             ##
##                     Shiqi Wang <sw3215@columbia.edu>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Yihan Wang <yihanwang@ucla.edu>                 ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import time
import numpy as np
import torch
from collections import defaultdict
from sortedcontainers import SortedList

from branching_domains_input_split import pick_out_batch, add_domain_parallel, InputDomain, input_split_batch
from attack_pgd import AdamClipping

Visited, Solve_slope = 0, False


def batch_verification(d, net, batch,  decision_thresh=0, lr_alpha=0.01, iteration=10, prop_mat=None, prop_rhs=None,
                       shape=None, branching_candidates=3, branching_method=None):

    relu_start = time.time()
    global Visited

    # prev_bounds = [dom.lower_bound for dom in d]
    slopes, dm_l_all, dm_u_all, selected_domains, selected_dims = pick_out_batch(d, decision_thresh, batch, device=net.x.device)
    # print('pick out time:', time.time()-relu_start)
    if dm_l_all is not None:
        time0 = time.time()

        new_dm_l_all, new_dm_u_all = input_split_batch(net, dm_l_all, dm_u_all, slopes, shape=shape,
                                                       selected_dims=selected_dims, branching_method=branching_method)

        print('decision time:', time.time()-time0)

        ret = net.get_lower_bound_naive(dm_l=new_dm_l_all, dm_u=new_dm_u_all, slopes=slopes, branching_candidates=branching_candidates,
                                        shortcut=False, lr_alpha=lr_alpha, iteration=iteration)

        dom_ub, dom_lb, slopes, selected_dims = ret
        # print('dom_lb parallel: ',  dom_lb)

        # For debugging bounds.
        """
        batch_size = min(len(prev_bounds), batch)
        for i in range(batch_size):
            prev_lb = prev_bounds[i]
            if prev_lb > dom_lb[i].item():
                print(f'prev={prev_lb} new_1={dom_lb[i]} l={new_dm_l_all[i].cpu().numpy()} u={new_dm_u_all[i].cpu().numpy()} old_l={dm_l_all[i].cpu().numpy()} old_u={dm_u_all[i].cpu().numpy()}')
            if prev_lb > dom_lb[i+batch_size].item():
                print(f'prev={prev_lb} new_2={dom_lb[i+batch_size]} l={new_dm_l_all[i+batch_size].cpu().numpy()} u={new_dm_u_all[i+batch_size].cpu().numpy()} old_l={dm_l_all[i].cpu().numpy()} old_u={dm_u_all[i].cpu().numpy()}')
        """

        time1 = time.time()

        add_domain_parallel(d, dom_lb, dom_ub, new_dm_l_all.detach(), new_dm_u_all.detach(), selected_domains, slopes,
                            selected_dims=selected_dims, decision_thresh=decision_thresh)
        Visited += len(selected_domains) * 2  # one unstable neuron split to two nodes

    relu_end = time.time()
    print('insert to domain / total batch time: {:4f}/{:4f}'.format(relu_end - time1, relu_end - relu_start))
    print('length of domains:', len(d))

    if len(d) > 0:
        global_lb = d[0].lower_bound
    else:
        print("No domains left, verification finished!")
        return torch.tensor(decision_thresh + 1e-7)

    print(f"Current lb:{global_lb.item()}")

    print('{} neurons visited\n'.format(Visited))

    return global_lb


def relu_bab_parallel(net, domain, x, batch=64, decision_thresh=0, iteration=20, max_subproblems_list=100000,
                      timeout=3600, record=False, lr_alpha=0.1, lr_init_alpha=0.5, model_ori=None, shape=None,
                      share_slopes=False, adv_check=0, prop_mat=None, prop_rhs=None,
                      branching_candidates=3, branching_method=None, all_prop=None):
    start = time.time()
    prop_mat = torch.tensor(prop_mat, dtype=torch.float, device=x.device)
    prop_rhs = torch.tensor(prop_rhs, dtype=torch.float, device=x.device)

    global Visited, Solve_slope
    Visited, Solve_slope = 0, False

    global_ub, global_lb, _,  lower_bounds, upper_bounds, pre_relu_indices, slope, dm_l, dm_u, selected_dims = net.build_the_model(
    domain, x, decision_thresh, lr_init_alpha, share_slopes=share_slopes, shape=shape,
    input_grad=branching_method=='input_grad')

    # if not opt_intermediate_beta:
    #     # If we are not optimizing intermediate layer bounds, we do not need to save all the intermediate alpha.
    #     # We only keep the alpha for the last layer.
    #     new_slope = defaultdict(dict)
    #     output_layer_name = net.net.final_name
    #     for relu_layer, alphas in slope.items():
    #         new_slope[relu_layer][output_layer_name] = alphas[output_layer_name]
    #     slope = new_slope

    print(global_lb)
    if len(global_lb.flatten()) > 1:
        if global_lb.max() > decision_thresh:
            return global_lb.max(), None, [[time.time() - start, global_lb.max().item()]], 0
        else:
            global_lb = global_lb.max()
    else:
        if global_lb > decision_thresh:
            return global_lb, None, [[time.time()-start, global_lb.item()]], 0

    # This is the first (initial) domain.
    candidate_domain = InputDomain(global_lb, global_ub, slope=slope, dm_l=dm_l, dm_u=dm_u, selected_dims=selected_dims)
    domains = SortedList()
    domains.add(candidate_domain)

    glb_record = [[time.time()-start, global_lb.item()]]
    while len(domains) > 0:
        last_glb = global_lb

        if Visited > adv_check:
            # adv_time = time.time()
            # check whether adv example found
            # adv_example = torch.cat([domains[0].dm_l, domains[0].dm_u, domains[-1].dm_l, domains[-1].dm_u])
            adv_example = torch.cat([torch.cat([domains[i].dm_l, domains[i].dm_u]) for i in range(min(10, len(domains)))])
            adv_example = torch.cat([adv_example, domains[-1].dm_l, domains[-1].dm_u])
            ret = model_ori(adv_example).detach()  # .cpu().numpy()
            vec = prop_mat.matmul(ret.t())
            sat = torch.all(vec <= prop_rhs.reshape(-1, 1), dim=0)
            # print(vec.shape, sat.shape)
            if (sat==True).any():
                idx = torch.where(sat==True)[0][0]
                adv_example = adv_example[idx]
                print('adversarial example found!', ret[idx].cpu().numpy())
                del domains
                return global_lb, adv_example, glb_record, Visited

        global_lb = batch_verification(domains, net, batch, iteration=iteration, decision_thresh=decision_thresh,
                                       lr_alpha=lr_alpha, prop_mat=prop_mat, prop_rhs=prop_mat, shape=shape,
                                       branching_candidates=branching_candidates, branching_method=branching_method)

        # once the lower bound stop improving we change to solve slope mode
        if not Solve_slope and time.time()-start > 20 and global_lb.cpu() <= last_glb.cpu():
            net.solve_slope = True
            _, global_lb, _, lower_bounds, upper_bounds, pre_relu_indices, slope, dm_l, dm_u, selected_dims = net.build_the_model(
                domain, x, decision_thresh, lr_init_alpha, share_slopes=share_slopes)

            global_lb = global_lb.max()

            # This is the first (initial) domain.
            candidate_domain = InputDomain(global_lb, global_ub, slope=slope, dm_l=dm_l, dm_u=dm_u, selected_dims=selected_dims)
            domains = SortedList()
            domains.add(candidate_domain)
            Solve_slope = True

        if time.time()-start > 80 and len(domains) > 3000:
            # perform attack with massively random starts finally
            sample_lower_limit = domain[0] - x
            sample_upper_limit = domain[1] - x
            delta = (torch.empty(size=(1000000, 5), device=x.device).uniform_() * (sample_upper_limit - sample_lower_limit) + sample_lower_limit).requires_grad_(True)
            opt = AdamClipping(params=[delta], lr=(domain[1] - domain[0]).max()/50.)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, 0.98)
            prop_rhs_all = []
            for p, m in all_prop:
                prop_rhs_all.append(torch.tensor(p, dtype=torch.float, device=x.device))

            for _ in range(100):
                inputs = x + delta
                output = model_ori(inputs)
                vec = prop_mat.matmul(output.t())
                loss = -vec
                loss.mean().backward()
                print('attacking loss', loss.mean())
                # print(sum(delta.grad != 0))

                if time.time() - start > timeout:
                    print('time out!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    del domains
                    # np.save('glb_record.npy', np.array(glb_record))
                    return global_lb, None, glb_record, Visited

                for p in prop_rhs_all:
                    sat = torch.all(p.matmul(output.t()) <= prop_rhs.reshape(-1, 1), dim=0) # .cpu()
                    # print(vec.shape, sat.shape)
                    if (sat == True).any(): # TODO COMPARE ON GPU
                        idx = torch.where(sat == True)[0][0]
                        adv_example = inputs[idx]
                        assert (adv_example >= domain[0]).all()
                        assert (adv_example <= domain[1]).all()

                        print('adversarial example found!', output[idx].detach().cpu().numpy())
                        del domains
                        return global_lb, adv_example, glb_record, Visited

                opt.step(clipping=True, lower_limit=sample_lower_limit, upper_limit=sample_upper_limit, sign=1)
                opt.zero_grad(set_to_none=True)
                scheduler.step()

        if len(domains) > max_subproblems_list:
            print("no enough memory for the domain list")
            del domains
            return global_lb, None, glb_record, Visited

        if time.time() - start > timeout:
            print('time out!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            del domains
            # np.save('glb_record.npy', np.array(glb_record))
            return global_lb, None, glb_record, Visited

        if record: glb_record.append([time.time() - start, global_lb.item()])

    del domains
    return global_lb, None, glb_record, Visited
