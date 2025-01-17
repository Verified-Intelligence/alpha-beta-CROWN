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
""" Utilities for cuts. Mainly for manual cuts and cplex-based cuts."""
import time
import os
import struct
import psutil
import torch
from psutil import NoSuchProcess
from collections import defaultdict
import numpy as np

def read_cut(cut_file):
    # read linear constraints from file
    f = open(cut_file, "r")
    lines = f.readlines()
    num_constr = len(lines)
    coeffs = []
    biases = []
    x_decision = [[] for _ in range(num_constr)]
    x_coeffs = [[] for _ in range(num_constr)]
    relu_decision = [[] for _ in range(num_constr)]
    relu_coeffs = [[] for _ in range(num_constr)]
    arelu_decision = [[] for _ in range(num_constr)]
    arelu_coeffs = [[] for _ in range(num_constr)]
    pre_decision = [[] for _ in range(num_constr)]
    pre_coeffs = [[] for _ in range(num_constr)]
    c = []
    SPECIAL_CHARS = ["*", ">", "<", "="]
    for constr_idx, line in enumerate(lines):
        line = line.replace(" ", "").replace("\n", "")
        # layer = []
        # neuron_idx = []
        # coeff = []
        print(line)
        stack = ""
        i = 0
        while i <= len(line) - 1:
            # print(i, ch)
            ch = line[i]
            if ch not in SPECIAL_CHARS:
                stack += ch
                i += 1
                continue
            if ch == "*":
                # print(i, ch, stack)
                # coeff.append(float(stack))
                coeff = float(stack)
                stack = ""
                if line[i + 1] == "r":
                    decision, coeffs = relu_decision[constr_idx], relu_coeffs[constr_idx]
                    i = i + len("*relu_")
                elif line[i + 1] == "a":
                    decision, coeffs = arelu_decision[constr_idx], arelu_coeffs[constr_idx]
                    i = i + len("*arelu_")
                elif line[i + 1] == "p":
                    decision, coeffs = pre_decision[constr_idx], pre_coeffs[constr_idx]
                    i = i + len("*pre_")
                elif line[i + 1] == "x":
                    decision, coeffs = x_decision[constr_idx], x_coeffs[constr_idx]
                    i = i + len("*x_")
                layer, neuron_idx = -1, -1
                while line[i] not in SPECIAL_CHARS:
                    ch = line[i]
                    if ch in ["+", "-"]:
                        break
                    if ch == "_":
                        layer = int(stack)
                        stack = ""
                    else:
                        stack += ch
                    i += 1
                neuron_idx = int(stack)
                stack = ""
                decision.append([layer, neuron_idx])
                coeffs.append(coeff)
            if ch in [">", "<"]:
                constr_c = 1 if ch == ">" else -1
                c.append(constr_c)
                i += 1
            if ch == '=':
                i += 1
        bias = float(stack)
        # print("final:", layer, neuron_idx, coeffs, bias)
        # coeffs.append(coeff)
        # decision.append([[l, idx] for l, idx in zip(layer, neuron_idx)])
        biases.append(bias)
        # print(decision, coeffs, biases, c)
    f.close()
    # format used for building lp solver and beta crown for general cut
    # each element constraint format {node type}_{relu layer index}_{flattened neuron index}
    # node type:
    # x: input in the cut constraints, x_decision [-1, idx]
    # relu: post activation var in the cut constraint
    # pre: pre activation var in the cut constraint
    # arelu: integer var in the cut constraint
    # cut is a list, each element in cut is a constraint
    cut = [{"x_decision": x_decision[i], "x_coeffs": x_coeffs[i],
                "relu_decision": relu_decision[i], "relu_coeffs": relu_coeffs[i],
                "arelu_decision": arelu_decision[i], "arelu_coeffs": arelu_coeffs[i],
                "pre_decision": pre_decision[i], "pre_coeffs": pre_coeffs[i],
                "bias": biases[i], "c": c[i]}
            for i in range(num_constr)]

    return cut


def read_cut_efficient(net, cut_bin, indx_bin):
    """
        An auxiliary function for efficient cut reading by first checking the timestamp of the cut to read and
        only reading when the cut is present and updated from net.net.cut_timestamp
        Used in cut_verification() and fetch_cut_from_cplex()
    :param net: auto_LiRPA module
    :param cut_bin: cut path
    :param indx_bin: cut indx path
    :return: cut, cut_timestamp --- if not updated, return None, -1
    """

    print(f"Trying to loading cuts from {cut_bin}")
    cut_timestamp = get_cplex_cut_timestamp(cut_bin)
    if cut_timestamp is None:
        # cut does not exist
        cuts, cut_timestamp = None, -1
        print("CPLEX cuts have not been generated yet.")
    else:
        if getattr(net.net, "var_names", None) is None:
            # if var_name has not been parsed yet, we parse it;
            # otherwise we skip since the indx does not change any more in following cuts
            var_names = parse_cplex_indx(indx_bin)
            net.net.var_names = var_names
            print("CPLEX cuts names loaded.")
        if cut_timestamp == net.net.cut_timestamp:
            # the cut is not updated, skip to read the cut
            # note: if there is no cut now, cut_timestamp == -1; if there is no previous cut, net.net.cut_timestamp == -1 by init
            # so "both have no cut" case is naturally handled
            print("Cuts have not been updated in CPLEX; still using old cuts.")
            cuts, cut_timestamp = None, -1
        else:
            relu_layer_names = [relu_layer.name for relu_layer in net.net.relus]
            pre_relu_layer_names = [relu_layer.inputs[0].name for relu_layer in net.net.relus]
            # record the cut file modification time as the timestamp to distinguish difference cut versions
            cuts, cut_timestamp = parse_cplex_cuts(cut_bin, net.net.var_names, relu_layer_names, pre_relu_layer_names)
    return cuts, cut_timestamp


def parse_cplex_indx(indx_bin):
    # typedef struct {
    #     int32_t signature = 0x58444e49;  // "INDX"
    #     int32_t first_col_num;  // first column index, usually 0.
    #     int32_t num_cols;
    #     int32_t names_offset;  // num_cols NULL terminated strings.
    # } IndexFileHeader;
    try:
        indx_file = open(indx_bin, "rb")
        indx_mbr = indx_file.read()
        assert indx_mbr[:4] == b"INDX", "indx_bin should be index binary"
        first_col_num, num_cols, names_offset = struct.unpack('iii', indx_mbr[4:16])
        assert names_offset == 16, "names_offset not match"
        names = struct.unpack(f'{len(indx_mbr) - names_offset}s', indx_mbr[names_offset:])[0].decode("ascii")
        names = names.split("\x00")[:-1]
        indx_file.close()
        return names
    except Exception as e:
        print('unable to parse indx from {}'.format(indx_bin))
        return None


def get_cplex_cut_timestamp(cut_bin):
    """
        get the modification time from cplex generated cut file and encode it to internal "signature" timestamp
    :param cut_bin: binary cut file path
    :return: encoded timestamp (original timestamp * 100 mod 1e8) or None (if the file does not exist or enable to access)
    """
    try:
        # get modification time of the cut
        # in case the timestamp number is too large, so use reminder
        # * 100.0 ensures the granularity of int is 0.01 s
        cut_timestamp = int(os.path.getmtime(cut_bin) * 100.0) % 100000000
    except Exception:
        cut_timestamp = None
    return cut_timestamp


def parse_cplex_cuts(cut_bin, var_names, relu_layer_names, pre_relu_layer_names):

    # typedef struct {
    #     int32_t signature = 0x53545543;  // "CUTS"
    #     int32_t num_rows;  // number of cut constraints
    #     int32_t num_elements;
    #     int32_t row_begin_idx_offset;  // length is num_rows;
    #     int32_t rhs_values_offset;  // length is num_rows;
    #     int32_t row_indices_offset;  // length is num_elements;
    #     int32_t row_values_offset;  // length is num_elements;
    # } CutFileHeader;

    try:
        cut_timestamp = get_cplex_cut_timestamp(cut_bin)
        # make sure the file exist, otherwise we can exit early
        assert cut_timestamp is not None

        cut_file = open(cut_bin, "rb")
        cut_mbr = cut_file.read()
        assert cut_mbr[:4] == b"CUTS", "cut_bin should be cuts binary"
        num_rows, num_elements, row_begin_idx_offset, rhs_values_offset, row_indices_offset, row_values_offset = struct.unpack('6i', cut_mbr[4:28])
        print(f"cut {cut_bin}: total {num_rows} constraints, {num_elements} nonzero elements")
        # import pdb; pdb.set_trace()
        ### header || row_begin_idx || rhs || nonzero var indices || nonzero values
        row_begin_idx = list(struct.unpack(f'{num_rows}Q', cut_mbr[row_begin_idx_offset:rhs_values_offset]))
        rhs = struct.unpack(f'{num_rows}d', cut_mbr[rhs_values_offset:row_indices_offset])
        row_indices = struct.unpack(f'{num_elements}i', cut_mbr[row_indices_offset:row_values_offset])
        row_values = struct.unpack(f'{num_elements}d', cut_mbr[row_values_offset:])
        cut_file.close()

        cuts = []
        row_begin_idx.append(num_elements)
        for cut_idx in range(num_rows):
            skip = False
            row_begin_start = row_begin_idx[cut_idx]
            row_begin_end = row_begin_idx[cut_idx+1]
            cut = {"x_decision": [], "x_coeffs": [],
                    "relu_decision": [], "relu_coeffs": [],
                    "arelu_decision": [], "arelu_coeffs": [],
                    "pre_decision": [], "pre_coeffs": [],
                    "bias": [], "c": []}
            # nonzero_vars = [var_names[indx] for indx in row_indices[row_begin_start:row_begin_end]]
            # nonzero_values = row_values[row_begin_start:row_begin_end]
            # print(nonzero_vars)
            for cut_var_idx in range(row_begin_start, row_begin_end):
                # row_indices[row_begin_start:row_begin_end]
                var_name = var_names[row_indices[cut_var_idx]]
                # var_name = var_hash_map[row_indices[cut_var_idx]]
                coeff = row_values[cut_var_idx]
                if "inp_" in var_name:
                    neuron_idx = int(var_name.replace("inp_", ""))
                    cut["x_decision"].append([-1, neuron_idx])
                    cut["x_coeffs"].append(coeff)
                elif "aReLU" in var_name:
                    relu_name, neuron_idx = var_name.replace("aReLU", "").split("_")
                    relu_idx = relu_layer_names.index(relu_name)
                    cut["arelu_decision"].append([relu_idx, int(neuron_idx)])
                    cut["arelu_coeffs"].append(coeff)
                elif "ReLU" in var_name:
                    relu_name, neuron_idx = var_name.replace("ReLU", "").split("_")
                    relu_idx = relu_layer_names.index(relu_name)
                    cut["relu_decision"].append([relu_idx, int(neuron_idx)])
                    cut["relu_coeffs"].append(coeff)
                elif "lay" in var_name:
                    layer_name, neuron_idx = var_name.replace("lay", "").split("_")
                    if layer_name not in pre_relu_layer_names:
                        # maybe it will add last layer neuron in the cut constraint, we need to skip that constraint!
                        print(f"Warning: var{var_name} not in pre_relu_layer_names {pre_relu_layer_names}, skip this constraint!")
                        skip = True
                        break
                    relu_idx = pre_relu_layer_names.index(layer_name)
                    cut["pre_decision"].append([relu_idx, int(neuron_idx)])
                    cut["pre_coeffs"].append(coeff)
                else:
                    print(f"Warning: var{var_name} not supported!")
                    exit()

            cut["bias"] = rhs[cut_idx]
            cut["c"] = -1

            if not skip:
                cuts.append(cut)
        # import pdb; pdb.set_trace()
        return cuts, cut_timestamp
    except Exception as e:
        print('unable to get the cut from filepath: {} '
              'maybe the first batch of cut has not come out yet from cplex'.format(cut_bin))
        return None, -1


def fetch_cut_from_cplex(net, sync_to_net=True):
    """
    :param net: AutoLiRPA instance
    :param sync_to_net:
    whether to inject the loaded cut into the instance, when called inside bab loops, it is true;
    when called right before the bab loop, it is false.
    :return:
    """
    start_time = time.time()
    ######## parse cplex cuts files ###########
    if (
        net.mip_building_proc is not None
        and net.mip_building_proc.exitcode is not None
        and net.mip_building_proc.exitcode != 0
    ):
        raise RuntimeError("MIP building process is not terminated correctly, please check the process. Return code", net.mip_building_proc.exitcode)
    read_from = 'cplex_processes' if getattr(net, 'cplex_processes', None) is not None else 'processes'
    process_dict = getattr(net, read_from, None)
    if process_dict is None:
        print('Fetch cut process: mps construction process is still running')
    else:
        for key, value in process_dict.items():
            if (value['c'] == net.c.detach().cpu()).all():
                print("Matched cut cplex process, internal label idx = {}".format(key))
                MODEL_NAME = value['_fname_stamped']

                indx_bin = f"{MODEL_NAME}.indx"
                cut_bin = f"{MODEL_NAME}.cuts"
                cuts, cut_timestamp = read_cut_efficient(net, cut_bin, indx_bin)

                if cuts is not None:
                    if sync_to_net:
                        # recreate the global cut_module
                        net.cutter.cuts = cuts
                        net.cutter.cut_timestamp = cut_timestamp

                        # construct new cut_module
                        cut_module = net.cutter.construct_cut_module()
                        net.net.cut_module = cut_module
                        net.net.cut_timestamp = cut_timestamp
                        for m in net.net.relus:
                            m.cut_module = cut_module
                else:
                    cuts, cut_timestamp = None, -1
                print(f'cuts preparing time: {time.time() - start_time:.4f}')
                return cuts, cut_timestamp
        print('Fetch cut process: mps for current label is not ready yet')
    return None, -1


def read_cut_pt(cut_file, use_float64_in_last_iteration=False):
    cut_raw = torch.load(cut_file)

    biases = []
    x_decision, x_coeffs = [], []
    relu_decision, relu_coeffs = [], []
    arelu_decision, arelu_coeffs = [], []
    pre_decision, pre_coeffs = [], []
    c = []
    for layer, d in enumerate(cut_raw):
        kact_cons = d['kact_cons']
        for con in kact_cons:
            # FIXME: many are duplicate
            # TODO: prune variables with zero coefficient
            k = con['k']
            if use_float64_in_last_iteration:
                con['cons'] = con['cons'].astype(np.float32)
            exists = {}
            for i in range(con['cons'].shape[0]):
                if tuple(con['cons'][i]) in exists:
                    # Duplicate
                    continue
                exists[tuple(con['cons'][i])] = True
                coeffs = con['cons'][i]

                pre_decision_ = []
                pre_coeffs_ = []
                for j in range(k):
                    if con['cons'][i][1+j] != 0:
                        pre_coeffs_.append(float(coeffs[1+j]))
                        pre_decision_.append([layer, con['varsid'][j]])
                pre_decision.append(pre_decision_)
                pre_coeffs.append(pre_coeffs_)

                relu_decision_ = []
                relu_coeffs_ = []
                for j in range(k):
                    if con['cons'][i][1+k+j] != 0:
                        relu_coeffs_.append(float(coeffs[1+k+j]))
                        relu_decision_.append([layer, con['varsid'][j]])
                relu_decision.append(relu_decision_)
                relu_coeffs.append(relu_coeffs_)

                # bias should be on the right-hand-side (ERAN's is on the left-hand-side)
                biases.append(float(-coeffs[0]))
                c.append(1)

                if False and ((con['cons'][i][1:1+k] != 0).sum() == 1 and (con['cons'][i][1+k:1+k*2] != 0).sum() == 1
                        and np.absolute(con['cons'][i][1:1+k]).argmax() ==  np.absolute(con['cons'][i][1+k:1+k*2]).argmax()):
                    # Check: compute lower bounds using pre-activation bounds
                    check_sum = 0
                    for j in range(k):
                        check_sum += min(
                            (d['preact_l'][ con['varsid'][j] ] * con['cons'][i][1+j]
                                + max(d['preact_l'][ con['varsid'][j] ], 0) * con['cons'][i][1+k+j]),
                            (d['preact_u'][ con['varsid'][j] ] * con['cons'][i][1+j]
                                + max(d['preact_u'][ con['varsid'][j] ], 0) * con['cons'][i][1+k+j])
                        )
                    check_sum += con['cons'][i][0]
                    print('check', layer, check_sum)
                    if check_sum < -1e-6:
                        import pdb
                        pdb.set_trace()

                arelu_decision.append([])
                arelu_coeffs.append([])
                x_decision.append([])
                x_coeffs.append([])

    num_constr = len(pre_decision)

    print(f'{num_constr} constraints read from ERAN')

    cut = [{"x_decision": x_decision[i], "x_coeffs": x_coeffs[i],
                "relu_decision": relu_decision[i], "relu_coeffs": relu_coeffs[i],
                "arelu_decision": arelu_decision[i], "arelu_coeffs": arelu_coeffs[i],
                "pre_decision": pre_decision[i], "pre_coeffs": pre_coeffs[i],
                "bias": biases[i], "c": c[i]}
            for i in range(num_constr)]

    pre_bounds = [ (item['preact_l'], item['preact_u']) for item in cut_raw]

    return cut, pre_bounds


def close_cut_log(processes, pidx):
    """Close log file. """
    if '_logfile' in processes[pidx] and processes[pidx]['_logfile'] is not None:
        try:
            os.close(processes[pidx]['_logfile'])
        except:
            pass


def remove_cut_files(processes, pidx):
    try:
        fname = processes[pidx]['_fname_stamped']
    except:
        print('cut file name not found, quit removing')
        return
    files_to_remove = [f"{fname}.mps", f"{fname}.cuts", f"{fname}.indx", f"{fname}.log"]
    for file_to_remove in files_to_remove:
        if os.path.exists(file_to_remove):
            print(f'removing {file_to_remove}')
            try:
                os.remove(file_to_remove)
            except:
                print(f'failed to remove {file_to_remove}')


def terminate_mip_processes(mip_building_proc, processes):
    """terminate mip processes. """
    # first, terminate the mip model building process
    while mip_building_proc.is_alive():
        print('the mip building process is not terminated yet, kill it')
        # Note: in a very extreme case, if it terminates between Popen and processes recording the pid, it may induces an orphaned process')
        # FIXME However, this case is really rare so we just skip it for now
        mip_building_proc.terminate()
        time.sleep(0.2)
    for pidx in processes:
        print('found process for pidx={}'.format(pidx))
        while True:
            if psutil.pid_exists(processes[pidx]['pid']):
                print('kill process for pidx={}'.format(pidx))
                try:
                    psutil.Process(processes[pidx]['pid']).kill()
                except NoSuchProcess as e:
                    print('process already terminated, no need to kill')
            else:
                break
            time.sleep(0.2)

        close_cut_log(processes, pidx)
        remove_cut_files(processes, pidx)
        # TODO: all tmp files

# deprecated
# def terminate_single_mip_process(processes, pidx):
#     """terminate one mip processes. """
#     pidx = int(pidx)
#     if pidx in processes:
#         print('kill process for pidx={}'.format(pidx))
#         if psutil.pid_exists(processes[pidx]['pid']):
#             psutil.Process(processes[pidx]['pid']).kill()
#         else:
#             pass
#
#         close_cut_log(processes, pidx)
#         # TODO: remove all tmp files


def terminate_mip_processes_by_c_matching(processes, c_list):
    """terminate one mip processes. """
    c_list = [c.cpu().detach() for c in c_list]
    for k, value in processes.items():
        if any([(value['c'] == c).all().item() for c in c_list]):
            print('found process to kill: terminal indx = {}'.format(k))
            if psutil.pid_exists(processes[k]['pid']):
                try:
                    psutil.Process(processes[k]['pid']).kill()
                except NoSuchProcess as e:
                    print('process already terminated, no need to kill')
            else:
                pass

            close_cut_log(processes, k)
            remove_cut_files(processes, k)


def generate_cplex_cuts(net):
    ######## parse cplex cuts files ###########
    cuts, cut_timestamp = fetch_cut_from_cplex(net, sync_to_net=False)
    # we manually assign the cut to net instance
    net.cutter.cuts = cuts
    net.cutter.cut_timestamp = cut_timestamp

def clean_net_mps_process(net):
    """
        If AutoLiRPA instance generates processes by itself, we need to close them before quit
    :param net:
    :return:
    """
    if net.cplex_processes is None and net.processes is not None:
        while net.mip_building_proc.is_alive():
            # if the building process is not terminated yet, we need to wait it to terminate so that
            # no new cut processes will be created
            # otherwise, new cur processes that are created will become orphaned
            print('the mip building process is not terminated yet, killing it')
            net.mip_building_proc.kill()
            net.mip_building_proc.join()
            time.sleep(0.2)
        for pidx in net.processes:
            print('found process for pidx={}'.format(pidx))
            while True:
                if psutil.pid_exists(net.processes[pidx]['pid']):
                    print('kill process for pidx={}'.format(pidx))
                    try:
                        psutil.Process(net.processes[pidx]['pid']).kill()
                    except NoSuchProcess as e:
                        print('process already terminated, no need to kill')
                else:
                    break
                time.sleep(0.2)
            # Close log file.
            if '_logfile' in net.processes[pidx] and net.processes[pidx]['_logfile'] is not None:
                try: os.close(net.processes[pidx]['_logfile'])
                except: pass
            remove_cut_files(net.processes, pidx)

def cplex_update_general_beta(net, selected_domains):
    for sd in selected_domains["split_history"]:
        if ("general_betas" in sd and sd["cut_timestamp"] != net.net.cut_timestamp and net.net.cut_module.general_beta is not None) or \
                ("general_betas" not in sd and getattr(net.net, "cut_module", None) is not None and net.net.cut_module.general_beta is not None):
            # copy initial general beta from the latest cut module
            # ensure that for each domain the stack batch = 1
            sd["general_betas"] = net.cutter.beta_init * torch.ones_like(net.net.cut_module.general_beta[net.net.final_name][:, :, 0:1, :])
            # need to attach the latest timestamp
            sd["cut_timestamp"] = net.net.cut_timestamp

def biccos_update_general_beta(net, selected_domains):
    for sd in selected_domains["split_history"]:
        if ("general_betas" in sd and net.net.cut_module.general_beta is not None) or \
                ("general_betas" not in sd and getattr(net.net, "cut_module", None) is not None and net.net.cut_module.general_beta is not None):
            # copy initial general beta from the latest cut module
            # ensure that for each domain the stack batch = 1
            # refers to ensuring the tensor's shape accommodates a batch of data 
            # with each element being processed independently yet 
            # identically shaped like the original general_beta
            sd["general_betas"] = net.cutter.beta_init * torch.ones_like(net.net.cut_module.general_beta[net.net.final_name][:, :, 0:1, :])

def cut_analysis(cuts, max_length=20, cluster_size=3) -> None:
    clusters = defaultdict(int)
    total_length = 0

    for cut in cuts:
        length = sum(len(cut[key]) for key in ['arelu_coeffs', 'pre_coeffs', 'x_coeffs', 'relu_coeffs'])
        total_length += length

        # Determine the cluster for this length
        if length >= max_length:
            clusters[max_length] += 1
        else:
            cluster_index = length // cluster_size
            clusters[cluster_index] += 1

    total_cuts = len(cuts)
    print(f'Total number of valid cuts: {total_cuts}.')

    for cluster_index in range((max_length // cluster_size) + 1):
        if cluster_index * cluster_size < max_length:
            lower_bound = cluster_index * cluster_size
            upper_bound = lower_bound + cluster_size - 1
            count = clusters[cluster_index]
            if count == 0:
                continue
            print(f'#cuts {lower_bound + 1}-{upper_bound + 1}: {count}')

    # Print the count of cuts with length greater than or equal to max_length
    print(f'#cuts >= {max_length}: {clusters[max_length]}')
