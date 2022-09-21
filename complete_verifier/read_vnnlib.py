'''
vnnlib simple utilities

Stanley Bak
June 2021
'''

from copy import deepcopy
import os
import arguments
import numpy as np
import re
import hashlib
import pickle

def read_statements(vnnlib_filename):
    '''process vnnlib and return a list of strings (statements)

    useful to get rid of comments and blank lines and combine multi-line statements
    '''

    with open(vnnlib_filename, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    assert len(lines) > 0

    # combine lines if case a single command spans multiple lines
    open_parentheses = 0
    statements = []
    current_statement = ''

    for line in lines:
        comment_index = line.find(';')

        if comment_index != -1:
            line = line[:comment_index].rstrip()

        if not line:
            continue

        new_open = line.count('(')
        new_close = line.count(')')

        open_parentheses += new_open - new_close

        assert open_parentheses >= 0, "mismatched parenthesis in vnnlib file"

        # add space
        current_statement += ' ' if current_statement else ''
        current_statement += line

        if open_parentheses == 0:
            statements.append(current_statement)
            current_statement = ''

    if current_statement:
        statements.append(current_statement)

    # remove repeated whitespace characters
    statements = [" ".join(s.split()) for s in statements]

    # remove space after '('
    statements = [s.replace('( ', '(') for s in statements]

    # remove space after ')'
    statements = [s.replace(') ', ')') for s in statements]

    return statements


def update_rv_tuple(rv_tuple, op, first, second, num_inputs, num_outputs):
    'update tuple from rv in read_vnnlib_simple, with the passed in constraint "(op first second)"'

    if first.startswith("X_"):
        # Input constraints
        index = int(first[2:])

        assert not second.startswith("X") and not second.startswith("Y"), \
            f"input constraints must be box ({op} {first} {second})"
        assert 0 <= index < num_inputs, f"{first} is invalid"

        limits = rv_tuple[0][index]

        if op == "<=":
            limits[1] = min(float(second), limits[1])
        else:
            limits[0] = max(float(second), limits[0])

        assert limits[0] <= limits[1], f"{first} range is empty: {limits}"

    else:
        # output constraint
        if op == ">=":
            # swap order if op is >=
            first, second = second, first

        row = [0.0] * num_outputs
        rhs = 0.0

        # assume op is <=
        if first.startswith("Y_") and second.startswith("Y_"):
            index1 = int(first[2:])
            index2 = int(second[2:])

            row[index1] = 1
            row[index2] = -1
        elif first.startswith("Y_"):
            index1 = int(first[2:])
            row[index1] = 1
            rhs = float(second)
        else:
            assert second.startswith("Y_")
            index2 = int(second[2:])
            row[index2] = -1
            rhs = -1 * float(first)

        mat, rhs_list = rv_tuple[1], rv_tuple[2]
        mat.append(row)
        rhs_list.append(rhs)


def make_input_box_dict(num_inputs):
    'make a dict for the input box'

    rv = {i: [-np.inf, np.inf] for i in range(num_inputs)}

    return rv


def read_vnnlib(vnnlib_filename, regression=False):
    '''process in a vnnlib file

    this is not a general parser, and assumes files are provided in a 'nice' format. Only a single disjunction
    is allowed

    output a list containing 2-tuples:
        1. input ranges (box), list of pairs for each input variable
        2. specification, provided as a list of pairs (mat, rhs), as in: mat * y <= rhs, where y is the output.
                          Each element in the list is a term in a disjunction for the specification.

    If regression=True, the specification is a regression problem rather than classification.
    
    Currently we support vnnlib loader with cache:
        1. For the first time loading, it will parse the entire file and generate a cache file with md5 code of original file into *.compiled.
        2. For the later loading, it will check *.compiled and see if the stored md5 matches the original one. If not, regeneration is needed for vnnlib changing cases. Otherwise return the cache file.
    '''

    # example: "(declare-const X_0 Real)"
    compiled_vnnlib_suffix = ".compiled"
    compiled_vnnlib_filename = vnnlib_filename + compiled_vnnlib_suffix
    with open(vnnlib_filename, "rb") as file:
        curfile_md5 = hashlib.md5(file.read()).hexdigest()
    if (os.path.exists(compiled_vnnlib_filename)):
        read_error = False
        try:
            with open(compiled_vnnlib_filename, "rb") as extf:
                final_rv, old_file_md5 = pickle.load(extf)
        except (pickle.PickleError, ValueError, EOFError):
            print("Cannot read compiled vnnlib file. Regenerating...")
            read_error = True
        
        if (read_error == False):
            if (curfile_md5 == old_file_md5):
                print(f"Precompiled vnnlib file found at {compiled_vnnlib_filename}")
                return final_rv
            else:
                print(f"{compiled_vnnlib_suffix} file md5: {old_file_md5} does not match the current vnnlib md5: {curfile_md5}. Regenerating...")
    regex_declare = re.compile(r"^\(declare-const (X|Y)_(\S+) Real\)$")

    # comparison sub-expression
    # example: "(<= Y_0 Y_1)" or "(<= Y_0 10.5)"
    comparison_str = r"\((<=|>=) (\S+) (\S+)\)"

    # example: "(and (<= Y_0 Y_2)(<= Y_1 Y_2))"
    dnf_clause_str = r"\(and (" + comparison_str + r")+\)"

    # example: "(assert (<= Y_0 Y_1))"
    regex_simple_assert = re.compile(r"^\(assert " + comparison_str + r"\)$")

    # disjunctive-normal-form
    # (assert (or (and (<= Y_3 Y_0)(<= Y_3 Y_1)(<= Y_3 Y_2))(and (<= Y_4 Y_0)(<= Y_4 Y_1)(<= Y_4 Y_2))))
    regex_dnf = re.compile(r"^\(assert \(or (" + dnf_clause_str + r")+\)\)$")

    lines = read_statements(vnnlib_filename)

    # Read lines to determine number of inputs and outputs
    num_inputs = num_outputs = 0
    for line in lines:
        declare = regex_declare.findall(line)
        if len(declare) == 0:
            continue
        elif len(declare) > 1:
            raise ValueError(f'There cannot be more than one declaration in one line: {line}')
        else:
            declare = declare[0]
            if declare[0] == 'X':
                num_inputs = max(num_inputs, int(declare[1]) + 1)
            elif declare[0] == 'Y':
                num_outputs = max(num_outputs, int(declare[1]) + 1)
            else:
                raise ValueError(f'Unknown declaration: {line}')
    print(f'{num_inputs} inputs and {num_outputs} outputs in vnnlib')
    
    rv = []  # list of 3-tuples, (box-dict, mat, rhs)
    rv.append((make_input_box_dict(num_inputs), [], []))

    if regression:
        # declare x0; declare y0; single assert
        assert len(lines) == 3

    for line in lines:
        if len(regex_declare.findall(line)) > 0:
            continue

        groups = regex_simple_assert.findall(line)

        if groups:
            assert len(groups[0]) == 3, f"groups was {groups}: {line}"
            op, first, second = groups[0]

            for rv_tuple in rv:
                update_rv_tuple(rv_tuple, op, first, second, num_inputs, num_outputs)

            continue

        ################
        groups = regex_dnf.findall(line)
        assert groups, f"failed parsing line: {line}"

        tokens = line.replace("(", " ").replace(")", " ").split()
        tokens = tokens[2:]  # skip 'assert' and 'or'

        conjuncts = " ".join(tokens).split("and")[1:]

        if regression:
            cases = []
            for c  in conjuncts:
                c_ = c.split()
                if c_[6] == '<=':
                    cases.append((float(c_[2]), float(c_[5]), float(c_[8]), 'lower'))
                elif c_[6] == '>=':
                    cases.append((float(c_[2]), float(c_[5]), float(c_[8]), 'upper'))
                else:
                    print(c_[6])
                    raise NotImplementedError
            return cases

        old_rv = rv
        rv = []

        for rv_tuple in old_rv:
            for c in conjuncts:
                rv_tuple_copy = deepcopy(rv_tuple)
                rv.append(rv_tuple_copy)

                c_tokens = [s for s in c.split(" ") if len(s) > 0]

                count = len(c_tokens) // 3

                for i in range(count):
                    op, first, second = c_tokens[3 * i:3 * (i + 1)]

                    update_rv_tuple(rv_tuple_copy, op, first, second, num_inputs, num_outputs)

    # merge elements of rv with the same input spec
    merged_rv = {}

    for rv_tuple in rv:
        boxdict = rv_tuple[0]
        matrhs = (rv_tuple[1], rv_tuple[2])

        key = str(boxdict)  # merge based on string representation of input box... accurate enough for now

        if key in merged_rv:
            merged_rv[key][1].append(matrhs)
        else:
            merged_rv[key] = (boxdict, [matrhs])

    # finalize objects (convert dicts to lists and lists to np.array)
    final_rv = []

    for rv_tuple in merged_rv.values():
        box_dict = rv_tuple[0]

        box = []

        for d in range(num_inputs):
            r = box_dict[d]

            assert r[0] != -np.inf and r[1] != np.inf, f"input X_{d} was unbounded: {r}"
            box.append(r)

        spec_list = []

        for matrhs in rv_tuple[1]:
            mat = np.array(matrhs[0], dtype=float)
            rhs = np.array(matrhs[1], dtype=float)
            spec_list.append((mat, rhs))
            # final_spec.append(mat)
            # final_rhs.append(rhs)

        final_rv.append((box, spec_list))

    with open(compiled_vnnlib_filename, "wb") as extf:
        pickle.dump((final_rv, curfile_md5), extf, protocol=pickle.HIGHEST_PROTOCOL)
    return final_rv


def batch_vnnlib(vnnlib):
    """reorganize original vnnlib file, make x, c and rhs batch wise"""
    # x_prop_length = np.array([len(i[1]) for i in vnnlib])
    final_merged_rv = []

    init_d = {'x': [], 'c': [], 'rhs': [], 'verify_criterion': [], 'attack_criterion': [] }
    true_labels, target_labels = [], []

    try:

        for vnn in vnnlib:
            for mat, rhs in vnn[1]:
                init_d['x'].append(np.array(vnn[0]))
                init_d['c'].append(mat)
                init_d['rhs'].append(rhs)
                # initial_d['verify_criterion'].append(np.array([i[0] for i in vnnlib]))
                # initial_d['attack_criterion'].append(np.array([i[0] for i in vnnlib]))
                tmp_true_labels, tmp_target_labels = [], []
                for m in mat:
                    true_label = np.where(m == 1)[-1]
                    if len(true_label) != 0:
                        assert len(true_label) == 1
                        tmp_true_labels.append(true_label[0])
                    else:
                        tmp_true_labels.append(None)

                    target_label = np.where(m == -1)[-1]
                    if len(target_label) != 0:
                        assert len(target_label) == 1
                        tmp_target_labels.append(target_label[0])
                    else:
                        tmp_target_labels.append(None)

                true_labels.append(np.array(tmp_true_labels))
                target_labels.append(np.array(tmp_target_labels))

        init_d['x'] = np.array(init_d['x'])  # n, shape, 2; the batch dim n is necessary, even if n = 1
        init_d['c'] = np.array(init_d['c'])  # n, c_shape
        init_d['rhs'] = np.array(init_d['rhs'])  # n, n_output
        true_labels = np.array(true_labels)
        target_labels = np.array(target_labels)

        batch_size = arguments.Config["bab"]["initial_max_domains"]
        total_batch = int(np.ceil(len(init_d['x']) / batch_size))
        print(f"Total VNNLIB file length: {len(init_d['x'])}, max property batch size: {batch_size}, total number of batches: {total_batch}")

        for i in range(total_batch):
            # [x, [(c, rhs, y, pidx)]]
            final_merged_rv.append([init_d['x'][i * batch_size: (i + 1) * batch_size], [
                (init_d['c'][i * batch_size: (i + 1) * batch_size], init_d['rhs'][i * batch_size: (i + 1) * batch_size],
                true_labels[i * batch_size: (i + 1) * batch_size], target_labels[i * batch_size: (i + 1) * batch_size])]])

    except Exception as e:
        print(e)
        print('Merge domains failed, may caused by different shape of x (input) or spec (c matrix)!')
        raise e

    return final_merged_rv
