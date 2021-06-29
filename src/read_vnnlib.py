'''
vnnlib simple utilities

Stanley Bak
June 2021
'''

from copy import deepcopy

import numpy as np
import re


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
        assert 0 <= index < num_inputs

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


def read_vnnlib_simple(vnnlib_filename, num_inputs, num_outputs, regression=False):
    '''process in a vnnlib file

    this is not a general parser, and assumes files are provided in a 'nice' format. Only a single disjunction
    is allowed

    output a list containing 2-tuples:
        1. input ranges (box), list of pairs for each input variable
        2. specification, provided as a list of pairs (mat, rhs), as in: mat * y <= rhs, where y is the output.
                          Each element in the list is a term in a disjunction for the specification.

    If regression=True, the specification is a regression problem rather than classification.
    '''

    # example: "(declare-const X_0 Real)"
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

    rv = []  # list of 3-tuples, (box-dict, mat, rhs)
    rv.append((make_input_box_dict(num_inputs), [], []))

    lines = read_statements(vnnlib_filename)

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

        final_rv.append((box, spec_list))

    # for i, (box, spec_list) in enumerate(final_rv):
    #    print(f"-----\n{i+1}. {box}\nspec:{spec_list}")

    return final_rv
