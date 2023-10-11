from attack.attack_pgd import test_conditions


def customized_gtrsb_condition(inputs, output, C_mat, rhs_mat, cond_mat, same_number_const,
    data_max, data_min, model, indices, num_or_spec, return_success_idx=False):
    # condition based on base size 1

    test_input = inputs[:, indices.item() // num_or_spec, indices.item() % num_or_spec, :]
    test_output = model(test_input)
    test_input = test_input.unsqueeze(0).unsqueeze(0)
    test_output = test_output.unsqueeze(0).unsqueeze(0)
    return test_conditions(test_input, test_output, C_mat, rhs_mat, cond_mat, same_number_const,
        data_max, data_min, return_success_idx)