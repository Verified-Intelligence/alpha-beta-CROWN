#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import numpy as np
import torch.nn.functional as F

def customized_gtrsb_saver(adv_example, adv_output, res_path):
    # almost the same as the original save_cex function
    # permute the input back before flattening the tensor
    # (See customized_Gtrsb_loader() from custom/custom_model_loader.py

    adv_example = adv_example.permute(0, 1, 3, 4, 2).contiguous()

    x = adv_example.view(-1).detach().cpu()
    adv_output = F.softmax(adv_output).detach().cpu().numpy()
    with open(res_path, 'w+') as f:
        input_dim = np.prod(adv_example[0].shape)
        f.write("(")
        for i in range(input_dim):
            f.write("(X_{}  {})\n".format(i, x[i].item()))

        for i in range(adv_output.shape[1]):
            if i == 0:
                f.write("(Y_{} {})".format(i, adv_output[0,i]))
            else:
                f.write("\n(Y_{} {})".format(i, adv_output[0,i]))
        f.write(")")
        f.flush()

