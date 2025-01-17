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
import numpy as np
import torch.nn.functional as F

def customized_gtrsb_saver(adv_example, adv_output, res_path):
    # almost the same as the original save_cex function
    # permute the input back before flattening the tensor
    # (See customized_Gtrsb_loader() from custom/custom_model_loader.py

    adv_example = adv_example.permute(0, 2, 3, 1).contiguous()

    x = adv_example.view(-1).detach().cpu()
    adv_output = F.softmax(adv_output, dim=1).detach().cpu().numpy()
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

