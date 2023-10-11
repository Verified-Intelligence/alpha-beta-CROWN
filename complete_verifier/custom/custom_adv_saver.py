import numpy as np
import torch.nn.functional as F

def customized_gtrsb_saver(adv_example, adv_output, res_path):
    # almost the same as the original save_cex fucntion
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

