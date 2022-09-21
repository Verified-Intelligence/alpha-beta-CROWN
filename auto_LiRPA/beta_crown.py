import torch


def beta_bias(self):
    batch_size = len(self.relus[-1].split_beta)
    batch = int(batch_size/2)
    bias = torch.zeros((batch_size, 1), device=self.device)
    for m in self.relus:
        if not m.used or not m.perturbed:
            continue
        if m.split_beta_used:
            bias[:batch] = bias[:batch] + m.split_bias*m.split_beta[:batch]*m.split_c[:batch]
            bias[batch:] = bias[batch:] + m.split_bias*m.split_beta[batch:]*m.split_c[batch:]
        if m.history_beta_used:
            bias = bias + (m.new_history_bias*m.new_history_beta*m.new_history_c).sum(1, keepdim=True)
        # No single node split here, because single node splits do not have bias.
    return bias


def print_optimized_beta(self, relus, intermediate_beta_enabled=False):
    masked_betas = []
    for model in relus:
        masked_betas.append(model.masked_beta)
        if model.history_beta_used:
            print(f"{model.name} history beta", model.new_history_beta.squeeze())
        if model.split_beta_used:
            print(f"{model.name} split beta:", model.split_beta.view(-1))
            print(f"{model.name} bias:", model.split_bias)


def save_best_intermediate_betas(self, relus, idx):
    for layer in relus:
        # The history split and current split is handled seperatedly.
        if layer.history_beta_used:
            # Each key in history_intermediate_betas for this layer is a dictionary, with all other pre-relu layers' names.
            for k, v in layer.history_intermediate_betas.items():
                # This is a tensor with shape (batch, *intermediate_layer_shape, number_of_beta)
                self.best_intermediate_betas[layer.name]['history'][k]["lb"][idx] = v["lb"][idx]
                self.best_intermediate_betas[layer.name]['history'][k]["ub"][idx] = v["ub"][idx]
        if layer.split_beta_used:
            for k, v in layer.split_intermediate_betas.items():
                # This is a tensor with shape (batch, *intermediate_layer_shape, 1)
                self.best_intermediate_betas[layer.name]['split'][k]["lb"][idx] = v["lb"][idx]
                self.best_intermediate_betas[layer.name]['split'][k]["ub"][idx] = v["ub"][idx]
        if layer.single_beta_used:
            for k, v in layer.single_intermediate_betas.items():
                self.best_intermediate_betas[layer.name]['single'][k]["lb"][idx] = v["lb"][idx]
                self.best_intermediate_betas[layer.name]['single'][k]["ub"][idx] = v["ub"][idx]