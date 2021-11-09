import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from .utils import logger, eyeC, LinearBound, Patches, BoundList, patches_to_matrix, inplace_unfold
import torch.nn.functional as F

class Perturbation:
    r"""
    Base class for a perturbation specification. Please see examples
    at `auto_LiRPA/perturbations.py`.    

    Examples: 

    * `PerturbationLpNorm`: Lp-norm (p>=1) perturbation.

    * `PerturbationL0Norm`: L0-norm perturbation.

    * `PerturbationSynonym`: Synonym substitution perturbation for NLP.
    """

    def __init__(self):
        pass

    def set_eps(self, eps):
        self.eps = eps
    
    def concretize(self, x, A, sign=-1, aux=None):
        r"""
        Concretize bounds according to the perturbation specification.

        Args:
            x (Tensor): Input before perturbation.

            A (Tensor) : A matrix from LiRPA computation.

            sign (-1 or +1): If -1, concretize for lower bound; if +1, concretize for upper bound.

            aux (object, optional): Auxilary information for concretization.

        Returns:
            bound (Tensor): concretized bound with the shape equal to the clean output.
        """
        raise NotImplementedError

    def init(self, x, aux=None, forward=False):
        r"""
        Initialize bounds before LiRPA computation.

        Args:
            x (Tensor): Input before perturbation.

            aux (object, optional): Auxilary information.

            forward (bool): It indicates whether forward mode LiRPA is involved. 

        Returns:
            bound (LinearBound): Initialized bounds.

            center (Tensor): Center of perturbation. It can simply be `x`, or some other value.

            aux (object, optional): Auxilary information. Bound initialization may modify or add auxilary information.
        """

        raise NotImplementedError


"""Perturbation constrained by the L_0 norm (assuming input data is in the range of 0-1)."""
class PerturbationL0Norm(Perturbation):
    def __init__(self, eps, x_L = None, x_U = None, ratio = 1.0):
        self.eps = eps
        self.x_U = x_U
        self.x_L = x_L
        self.ratio = ratio

    def concretize(self, x, A, sign = -1, aux = None):
        if A is None:
            return None
        
        eps = math.ceil(self.eps)
        x = x.reshape(x.shape[0], -1, 1)
        center = A.matmul(x)

        x = x.reshape(x.shape[0], 1, -1)

        original = A * x.expand(x.shape[0], A.shape[-2], x.shape[2])
        neg_mask = A < 0
        pos_mask = A >= 0


        if sign == 1:
            A_diff = torch.zeros_like(A)
            A_diff[pos_mask] = A[pos_mask] - original[pos_mask]# changes that one weight can contribute to the value
            A_diff[neg_mask] = - original[neg_mask]
        else:
            A_diff = torch.zeros_like(A)
            A_diff[pos_mask] = original[pos_mask]
            A_diff[neg_mask] = original[neg_mask] - A[neg_mask]

        # FIXME: this assumes the input pixel range is between 0 and 1!
        A_diff, _= torch.sort(A_diff, dim = 2, descending=True)

        bound = center + sign * A_diff[:, :, :eps].sum(dim = 2).unsqueeze(2) * self.ratio

        return bound.squeeze(2)

    def init(self, x, aux=None, forward=False):
        # For other norms, we pass in the BoundedTensor objects directly.
        x_L = x
        x_U = x
        if not forward:
            return LinearBound(None, None, None, None, x_L, x_U), x, None
        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]
        eye = torch.eye(dim).to(x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        lw = eye.reshape(batch_size, dim, *x.shape[1:])
        lb = torch.zeros_like(x).to(x.device)
        uw, ub = lw.clone(), lb.clone()       
        return LinearBound(lw, lb, uw, ub, x_L, x_U), x, None

    def __repr__(self):
        return 'PerturbationLpNorm(norm=0, eps={})'.format(self.eps)

"""Perturbation constrained by the L_p norm."""
class PerturbationLpNorm(Perturbation):
    def __init__(self, eps=0, norm=np.inf, x_L=None, x_U=None, relative=False):
        self.eps = eps
        self.norm = norm
        self.dual_norm = 1 if (norm == np.inf) else (np.float64(1.0) / (1 - 1.0 / self.norm))
        self.x_L = x_L
        self.x_U = x_U
        self.relative = relative

    """Given an variable x and its bound matrix A, compute worst case bound according to Lp norm."""
    def concretize(self, x, A, sign=-1, aux=None, extra_constr=None):
        if A is None:
            return None
        # If A is an identity matrix, we will handle specially.
        def concretize_matrix(A):
            nonlocal x
            if not isinstance(A, eyeC):
                # A has (Batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
                A = A.reshape(A.shape[0], A.shape[1], -1)

                if extra_constr is not None:
                    # For each neuron, we have a beta, so beta size is (Batch, *neuron_size, n_beta) (in A, spec is *neuron_size).
                    # For intermediate layer neurons, A has *neuron_size specifications.
                    beta = extra_constr['beta']
                    beta = beta.view(beta.size(0), -1, beta.size(-1))
                    # coeffs are linear relationships between split neurons and x. They have size (batch, n_beta, *input_size), and unreated to neuron_size.
                    beta_coeffs = extra_constr['coeffs']
                    beta_coeffs = beta_coeffs.view(beta_coeffs.size(0), beta_coeffs.size(1), -1)
                    # biases are added for each batch each spec, size is (batch, n_beta), and unrelated to neuron_size.
                    beta_bias = extra_constr['bias']
                    # Merge beta into extra A and bias. Extra A has size (batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
                    extra_A = torch.einsum('ijk,ikl->ijl', beta, beta_coeffs)
                    # Merge beta into the bias term. Output has size (batch, spec).
                    extra_bias = torch.einsum('ijk,ik->ij', beta, beta_bias)

            if self.norm == np.inf:
                # For Linfinity distortion, when an upper and lower bound is given, we use them instead of eps.
                x_L = x - self.eps if self.x_L is None else self.x_L
                x_U = x + self.eps if self.x_U is None else self.x_U
                x_ub = x_U.reshape(x_U.shape[0], -1, 1)
                x_lb = x_L.reshape(x_L.shape[0], -1, 1)
                # Find the uppwer and lower bound similarly to IBP.
                center = (x_ub + x_lb) / 2.0
                diff = (x_ub - x_lb) / 2.0
                if not isinstance(A, eyeC):
                    if extra_constr is not None:
                        # Extra linear and bias terms from constraints.
                        print(
                            f'A extra: {(sign * extra_A).abs().sum().item()}, b extra: {(sign * extra_bias).abs().sum().item()}')
                        A = A - sign * extra_A
                        bound = A.matmul(center) - sign * extra_bias.unsqueeze(-1) + sign * A.abs().matmul(diff)
                    else:
                        bound = A.matmul(center) + sign * A.abs().matmul(diff)
                else:
                    assert extra_constr is None
                    # A is an identity matrix. No need to do this matmul.
                    bound = center + sign * diff
            else:
                assert extra_constr is None
                x = x.reshape(x.shape[0], -1, 1)
                if not isinstance(A, eyeC):
                    # Find the upper and lower bounds via dual norm.
                    deviation = A.norm(self.dual_norm, -1) * self.eps
                    bound = A.matmul(x) + sign * deviation.unsqueeze(-1)
                else:
                    # A is an identity matrix. Its norm is all 1.
                    bound = x + sign * self.eps
            bound = bound.squeeze(-1)
            return bound

        def concretize_patches(A):
            nonlocal x
            if self.norm == np.inf:
                # For Linfinity distortion, when an upper and lower bound is given, we use them instead of eps.
                x_L = x - self.eps if self.x_L is None else self.x_L
                x_U = x + self.eps if self.x_U is None else self.x_U

                # Here we should not reshape
                # Find the uppwer and lower bound similarly to IBP.
                center = (x_U + x_L) / 2.0
                diff = (x_U - x_L) / 2.0
                if not A.identity == 1:
                    # last_A shape: [out_c, batch_size, out_h, out_w, in_c, H, W] or [unstable_size, batch_size, in_c, H, W]. Here out_c is the spec dimension.
                    patches = A.patches

                    # unfold the input as [batch_size, out_h, out_w, in_c, H, W]
                    unfold_input = inplace_unfold(center, kernel_size=A.patches.shape[-2:], padding = A.padding, stride = A.stride)
                    if A.unstable_idx is not None:
                        # We need to add a out_c dimension and select from it.
                        unfold_input = unfold_input.unsqueeze(0).expand(A.output_shape[1], -1, -1, -1, -1, -1, -1)
                        # When A is sparse, the shape is [unstable_size, batch_size, in_c, H, W]. Here unfold_input will match this shape.
                        unfold_input = unfold_input[A.unstable_idx[0], :, A.unstable_idx[1], A.unstable_idx[2]]
                        # size of bound: [batch_size, unstable_size].
                        bound = torch.einsum('sbchw,sbchw->bs', unfold_input, patches)
                    else:
                        # size of bound: [batch_size, out_c, out_h, out_w].
                        bound = torch.einsum('bijchw,sbijchw->bsij', unfold_input, patches)

                    # unfold the diff as [batch_size, out_h, out_w, in_c, H, W]
                    unfold_diff = inplace_unfold(diff, kernel_size=A.patches.shape[-2:], padding = A.padding, stride = A.stride)
                    if A.unstable_idx is not None:
                        # We need to add a out_c dimension and select from it.
                        unfold_diff = unfold_diff.unsqueeze(0).expand(A.output_shape[1], -1, -1, -1, -1, -1, -1)
                        # When A is sparse, the shape is [unstable_size, batch_size, in_c, H, W]
                        unfold_diff = unfold_diff[A.unstable_idx[0], :, A.unstable_idx[1], A.unstable_idx[2]]
                        # size of diff: [batch_size, unstable_size].
                        bound_diff = torch.einsum('sbchw,sbchw->bs', unfold_diff, patches.abs())
                    else:
                        # size of diff: [batch_size, out_c, out_h, out_w].
                        bound_diff = torch.einsum('bijchw,sbijchw->bsij', unfold_diff, patches.abs())

                    if sign == 1:
                        bound += bound_diff
                    elif sign == -1:
                        bound -= bound_diff
                    else:
                        raise ValueError("Unsupported Sign")

                    # The extra bias term from beta term.
                    if extra_constr is not None:
                        bound += extra_constr
                else:
                    assert extra_constr is None
                    # A is an identity matrix. No need to do this matmul.
                    bound = center + sign * diff
                return bound
            else:  # Lp norm
                # x_L = x - self.eps if self.x_L is None else self.x_L
                # x_U = x + self.eps if self.x_U is None else self.x_U

                input_shape = x.shape
                if not A.identity:
                    # Find the upper and lower bounds via dual norm.
                    # matrix has shape (batch_size, out_c * out_h * out_w, input_c, input_h, input_w) or (batch_size, unstable_size, input_c, input_h, input_w)
                    matrix = patches_to_matrix(A.patches, input_shape, A.stride, A.padding, A.output_shape, A.unstable_idx)
                    # Note that we should avoid reshape the matrix. Due to padding, matrix cannot be reshaped without copying.
                    deviation = matrix.norm(p=self.dual_norm, dim=(-3,-2,-1)) * self.eps
                    # Bound has shape (batch, out_c * out_h * out_w) or (batch, unstable_size).
                    bound = torch.einsum('bschw,bchw->bs', matrix, x) + sign * deviation
                    if A.unstable_idx is None:
                        # Reshape to (batch, out_c, out_h, out_w).
                        bound = bound.view(matrix.size(0), A.patches.size(0), A.patches.size(2), A.patches.size(3))
                else:
                    # A is an identity matrix. Its norm is all 1.
                    bound = x + sign * self.eps
                return bound

        if isinstance(A, eyeC) or isinstance(A, torch.Tensor):
            return concretize_matrix(A)
        elif isinstance(A, Patches):
            return concretize_patches(A)
        elif isinstance(A, BoundList):
            for b in A.bound_list:
                if isinstance(b, eyeC) or isinstance(b, torch.Tensor):
                    pass
        else:
            raise NotImplementedError()

    def init(self, x, aux=None, forward=False):
        if self.norm == np.inf:
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
        else:
            # For other norms, we pass in the BoundedTensor objects directly.
            x_L = x
            x_U = x
        if self.relative:
            nominal = x
            lower_offset = torch.max(x_L - x - 1e-8, torch.ones_like(x_L) * (-self.eps))
            upper_offset = torch.min(x_U - x + 1e-8, torch.ones_like(x_U) * (self.eps))
        else:
            nominal = lower_offset = upper_offset = None     
        if not forward:
            return LinearBound(
                None, None, None, None, x_L, x_U,
                nominal=nominal, lower_offset=lower_offset, upper_offset=upper_offset), x, None
        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]
        eye = torch.eye(dim).to(x).expand(batch_size, dim, dim)
        lw = eye.reshape(batch_size, dim, *x.shape[1:])
        lb = torch.zeros_like(x).to(x.device)
        uw, ub = lw.clone(), lb.clone()     
        return LinearBound(
            lw, lb, uw, ub, x_L, x_U, 
            nominal=nominal, lower_offset=lower_offset, upper_offset=upper_offset), x, None

    def __repr__(self):
        if self.norm == np.inf:
            if self.x_L is None and self.x_U is None:
                return 'PerturbationLpNorm(norm=inf, eps={})'.format(self.eps)
            else:
                return 'PerturbationLpNorm(norm=inf, eps={}, x_L={}, x_U={})'.format(self.eps, self.x_L, self.x_U)
        else:
            return 'PerturbationLpNorm(norm={}, eps={})'.format(self.norm, self.eps)

class PerturbationSynonym(Perturbation):
    def __init__(self, budget, eps=1.0, use_simple=False):
        super(PerturbationSynonym, self).__init__()
        self._load_synonyms()
        self.budget = budget
        self.eps = eps
        self.use_simple = use_simple
        self.model = None
        self.train = False

    def __repr__(self):
        return 'perturbation(Synonym-based word substitution budget={}, eps={})'.format(
            self.budget, self.eps)

    def _load_synonyms(self, path='data/synonyms.json'):
        with open(path) as file:
            self.synonym = json.loads(file.read())
        logger.info('Synonym list loaded for {} words'.format(len(self.synonym)))

    def set_train(self, train):
        self.train = train

    def concretize(self, x, A, sign, aux):
        assert(self.model is not None)

        x_rep, mask, can_be_replaced = aux
        batch_size, length, dim_word = x.shape[0], x.shape[1], x.shape[2]
        dim_out = A.shape[1]
        max_num_cand = x_rep.shape[2]

        mask_rep = torch.tensor(can_be_replaced, dtype=torch.float32, device=A.device)

        num_pos = int(np.max(np.sum(can_be_replaced, axis=-1)))
        update_A = A.shape[-1] > num_pos * dim_word
        if update_A:
            bias = torch.bmm(A, (x * (1 - mask_rep).unsqueeze(-1)).reshape(batch_size, -1, 1)).squeeze(-1)        
        else:
            bias = 0.
        A = A.reshape(batch_size, dim_out, -1, dim_word)

        A_new, x_new, x_rep_new, mask_new = [], [], [], []
        zeros_A = torch.zeros(dim_out, dim_word, device=A.device)
        zeros_w = torch.zeros(dim_word, device=A.device)
        zeros_rep = torch.zeros(max_num_cand, dim_word, device=A.device)
        zeros_mask = torch.zeros(max_num_cand, device=A.device)
        for t in range(batch_size):
            cnt = 0
            for i in range(0, length):
                if can_be_replaced[t][i]:
                    if update_A:
                        A_new.append(A[t, :, i, :])
                    x_new.append(x[t][i])
                    x_rep_new.append(x_rep[t][i])
                    mask_new.append(mask[t][i])
                    cnt += 1
            if update_A:
                A_new += [zeros_A] * (num_pos - cnt)
            x_new += [zeros_w] * (num_pos - cnt)
            x_rep_new += [zeros_rep] * (num_pos - cnt)
            mask_new += [zeros_mask] * (num_pos - cnt)
        if update_A:
            A = torch.cat(A_new).reshape(batch_size, num_pos, dim_out, dim_word).transpose(1, 2)
        x = torch.cat(x_new).reshape(batch_size, num_pos, dim_word)
        x_rep = torch.cat(x_rep_new).reshape(batch_size, num_pos, max_num_cand, dim_word)
        mask = torch.cat(mask_new).reshape(batch_size, num_pos, max_num_cand)
        length = num_pos

        A = A.reshape(batch_size, A.shape[1], length, -1).transpose(1, 2) 
        x = x.reshape(batch_size, length, -1, 1)

        if sign == 1:
            cmp, init = torch.max, -1e30
        else:
            cmp, init = torch.min, 1e30

        init_tensor = torch.ones(batch_size, dim_out).to(x.device) * init
        dp = [[init_tensor] * (self.budget + 1) for i in range(0, length + 1)]
        dp[0][0] = torch.zeros(batch_size, dim_out).to(x.device)     
 
        A = A.reshape(batch_size * length, A.shape[2], A.shape[3])
        Ax = torch.bmm(
            A,
            x.reshape(batch_size * length, x.shape[2], x.shape[3])
        ).reshape(batch_size, length, A.shape[1])

        Ax_rep = torch.bmm(
            A,
            x_rep.reshape(batch_size * length, max_num_cand, x.shape[2]).transpose(-1, -2)
        ).reshape(batch_size, length, A.shape[1], max_num_cand)
        Ax_rep = Ax_rep * mask.unsqueeze(2) + init * (1 - mask).unsqueeze(2)
        Ax_rep_bound = cmp(Ax_rep, dim=-1).values

        if self.use_simple and self.train:
            return torch.sum(cmp(Ax, Ax_rep_bound), dim=1) + bias

        for i in range(1, length + 1):
            dp[i][0] = dp[i - 1][0] + Ax[:, i - 1]
            for j in range(1, self.budget + 1):
                dp[i][j] = cmp(
                    dp[i - 1][j] + Ax[:, i - 1], 
                    dp[i - 1][j - 1] + Ax_rep_bound[:, i - 1]
                )
        dp = torch.cat(dp[length], dim=0).reshape(self.budget + 1, batch_size, dim_out)        

        return cmp(dp, dim=0).values + bias

    def init(self, x, aux=None, forward=False):
        tokens, batch = aux
        self.tokens = tokens # DEBUG
        assert(len(x.shape) == 3)
        batch_size, length, dim_word = x.shape[0], x.shape[1], x.shape[2]

        max_pos = 1
        can_be_replaced = np.zeros((batch_size, length), dtype=np.bool)

        self._build_substitution(batch)

        for t in range(batch_size):
            cnt = 0
            candidates = batch[t]['candidates']
            # for transformers
            if tokens[t][0] == '[CLS]':
                candidates = [[]] + candidates + [[]]
            for i in range(len(tokens[t])):
                if tokens[t][i] == '[UNK]' or \
                        len(candidates[i]) == 0 or tokens[t][i] != candidates[i][0]:
                    continue
                for w in candidates[i][1:]:
                    if w in self.model.vocab:
                        can_be_replaced[t][i] = True
                        cnt += 1
                        break
            max_pos = max(max_pos, cnt)

        dim = max_pos * dim_word
        if forward:
            eye = torch.eye(dim_word).to(x.device)
            lw = torch.zeros(batch_size, dim, length, dim_word).to(x.device)
            lb = torch.zeros_like(x).to(x.device)   
        x_new = []     
        word_embeddings = self.model.word_embeddings.weight
        vocab = self.model.vocab
        x_rep = [[[] for i in range(length)] for t in range(batch_size)]
        max_num_cand = 1
        for t in range(batch_size):
            candidates = batch[t]['candidates']
            # for transformers
            if tokens[t][0] == '[CLS]':
                candidates = [[]] + candidates + [[]]  
            cnt = 0    
            for i in range(length):
                if can_be_replaced[t][i]:
                    word_embed = word_embeddings[vocab[tokens[t][i]]]
                    # positional embedding and token type embedding
                    other_embed = x[t, i] - word_embed
                    if forward:
                        lw[t, (cnt * dim_word):((cnt + 1) * dim_word), i, :] = eye
                        lb[t, i, :] = torch.zeros_like(word_embed)
                    for w in candidates[i][1:]:
                        if w in self.model.vocab:
                            x_rep[t][i].append(
                                word_embeddings[self.model.vocab[w]] + other_embed)
                    max_num_cand = max(max_num_cand, len(x_rep[t][i]))
                    cnt += 1
                else:
                    if forward:
                        lb[t, i, :] = x[t, i, :]        
        if forward:
            uw, ub = lw, lb
        else:
            lw = lb = uw = ub = None
        zeros = torch.zeros(dim_word, device=x.device)
        
        x_rep_, mask = [], []
        for t in range(batch_size):
            for i in range(length):
                x_rep_ += x_rep[t][i] + [zeros] * (max_num_cand - len(x_rep[t][i]))
                mask += [1] * len(x_rep[t][i]) + [0] * (max_num_cand - len(x_rep[t][i]))
        x_rep_ = torch.cat(x_rep_).reshape(batch_size, length, max_num_cand, dim_word)
        mask = torch.tensor(mask, dtype=torch.float32, device=x.device)\
            .reshape(batch_size, length, max_num_cand)
        x_rep_ = x_rep_ * self.eps + x.unsqueeze(2) * (1 - self.eps)
        
        inf = 1e20
        lower = torch.min(mask.unsqueeze(-1) * x_rep_ + (1 - mask).unsqueeze(-1) * inf, dim=2).values
        upper = torch.max(mask.unsqueeze(-1) * x_rep_ + (1 - mask).unsqueeze(-1) * (-inf), dim=2).values
        lower = torch.min(lower, x)
        upper = torch.max(upper, x)

        return LinearBound(lw, lb, uw, ub, lower, upper), x, (x_rep_, mask, can_be_replaced)

    def _build_substitution(self, batch):
        for t, example in enumerate(batch):
            if not 'candidates' in example or example['candidates'] is None:
                candidates = []
                tokens = example['sentence'].strip().lower().split(' ')
                for i in range(len(tokens)):
                    _cand = []
                    if tokens[i] in self.synonym:
                        for w in self.synonym[tokens[i]]:
                            if w in self.model.vocab:
                                _cand.append(w)
                    if len(_cand) > 0:
                        _cand = [tokens[i]] + _cand
                    candidates.append(_cand)
                example['candidates'] = candidates
