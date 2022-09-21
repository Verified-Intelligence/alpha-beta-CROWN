import torch
import torch.nn.functional as F
from torch import Tensor


def insert_zeros(image, s):
    """
    Insert s columns and rows 0 between every pixel in the image. For example:
    image = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
    s = 2
    output = [[1, 0, 0, 2, 0, 0, 3],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [4, 0, 0, 5, 0, 0, 6],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [7, 0, 0, 8, 0, 0, 9]]
    """
    if s <= 0:
        return image
    matrix = torch.zeros(size=(image.size(0), image.size(1), image.size(2) * (s+1) - s, image.size(3) * (s+1) - s), dtype=image.dtype, device=image.device)
    matrix_stride = matrix.stride()
    selected_matrix = torch.as_strided(matrix, [
          # Shape of the output matrix.
          matrix.size(0),  # Batch size.
          matrix.size(1),  # Channel.
          image.size(2),  # H (without zeros)
          image.size(3),  # W (without zeros)
          ], [
          # Stride of the output matrix.
          matrix_stride[0],  # Batch size dimension, keep using the old stride.
          matrix_stride[1],  # Channel dimension.
          matrix_stride[2] * (s + 1),  # Move s+1 rows.
          s+1,  # Move s+1 pixels.
    ])  # Move a pixel (on the width direction).
    selected_matrix[:] = image
    return matrix


def remove_zeros(image, s):
    if s <= 0:
        return image
    matrix_stride = image.stride()
    return torch.as_strided(image, [
        # Shape of the output matrix.
        *image.shape[:-2],
        (image.size(-2) + 1) // 2,  # H (without zeros)
        (image.size(-1) + 1) // 2,  # W (without zeros)
        ], [
        # Stride of the output matrix.
        *matrix_stride[:-2],
        matrix_stride[-2] * (s + 1),  # Move s+1 rows.
        matrix_stride[-1] * (s + 1),  # Move s+1 pixels.
    ])


def unify_shape(shape):
    """Convert shapes to 4-tuple."""
    if shape is not None:
        if isinstance(shape, int):
            shape = (shape, shape, shape, shape)
        if len(shape) == 2:
            shape = (shape[1], shape[1], shape[0], shape[0])
        assert len(shape) == 4
    return shape


def is_shape_used(shape, expected=0):
    if isinstance(shape, int):
        return shape != expected
    else:
        return sum(shape) != expected


class Patches:
    """
    A special class which denotes a convoluntional operator as a group of patches
    the shape of Patches.patches is [batch_size, num_of_patches, out_channel, in_channel, M, M]
    M is the size of a single patch
    Assume that we have a conv2D layer with w.weight(out_channel, in_channel, M, M), stride and padding applied on an image (N * N)
    num_of_patches = ((N + padding * 2 - M)//stride + 1) ** 2
    Here we only consider kernels with the same H and W
    """
    def __init__(
            self, patches=None, stride=1, padding=0, shape=None, identity=0,
            unstable_idx=None, output_shape=None, inserted_zeros=0, output_padding=0, input_shape=None):
        # Shape: [batch_size, num_of_patches, out_channel, in_channel, M, M]
        # M is the size of a single patch
        # Assume that we have a conv2D layer with w.weight(out_channel, in_channel, M, M), stride and padding applied on an image (N * N)
        # num_of_patches = ((N + padding * 2 - M)//stride + 1) ** 2
        # Here we only consider kernels with the same H and W
        self.patches = patches
        self.stride = stride
        self.padding = padding
        self.shape = shape
        self.identity = identity
        self.unstable_idx = unstable_idx
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.inserted_zeros = inserted_zeros
        self.output_padding = output_padding
        self.simplify()

    def __add__(self, other):
        if isinstance(other, Patches):
            # Insert images with zero to make stride the same, if necessary.
            assert self.stride == other.stride
            if self.unstable_idx is not None or other.unstable_idx is not None:
                if self.unstable_idx is not other.unstable_idx:  # Same tuple object.
                    raise ValueError('Please set bound option "sparse_conv_intermediate_bounds" to False to run this model.')
                assert self.output_shape == other.output_shape
            A1 = self.patches
            A2 = other.patches
            # change paddings to merge the two patches
            sp = torch.tensor(unify_shape(self.padding))
            op = torch.tensor(unify_shape(other.padding))
            if (sp - op).abs().sum().item() > 0:
                if (sp - op >= 0).all():
                    A2 = F.pad(A2, (sp - op).tolist())
                    pass
                elif (sp - op <= 0).all():
                    A1 = F.pad(A1, (op - sp).tolist())
                else:
                    raise ValueError("Unsupported padding size")
            ret = A1 + A2
            return Patches(ret, other.stride, torch.max(sp, op).tolist(),
                           ret.shape, unstable_idx=self.unstable_idx, output_shape=self.output_shape,
                           inserted_zeros=self.inserted_zeros, output_padding=self.output_padding)
        else:
            assert self.inserted_zeros == 0
            assert not is_shape_used(self.output_padding)
            # Patches has shape (out_c, batch, out_h, out_w, in_c, h, w).
            input_shape = other.shape[3:]
            matrix = other
            pieces = self.patches
            if pieces.ndim == 9:
                pieces = pieces.transpose(0, 1)
                pieces = pieces.view(pieces.shape[0], -1, pieces.shape[3], pieces.shape[4], pieces.shape[5]*pieces.shape[6], pieces.shape[7], pieces.shape[8]).transpose(0,1)
            if pieces.ndim == 8:
                pieces = pieces.transpose(0, 1)
                pieces = pieces.view(pieces.shape[0], -1, pieces.shape[3], pieces.shape[4], pieces.shape[5], pieces.shape[6], pieces.shape[7]).transpose(0,1)
            A1_matrix = patches_to_matrix(
                pieces, input_shape, self.stride, self.padding,
                output_shape=self.output_shape, unstable_idx=self.unstable_idx)
            return A1_matrix.transpose(0, 1) + matrix

    @property
    def device(self):
        if self.patches is not None:
            return self.patches.device
        if self.unstable_idx is not None:
            if isinstance(self.unstable_idx, tuple):
                return self.unstable_idx[0].device
            else:
                return self.unstable_idx.device
        raise RuntimeError("Patches object is unintialized and cannot determine its device.")

    def create_similar(self, patches=None, stride=None, padding=None, identity=None,
                       unstable_idx=None, output_shape=None, inserted_zeros=None, output_padding=None,
                       input_shape=None):
        """
        Create a new Patches object with new patches weights, and keep other properties the same.
        """
        new_patches = self.patches if patches is None else patches
        return Patches(
            new_patches,
            stride=self.stride if stride is None else stride,
            padding=self.padding if padding is None else padding,
            shape=new_patches.shape,
            identity=self.identity if identity is None else identity,
            unstable_idx=self.unstable_idx if unstable_idx is None else unstable_idx,
            output_shape=self.output_shape if output_shape is None else output_shape,
            inserted_zeros=self.inserted_zeros if inserted_zeros is None else inserted_zeros,
            output_padding=self.output_padding if output_padding is None else output_padding,
            input_shape=self.input_shape if input_shape is None else input_shape,
        )

    def to_matrix(self, input_shape):
        assert self.inserted_zeros == 0
        assert not is_shape_used(self.output_padding)
        return patches_to_matrix(self.patches, input_shape, self.stride, self.padding, self.output_shape, self.unstable_idx)

    def simplify(self):
        """Merge stride and inserted_zeros; if they are the same they can cancel out."""
        stride = [self.stride, self.stride] if isinstance(self.stride, int) else self.stride
        if self.inserted_zeros > 0 and self.inserted_zeros + 1 == stride[0] and stride[0] == stride[1]:
            # print(f'before simplify: patches={self.patches.size()} padding={self.padding}, stride={self.stride}, output_padding={self.output_padding}, inserted_zeros={self.inserted_zeros}')
            full_stride = [stride[1], stride[1], stride[0], stride[0]]
            # output_padding = tuple(p // s for p, s in zip(output_padding, full_stride))
            self.padding = tuple(p // s - o for p, s, o in zip(self.padding, full_stride, unify_shape(self.output_padding)))
            self.patches = remove_zeros(self.patches, self.inserted_zeros)
            self.stride = 1
            self.inserted_zeros = 0
            self.output_padding = 0
            # print(f'after simplify: patches={self.patches.size()} padding={self.padding}, stride={self.stride}, output_padding={self.output_padding}, inserted_zeros={self.inserted_zeros}')

    def matmul(self, input, patch_abs=False, input_shape=None):
        """
        Broadcast multiplication for patches and a matrix.

        Input shape: (batch_size, in_c, in_h, in_w).
        If the dim of in_c, in_h, in_w = 1, the the input will be expand by given input_shape to support broadcast

        Output shape: [batch_size, unstable_size] when unstable_idx is not None,
                      [batch_size, out_c, out_h, out_w] when unstable_idx is None,
        """

        patches = self.patches
        if patch_abs:
            patches = patches.abs()

        if input_shape is not None:
            # For cases that input only has fewer dimensions like (1, in_c, 1, 1)
            input = input.expand(input_shape)
            # Expand to (batch_size, in_c, in_h, in_w)

        # unfold the input as [batch_size, out_h, out_w, in_c, H, W]
        unfold_input = inplace_unfold(
            input, kernel_size=patches.shape[-2:],
            padding=self.padding, stride=self.stride,
            inserted_zeros=self.inserted_zeros, output_padding=self.output_padding)
        if self.unstable_idx is not None:
            # We need to add a out_c dimension and select from it.
            unfold_input = unfold_input.unsqueeze(0).expand(self.output_shape[1], -1, -1, -1, -1, -1, -1)
            # Shape: [unstable_size, batch_size, in_c, H, W].
            # Here unfold_input will match this shape.
            unfold_input = unfold_input[self.unstable_idx[0], :, self.unstable_idx[1], self.unstable_idx[2]]
            # shape: [batch_size, unstable_size].
            return torch.einsum('sbchw,sbchw->bs', unfold_input, patches)
        else:
            # shape: [batch_size, out_c, out_h, out_w].
            return torch.einsum('bijchw,sbijchw->bsij', unfold_input, patches)


def compute_patches_stride_padding(input_shape, patches_padding, patches_stride, op_padding, op_stride, inserted_zeros=0, output_padding=0, simplify=True):
    """
    Compute stride and padding after a conv layer with patches mode.
    """
    for p in (patches_padding, patches_stride, op_padding, op_stride):
        assert isinstance(p, int) or (isinstance(p, (list, tuple)) and (len(p) == 2 or len(p) == 4))
    # If p is int, then same padding on all 4 sides.
    # If p is 2-tuple, then it is padding p[0] on both sides of H, p[1] on both sides of W
    # If p is 4-tuple, then it is padding p[2], p[3] on top and bottom sides of H, p[0] and p[1] on left and right sides of W

    # If any of the inputs are not tuple/list, we convert them to tuple.
    full_patch_padding, full_op_padding, full_patch_stride, full_op_stride = [
            (p, p) if isinstance(p, int) else p for p in [patches_padding, op_padding, patches_stride, op_stride]]
    full_patch_padding, full_op_padding, full_patch_stride, full_op_stride = [
            (p[1], p[1], p[0], p[0]) if len(p) == 2 else p for p in [full_patch_padding, full_op_padding, full_patch_stride, full_op_stride]]
    # Compute the new padding and stride after this layer.
    new_padding = tuple(pp * os + op * (inserted_zeros + 1) for pp, op, os in zip(full_patch_padding, full_op_padding, full_op_stride))
    new_stride = tuple(ps * os for ps, os in zip(full_patch_stride, full_op_stride))

    output_padding = unify_shape(output_padding)
    new_output_padding = (output_padding[0],  # Left
          output_padding[1] + inserted_zeros * input_shape[3] % full_op_stride[2],  # Right
          output_padding[2],  # Top
          output_padding[3] + inserted_zeros * input_shape[2] % full_op_stride[0])  # Bottom

    # Merge into a single number if all numbers are identical.
    if simplify:
        if new_padding.count(new_padding[0]) == len(new_padding):
            new_padding = new_padding[0]
        if new_stride.count(new_stride[0]) == len(new_stride):
            new_stride = new_stride[0]

    return new_padding, new_stride, new_output_padding


def patches_to_matrix(pieces, input_shape, stride, padding, output_shape=None, unstable_idx=None):
    """Converting a Patches piece into a full dense matrix."""
    if type(padding) == int:
        padding = (padding, padding, padding, padding)

    if pieces.ndim == 9:
        # Squeeze two additional dimensions for output and input respectively
        assert pieces.shape[1] == 1 and pieces.shape[5] == 1
        pieces = pieces.reshape(
            pieces.shape[0], *pieces.shape[2:5],
            *pieces.shape[6:]
        )

    if unstable_idx is None:
        assert pieces.ndim == 7
        # Non-sparse pieces, with shape (out_c, batch, out_h, out_w, c, h, w).
        output_channel, batch_size, output_x, output_y = pieces.shape[:4]
    else:
        batch_size = pieces.shape[1]
        output_channel, output_x, output_y = output_shape[1:]
    input_channel, kernel_x, kernel_y = pieces.shape[-3:]
    input_x, input_y = input_shape[-2:]

    if unstable_idx is None:
        # Fix all patches in a full A matrix.
        A_matrix = torch.zeros(batch_size, output_channel, output_x, output_y, input_channel, (input_x + padding[2] + padding[3]) * (input_y + padding[0] + padding[1]), device=pieces.device, dtype=pieces.dtype)
        # Save its orignal stride.
        orig_stride = A_matrix.stride()
        # This is the main trick - we create a *view* of the original matrix, and it contains all sliding windows for the convolution.
        # Since we only created a view (in fact, only metadata of the matrix changed), it should be very efficient.
        matrix_strided = torch.as_strided(A_matrix, [batch_size, output_channel, output_x, output_y, output_x, output_y, input_channel, kernel_x, kernel_y], [orig_stride[0], orig_stride[1], orig_stride[2], orig_stride[3], (input_x + padding[2] + padding[3]) * stride, stride, orig_stride[4], input_y + padding[0] + padding[1], 1])
        # Now we need to fill the conv kernel parameters into the last three dimensions of matrix_strided.
        first_indices = torch.arange(output_x * output_y, device=pieces.device)
        second_indices = torch.div(first_indices, output_y, rounding_mode="trunc")
        third_indices = torch.fmod(first_indices, output_y)
        # pieces have shape (out_c, batch, out_h, out_w, c, h, w).
        pieces = pieces.transpose(0, 1)   # pieces has the out_c dimension at the front, need to move it to the second.
        matrix_strided[:,:,second_indices,third_indices,second_indices,third_indices,:,:,:] = pieces.reshape(*pieces.shape[:2], -1, *pieces.shape[4:])
        A_matrix = A_matrix.view(batch_size, output_channel * output_x * output_y, input_channel, input_x + padding[2] + padding[3], input_y + padding[0] + padding[1])
    else:
        # Fill only a selection of patches.
        # Create only a partial A matrix.
        unstable_size = unstable_idx[0].numel()
        A_matrix = torch.zeros(batch_size, unstable_size, input_channel, (input_x + padding[2] + padding[3]) * (input_y + padding[0] + padding[1]), device=pieces.device, dtype=pieces.dtype)
        # Save its orignal stride.
        orig_stride = A_matrix.stride()
        # This is the main trick - we create a *view* of the original matrix, and it contains all sliding windows for the convolution.
        # Since we only created a view (in fact, only metadata of the matrix changed), it should be very efficient.
        matrix_strided = torch.as_strided(A_matrix, [batch_size, unstable_size, output_x, output_y, input_channel, kernel_x, kernel_y], [orig_stride[0], orig_stride[1], (input_x + padding[2] + padding[3]) * stride, stride, orig_stride[2], input_y + padding[0] + padding[1], 1])
        # pieces have shape (unstable_size, batch, c, h, w).
        first_indices = torch.arange(unstable_size, device=pieces.device)
        matrix_strided[:,first_indices,unstable_idx[1],unstable_idx[2],:,:,:] = pieces.transpose(0, 1).to(matrix_strided)
        A_matrix = A_matrix.view(batch_size, unstable_size, input_channel, input_x + padding[2] + padding[3], input_y + padding[0] + padding[1])

    A_matrix = A_matrix[:,:,:,padding[2]:input_x + padding[2],padding[0]:input_y + padding[0]]

    return A_matrix


def check_patch_biases(lb, ub, lower_b, upper_b):
    # When we use patches mode, it's possible that we need to add two bias
    # one is from the Tensor mode and one is from the patches mode
    # And we need to detect this case and reshape the bias
    if lower_b.ndim < lb.ndim:
        lb = lb.transpose(0,1).reshape(lb.size(1), lb.size(0), -1)
        lb = lb.expand(lb.size(0), lb.size(1), lower_b.size(0)//lb.size(1))
        lb = lb.reshape(lb.size(0), -1).t()
        ub = ub.transpose(0,1).reshape(ub.size(1), ub.size(0), -1)
        ub = ub.expand(ub.size(0), ub.size(1), upper_b.size(0)//ub.size(1))
        ub = ub.reshape(ub.size(0), -1).t()
    elif lower_b.ndim > lb.ndim:
        lower_b = lower_b.transpose(0,1).reshape(lower_b.size(1), -1).t()
        upper_b = upper_b.transpose(0,1).reshape(upper_b.size(1), -1).t()
    return lb, ub, lower_b, upper_b


def inplace_unfold(image, kernel_size, stride=1, padding=0, inserted_zeros=0, output_padding=0):
    # Image has size (batch_size, channel, height, width).
    assert image.ndim == 4
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)  # (left, right, top, bottom).
    if len(padding) == 2:  # (height direction, width direction).
        padding = (padding[1], padding[1], padding[0], padding[0])
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding, output_padding, output_padding)  # (left, right, top, bottom).
    if len(output_padding) == 2:  # (height direction, width direction).
        output_padding = (output_padding[1], output_padding[1], output_padding[0], output_padding[0])
    if isinstance(stride, int):
        stride = (stride, stride)  # (height direction, width direction).
    assert len(kernel_size) == 2 and len(padding) == 4 and len(stride) == 2
    # Make sure the image is large enough for the kernel.
    assert image.size(2) + padding[2] + padding[3] >= kernel_size[0] and image.size(3) + padding[0] + padding[1] >= kernel_size[1]
    if inserted_zeros > 0:
        # We first need to insert zeros in the image before unfolding.
        image = insert_zeros(image, inserted_zeros)
        # padding = (padding[0], padding[1] + 1, padding[2], padding[3] + 1)
    # Compute the number of patches.
    # Formulation: https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold
    patches_h = int((image.size(2) + padding[2] + padding[3] - (kernel_size[0] - 1) - 1) / stride[0] + 1)
    patches_w = int((image.size(3) + padding[0] + padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1)
    # Pad image.
    if sum(padding) != 0:
        image = torch.nn.functional.pad(image, padding)
    # Save its orignal stride.
    image_stride = image.stride()
    matrix_strided = torch.as_strided(image, [
        # Shape of the output matrix.
        image.size(0),  # Batch size.
        patches_h,  # indices for each patch.
        patches_w,
        image.size(1),  # Channel.
        kernel_size[0],   # indices for each pixel on a patch.
        kernel_size[1]], [
        # Stride of the output matrix.
        image_stride[0],  # Batch size dimension, keep using the old stride.
        image_stride[2] * stride[0],  # Move patch in the height dimension.
        image_stride[3] * stride[1],  # Move patch in the width dimension.
        image_stride[1],  # Move to the next channel.
        image_stride[2],  # Move to the next row.
        image_stride[3]])  # Move a pixel (on the width direction).
    # Output shape is (batch_size, patches_h, patches_w, channel, kernel_height, kernel_width)
    if sum(output_padding) > 0:
      output_padding = tuple(p if p > 0 else None for p in output_padding)
      output_padding = (output_padding)
      matrix_strided = matrix_strided[:, output_padding[2]:-output_padding[3] if output_padding[3] is not None else None,
                                      output_padding[0]:-output_padding[1] if output_padding[1] is not None else None, :, :, :]
    return matrix_strided


def maybe_unfold_patches(d_tensor, last_A, alpha_lookup_idx=None):
    """
    Utility function to handle patch mode bound propagation in activation functions.
    In patches mode, we need to unfold lower and upper slopes (as input "d_tensor"). In matrix mode we simply return.
    """
    if d_tensor is None or last_A is None or isinstance(last_A, Tensor):
        return d_tensor

    # Shape for d_tensor:
    #   sparse: [spec, batch, in_c, in_h, in_w]
    #   non-sparse (partially shared): [out_c, batch, in_c, in_h, in_w]
    #   non-sparse (not shared): [out_c*out_h*out_w, batch, in_c, in_h, in_w]
    #   shared (independent of output spec): [1, batch, in_c, in_h, in_w]
    # The in_h, in_w dimensions must be unfolded as patches.
    origin_d_shape = d_tensor.shape
    if d_tensor.ndim == 6:
        # Merge the (out_h, out_w) dimensions.
        d_tensor = d_tensor.view(*origin_d_shape[:2], -1, *origin_d_shape[-2:])
    d_shape = d_tensor.size()
    # Reshape to 4-D tensor to unfold.
    d_tensor = d_tensor.view(-1, *d_tensor.shape[-3:])
    # unfold the slope matrix as patches. Patch shape is [spec * batch, out_h, out_w, in_c, H, W).
    d_unfolded = inplace_unfold(
        d_tensor, kernel_size=last_A.patches.shape[-2:], stride=last_A.stride,
        padding=last_A.padding, inserted_zeros=last_A.inserted_zeros,
        output_padding=last_A.output_padding)
    # Reshape to the original shape of d, e.g., for non-sparse it is (out_c, batch, out_h, out_w, in_c, H, W).
    d_unfolded_r = d_unfolded.view(*d_shape[:-3], *d_unfolded.shape[1:])
    if last_A.unstable_idx is not None:
        # Here we have d for all output neurons, but we only need to select unstable ones.
        if d_unfolded_r.size(0) == 1:
            # Shared alpha, spasre alpha should not be used.
            assert alpha_lookup_idx is None
            if len(last_A.unstable_idx) == 3:
                # Broadcast the spec shape, so only need to select the rest dimensions.
                # Change shape to (out_h, out_w, batch, in_c, H, W) or (out_h, out_w, in_c, H, W).
                d_unfolded_r = d_unfolded_r.squeeze(0).permute(1, 2, 0, 3, 4, 5)
                d_unfolded_r = d_unfolded_r[last_A.unstable_idx[1], last_A.unstable_idx[2]]
            elif len(last_A.unstable_idx) == 4:
                # [spec, batch, output_h, output_w, input_c, H, W]
                # to [output_h, output_w, batch, in_c, H, W]
                d_unfolded_r = d_unfolded_r.squeeze(0).permute(1, 2, 0, 3, 4, 5)
                d_unfolded_r = d_unfolded_r[last_A.unstable_idx[2], last_A.unstable_idx[3]]
            else:
                raise NotImplementedError()
            # output shape: (unstable_size, batch, in_c, H, W).
        else:
            # The spec dimension may be sparse and contains unstable neurons for the spec layer only.
            if alpha_lookup_idx is None:
                # alpha is spec-dense. Possible because the number of unstable neurons may decrease.
                if last_A.size(0) == d_unfolded_r.size(0):
                    # Non spec-sparse, partially shared alpha among output channel dimension.
                    # Shape after unfolding is (out_c, batch, out_h, out_w, in_c, patch_h, patch_w).
                    d_unfolded_r = d_unfolded_r[last_A.unstable_idx[0], :, last_A.unstable_idx[1], last_A.unstable_idx[2]]
                else:
                    # Non spec-sparse, non-shared alpha.
                    # Shape after unfolding is (out_c*out_h*out_w, batch, out_h, out_w, in_c, patch_h, patch_w).
                    # Reshaped to (out_c, out_h, out_w, batch, out_h, out_w, in_c, patch_h, patch_w).
                    d_unfolded_r = d_unfolded_r.view(last_A.shape[0], last_A.shape[2], last_A.shape[3], -1, *d_unfolded_r.shape[2:])
                    # Select on all out_c, out_h, out_w dimensions.
                    d_unfolded_r = d_unfolded_r[last_A.unstable_idx[0], last_A.unstable_idx[1],
                            last_A.unstable_idx[2], :, last_A.unstable_idx[1], last_A.unstable_idx[2]]
            elif alpha_lookup_idx.ndim == 1:
                # sparse alpha: [spec, batch, in_c, in_h, in_w]
                # Partially shared alpha on the spec dimension - all output neurons on the same channel use the same alpha.
                # If alpha_lookup_idx is not None, we need to convert the sparse indices using alpha_lookup_idx.
                _unstable_idx = alpha_lookup_idx[last_A.unstable_idx[0]]
                # The selection is only used on the channel dimension.
                d_unfolded_r = d_unfolded_r[_unstable_idx, :, last_A.unstable_idx[1], last_A.unstable_idx[2]]
            elif alpha_lookup_idx is not None and alpha_lookup_idx.ndim == 3:
                # sparse alpha: [spec, batch, in_c, in_h, in_w]
                # We created alpha as full output shape; alpha not shared among channel dimension.
                # Shape of alpha is (out_c*out_h*out_w, batch, in_c, in_h, in_w), note that the first 3 dimensions
                # is merged into one to allow simpler selection.
                _unstable_idx = alpha_lookup_idx[
                    last_A.unstable_idx[0],
                    last_A.unstable_idx[1],
                    last_A.unstable_idx[2]]
                # d_unfolded_r shape from (out_c, batch, out_h, out_w, in_c, in_h, in_w)
                # to (out_c * out_h * out_w(sparse), batch, in_c, in_h, in_w)
                # Note that the dimensions out_h, out_w come from unfolding, not specs in alpha, so they will be selected
                # directly without translating using the lookup table.
                d_unfolded_r = d_unfolded_r[_unstable_idx, :, last_A.unstable_idx[1], last_A.unstable_idx[2]]
                # after selection we return (unstable_size, batch_size, in_c, H, W)
                return d_unfolded_r
            else:
                raise ValueError
    else:
        # A is not sparse. Alpha shouldn't be sparse as well.
        assert alpha_lookup_idx is None
        if last_A.patches.size(0) != d_unfolded_r.size(0) and d_unfolded_r.size(0) != 1:
            # Non-shared alpha, shape after unfolding is (out_c*out_h*out_w, batch, out_h, out_w, in_c, patch_h, patch_w).
            # Reshaped to (out_c, out_h*out_w, batch, out_h*out_w, in_c, patch_h, patch_w).
            d_unfolded_r = d_unfolded_r.reshape(last_A.shape[0], last_A.shape[2] * last_A.shape[3], -1,
                    d_unfolded_r.shape[2] * d_unfolded_r.shape[3], *d_unfolded_r.shape[4:])
            # Select the "diagonal" elements in the out_h*out_w dimension.
            # New shape is (out_c, batch, in_c, patch_h, patch_w, out_h*out_w)
            d_unfolded_r = d_unfolded_r.diagonal(offset=0, dim1=1, dim2=3)
            # New shape is (out_c, batch, in_c, patch_h, patch_w, out_h, out_w)
            d_unfolded_r = d_unfolded_r.view(*d_unfolded_r.shape[:-1], last_A.shape[2], last_A.shape[3])
            # New shape is (out_c, batch, out_h, out_w, in_c, patch_h, patch_w)
            d_unfolded_r = d_unfolded_r.permute(0, 1, 5, 6, 2, 3, 4)


    # For sparse patches, the shape after unfold is (unstable_size, batch_size, in_c, H, W).
    # For regular patches, the shape after unfold is (out_c, batch, out_h, out_w, in_c, H, W).
    if d_unfolded_r.ndim != last_A.patches.ndim:
        # For the situation of d independent of output neuron (e.g., vanilla crown bound), which does not have
        # the out_h, out_w dimension and out_c = 1 (sepc). We added 1s for the out_h, out_w dimensions.
        d_unfolded_r = d_unfolded_r.unsqueeze(2).unsqueeze(-4)
    return d_unfolded_r
