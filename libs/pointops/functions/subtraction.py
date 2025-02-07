import torch
from torch.autograd import Function

from pointops._C import subtraction_forward_cuda, subtraction_backward_cuda


class Subtraction(Function):
    """
    Custom autograd Function implementing feature subtraction for point clouds.
    This class computes the difference between two sets of features based on provided indices,
    useful for computing relative features or displacements.
    """

    @staticmethod
    def forward(ctx, input1, input2, idx):
        """
        Forward pass of feature subtraction.

        Args:
            ctx: Context object for autograd.
            input1 (torch.Tensor): First input features (n, c).
            input2 (torch.Tensor): Second input features (n, c).
            idx (torch.Tensor): Sample indices (n, nsample).

        Returns:
            torch.Tensor: Feature differences (n, nsample, c).
        """
        assert input1.is_contiguous() and input2.is_contiguous()
        n, c = input1.shape
        nsample = idx.shape[-1]
        output = torch.cuda.FloatTensor(n, nsample, c).zero_()
        subtraction_forward_cuda(n, nsample, c, input1, input2, idx, output)
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of feature subtraction.

        Args:
            ctx: Context object for autograd.
            grad_output (torch.Tensor): Gradient of the loss with respect to output (n, nsample, c).

        Returns:
            tuple: (grad_input1, grad_input2, None) containing gradients for:
                - grad_input1: First input features (n, c)
                - grad_input2: Second input features (n, c)
                - idx: No gradient
        """
        (idx,) = ctx.saved_tensors
        n, nsample, c = grad_output.shape
        grad_input1 = torch.cuda.FloatTensor(n, c).zero_()
        grad_input2 = torch.cuda.FloatTensor(n, c).zero_()
        subtraction_backward_cuda(
            n, nsample, c, idx, grad_output, grad_input1, grad_input2
        )
        return grad_input1, grad_input2, None


subtraction = Subtraction.apply
