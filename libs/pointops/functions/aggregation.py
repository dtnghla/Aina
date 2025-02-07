import torch
from torch.autograd import Function

from pointops._C import aggregation_forward_cuda, aggregation_backward_cuda


class Aggregation(Function):
    """
    Custom autograd Function implementing feature aggregation for point clouds.
    This class performs weighted aggregation of features based on spatial positions and indices.
    """

    @staticmethod
    def forward(ctx, input, position, weight, idx):
        """
        Forward pass of feature aggregation.

        Args:
            ctx: Context object for autograd.
            input (torch.Tensor): Input features (n, c).
            position (torch.Tensor): Position features (n, nsample, c).
            weight (torch.Tensor): Weight features (n, nsample, c').
            idx (torch.Tensor): Sample indices (n, nsample).

        Returns:
            torch.Tensor: Aggregated features (n, c).
        """
        assert (
            input.is_contiguous()
            and position.is_contiguous()
            and weight.is_contiguous()
        )
        n, nsample, c = position.shape
        w_c = weight.shape[-1]
        output = torch.cuda.FloatTensor(n, c).zero_()
        aggregation_forward_cuda(
            n, nsample, c, w_c, input, position, weight, idx, output
        )
        ctx.save_for_backward(input, position, weight, idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of feature aggregation.

        Args:
            ctx: Context object for autograd.
            grad_output (torch.Tensor): Gradient of the loss with respect to output (n, c).

        Returns:
            tuple: (grad_input, grad_position, grad_weight, None) containing gradients for:
                - input: Input features (n, c)
                - position: Position features (n, nsample, c)
                - weight: Weight features (n, nsample, c')
                - idx: No gradient for indices
        """
        input, position, weight, idx = ctx.saved_tensors
        n, nsample, c = position.shape
        w_c = weight.shape[-1]
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        grad_position = torch.cuda.FloatTensor(n, nsample, c).zero_()
        grad_weight = torch.cuda.FloatTensor(n, nsample, w_c).zero_()
        aggregation_backward_cuda(
            n,
            nsample,
            c,
            w_c,
            input,
            position,
            weight,
            idx,
            grad_output,
            grad_input,
            grad_position,
            grad_weight,
        )
        return grad_input, grad_position, grad_weight, None


aggregation = Aggregation.apply
