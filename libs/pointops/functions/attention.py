import torch
from torch.autograd import Function

from pointops._C import (
    attention_relation_step_forward_cuda,
    attention_relation_step_backward_cuda,
    attention_fusion_step_forward_cuda,
    attention_fusion_step_backward_cuda,
)


class AttentionRelationStep(Function):
    """
    Custom autograd Function implementing the relation step of attention mechanism for point clouds.
    This class computes attention weights between pairs of points based on their query and key features.
    """

    @staticmethod
    def forward(ctx, query, key, weight, index_target, index_refer):
        """
        Forward pass of attention relation step.

        Args:
            ctx: Context object for autograd.
            query (torch.Tensor): Query features (n, g, c), where n is number of points,
                                g is number of groups, and c is feature dimension.
            key (torch.Tensor): Key features (n, g, c).
            weight (torch.Tensor): Weight vector (c) for scatter attention.
            index_target (torch.Tensor): Target indices (m).
            index_refer (torch.Tensor): Reference indices (m).

        Returns:
            torch.Tensor: Attention relation weights (m, g).
        """

        assert (
            query.is_contiguous()
            and key.is_contiguous()
            and index_target.is_contiguous()
            and index_refer.is_contiguous()
            and weight.is_contiguous()
        )

        assert index_target.shape[0] == index_refer.shape[0]

        _, g, c = query.shape
        m = index_target.shape[0]
        output = torch.cuda.FloatTensor(m, g).zero_()
        attention_relation_step_forward_cuda(
            m, g, c, query, key, weight, index_target.int(), index_refer.int(), output
        )
        ctx.save_for_backward(query, key, weight, index_target, index_refer)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of attention relation step.

        Args:
            ctx: Context object for autograd.
            grad_output (torch.Tensor): Gradient of the loss with respect to output (m, g).

        Returns:
            tuple: (grad_query, grad_key, None, None, None) containing gradients for:
                - grad_query: Query features (n, g, c)
                - grad_key: Key features (n, g, c)
                - weight: No gradient
                - index_target: No gradient
                - index_refer: No gradient
        """
        query, key, weight, index_target, index_refer = ctx.saved_tensors
        n, g, c = query.shape
        m = index_target.shape[0]
        grad_query = torch.cuda.FloatTensor(n, g, c).zero_()
        grad_key = torch.cuda.FloatTensor(n, g, c).zero_()
        grad_weight = torch.cuda.FloatTensor(c).zero_()
        attention_relation_step_backward_cuda(
            m,
            g,
            c,
            query,
            grad_query,
            key,
            grad_key,
            weight,
            grad_weight,
            index_target.int(),
            index_refer.int(),
            grad_output,
        )
        return grad_query, grad_key, None, None, None


class AttentionFusionStep(Function):
    """
    Custom autograd Function implementing the fusion step of attention mechanism for point clouds.
    This class fuses features based on computed attention weights.
    """

    @staticmethod
    def forward(ctx, weight, value, index_target, index_refer):
        """
        Forward pass of attention fusion step.

        Args:
            ctx: Context object for autograd.
            weight (torch.Tensor): Attention weights (m, g).
            value (torch.Tensor): Value features (n, g, c).
            index_target (torch.Tensor): Target indices (m).
            index_refer (torch.Tensor): Reference indices (m).

        Returns:
            torch.Tensor: Fused features (n, g, c).
        """

        assert (
            weight.is_contiguous()
            and value.is_contiguous()
            and index_target.is_contiguous()
            and index_refer.is_contiguous()
            and weight.is_contiguous()
        )

        assert index_target.shape[0] == index_refer.shape[0]

        n, g, c = value.shape
        m = index_refer.shape[0]
        output = torch.cuda.FloatTensor(n, g, c).zero_()
        attention_fusion_step_forward_cuda(
            m, g, c, weight, value, index_target.int(), index_refer.int(), output
        )
        ctx.save_for_backward(weight, value, index_target, index_refer)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of attention fusion step.

        Args:
            ctx: Context object for autograd.
            grad_output (torch.Tensor): Gradient of the loss with respect to output (n, g, c).

        Returns:
            tuple: (grad_weight, grad_value, None, None) containing gradients for:
                - grad_weight: Attention weights (m, g)
                - grad_value: Value features (n, g, c)
                - index_target: No gradient
                - index_refer: No gradient
        """
        weight, value, index_target, index_refer = ctx.saved_tensors
        n, g, c = value.shape
        m = index_target.shape[0]
        grad_weight = torch.cuda.FloatTensor(m, g).zero_()
        grad_value = torch.cuda.FloatTensor(n, g, c).zero_()
        attention_fusion_step_backward_cuda(
            m,
            g,
            c,
            weight,
            grad_weight,
            value,
            grad_value,
            index_target.int(),
            index_refer.int(),
            grad_output,
        )
        return grad_weight, grad_value, None, None


attention_relation_step = AttentionRelationStep.apply
attention_fusion_step = AttentionFusionStep.apply
