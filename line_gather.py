#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Function
from torch import nn

try:
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
finally:
    import line_convolution_cuda as lc


class GatherVerticalFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias):
        ctx.save_for_backward(inputs, weights, bias)
        outputs = lc.gather.forward_vertical(inputs, weights, bias)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs, weights, bias = ctx.saved_tensors
        grad_intpus, grad_weights, grad_bias = lc.gather.backward_vertical(
            grad_outputs,inputs, weights, bias
        )
        return grad_intpus, grad_weights, grad_bias


class GatherVertical(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(9, out_channels, in_channels)
        )
        self.bias = nn.Parameter(torch.randn(9, out_channels))

    def forward(self, inputs):
        return GatherVerticalFunction.apply(inputs, self.weights, self.bias)


class GatherHorizontalFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias):
        ctx.save_for_backward(inputs, weights, bias)
        outputs = lc.gather.forward_horizontal(inputs, weights, bias)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs, weights, bias = ctx.saved_tensors
        grad_intpus, grad_weights, grad_bias = lc.gather.backward_horizontal(
            grad_outputs,inputs, weights, bias
        )
        return grad_intpus, grad_weights, grad_bias


class GatherHorizontal(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(9, out_channels, in_channels)
        )
        self.bias = nn.Parameter(torch.randn(9, out_channels))

    def forward(self, inputs):
        return GatherHorizontalFunction.apply(inputs, self.weights, self.bias)


class GatherDiagonal1Function(Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias):
        ctx.save_for_backward(inputs, weights, bias)
        outputs = lc.gather.forward_diagonal1(inputs, weights, bias)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs, weights, bias = ctx.saved_tensors
        grad_intpus, grad_weights, grad_bias = lc.gather.backward_diagonal1(
            grad_outputs,inputs, weights, bias
        )
        return grad_intpus, grad_weights, grad_bias


class GatherDiagonal1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(17, out_channels, in_channels)
        )
        self.bias = nn.Parameter(torch.randn(17, out_channels))

    def forward(self, inputs):
        return GatherDiagonal1Function.apply(inputs, self.weights, self.bias)


class GatherDiagonal2Function(Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias):
        ctx.save_for_backward(inputs, weights, bias)
        outputs = lc.gather.forward_diagonal2(inputs, weights, bias)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs, weights, bias = ctx.saved_tensors
        grad_intpus, grad_weights, grad_bias = lc.gather.backward_diagonal2(
            grad_outputs,inputs, weights, bias
        )
        return grad_intpus, grad_weights, grad_bias


class GatherDiagonal2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(17, out_channels, in_channels)
        )
        self.bias = nn.Parameter(torch.randn(17, out_channels))

    def forward(self, inputs):
        return GatherDiagonal2Function.apply(inputs, self.weights, self.bias)

