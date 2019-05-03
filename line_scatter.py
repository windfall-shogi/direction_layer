#!/usr/bin/env python3
# -*- conding: utf-8 -*-

import torch
from torch.autograd import Function
from torch import nn

try:
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
finally:
    import line_convolution_cuda as lc


class ScatterVerticalFunction(Function):
    @staticmethod
    def forward(ctx, weights, bias, *inputs):
        ctx.save_for_backward(weights, bias, *inputs)
        outputs = lc.scatter.forward_vertical(inputs, weights, bias)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        tmp = ctx.saved_tensors
        weights, bias = tmp[:2]
        inputs = tmp[2:]
        tmp = lc.scatter.backward_vertical(
            grad_outputs, inputs, weights, bias
        )
        grad_weights, grad_bias = tmp[-2:]
        grad = (grad_weights, grad_bias) + tuple(tmp[:-2])
        return grad


class ScatterVertical(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(9, out_channels, in_channels)
        )
        self.bias = nn.Parameter(torch.randn(9, out_channels))

    def forward(self, inputs):
        return ScatterVerticalFunction.apply(self.weights, self.bias, *inputs)


class ScatterHorizontalFunction(Function):
    @staticmethod
    def forward(ctx, weights, bias, *inputs):
        ctx.save_for_backward(weights, bias, *inputs)
        outputs = lc.scatter.forward_horizontal(inputs, weights, bias)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        tmp = ctx.saved_tensors
        weights, bias = tmp[:2]
        inputs = tmp[2:]
        tmp = lc.scatter.backward_horizontal(
            grad_outputs,inputs, weights, bias
        )
        grad_weights, grad_bias = tmp[-2:]
        grad = (grad_weights, grad_bias) + tuple(tmp[:-2])
        return grad


class ScatterHorizontal(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(9, out_channels, in_channels)
        )
        self.bias = nn.Parameter(torch.randn(9, out_channels))

    def forward(self, inputs):
        return ScatterHorizontalFunction.apply(self.weights, self.bias, *inputs)


class ScatterDiagonal1Function(Function):
    @staticmethod
    def forward(ctx, weights, bias, *inputs):
        ctx.save_for_backward(weights, bias, *inputs)
        outputs = lc.scatter.forward_diagonal1(inputs, weights, bias)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        tmp = ctx.saved_tensors
        weights, bias = tmp[:2]
        inputs = tmp[2:]
        tmp = lc.scatter.backward_diagonal1(
            grad_outputs, inputs, weights, bias
        )
        grad_weights, grad_bias = tmp[-2:]
        grad = (grad_weights, grad_bias) + tuple(tmp[:-2])
        return grad


class ScatterDiagonal1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(17, out_channels, in_channels)
        )
        self.bias = nn.Parameter(torch.randn(17, out_channels))

    def forward(self, inputs):
        return ScatterDiagonal1Function.apply(self.weights, self.bias, *inputs)


class ScatterDiagonal2Function(Function):
    @staticmethod
    def forward(ctx, weights, bias, *inputs):
        ctx.save_for_backward(weights, bias, *inputs)
        outputs = lc.scatter.forward_diagonal2(inputs, weights, bias)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        tmp = ctx.saved_tensors
        weights, bias = tmp[:2]
        inputs = tmp[2:]
        tmp = lc.scatter.backward_diagonal2(
            grad_outputs,inputs, weights, bias
        )
        grad_weights, grad_bias = tmp[-2:]
        grad = (grad_weights, grad_bias) + tuple(tmp[:-2])
        return grad


class ScatterDiagonal2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(17, out_channels, in_channels)
        )
        self.bias = nn.Parameter(torch.randn(17, out_channels))

    def forward(self, inputs):
        return ScatterDiagonal2Function.apply(self.weights, self.bias, *inputs)

