import torch
from torch.autograd import Function, Variable
from torch import nn
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(__file__))

import area_cuda


class ScatterFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias):
        ctx.save_for_backward(inputs, weights, bias)
        return area_cuda.scatter.forward(inputs, weights, bias)[0]

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights, bias = ctx.saved_variables
        grad_intpus, grad_weights, grad_bias = area_cuda.scatter.backward(
            inputs, weights, bias, grad_output
        )
        return grad_intpus, grad_weights, grad_bias


class Scatter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels).double())
        self.bias = nn.Parameter(torch.randn(out_channels).double())

    def forward(self, inputs):
        return ScatterFunction.apply(inputs, self.weights, self.bias)


def main():
    in_channels, out_channels = 3, 4

    w = np.random.randn(2, in_channels, 9)
    w = np.zeros_like(w)
    w[0, 0, 0] = 1
    w[0, 1, 1] = 1
    w[0, 2, 2] = 1
    print(w)
    x = torch.from_numpy(w).double().cuda()
    x = Variable(x, requires_grad=False)

    scatter = Scatter(in_channels, out_channels)
    scatter.cuda()

    y = scatter(x)
    print(y.dtype)

    z = y.cpu().detach().numpy()
    print(z.shape)
    print(z)


if __name__ == '__main__':
    main()
