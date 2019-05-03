import torch
from torch.autograd import Function, Variable
from torch import nn
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(__file__))

import area_cuda


class GatherVerticalFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias):
        ctx.save_for_backward(inputs, weights, bias)
        outputs = area_cuda.gather.forward_vertical(inputs, weights, bias)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs, weights, bias = ctx.saved_tensors
        grad_intpus, grad_weights, grad_bias = area_cuda.gather.backward_vertical(
            grad_outputs,inputs, weights, bias
        )
        return grad_intpus, grad_weights, grad_bias


class GatherVertical(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(9, out_channels, in_channels).double())
        self.bias = nn.Parameter(torch.randn(9, out_channels).double())

    def forward(self, inputs):
        return GatherVerticalFunction.apply(inputs, self.weights, self.bias)


def main():
    in_channels, out_channels = 3, 4

    w = np.random.randn(2, in_channels, 9, 9)
    w = np.zeros_like(w)
    w[0, 0, 0, 0] = 1
    w[0, 1, 1, 1] = 1
    w[0, 2, 2, 2] = 1
    print(w)
    x = torch.from_numpy(w).double().cuda()
    x = Variable(x, requires_grad=False)

    gather = GatherVertical(in_channels, out_channels)
    gather.cuda()

    y = gather(x)
    print(y[0].dtype)

    z = y[0].cpu().detach().numpy()
    print(z.shape)
    print(z)


if __name__ == '__main__':
    main()
