import torch
from torch.autograd import Function, Variable
from torch import nn
import numpy as np

import area_cuda


class OnehotFunction(Function):
    @staticmethod
    def forward(ctx, inputs):
        return area_cuda.onehot.forward(inputs)[0]


class Onehot(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return OnehotFunction.apply(inputs)


def main():
    w = np.random.randint(0, 29, size=(2, 9, 9))
    x = torch.from_numpy(w).long().cuda()
    x = Variable(x, requires_grad=False)
    y = Onehot()(x)

    z = y.cpu().numpy()
    print(z.shape)
    print(w[0, 0, 0])
    print(z[0, :, 0, 0])
    print(np.all(z == 0))


if __name__ == '__main__':
    main()
