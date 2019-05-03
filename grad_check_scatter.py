import argparse
import torch

from torch.autograd import Variable, gradcheck

from check_scatter import ScatterFunction

def main():
    in_channels, out_channels = 4, 5
    batch_size = 3
    
    x = torch.randn(batch_size, in_channels, 9)
    weights = torch.randn(out_channels, in_channels)
    bias = torch.randn(out_channels)

    variables = [x, weights, bias]
    for i, v in enumerate(variables):
        v = v.cuda()
        variables[i] = Variable(v.double(), requires_grad=True)

    if gradcheck(ScatterFunction.apply, variables):
        print('OK')


if __name__ == "__main__":
    main()
