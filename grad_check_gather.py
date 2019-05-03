import argparse
import torch

from torch.autograd import Variable, gradcheck

from check_gather import GatherFunction

def main():
    in_channels, out_channels = 3, 4
    batch_size = 2
    
    x = torch.randn(batch_size, in_channels, 9, 9)
    weights = torch.randn(9, out_channels, in_channels)
    bias = torch.randn(9, out_channels)

    variables = [x, weights, bias]
    for i, v in enumerate(variables):
        v = v.cuda()
        variables[i] = Variable(v.double(), requires_grad=True)

    if gradcheck(GatherFunction.apply, variables):
        print('OK')


if __name__ == "__main__":
    main()
