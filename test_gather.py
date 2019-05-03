#!/usr/bin/env python3
# -*- conding: utf-8 -*-
"""
gatherの動作を確認する
スライスごとのforwardが正しいことを確認する
backwardも確認する
"""

import unittest
from itertools import product

import numpy as np
import torch
from torch.autograd import Function, Variable, gradcheck
from torch import nn

try:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
finally:
    from line_gather import GatherVerticalFunction, GatherVertical
    from line_gather import GatherHorizontalFunction, GatherHorizontal
    from line_gather import GatherDiagonal1Function, GatherDiagonal1
    from line_gather import GatherDiagonal2Function, GatherDiagonal2


class BaseGather(unittest.TestCase):
    def setUp(self):
        self.in_channels = 4
        self.out_channels = 5
        self.batch_size = 3

        self.inputs = np.random.randn(self.batch_size, self.in_channels, 9, 9)

    def _compute_result_weight(self, input_slice, weight):
        self.assertEqual(weight.shape[1], input_slice.shape[0])
        return np.matmul(weight, input_slice)


class TestGatherVertical(BaseGather):
    def test_line_weight(self):
        # 筋ごとにweightについての計算が正しいかを確認

        x = torch.from_numpy(self.inputs).double().cuda()

        weight = np.random.randn(9, self.out_channels, self.in_channels)
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.zeros([9, self.out_channels])
        b = torch.from_numpy(bias).double().cuda()
        
        gather = GatherVertical(in_channels=self.in_channels, 
                                out_channels=self.out_channels)
        # パラメータを差し替える
        gather.weights = nn.Parameter(w)
        gather.bias = nn.Parameter(b)
        gather = gather.cuda()

        y = gather(x)

        for batch, index in product(range(self.batch_size), range(9)):
            with self.subTest(batch=batch, index=index):
                actual = y[index][batch].cpu().detach().numpy()
                expected = self._compute_result_weight(
                    self.inputs[batch, :, index], weight[index]
                )
                np.testing.assert_allclose(actual, expected)

    def test_line_bias(self):
        # 筋ごとにbiasについての計算が正しいかを確認

        x = torch.from_numpy(self.inputs).double().cuda()

        # weightの影響をなくすために0
        weight = np.zeros([9, self.out_channels, self.in_channels])
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.random.randn(9, self.out_channels)
        b = torch.from_numpy(bias).double().cuda()
        
        gather = GatherVertical(in_channels=self.in_channels, 
                                out_channels=self.out_channels)
        # パラメータを差し替える
        gather.weights = nn.Parameter(w)
        gather.bias = nn.Parameter(b)
        gather = gather.cuda()

        y = gather(x)

        for batch, index in product(range(self.batch_size), range(9)):
            with self.subTest(batch=batch, index=index):
                actual = y[index][batch].cpu().detach().numpy()
                # バイアスの形が合うように変更
                expected = np.tile(np.reshape(bias[index], [-1, 1]), [1, 9])

                np.testing.assert_allclose(actual, expected)

    def test_backward(self):
        x = torch.randn(self.batch_size, self.in_channels, 9, 9)
        weights = torch.randn(9, self.out_channels, self.in_channels)
        bias = torch.randn(9, self.out_channels)

        variables = [x, weights, bias]
        for i, v in enumerate(variables):
            v = v.cuda()
            variables[i] = Variable(v.double(), requires_grad=True)

        if gradcheck(GatherVerticalFunction.apply, variables):
            self.assertTrue(True)


class TestGatherHorizontal(BaseGather):
    def test_line_weight(self):
        # 段ごとにweightについての計算が正しいかを確認

        x = torch.from_numpy(self.inputs).double().cuda()

        weight = np.random.randn(9, self.out_channels, self.in_channels)
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.zeros([9, self.out_channels])
        b = torch.from_numpy(bias).double().cuda()
        
        gather = GatherHorizontal(in_channels=self.in_channels, 
                                  out_channels=self.out_channels)
        # パラメータを差し替える
        gather.weights = nn.Parameter(w)
        gather.bias = nn.Parameter(b)
        gather = gather.cuda()

        y = gather(x)

        for batch, index in product(range(self.batch_size), range(9)):
            with self.subTest(batch=batch, index=index):
                actual = y[index][batch].cpu().detach().numpy()
                expected = self._compute_result_weight(
                    self.inputs[batch, :, :, index], weight[index]
                )
                np.testing.assert_allclose(actual, expected)

    def test_line_bias(self):
        # 段ごとにbiasについての計算が正しいかを確認

        x = torch.from_numpy(self.inputs).double().cuda()

        # weightの影響をなくすために0
        weight = np.zeros([9, self.out_channels, self.in_channels])
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.random.randn(9, self.out_channels)
        b = torch.from_numpy(bias).double().cuda()
        
        gather = GatherHorizontal(in_channels=self.in_channels, 
                                  out_channels=self.out_channels)
        # パラメータを差し替える
        gather.weights = nn.Parameter(w)
        gather.bias = nn.Parameter(b)
        gather = gather.cuda()

        y = gather(x)

        for batch, index in product(range(self.batch_size), range(9)):
            with self.subTest(batch=batch, index=index):
                actual = y[index][batch].cpu().detach().numpy()
                # バイアスの形が合うように変更
                expected = np.tile(np.reshape(bias[index], [-1, 1]), [1, 9])

                np.testing.assert_allclose(actual, expected)

    def test_backward(self):
        x = torch.randn(self.batch_size, self.in_channels, 9, 9)
        weights = torch.randn(9, self.out_channels, self.in_channels)
        bias = torch.randn(9, self.out_channels)

        variables = [x, weights, bias]
        for i, v in enumerate(variables):
            v = v.cuda()
            variables[i] = Variable(v.double(), requires_grad=True)

        if gradcheck(GatherHorizontalFunction.apply, variables):
            self.assertTrue(True)

class TestGatherDiagonal1(BaseGather):
    def setUp(self):
        super().setUp()
        self.line_index = np.array([[8, 9, 10, 11, 12, 13, 14, 15, 16],
                                    [7, 8,  9, 10, 11, 12, 13, 14, 15],
                                    [6, 7,  8,  9, 10, 11, 12, 13, 14],
                                    [5, 6,  7,  8,  9, 10, 11, 12, 13],
                                    [4, 5,  6,  7,  8,  9, 10, 11, 12],
                                    [3, 4,  5,  6,  7,  8,  9, 10, 11],
                                    [2, 3,  4,  5,  6,  7,  8,  9, 10],
                                    [1, 2,  3,  4,  5,  6,  7,  8,  9],
                                    [0, 1,  2,  3,  4,  5,  6,  7,  8]])
        self.space_index = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 1, 1, 1, 1, 1, 1, 1],
                                     [0, 1, 2, 2, 2, 2, 2, 2, 2],
                                     [0, 1, 2, 3, 3, 3, 3, 3, 3],
                                     [0, 1, 2, 3, 4, 4, 4, 4, 4],
                                     [0, 1, 2, 3, 4, 5, 5, 5, 5],
                                     [0, 1, 2, 3, 4, 5, 6, 6, 6],
                                     [0, 1, 2, 3, 4, 5, 6, 7, 7],
                                     [0, 1, 2, 3, 4, 5, 6, 7, 8]])
                            
    def test_line_weight(self):
        # 斜めのラインごとにweightについての計算が正しいかを確認

        x = torch.from_numpy(self.inputs).double().cuda()

        weight = np.random.randn(17, self.out_channels, self.in_channels)
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.zeros([17, self.out_channels])
        b = torch.from_numpy(bias).double().cuda()
        
        gather = GatherDiagonal1(in_channels=self.in_channels, 
                                 out_channels=self.out_channels)
        # パラメータを差し替える
        gather.weights = nn.Parameter(w)
        gather.bias = nn.Parameter(b)
        gather = gather.cuda()

        y = gather(x)

        for batch, i, j in product(range(self.batch_size), range(9), range(9)):
            with self.subTest(batch=batch, i=i, j=j):
                line = self.line_index[i, j]
                space = self.space_index[i, j]

                actual = y[line][batch, :, space].cpu().detach().numpy()
                expected = self._compute_result_weight(
                    np.reshape(self.inputs[batch, :, i, j], [-1, 1]),
                    weight[line]
                ).squeeze()
                np.testing.assert_allclose(actual, expected)

    def test_line_bias(self):
        # 筋ごとにbiasについての計算が正しいかを確認

        x = torch.from_numpy(self.inputs).double().cuda()

        # weightの影響をなくすために0
        weight = np.zeros([17, self.out_channels, self.in_channels])
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.random.randn(17, self.out_channels)
        b = torch.from_numpy(bias).double().cuda()
        
        gather = GatherDiagonal1(in_channels=self.in_channels, 
                                 out_channels=self.out_channels)
        # パラメータを差し替える
        gather.weights = nn.Parameter(w)
        gather.bias = nn.Parameter(b)
        gather = gather.cuda()

        y = gather(x)

        for batch, i, j in product(range(self.batch_size), range(9), range(9)):
            with self.subTest(batch=batch, i=i, j=j):
                line = self.line_index[i, j]
                space = self.space_index[i, j]
                
                actual = y[line][batch, :, space].cpu().detach().numpy()
                expected = bias[line]

                np.testing.assert_allclose(actual, expected)

    def test_backward(self):
        x = torch.randn(self.batch_size, self.in_channels, 9, 9)
        weights = torch.randn(17, self.out_channels, self.in_channels)
        bias = torch.randn(17, self.out_channels)

        variables = [x, weights, bias]
        for i, v in enumerate(variables):
            v = v.cuda()
            variables[i] = Variable(v.double(), requires_grad=True)

        if gradcheck(GatherDiagonal1Function.apply, variables):
            self.assertTrue(True)


class TestGatherDiagonal2(BaseGather):
    def setUp(self):
        super().setUp()
        self.line_index = np.array([[0, 1,  2,  3,  4,  5,  6,  7,  8],
                                    [1, 2,  3,  4,  5,  6,  7,  8,  9],
                                    [2, 3,  4,  5,  6,  7,  8,  9, 10],
                                    [3, 4,  5,  6,  7,  8,  9, 10, 11],
                                    [4, 5,  6,  7,  8,  9, 10, 11, 12],
                                    [5, 6,  7,  8,  9, 10, 11, 12, 13],
                                    [6, 7,  8,  9, 10, 11, 12, 13, 14],
                                    [7, 8,  9, 10, 11, 12, 13, 14, 15],
                                    [8, 9, 10, 11, 12, 13, 14, 15, 16]])
        self.space_index = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 0],
                                     [2, 2, 2, 2, 2, 2, 2, 1, 0],
                                     [3, 3, 3, 3, 3, 3, 2, 1, 0],
                                     [4, 4, 4, 4, 4, 3, 2, 1, 0],
                                     [5, 5, 5, 5, 4, 3, 2, 1, 0],
                                     [6, 6, 6, 5, 4, 3, 2, 1, 0],
                                     [7, 7, 6, 5, 4, 3, 2, 1, 0],
                                     [8, 7, 6, 5, 4, 3, 2, 1, 0]])
                            
    def test_line_weight(self):
        # 斜めのラインごとにweightについての計算が正しいかを確認

        x = torch.from_numpy(self.inputs).double().cuda()

        weight = np.random.randn(17, self.out_channels, self.in_channels)
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.zeros([17, self.out_channels])
        b = torch.from_numpy(bias).double().cuda()
        
        gather = GatherDiagonal2(in_channels=self.in_channels, 
                                 out_channels=self.out_channels)
        # パラメータを差し替える
        gather.weights = nn.Parameter(w)
        gather.bias = nn.Parameter(b)
        gather = gather.cuda()

        y = gather(x)

        for batch, i, j in product(range(self.batch_size), range(9), range(9)):
            with self.subTest(batch=batch, i=i, j=j):
                line = self.line_index[i, j]
                space = self.space_index[i, j]

                actual = y[line][batch, :, space].cpu().detach().numpy()
                expected = self._compute_result_weight(
                    np.reshape(self.inputs[batch, :, i, j], [-1, 1]),
                    weight[line]
                ).squeeze()
                np.testing.assert_allclose(actual, expected)

    def test_line_bias(self):
        # 筋ごとにbiasについての計算が正しいかを確認

        x = torch.from_numpy(self.inputs).double().cuda()

        # weightの影響をなくすために0
        weight = np.zeros([17, self.out_channels, self.in_channels])
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.random.randn(17, self.out_channels)
        b = torch.from_numpy(bias).double().cuda()
        
        gather = GatherDiagonal2(in_channels=self.in_channels, 
                                 out_channels=self.out_channels)
        # パラメータを差し替える
        gather.weights = nn.Parameter(w)
        gather.bias = nn.Parameter(b)
        gather = gather.cuda()

        y = gather(x)

        for batch, i, j in product(range(self.batch_size), range(9), range(9)):
            with self.subTest(batch=batch, i=i, j=j):
                line = self.line_index[i, j]
                space = self.space_index[i, j]
                
                actual = y[line][batch, :, space].cpu().detach().numpy()
                expected = bias[line]

                np.testing.assert_allclose(actual, expected)

    def test_backward(self):
        x = torch.randn(self.batch_size, self.in_channels, 9, 9)
        weights = torch.randn(17, self.out_channels, self.in_channels)
        bias = torch.randn(17, self.out_channels)

        variables = [x, weights, bias]
        for i, v in enumerate(variables):
            v = v.cuda()
            variables[i] = Variable(v.double(), requires_grad=True)

        if gradcheck(GatherDiagonal2Function.apply, variables):
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
