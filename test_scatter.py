#!/usr/bin/env python3
# -*- conding: utf-8 -*-
"""
scatterの動作を確認する
スライスごとのforwardが正しいことを確認する
backwardも確認する
"""

import unittest
from itertools import product, chain

import numpy as np
import torch
from torch.autograd import Function, Variable, gradcheck
from torch import nn

try:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
finally:
    from line_scatter import ScatterVerticalFunction, ScatterVertical
    from line_scatter import ScatterHorizontalFunction, ScatterHorizontal
    from line_scatter import ScatterDiagonal1Function, ScatterDiagonal1
    from line_scatter import ScatterDiagonal2Function, ScatterDiagonal2


class BaseScatter(unittest.TestCase):
    def setUp(self):
        self.in_channels = 4
        self.out_channels = 2
        self.batch_size = 3

    def _compute_result_weight(self, input_slice, weight):
        self.assertEqual(weight.shape[1], input_slice.shape[0])
        return np.matmul(weight, input_slice)

class TestScatterVertical(BaseScatter):
    def setUp(self):
        super().setUp()

        inputs = [np.random.randn(self.batch_size, self.in_channels, 9) 
                  for _ in range(9)]
        self.inputs = inputs

    def test_line_weight(self):
        # 筋ごとにweightについての計算が正しいかを確認

        x = [torch.from_numpy(i).double().cuda() for i in self.inputs]

        weight = np.random.randn(9, self.out_channels, self.in_channels)
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.zeros([9, self.out_channels])
        b = torch.from_numpy(bias).double().cuda()

        scatter = ScatterVertical(in_channels=self.in_channels,
                                  out_channels=self.out_channels)
        scatter.weights = nn.Parameter(w)
        scatter.bias = nn.Parameter(b)
        scatter = scatter.cuda()

        y = scatter(x)

        for batch, index in product(range(self.batch_size), range(9)):
            with self.subTest(batch=batch, index=index):
                actual = y[batch, :, index, :].cpu().detach().numpy()
                expected = self._compute_result_weight(
                    self.inputs[index][batch], weight[index]
                )
                np.testing.assert_allclose(actual, expected)

    def test_line_bias(self):
        # 筋ごとにbiasについての計算が正しいかを確認

        x = [torch.from_numpy(i).double().cuda() for i in self.inputs]

        # weightの影響をなくすために0
        weight = np.zeros([9, self.out_channels, self.in_channels])
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.random.randn(9, self.out_channels)
        b = torch.from_numpy(bias).double().cuda()
        
        gather = ScatterVertical(in_channels=self.in_channels, 
                                 out_channels=self.out_channels)
        # パラメータを差し替える
        gather.weights = nn.Parameter(w)
        gather.bias = nn.Parameter(b)
        gather = gather.cuda()

        y = gather(x)

        for batch, index in product(range(self.batch_size), range(9)):
            with self.subTest(batch=batch, index=index):
                actual = y[batch, :, index, :].cpu().detach().numpy()
                # バイアスの形が合うように変更
                expected = np.tile(np.reshape(bias[index], [-1, 1]), [1, 9])

                np.testing.assert_allclose(actual, expected)

    def test_backward(self):
        x = [torch.from_numpy(i).double().cuda() for i in self.inputs]
        weights = torch.randn(9, self.out_channels, self.in_channels)
        bias = torch.randn(9, self.out_channels)

        variables = [weights, bias] + x
        for i, v in enumerate(variables):
            v = v.cuda()
            variables[i] = Variable(v.double(), requires_grad=True)

        if gradcheck(ScatterVerticalFunction.apply, variables):
            self.assertTrue(True)


class TestScatterHorizontal(BaseScatter):
    def setUp(self):
        super().setUp()

        inputs = [np.random.randn(self.batch_size, self.in_channels, 9) 
                  for _ in range(9)]
        self.inputs = inputs

    def test_line_weight(self):
        # 段ごとにweightについての計算が正しいかを確認

        x = [torch.from_numpy(i).double().cuda() for i in self.inputs]

        weight = np.random.randn(9, self.out_channels, self.in_channels)
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.zeros([9, self.out_channels])
        b = torch.from_numpy(bias).double().cuda()

        scatter = ScatterHorizontal(in_channels=self.in_channels,
                                    out_channels=self.out_channels)
        scatter.weights = nn.Parameter(w)
        scatter.bias = nn.Parameter(b)
        scatter = scatter.cuda()

        y = scatter(x)

        for batch, index in product(range(self.batch_size), range(9)):
            with self.subTest(batch=batch, index=index):
                actual = y[batch, :, :, index].cpu().detach().numpy()
                expected = self._compute_result_weight(
                    self.inputs[index][batch], weight[index]
                )
                np.testing.assert_allclose(actual, expected)

    def test_line_bias(self):
        # 段ごとにbiasについての計算が正しいかを確認

        x = [torch.from_numpy(i).double().cuda() for i in self.inputs]

        # weightの影響をなくすために0
        weight = np.zeros([9, self.out_channels, self.in_channels])
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.random.randn(9, self.out_channels)
        b = torch.from_numpy(bias).double().cuda()
        
        gather = ScatterHorizontal(in_channels=self.in_channels, 
                                   out_channels=self.out_channels)
        # パラメータを差し替える
        gather.weights = nn.Parameter(w)
        gather.bias = nn.Parameter(b)
        gather = gather.cuda()

        y = gather(x)

        for batch, index in product(range(self.batch_size), range(9)):
            with self.subTest(batch=batch, index=index):
                actual = y[batch, :, :, index].cpu().detach().numpy()
                # バイアスの形が合うように変更
                expected = np.tile(np.reshape(bias[index], [-1, 1]), [1, 9])

                np.testing.assert_allclose(actual, expected)

    def test_backward(self):
        x = [torch.from_numpy(i).double().cuda() for i in self.inputs]
        weights = torch.randn(9, self.out_channels, self.in_channels)
        bias = torch.randn(9, self.out_channels)

        variables = [weights, bias] + x
        for i, v in enumerate(variables):
            v = v.cuda()
            variables[i] = Variable(v.double(), requires_grad=True)

        if gradcheck(ScatterHorizontalFunction.apply, variables):
            self.assertTrue(True)


class TestScatterDiagonal1(BaseScatter):
    def setUp(self):
        super().setUp()

        inputs = [np.random.randn(self.batch_size, self.in_channels, i) 
                  for i in chain(range(1, 10), range(8, 0, -1))]
        self.inputs = inputs

    def test_line_weight(self):
        # 斜めのラインごとにweightについての計算が正しいかを確認

        x = [torch.from_numpy(i).double().cuda() for i in self.inputs]

        weight = np.random.randn(len(x), self.out_channels, self.in_channels)
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.zeros([len(x), self.out_channels])
        b = torch.from_numpy(bias).double().cuda()

        scatter = ScatterDiagonal1(in_channels=self.in_channels,
                                   out_channels=self.out_channels)
        scatter.weights = nn.Parameter(w)
        scatter.bias = nn.Parameter(b)
        scatter = scatter.cuda()

        y = scatter(x)

        for batch, index in product(range(self.batch_size), range(len(x))):
            with self.subTest(batch=batch, index=index):
                if index <= 8:
                    x_index = np.array(range(8 - index, 8 + 1))
                    y_index = np.array(range(index + 1))
                else:
                    x_index = np.array(range(17 - index))
                    y_index = np.array(range(index - 8, 9))
                if index == 16:
                    self.assertEqual(len(x_index), 1)
                    self.assertEqual(len(y_index), 1)
                
                actual = np.empty([self.out_channels, len(x_index)])
                tmp = y[batch].cpu().detach().numpy()
                for i, (f, r) in enumerate(zip(x_index, y_index)):
                    actual[:, i] = tmp[:, f, r]
                expected = self._compute_result_weight(
                    self.inputs[index][batch], weight[index]
                )
                np.testing.assert_allclose(actual, expected)

    def test_line_bias(self):
        # 段ごとにbiasについての計算が正しいかを確認

        x = [torch.from_numpy(i).double().cuda() for i in self.inputs]

        # weightの影響をなくすために0
        weight = np.zeros([len(x), self.out_channels, self.in_channels])
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.random.randn(len(x), self.out_channels)
        b = torch.from_numpy(bias).double().cuda()
        
        gather = ScatterDiagonal1(in_channels=self.in_channels, 
                                  out_channels=self.out_channels)
        # パラメータを差し替える
        gather.weights = nn.Parameter(w)
        gather.bias = nn.Parameter(b)
        gather = gather.cuda()

        y = gather(x)

        for batch, index in product(range(self.batch_size), range(9)):
            with self.subTest(batch=batch, index=index):
                if index <= 8:
                    x_index = np.array(range(8 - index, 8 + 1))
                    y_index = np.array(range(index + 1))
                else:
                    x_index = np.array(range(17 - index))
                    y_index = np.array(range(index - 8, 9))
                
                actual = np.empty([self.out_channels, len(x_index)])
                tmp = y[batch].cpu().detach().numpy()
                for i, (f, r) in enumerate(zip(x_index, y_index)):
                    actual[:, i] = tmp[:, f, r]
                # バイアスの形が合うように変更
                expected = np.tile(
                    np.reshape(bias[index], [-1, 1]), 
                    [1, len(x_index)]
                )

                np.testing.assert_allclose(actual, expected)

    def test_backward(self):
        x = [torch.from_numpy(i).double().cuda() for i in self.inputs]
        weights = torch.randn(len(x), self.out_channels, self.in_channels)
        bias = torch.randn(len(x), self.out_channels)

        variables = [weights, bias] + x
        for i, v in enumerate(variables):
            v = v.cuda()
            variables[i] = Variable(v.double(), requires_grad=True)

        if gradcheck(ScatterDiagonal1Function.apply, variables):
            self.assertTrue(True)


class TestScatterDiagonal2(BaseScatter):
    def setUp(self):
        super().setUp()

        inputs = [np.random.randn(self.batch_size, self.in_channels, i) 
                  for i in chain(range(1, 10), range(8, 0, -1))]
        self.inputs = inputs

    def test_line_weight(self):
        # 斜めのラインごとにweightについての計算が正しいかを確認

        x = [torch.from_numpy(i).double().cuda() for i in self.inputs]

        weight = np.random.randn(len(x), self.out_channels, self.in_channels)
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.zeros([len(x), self.out_channels])
        b = torch.from_numpy(bias).double().cuda()

        scatter = ScatterDiagonal2(in_channels=self.in_channels,
                                   out_channels=self.out_channels)
        scatter.weights = nn.Parameter(w)
        scatter.bias = nn.Parameter(b)
        scatter = scatter.cuda()

        y = scatter(x)

        for batch, index in product(range(self.batch_size), range(len(x))):
            with self.subTest(batch=batch, index=index):
                if index <= 8:
                    x_index = np.array(range(index + 1))
                    y_index = np.array(range(index, -1, -1))
                else:
                    x_index = np.array(range(index - 8, 9))
                    y_index = np.array(range(8, index - 9, -1))
                if index == 0:
                    self.assertListEqual(list(x_index), [0])
                    self.assertListEqual(list(y_index), [0])
                if index == 16:
                    self.assertEqual(len(x_index), 1)
                    self.assertEqual(len(y_index), 1)
                
                actual = np.empty([self.out_channels, len(x_index)])
                tmp = y[batch].cpu().detach().numpy()
                for i, (f, r) in enumerate(zip(x_index, y_index)):
                    actual[:, i] = tmp[:, f, r]
                expected = self._compute_result_weight(
                    self.inputs[index][batch], weight[index]
                )
                np.testing.assert_allclose(actual, expected)

    def test_line_bias(self):
        # 段ごとにbiasについての計算が正しいかを確認

        x = [torch.from_numpy(i).double().cuda() for i in self.inputs]

        # weightの影響をなくすために0
        weight = np.zeros([len(x), self.out_channels, self.in_channels])
        w = torch.from_numpy(weight).double().cuda()

        # biasの影響をなくすために0
        bias = np.random.randn(len(x), self.out_channels)
        b = torch.from_numpy(bias).double().cuda()
        
        gather = ScatterDiagonal2(in_channels=self.in_channels, 
                                  out_channels=self.out_channels)
        # パラメータを差し替える
        gather.weights = nn.Parameter(w)
        gather.bias = nn.Parameter(b)
        gather = gather.cuda()

        y = gather(x)

        for batch, index in product(range(self.batch_size), range(9)):
            with self.subTest(batch=batch, index=index):
                if index <= 8:
                    x_index = np.array(range(index + 1))
                    y_index = np.array(range(index, -1, -1))
                else:
                    x_index = np.array(range(index - 8, 9))
                    y_index = np.array(range(8, index - 9, -1))
                
                actual = np.empty([self.out_channels, len(x_index)])
                tmp = y[batch].cpu().detach().numpy()
                for i, (f, r) in enumerate(zip(x_index, y_index)):
                    actual[:, i] = tmp[:, f, r]
                # バイアスの形が合うように変更
                expected = np.tile(
                    np.reshape(bias[index], [-1, 1]), 
                    [1, len(x_index)]
                )

                np.testing.assert_allclose(actual, expected)

    def test_backward(self):
        x = [torch.from_numpy(i).double().cuda() for i in self.inputs]
        weights = torch.randn(len(x), self.out_channels, self.in_channels)
        bias = torch.randn(len(x), self.out_channels)

        variables = [weights, bias] + x
        for i, v in enumerate(variables):
            v = v.cuda()
            variables[i] = Variable(v.double(), requires_grad=True)

        if gradcheck(ScatterDiagonal2Function.apply, variables):
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
