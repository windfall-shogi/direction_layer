#pragma once
#ifndef WRAPPER_HPP_INCLUDED
#define WRAPPER_HPP_INCLUDED

#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "direction.hpp"

inline int LeadingDimension(const at::Tensor& t) {
  return t.size(2) * (t.dim() == 3 ? 1 : t.size(3));
}

template <typename T, typename scalar_t = typename std::remove_const<T>::type>
struct BaseType {
  using pointer = typename std::conditional<std::is_const<T>::value,
                                            const scalar_t*, scalar_t*>::type;
  using value_type = scalar_t;
};

// 空間方向を分割
// batch x in_channels x 9 x 9の配列のうちbatch x in_channels x 1 x
// 1の領域を操作 もしくはbatch x in_channels x sizeのうちbatch x in_channels x
// 1の領域を操作 チャネル方向を行列の行方向とみなす 1 x in_channelsの行列
template <typename T>
struct SpaceSlicedInput : BaseType<T> {
  using pointer = typename BaseType<T>::pointer;
  using value_type = typename BaseType<T>::value_type;
  pointer data;
  int rows, cols;
  int ld;
  int stride;
  // 空間方向の移動量
  int delta;

  // scatterのbackward(input)
  SpaceSlicedInput(at::Tensor& input)
      : data(input.data<value_type>()),
        rows(1),
        cols(input.size(1)),
        ld(LeadingDimension(input)),
        stride(input.size(1) * ld),
        delta(1) {}
  // scatterのforwardで利用
  SpaceSlicedInput(const at::Tensor& input)
      : data(input.data<value_type>()),
        rows(1),
        cols(input.size(1)),
        ld(LeadingDimension(input)),
        stride(input.size(1) * ld),
        delta(1) {}

  // gatherのbackwardで利用(input)
  SpaceSlicedInput(at::Tensor& input, const Direction direction,
                   const int offset)
      : data(input.data<value_type>() + offset),
        rows(1),
        cols(input.size(1)),
        ld(LeadingDimension(input)),
        stride(input.size(1) * ld),
        delta(direction) {}
  // gatherのforwardで利用
  SpaceSlicedInput(const at::Tensor& input, const Direction direction,
                   const int offset)
      : data(input.data<value_type>() + offset),
        rows(1),
        cols(input.size(1)),
        ld(LeadingDimension(input)),
        stride(input.size(1) * ld),
        delta(direction) {}
};

// 深さ方向を分割
// batch x in_channels x 9 x 9の配列のうちbatch x 1 x sizeの領域を操作
// もしくはbatch x in_channels x sizeのうちbatch x 1 x sizeの領域を操作
// 1 x sizeの行列として空間方向を操作
// in_channelsはループ処理で対応
template <typename T>
struct DepthSlicedInput : BaseType<T> {
  using pointer = typename BaseType<T>::pointer;
  using value_type = typename BaseType<T>::value_type;
  pointer data;
  int rows, cols;
  int ld;
  int stride;
  // 深さ方向の移動量
  int delta;

  // scatterのbackward(weights)
  DepthSlicedInput(const at::Tensor& input)
      : data(input.data<value_type>()),
        rows(input.size(2)),
        cols(1),
        ld(rows),
        stride(input.size(1) * rows),
        delta(rows) {}

  // gatherのweightsでの微分で利用
  DepthSlicedInput(const at::Tensor& input, const Direction direction,
                   const int offset, const int size)
      : data(input.data<value_type>() + offset),
        rows(1),
        cols(size),
        ld(direction),
        stride(input.size(1) * 81),
        delta(81) {}
};

// 空間方向を分割
// batch x out_channels x sizeのうち、batch x out_channels x 1の領域を操作
// もしくはbatch x out_channels x 9 x 9のうちbatch x out_channels x 1 x
// 1の領域を操作 チャネル方向を行列の行方向とみなす 1 x out_channelsの行列
template <typename T>
struct SpaceSlicedOutput : BaseType<T> {
  using pointer = typename BaseType<T>::pointer;
  using value_type = typename BaseType<T>::value_type;
  pointer data;
  int rows, cols;
  int ld;
  int stride;
  // 空間方向の移動量
  int delta;

  // scatterのforwardで利用
  SpaceSlicedOutput(at::Tensor& output, const Direction direction,
                    const int offset)
      : data(output.data<value_type>() + offset),
        rows(1),
        cols(output.size(1)),
        ld(LeadingDimension(output)),
        stride(output.size(1) * ld),
        delta(direction) {}
  // scatterのbackward(input)
  SpaceSlicedOutput(const at::Tensor& output, const Direction direction,
                    const int offset)
      : data(output.data<value_type>() + offset),
        rows(1),
        cols(output.size(1)),
        ld(LeadingDimension(output)),
        stride(output.size(1) * ld),
        delta(direction) {}

  // メモリが連続している3次元配列
  // gatherのforwardで利用
  SpaceSlicedOutput(at::Tensor& output)
      : data(output.data<value_type>()),
        rows(1),
        cols(output.size(1)),
        ld(output.size(2)),
        stride(output.size(1) * output.size(2)),
        delta(1) {}
  // gatherのbackwardで利用(input)
  SpaceSlicedOutput(const at::Tensor& output)
      : data(output.data<value_type>()),
        rows(1),
        cols(output.size(1)),
        ld(output.size(2)),
        stride(output.size(1) * output.size(2)),
        delta(1) {}
};

// 深さ方向を分割
// batch x out_channels x sizeのうちbatch x 1 x sizeの領域を操作
// もしくはbatch x out_channels x 9 x 9の配列のうちbatch x 1 x 1 x
// sizeの領域を操作 1 x sizeの行列として空間方向を操作
// out_channelsはループ処理で対応
template <typename T>
struct DepthSlicedOutput : BaseType<T> {
  using pointer = typename BaseType<T>::pointer;
  using value_type = typename BaseType<T>::value_type;
  pointer data;
  int rows, cols;
  int ld;
  int stride;
  // 深さ方向の移動量
  int delta;

  // scatterのbackward(weighs, bias)
  DepthSlicedOutput(const at::Tensor& output, const Direction direction,
                    const int size, const int offset)
      : data(output.data<value_type>() + offset),
        rows(1),
        cols(size),
        ld(direction),
        stride(output.size(1) * 81),
        delta(81) {}
  // scatterのbackward(bias)
  DepthSlicedOutput(const at::Tensor& output, const Direction direction,
                    const int size, const int offset, const int dummy)
      : data(output.data<value_type>() + offset),
        rows(1),
        cols(size),
        ld(direction),
        stride(81),
        delta(81) {}

  // gatherでweightsについて微分する時に利用
  DepthSlicedOutput(const at::Tensor& output)
      : data(output.data<value_type>()),
        rows(output.size(2)),
        cols(1),
        ld(rows),
        stride(output.size(1) * rows),
        delta(rows) {}
  // scatterでbiasについて微分する時に利用
  DepthSlicedOutput(const at::Tensor& output, const int space_size,
                    const int step)
      : data(output.data<value_type>()),
        rows(1),
        cols(space_size),
        ld(step),
        stride(81),
        delta(0) {}
};

// 確保されている行列はrow majorだが、blasで要求されるのはcolumn majorなので、
// ややこしいのでラッパー
template <typename T>
struct Weight : BaseType<T> {
  using pointer = typename BaseType<T>::pointer;
  using value_type = typename BaseType<T>::value_type;
  pointer data;
  // メモリが連続な方向が列方向
  // rows: in_channels, cols: out_channels
  int rows, cols;
  int ld;
  int stride;

  Weight(at::Tensor& weighit)
      : data(weighit.data<value_type>()),
        rows(weighit.size(1)),  // メモリが連続
        cols(weighit.size(0)),  // メモリが不連続
        ld(weighit.size(1)),
        stride(0) {}
  Weight(const at::Tensor& weighit)
      : data(weighit.data<value_type>()),
        rows(weighit.size(1)),  // メモリが連続
        cols(weighit.size(0)),  // メモリが不連続
        ld(weighit.size(1)),
        stride(0) {}
};

// vectorをmatrixとして扱う
// 一時バッファで利用
template <typename T>
struct Matrix : BaseType<T> {
  using pointer = typename BaseType<T>::pointer;
  using value_type = typename BaseType<T>::value_type;
  pointer data;
  int rows, cols;
  int ld;
  int stride;
  // 空間方向の移動量
  int delta;

  Matrix(at::Tensor& x)
      : data(x.data<value_type>()),
        rows(1),
        cols(x.size(0)),
        ld(1),
        stride(1),
        delta(1) {}
  Matrix(const at::Tensor& x)
      : data(x.data<value_type>()),
        rows(1),
        cols(x.size(0)),
        ld(1),
        stride(1),
        delta(1) {}

  Matrix(at::Tensor& x, const int r, const int c)
      : data(x.data<value_type>()),
        rows(r),
        cols(c),
        ld(r),
        stride(r),
        delta(1) {}
  Matrix(const at::Tensor& x, const int r, const int c)
      : data(x.data<value_type>()),
        rows(r),
        cols(c),
        ld(r),
        stride(r),
        delta(1) {}

  Matrix(at::Tensor& x, const int r, const int c, const int s)
      : data(x.data<value_type>()),
        rows(r),
        cols(c),
        ld(r),
        stride(s),
        delta(1) {}
  Matrix(const at::Tensor& x, const int r, const int c, const int s)
      : data(x.data<value_type>()),
        rows(r),
        cols(c),
        ld(r),
        stride(s),
        delta(1) {}
};

template <typename T>
struct Vector : BaseType<T> {
  using pointer = typename BaseType<T>::pointer;
  using value_type = typename BaseType<T>::value_type;
  pointer data;
  int size;
  int inc;

  Vector(at::Tensor& v) : data(v.data<value_type>()), size(v.size(0)), inc(1) {}
  Vector(const at::Tensor& v)
      : data(v.data<value_type>()), size(v.size(0)), inc(1) {}
  // one vectorを使いまわす時に長さを短いものとして宣言
  Vector(at::Tensor& v, const int s)
      : data(v.data<value_type>()), size(s), inc(1) {}
  Vector(const at::Tensor& v, const int s)
      : data(v.data<value_type>()), size(s), inc(1) {}
};
#endif
