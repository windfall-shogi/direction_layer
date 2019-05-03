#pragma once
#ifndef DIRECTION_HPP_INCLUDED
#define DIRECTION_HPP_INCLUDED

enum Direction{
  // 値は次の升へ行くための移動量
  VERTICAL=1, HORIZONTAL=9, DIAGONAL1=10, DIAGONAL2=8
};
template<Direction D>
constexpr int GetNumSlices(){
  return D%2?9:17;
}

// Offset
namespace {
constexpr int diagonal1_offset[] = {
    // 上側の辺
    9 * 8, 9 * 7, 9 * 6, 9 * 5, 9 * 4, 9 * 3, 9 * 2, 9 * 1,
    // 角
    0,
    // 右側の辺
    1, 2, 3, 4, 5, 6, 7, 8};
constexpr int diagonal2_offset[] = {
    // 右側の辺
    0, 1, 2, 3, 4, 5, 6, 7,
    // 角
    8,
    // 下側の辺
    8 + 9 * 1, 8 + 9 * 2, 8 + 9 * 3, 8 + 9 * 4, 8 + 9 * 5, 8 + 9 * 6, 8 + 9 * 7,
    8 + 9 * 8};

constexpr int GetVerticalOffset(const int position_index){
  return position_index * 9;
}
constexpr int GetHorizontalOffset(const int position_index){
  return position_index;
}
constexpr int GetDiagonal1Offset(const int position_index){
  return diagonal1_offset[position_index];
}
constexpr int GetDiagonal2Offset(const int position_index){
  return diagonal2_offset[position_index];
}
}  // namespace
constexpr int GetOffset(const Direction direction, const int position_index) {
  // directionが都合が良いことに奇数と偶数で別れている
  return (direction % 2)
             ? (direction == VERTICAL ? GetVerticalOffset(position_index)
                                      : GetHorizontalOffset(position_index))
             : (direction == DIAGONAL1 ? GetDiagonal1Offset(position_index)
                                       : GetDiagonal2Offset(position_index));
}

namespace {
constexpr int diagonal_size[]={
  // 上側の辺 or 右側の辺
    1, 2, 3, 4, 5, 6, 7, 8,
    // 角
    9,
    // 右側の辺 or 下側の辺
    8, 7, 6, 5, 4, 3, 2, 1
};
}  // namespace
constexpr int GetSize(const Direction direction, const int position_index) {
  // directionが都合が良いことに奇数と偶数で別れている
  return (direction % 2) ? 9 : diagonal_size[position_index];
}

#endif //DIRECTION_HPP_INCLUDED
