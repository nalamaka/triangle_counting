#ifndef GRAPH_OPERATIONS_H
#define GRAPH_OPERATIONS_H
// #include "set_intersection.cuh"
#include "cuda_utils.h"

 DEV_INLINE int warp_reduce(int val) {
  int sum = val;
  sum += SHFL_DOWN(sum, 16);
  sum += SHFL_DOWN(sum, 8);
  sum += SHFL_DOWN(sum, 4);
  sum += SHFL_DOWN(sum, 2);
  sum += SHFL_DOWN(sum, 1);
  sum = SHFL(sum, 0);
  return sum;
}

 DEV_INLINE void warp_reduce_iterative(int &val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  val = SHFL(val, 0);
}


#endif