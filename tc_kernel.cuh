#pragma once
#include "cuda_utils.h"
#include "utils.h"

__forceinline__ __device__ bool binary_search(int *list, int key, int size) {
  int l = 0;
  int r = size - 1;
  while (r >= l) {
    int mid = l + (r - l) / 2;
    int value = list[mid];
    if (value == key)
      return true;
    if (value < key)
      l = mid + 1;
    else
      r = mid - 1;
  }
  return false;
}

DEV_INLINE int intersect_num_bs(int *a, int size_a, int *b, int size_b) {
  if (size_a == 0 || size_b == 0)
    return 0;
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);       // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE; // warp index within the CTA
  int num = 0;
  int *lookup = a;
  int *search = b;
  int lookup_size = size_a;
  int search_size = size_b;
  __syncwarp();

  for (auto i = 0; i < lookup_size; i += 1) {
    auto key = lookup[i]; // each thread picks a vertex as the key
    if (binary_search(search, key, search_size))
      num += 1;
  }
  return num;
}

DEV_INLINE int intersect_num(int *a, int size_a, int *b, int size_b) {
  return intersect_num_bs(a, size_a, b, size_b);
}

__global__ void tc_base(int nv, dev::Graph g, AccType *total) {
  
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int num_threads =  gridDim.x * blockDim.x;

  
  AccType count = 0;
  for (int vid = thread_id; vid < nv; vid += num_threads) {
    auto v = vid;
    int *v_ptr = g.getNeighbor(v);
    int v_size = g.getOutDegree(v);
    for (auto e = 0; e < v_size; e ++) {
      auto u = v_ptr[e];
      int u_size = g.getOutDegree(u);
      count += intersect_num(v_ptr, v_size, g.getNeighbor(u), u_size);
    }
  }
 
    atomicAdd(total, count);
}




