#include "operations.cuh"

// #include "omp.h"
using namespace project_GraphFold;

#include "tc_kernel.cuh"
#include "cuda_utils.h"
#include "utils.h"

void tc_solver(project_GraphFold::Graph &hg, uint64_t &result) {
  
  
  int ne = hg.get_enum();
  int nv = hg.get_vnum();
  int max_degree = hg.getMaxDegree();

  int list_num = 6;
  size_t per_block_vlist_size =
      // WARPS_PER_BLOCK * size_t(k - 3) * size_t(max_degree) * sizeof(int);
      WARPS_PER_BLOCK * list_num * size_t(max_degree) * sizeof(int);

  AccType *d_total;
  int count_length = 6;
  AccType h_total[count_length] = {0,0,0};
  DMALLOC(d_total, count_length * sizeof(AccType));
  TODEV(d_total, &h_total, count_length * sizeof(AccType));
  // CLEAN(d_total, 4*sizeof(AccType));
  WAIT();
  auto d_g = hg.DeviceObject();
  int grid_size, block_size; // uninitialized

  H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               tc_opt, 0,
                                               (int)MAX_BLOCK_SIZE));
  size_t flist_size = grid_size * per_block_vlist_size;
  std::cout << "flist_size is " << flist_size / (1024 * 1024)
            << " MB, grid_size is " << grid_size << ", per_block_vlist_size is "
            << per_block_vlist_size << std::endl;
  int *d_frontier_list;
  DMALLOC(d_frontier_list, flist_size);

  double start = wtime();
  tc_base<<<grid_size, block_size>>>(nv, d_g, d_total);


  WAIT();
  double end = wtime();
  std::cout << "Triangle counting base " << " matching  time: " << (end - start)
            << " seconds" << std::endl;

  start = wtime();
  tc_opt_reduce_diff<<<grid_size, block_size>>>(nv, d_g, d_total+1);


  WAIT();
  end = wtime();
  std::cout << "Triangle counting opt reduce diff" << " matching  time: " << (end - start)
            << " seconds" << std::endl;

  start = wtime();
  tc_opt_merge_calc<<<grid_size, block_size>>>(nv, d_g, d_total+2);


  WAIT();
  end = wtime();
  std::cout << "Triangle counting opt merge calc" << " matching  time: " << (end - start)
            << " seconds" << std::endl;

  TOHOST(d_total, &h_total, count_length * sizeof(AccType));

  // calc_connect_itself<<<grid_size, block_size>>>(nv, d_g, d_total+3);
  result = h_total[1];

 

  FREE(d_total);
}