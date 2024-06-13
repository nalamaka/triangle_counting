#ifndef GRAPH_GRAPH_H
#define GRAPH_GRAPH_H

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "buffer.h"
#include "cuda_utils.h"
#include "logging.h"
#include "operations.cuh"
#include "scan.h"
#include "timer.h"
#include "utils.h"

namespace project_GraphFold {

// int
class Graph;

namespace dev {
class Graph {
private:
  size_t vsize_;  // uninitialized
  size_t esize_;  // uninitialized
  int max_degree; // uninitialized
  Buffer vlabels_;
  Buffer row_ptr_;
  Buffer col_idx_;
  Buffer odegs_;
  Buffer src_list_;
  Buffer dst_list_;

friend class project_GraphFold::Graph;

public:
  // Graph(project_GraphFold::Graph& hg) { init(hg);}
  int get_vnum() const { return vsize_; }
  int get_enum() const { return esize_; }
  DEV_INLINE int get_src(int edge) const { return src_list_.data()[edge]; }
  DEV_INLINE int get_dst(int edge) const { return dst_list_.data()[edge]; }
  DEV_INLINE int getOutDegree(int src) {
    return col_idx_.data()[src + 1] - col_idx_.data()[src];
  } // check
  DEV_INLINE size_t get_colidx_size() const { return col_idx_.size(); }
  DEV_INLINE int edge_begin(int src) const { return col_idx_.data()[src]; }
  DEV_INLINE int edge_end(int src) const { return col_idx_.data()[src + 1]; }
  DEV_INLINE int get_edge_dst(int idx) const { return row_ptr_.data()[idx]; }
  // Test and dump COO
  DEV_INLINE void DumpCO() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf("Dump COO: src_list size: %d, dst_list size: %d.\n",
             src_list_.size(), dst_list_.size());
      for (int i = 0; i < src_list_.size(); i++) {
        printf("%d ", src_list_.data()[i]);
      }
      printf("\n");
      for (int i = 0; i < dst_list_.size(); i++) {
        printf("%d ", dst_list_.data()[i]);
      }
    }
  }

  DEV_INLINE int *getNeighbor(int vid) const {
    return const_cast<int *>(row_ptr_.data()) + col_idx_.data()[vid];
  }
};
} // namespace dev

 class Graph {
public:
  using device_t = dev::Graph;
  // TODO: To support multiple partition in vertex-cut manner.
  // To this end, we have to store the vertex mapping(original_id->local_id)
  // get the neighborlist START pointer
  int *getNeighbor(int vid) const {
    return const_cast<int *>(row_ptr_.data()) + col_idx_.data()[vid];
  }
  int edge_begin(int src) const { return col_idx_.data()[src]; }
  int edge_end(int src) const { return col_idx_.data()[src + 1]; }

  int get_src(int idx) const { return src_list_[idx]; }
  int get_dst(int idx) const { return dst_list_[idx]; }

  int get_vnum() const { return vsize_; }
  int get_enum() const { return esize_; }
  int getMaxDegree() { return max_degree_; }
  int *getSrcPtr(int start) { return src_list_.data() + start; }
  int *getDstPtr(int start) { return dst_list_.data() + start; }
  int getOutDegree(int src) {
    return col_idx_.data()[src + 1] - col_idx_.data()[src];
  }
  size_t getNNZ() { return nnz; }
  int CalMaxDegree(std::vector<int> out_degs) {
    auto maxPosition = max_element(out_degs.begin(), out_degs.end());
    return *maxPosition;
  }

  
  // USE_DAG on with orientation
  void orientation(bool NeedToLoadToDevice = true) {
    std::cout << "Orientation enabled, DAG generated.\n" << std::endl;
    double start = wtime();
    std::vector<int> new_odegs_(vsize_, 0);
#pragma omp parallel for
    // Dump(std::cout);
    for (int src = 0; src < vsize_; ++src) {
      int *neighlist = getNeighbor(src);
      Buffer tmp(neighlist, getOutDegree(src));
      // std::cout << " size of neighlist: " << sizeof(neighlist);
      for (auto dst : tmp) {
        // std::cout << "i is " << i << ", dst is " << dst;
        if (odegs_[dst] > odegs_[src] ||
            (odegs_[dst] == odegs_[src] && dst > src)) {
          new_odegs_[src]++;
        }
      }
    }

    int new_max_degree_ = CalMaxDegree(new_odegs_);
    std::cout << "Orientation Generating: New max degree is: "
              << new_max_degree_ << std::endl;
    // vector type: this.row_ptr_; this.col_idx_;
    std::vector<int> new_row_ptr_;
    std::vector<int> new_col_idx_;
    std::vector<int> new_src_list_;
    new_col_idx_.resize(vsize_ + 1);
    parallel_prefix_sum(new_odegs_,
                                  new_col_idx_.data()); // vector satisfied
    auto n_edges_ = new_col_idx_[vsize_];
    new_row_ptr_.resize(n_edges_);
    new_src_list_.resize(n_edges_);
#pragma omp parallel for
    for (int src = 0; src < vsize_; ++src) {
      int *neighlist = getNeighbor(src);
      Buffer tmp(neighlist, getOutDegree(src));
      auto begin = new_col_idx_[src];
      int offset = 0;
      for (auto dst : tmp) {
        if (odegs_[dst] > odegs_[src] ||
            (odegs_[dst] == odegs_[src] && dst > src)) {
          new_row_ptr_[begin + offset] = dst;
          new_src_list_[begin + offset] = src;
          offset++;
        }
      }
    }
    // Update graph info
    row_ptr_ = new_row_ptr_;
    col_idx_ = new_col_idx_;
    esize_ = n_edges_;
    max_degree_ = new_max_degree_;

    double end = wtime();
    std::cout << "Orientation Generating time: " << (end - start) << " seconds"
              << std::endl;

    src_list_ = new_src_list_;
    dst_list_ = new_row_ptr_;
  }

  void SortCSRGraph(bool NeedToLoadToDevice = true) {
    std::vector<int> index(vsize_);
    std::vector<int> r_index(vsize_);
    for (int i = 0; i < index.size(); i++)
      index[i] = i;
    std::stable_sort(index.begin(), index.end(), [&](int a, int b) {
      return getOutDegree(a) > getOutDegree(b);
    });

    std::vector<int> new_col_idx_(vsize_ + 1);
    std::vector<int> new_row_ptr_(esize_);
    std::vector<int> new_odegs_(vsize_, 0);

    for (int src = 0; src < vsize_; src++) {
      int v = index[src];
      r_index[v] = src;
    }

    for (int src = 0; src < vsize_; src++) {
      int v = index[src];
      new_odegs_[src] = getOutDegree(v);
    }
    parallel_prefix_sum(new_odegs_,
                                  new_col_idx_.data()); // vector satisfied
    for (int src = 0; src < vsize_; src++) {
      int v = index[src];
      int *neighlist = getNeighbor(v);
      Buffer tmp(neighlist, getOutDegree(v));
      auto begin = new_col_idx_[src];
      int offset = 0;
      for (auto dst : tmp) {
        new_row_ptr_[begin + offset] = r_index[dst];
        offset++;
      }
      std::sort(&new_row_ptr_[begin], &new_row_ptr_[begin + offset]);
    }

    col_idx_ = new_col_idx_;
    row_ptr_ = new_row_ptr_;
    odegs_ = new_odegs_;
  }

  // initialize the size of device pointer vector
  void resizeDeviceVector(int n_dev) {
    d_row_ptr_.resize(n_dev);
    d_odegs_.resize(n_dev);
    d_col_idx_.resize(n_dev);
    d_src_list_.resize(n_dev);
    d_dst_list_.resize(n_dev);
    d_vlabels_.resize(n_dev);
  }

  void copyToDevice(size_t start, size_t end, int n_dev, bool sym_break = false,
                    bool use_label = false) {
    resizeDeviceVector(n_dev);
    auto n = end - start;
    int n_tasks_per_gpu = (n - 1) / n_dev + 1;
    for (int i = 0; i < n_dev; ++i) {
      SetDevice(i);
      if (use_label) {
        d_vlabels_[i].resize(vsize_);
        TODEV(thrust::raw_pointer_cast(d_vlabels_.data()), vlabels_.data(),
              sizeof(int) * vsize_);
      }

      int begin = start + i * n_tasks_per_gpu;
      // Note: Test only.
      // if (!sym_break) d_dst_list_[i] = row_ptr_.data() + begin;
      int num = n_tasks_per_gpu;
      if (begin + num > end)
        num = end - begin; // begin is the index for copy starting
      // initialize CSR
      d_row_ptr_[i].resize(esize_);
      d_odegs_[i].resize(vsize_);
      d_col_idx_[i].resize(vsize_ + 1);
      // initialize COO task list with size 'num'
      d_src_list_[i].resize(num);
      d_dst_list_[i].resize(num);
      // copy all CSR
      TODEV(thrust::raw_pointer_cast(d_row_ptr_[i].data()), row_ptr_.data(),
            sizeof(int) * esize_);
      TODEV(thrust::raw_pointer_cast(d_odegs_[i].data()), odegs_.data(),
            sizeof(int) * vsize_);
      TODEV(thrust::raw_pointer_cast(d_col_idx_[i].data()), col_idx_.data(),
            sizeof(int) * (vsize_ + 1)); // size_ to int
      // copy partial
      TODEV(thrust::raw_pointer_cast(d_src_list_[i].data()),
            src_list_.data() + begin, sizeof(int) * num);
      if (!sym_break) {
        TODEV(thrust::raw_pointer_cast(d_dst_list_[i].data()),
              row_ptr_.data() + begin, sizeof(int) * num);
      } else {
        TODEV(thrust::raw_pointer_cast(d_dst_list_[i].data()),
              dst_list_.data() + begin, sizeof(int) * num);
      } // sym_break_copy

      WAIT();
      std::cout << "Successful fill into GPU[" << i << "]." << std::endl;
    }
  }

  void copyToDevice(int n_dev, std::vector<int> tasks, std::vector<int *> &srcs,
                    std::vector<int *> &dsts, bool use_label = false) {
    // Timer t;
    // t.Start();
    resizeDeviceVector(n_dev);
    for (int i = 0; i < n_dev; ++i) {
      SetDevice(i);
      if (use_label) {
        d_vlabels_[i].resize(vsize_);
        TODEV(thrust::raw_pointer_cast(d_vlabels_.data()), vlabels_.data(),
              sizeof(int) * vsize_);
      }
      // initialize CSR
      d_row_ptr_[i].resize(esize_);
      d_odegs_[i].resize(vsize_);
      d_col_idx_[i].resize(vsize_ + 1);

      // initialize COO task list with size 'num'
      auto num = tasks[i];
      d_src_list_[i].resize(num);
      d_dst_list_[i].resize(num);

      // copy all CSR
      TODEV(thrust::raw_pointer_cast(d_row_ptr_[i].data()), row_ptr_.data(),
            sizeof(int) * esize_);
      TODEV(thrust::raw_pointer_cast(d_odegs_[i].data()), odegs_.data(),
            sizeof(int) * vsize_);
      TODEV(thrust::raw_pointer_cast(d_col_idx_[i].data()), col_idx_.data(),
            sizeof(int) * (vsize_ + 1)); // size_ to int

      // copy partial
      int *src_ptr = srcs[i];
      int *dst_ptr = dsts[i];

      // printf("hi, %d",int(srcs[i]));
      TODEV(thrust::raw_pointer_cast(d_src_list_[i].data()), src_ptr,
            sizeof(int) * num); // srcs[i]
      TODEV(thrust::raw_pointer_cast(d_dst_list_[i].data()), dst_ptr,
            sizeof(int) * num); // dsts[i]
      WAIT();
      std::cout << "Successful fill into GPU[" << i << "]." << std::endl;
    }
  }

  void copyToDevice(int n_dev, std::vector<int> tasks,
                    std::vector<std::vector<int>> &srcs,
                    std::vector<std::vector<int>> &dsts,
                    bool use_label = false) {
    resizeDeviceVector(n_dev);
    for (int i = 0; i < n_dev; ++i) {
      SetDevice(i);
      if (use_label) {
        d_vlabels_[i].resize(vsize_);
        TODEV(thrust::raw_pointer_cast(d_vlabels_.data()), vlabels_.data(),
              sizeof(int) * vsize_);
      }
      // initialize CSR
      d_row_ptr_[i].resize(esize_);
      d_odegs_[i].resize(vsize_);
      d_col_idx_[i].resize(vsize_ + 1);

      // initialize COO task list with size 'num'
      auto num = tasks[i];
      d_src_list_[i].resize(num);
      d_dst_list_[i].resize(num);

      // copy all CSR
      TODEV(thrust::raw_pointer_cast(d_row_ptr_[i].data()), row_ptr_.data(),
            sizeof(int) * esize_);
      TODEV(thrust::raw_pointer_cast(d_odegs_[i].data()), odegs_.data(),
            sizeof(int) * vsize_);
      TODEV(thrust::raw_pointer_cast(d_col_idx_[i].data()), col_idx_.data(),
            sizeof(int) * (vsize_ + 1)); // size_ to int

      // copy partial
      int *src_ptr = srcs[i].data();
      int *dst_ptr = dsts[i].data();

      // printf("hi, %d",int(srcs[i]));
      TODEV(thrust::raw_pointer_cast(d_src_list_[i].data()), src_ptr,
            sizeof(int) * num); // srcs[i]?
      TODEV(thrust::raw_pointer_cast(d_dst_list_[i].data()), dst_ptr,
            sizeof(int) * num); // dsts[i]?
      WAIT();
      std::cout << "Successful fill into GPU[" << i << "]." << std::endl;
    }
  }

  void Init(std::vector<int> const &vids, std::vector<int> const &vlabels,
            std::vector<std::pair<int, int>> const &edges, int n_dev,
            bool use_label = false) {
    std::cout << "Initializing graph..." << std::endl;
    double start = wtime();

    vsize_ = vids.size();
    esize_ = edges.size();
    if (use_label)
      vlabels_ = std::move(vlabels);
    odegs_.resize(vsize_);
    col_idx_.resize(vsize_ + 1);
    row_ptr_.resize(esize_);

    src_list_.resize(esize_);
    dst_list_.resize(esize_);

    for (size_t i = 0; i < edges.size(); ++i) {
      odegs_[edges[i].first]++;
    }

    col_idx_[0] = 0;
    for (size_t i = 0; i < vsize_; ++i) {
      col_idx_[i + 1] = col_idx_[i] + odegs_[i];
      odegs_[i] = 0;
    }

    // directed edges
    for (size_t i = 0; i < esize_; ++i) {
      int v0 = edges[i].first;
      int v1 = edges[i].second;
      row_ptr_[col_idx_[v0] + odegs_[v0]] = v1;
      odegs_[v0]++;
    }

    double end = wtime();
    std::cout << "CSR transforming time: " << end - start << "s" << std::endl;
    std::cout << " -- vsize: " << vsize_ << " esize: " << esize_ << "\n"
              << std::endl;
    // calculate max degree
    max_degree_ = CalMaxDegree(odegs_); // int

    // generating COO
    // Note: May use vector<std::pair<int, int>> instead.
    double start_coo = wtime();
    nnz = esize_; // no sym_break, no ascend.
    for (size_t i = 0; i < esize_; ++i) {
      src_list_[i] = edges[i].first;
      dst_list_[i] = edges[i].second;
    }
    double end_coo = wtime();
    std::cout << "COO loading time: " << end_coo - start_coo << "s"
              << std::endl;
  }

  // Only for single GPU
  device_t DeviceObject() const {
    device_t dg;
    // if (use_label)
    // dg.vlabels_ = Buffer(d_vlabels_[0]);
    dg.row_ptr_ = Buffer(d_row_ptr_[0]);
    dg.odegs_ = Buffer(d_odegs_[0]);
    dg.col_idx_ = Buffer(d_col_idx_[0]);
    dg.src_list_ = Buffer(d_src_list_[0]);
    dg.dst_list_ = Buffer(d_dst_list_[0]);

    return dg;
  }

  device_t DeviceObject(int dev_id,
                        bool use_label = false) const { // DEV_HOST, now is HOST
    device_t dg;
    if (use_label)
      dg.vlabels_ = Buffer(d_vlabels_[dev_id]);
    dg.row_ptr_ = Buffer(d_row_ptr_[dev_id]);
    dg.odegs_ = Buffer(d_odegs_[dev_id]);
    dg.col_idx_ = Buffer(d_col_idx_[dev_id]);
    dg.src_list_ = Buffer(d_src_list_[dev_id]);
    dg.dst_list_ = Buffer(d_dst_list_[dev_id]);

    return dg;
  }

  void Dump(std::ostream &out) {
    out << "vsize: " << vsize_ << " esize: " << esize_ << "\n";
    out << "labels: ";
    for (size_t i = 0; i < vsize_; ++i) {
      out << vlabels_[i] << " ";
    }
    out << "\n";
    out << "row_ptr: ";
    for (size_t i = 0; i < esize_; ++i) {
      out << row_ptr_[i] << " ";
    }
    out << "\n";
    out << "col_idx: ";
    for (size_t i = 0; i < vsize_ + 1; ++i) {
      out << col_idx_[i] << " ";
    }
    out << "\n";
  }

  void DumpCOO(std::ostream &out) {
    out << "vsize: " << vsize_ << " esize: " << esize_ << "\n";
    out << "labels: ";
    for (size_t i = 0; i < vsize_; ++i) {
      out << vlabels_[i] << " ";
    }
    out << "\n";
    out << "src_list: ";
    for (size_t i = 0; i < esize_; ++i) {
      out << src_list_[i] << " ";
    }
    out << "\n";
    out << "dst_list: ";
    for (size_t i = 0; i < esize_; ++i) {
      out << dst_list_[i] << " ";
    }
    out << "\n";
  }

  bool query_dense_graph() { return is_dense_graph; }

private:
  // Warning: NOT support device_id & n_gpu yet.
  size_t fid_; // ?
  size_t vsize_;
  size_t esize_;
  bool is_dense_graph;
  int max_degree_;
  std::vector<int> vlabels_;

  // int num_vertex_classes; // int classes count
  // may used by filter
  // std::vector<int> vlabels_frequency_;
  // int max_int_frequency_;
  // int max_int;
  // std::vector<nlf_map> nlf_;
  // std::vector<int> sizes;
  // CSR
  std::vector<int> row_ptr_;
  std::vector<int> col_idx_;
  std::vector<int> odegs_; // <size_t>
  // add evlabels_
  // COO
  int nnz;
  std::vector<int> src_list_; // <size_t> thrust host vector?
  std::vector<int> dst_list_; // <size_t>

  // Warning: More supported format may increase the storage.
  // Every GPU has its device vector.
  std::vector<thrust::device_vector<int>> d_vlabels_;
  std::vector<thrust::device_vector<int>> d_row_ptr_;
  std::vector<thrust::device_vector<int>> d_odegs_;
  std::vector<thrust::device_vector<int>> d_col_idx_;
  // assign tasks
  std::vector<thrust::device_vector<int>> d_src_list_;
  std::vector<thrust::device_vector<int>> d_dst_list_;
};

} // namespace project_GraphFold

#endif // endif
