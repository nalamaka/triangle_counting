#ifndef UTILS_ARRAY_VIEW_H
#define UTILS_ARRAY_VIEW_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/swap.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include "cuda_utils.h"
#include "utils.h"

namespace project_GraphFold {


class Buffer {
 public:
  Buffer() = default;

  explicit Buffer(const thrust::device_vector<int>& vec)
      : data_(const_cast<int*>(thrust::raw_pointer_cast(vec.data()))),
        size_(vec.size()) {}

  explicit Buffer(const thrust::host_vector<
                  int, thrust::cuda::experimental::pinned_allocator<int>>& vec)
      : data_(const_cast<int*>(thrust::raw_pointer_cast(vec.data()))),
        size_(vec.size()) {}

  DEV_HOST Buffer(int* data, size_t size) : data_(data), size_(size) {}

  DEV_HOST_INLINE int* data() { return data_; }

  DEV_HOST_INLINE const int* data() const { return data_; }

  DEV_HOST_INLINE size_t size() const { return size_; }

  DEV_HOST_INLINE bool empty() const { return size_ == 0; }

  DEV_HOST_INLINE int& operator[](size_t i) { return data_[i]; }

  DEV_HOST_INLINE const int& operator[](size_t i) const { return data_[i]; }

  DEV_HOST_INLINE void Swap(Buffer& rhs) {
    thrust::swap(data_, rhs.data_);
    thrust::swap(size_, rhs.size_);
  }

  DEV_HOST_INLINE int* begin() { return data_; }

  DEV_HOST_INLINE int* end() { return data_ + size_; }

  DEV_HOST_INLINE const int* begin() const { return data_; }

  DEV_HOST_INLINE const int* end() const { return data_ + size_; }

 private:
  int* data_{};
  size_t size_{};
};

}  // namespace project_GraphFold

#endif  // UTILS_ARRAY_VIEW_H
