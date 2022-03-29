#ifndef _DEVICE_SCAN_
#define _DEVICE_SCAN_

#include <CL/sycl.hpp>

constexpr size_t LOG2(size_t n) { return n <= 1 ? 0 : 1 + LOG2((n + 1) / 2); }

template <size_t NUM> constexpr size_t CONST_LOG2 = LOG2(NUM);

template <typename T>
T sg_scan(T value, sycl::nd_item<1> it) {
  /* auto local_id = it.get_local_id(0); */
  auto sub_group = it.get_sub_group();
  auto sub_local_id = sub_group.get_local_id();

  for (size_t i = 1; i < sub_group.get_local_range()[0] ; i = i << 1) {
    T v = sub_group.shuffle_up(value, i);
    if(i <= sub_local_id) {
      value += v;
    }
  }

  return value;
}

#endif
