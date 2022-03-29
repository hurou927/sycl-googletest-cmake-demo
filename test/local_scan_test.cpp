
#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include <device_scan.hpp>

using namespace cl::sycl;

template <typename T, size_t NUM_THREADS_PER_GROUP>
// return sub_group_size
size_t run_kernel(queue &q, T *host_input_buf, T *host_output_buf,
                  size_t num_items) {
  T *DEVICE_IN = static_cast<T *>(malloc_device(num_items * sizeof(T), q));
  T *DEVICE_OUT = static_cast<T *>(malloc_device(num_items * sizeof(T), q));
  T *DEVICE_SUB_GROUP_SIZE = static_cast<T *>(malloc_device(sizeof(size_t), q));
  q.memcpy(DEVICE_IN, host_input_buf, sizeof(T) * num_items).wait();
  event kernel_event = q.submit([&](handler &cgh) {
    auto localRange = range<1>(NUM_THREADS_PER_GROUP);

    auto kernel = [=](nd_item<1> it) {
      auto global_id = it.get_global_id(0);

      T value = DEVICE_IN[global_id];
      DEVICE_OUT[global_id] = sg_scan<T>(value, it);
      if (global_id == 0) {
        DEVICE_SUB_GROUP_SIZE[0] = it.get_sub_group().get_local_range()[0];
      }
    };
    cgh.parallel_for<class pm>(nd_range<1>{range<1>(num_items), localRange},
                               kernel);
  });
  kernel_event.wait();
  q.memcpy(host_output_buf, DEVICE_OUT, sizeof(T) * num_items).wait();
  size_t sub_group_size = 0;
  q.memcpy(&sub_group_size, DEVICE_SUB_GROUP_SIZE, sizeof(size_t)).wait();
  free(DEVICE_IN, q);
  free(DEVICE_OUT, q);

  return sub_group_size;
}

TEST(SubGroupScan, BasicAssertions) {

  gpu_selector d_selector;
  queue q(d_selector);
  size_t num_items = 1024 * 1024;
  int *host_in = static_cast<int *>(malloc(num_items * sizeof(int)));
  int *host_out = static_cast<int *>(malloc(num_items * sizeof(int)));
  for (int i = 0; i < num_items; i++) {
    host_in[i] = 1;
  }
  size_t sub_group_size = run_kernel<int, 256>(q, host_in, host_out, num_items);
  EXPECT_TRUE(sub_group_size > 0);
  for (int i = 0; i < 64; i = i + sub_group_size) {
    for (int s = 0; s < sub_group_size; s++) {
      EXPECT_TRUE(host_out[i + s] == s + 1);
    }
  }

  free(host_in);
  free(host_out);
}
