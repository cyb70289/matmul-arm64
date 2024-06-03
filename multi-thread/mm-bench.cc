#include <iostream>
#include <thread>
#include <vector>
#include <unistd.h>

using reorder_func = void(*)(const float*, const float*, float*, float*,
                             int, int, int);
using mm_func = void(*)(const float*, const float*, float*, int, int, int);
extern reorder_func _reorder;
extern mm_func _mm_tile_8x8;

int main(int argc, char* argv[]) {
  constexpr int batch = 2048;
  constexpr int m = 512, n = 256, k = 128;

  const int n_threads = []() {
    const char* threads = std::getenv("MM_NUM_THREADS");
    if (!threads) return 1;
    return std::atoi(threads);
  }();
  if (n_threads == 0) {
    std::cerr << "invalid thread count\n";
    return 1;
  } else if (batch % n_threads != 0) {
    std::cerr << "thread count must divide batch size " << batch << " \n";
    return 1;
  }
  const int mini_batch = batch / n_threads;

  // initialize data
  float *a = new float[batch*m*k];
  float *b = new float[batch*k*n];
  float *c = new float[batch*m*n]{};
  float *a_tx = new float[batch*m*k]{};
  float *b_tx = new float[batch*k*n]{};

  auto init_data = [](float* data, int size) {
    for (int i = 0; i < size; ++i) {
      data[i] = static_cast<float>(i);
    }
  };
  init_data(a, batch*m*k);
  init_data(b, batch*k*n);

  // define reorder and multiplier functor
  auto reorder = [a, b, a_tx, b_tx, mini_batch](int idx) {
    const int si = idx * mini_batch;
    const int ei = si + mini_batch;
    for (long i = si; i < ei; ++i) {
      _reorder(a + i*m*k, b + i*k*n, a_tx + i*m*k, b_tx + i*k*n, m, n, k);
    }
  };
  auto multiplier = [a_tx, b_tx, c, mini_batch](int idx) {
    const int si = idx * mini_batch;
    const int ei = si + mini_batch;
    for (long i = si; i < ei; ++i) {
      _mm_tile_8x8(a_tx + i*m*k, b_tx + i*k*n, c + i*m*n, m, n, k);
    }
  };

  // run the benchmark
  std::vector<std::thread> worker(n_threads);
  while (true) {
    const auto start = std::chrono::high_resolution_clock::now();

    const int bench_loops = n_threads;
    for (int i = 0; i < bench_loops; ++i) {
      // - do reorder with n_threads in parallel
      for (int i = 0; i < n_threads; ++i) {
        worker[i] = std::thread(reorder, i);
      }
      for (int i = 0; i < n_threads; ++i) {
        worker[i].join();
      }
      // - do matrix multiplication with n_threads in parallel
      for (int i = 0; i < n_threads; ++i) {
        worker[i] = std::thread(multiplier, i);
      }
      for (int i = 0; i < n_threads; ++i) {
        worker[i].join();
      }
    }

    const auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    const long ops = static_cast<long>(n_threads * batch / duration.count());
    std::cout << "pid=" << getpid() << ", threads=" << n_threads \
              << ", ops=" << ops << '\n';
#if 0
    std::cerr << "c[0]    = " << c[0] << '\n';
    std::cerr << "c[9973] = " << c[9973] << '\n';
    std::cerr << "c[-1]   = " << c[batch*m*n - 1] << '\n';
#endif
  }

  delete[] a;
  delete[] b;
  delete[] c;
  delete[] a_tx;
  delete[] b_tx;
  return 0;
}
