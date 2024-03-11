// built with clang-16, tested on bluewhale
// run time (the shorter the better)
// - onednn + acl:   1194 ms
// - baseline:       3262 ms
// - panel:          1368 ~ 1510 ms
// - panel-asm:      1171 ~ 1326 ms
// - tile:           1170 ~ 1199 ms
// - tile-asm:       1168 ~ 1202 ms
// - tile-transpose: 1143 ms

#include <cfloat>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_set>

using mm_func = void(*)(const float*, const float*, float*, int, int, int);

extern mm_func _mm_baseline;
extern mm_func _mm_panel_24;
extern mm_func _mm_tile_8x8;
extern mm_func _mm_tile_8x8_T;
extern mm_func _mm_panel_24_asm;
extern mm_func _mm_tile_8x8_asm;

struct {
  const char* name;
  mm_func func;
} mm_funcs[] {
  {"baseline",       _mm_baseline     },
  {"panel",          _mm_panel_24     },
  {"panel-asm",      _mm_panel_24_asm },
  {"tile",           _mm_tile_8x8     },
  {"tile-asm",       _mm_tile_8x8_asm },
  {"tile-transpose", _mm_tile_8x8_T   },
};
const int n_funcs = sizeof(mm_funcs) / sizeof(mm_funcs[0]);

void init_data(float* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = static_cast<float>(i);
  }
}

int main(int argc, char* argv[]) {
  bool verify = false;
  std::unordered_set<std::string> test_names;
  // run last test if no specified
  std::string test_name = argc > 1 ? argv[1] : mm_funcs[n_funcs-1].name;
  if (test_name == "list") {
    // list all benchmarks
    for (const auto [name, _] : mm_funcs) {
      std::cout << name << '\n';
    }
    return 0;
  } else if (test_name == "all" || test_name == "test") {
    // run all benchmarks
    for (const auto [name, _] : mm_funcs) {
      test_names.insert(name);
    }
    // verify test results
    verify = test_name == "test";
  } else {
    // run specific benchmark
    for (const auto [name, _] : mm_funcs) {
      if (test_name == name) {
        test_names.insert(name);
        break;
      }
    }
    if (test_names.empty()) {
      std::cerr << "unknown benchmark: " << test_name << '\n';
      std::cerr << "supported options: \n";
      std::cerr << "- list:   list all benchmark name\n";
      std::cerr << "- all:    run all benchmarks\n";
      std::cerr << "- test:   verify all benchmarks\n";
      std::cerr << "- [name]: specify valid benchmark name\n";
      return 1;
    }
  }

  const int batch = 512;
  const int m = 1000, n = 240, k = 200;

  float *a = new float[batch*m*k];
  float *b = new float[batch*k*n];
  float *c = new float[batch*m*n];

  init_data(a, batch*m*k);
  init_data(b, batch*k*n);

  float *t = nullptr;
  if (verify) {
    std::cout << "calculate baseline result as ground truth\n";
    t = new float[batch*m*n];
    for (long i = 0; i < batch; ++i) {
      _mm_baseline(a + i*m*k, b + i*k*n, t + i*m*n, m, n, k);
    }
  }

  for (auto [name, func] : mm_funcs) {
    if (test_names.find(name) == test_names.end()) continue;
    if (verify && std::string(name) == "baseline") continue;
    std::cout << "========== " << name << " ==========\n";

    // warmup
    for (long i = 0; i < batch; ++i) {
      func(a + i*m*k, b + i*k*n, c + i*m*n, m, n, k);
    }

    if (verify) {
      // compare against baseline test result
      for (long i = 0; i < static_cast<long>(batch)*m*n; ++i) {
        if (std::fabs(c[i] - t[i]) > FLT_MIN) {
          std::cerr << "FAILED! " << i << ": " << c[i] << " != " << t[i];
          return 1;
        }
      }
      std::cout << "OK\n";
    } else {
      // benchnmark
      const auto start = std::chrono::high_resolution_clock::now();
      for (long i = 0; i < batch; ++i) {
        func(a + i*m*k, b + i*k*n, c + i*m*n, m, n, k);
      }
      const auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> duration = end - start;
      std::cout << "time: " << duration.count() << " ms\n";

      // print some results for quick debugging
      std::cerr << "c[0]    = " << c[0] << '\n';
      std::cerr << "c[9973] = " << c[9973] << '\n';
      std::cerr << "c[-1]   = " << c[batch*m*n - 1] << '\n';
    }
  }

  delete[] a;
  delete[] b;
  delete[] c;
  delete[] t;
  return 0;
}
