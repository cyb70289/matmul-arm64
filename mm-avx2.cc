#include <cstdlib>
#include <cstring>
#include <immintrin.h>

template <int panel_width>
static void mm_panel(const float* __restrict a, const float* __restrict b,
                     float* __restrict c, int m, int n, int k) {
  // XXX: ignore edge case for now
  static_assert(panel_width % 8 == 0);
  if (n % panel_width != 0) std::abort();

  __m256 av;
  __m256 bv[panel_width / 8];
  __m256 cv[panel_width / 8];
  static_assert(sizeof(av) + sizeof(bv) + sizeof(cv) <= 16 * 32);

  // transpose b in panels
  float* bt = new float[k * n];
  {
    float* bt_ptr = bt;
    for (int nn = 0; nn < n; nn += panel_width) {
      const float* b_ptr = b + nn;
      for (int kk = 0; kk < k; ++kk) {
        std::memcpy(bv, b_ptr, sizeof(bv));
        std::memcpy(bt_ptr, bv, sizeof(bv));
        b_ptr += n;
        bt_ptr += panel_width;
      }
    }
  }

  for (int nn = 0; nn < n; nn += panel_width) {
    const float* a_ptr = a;
    float* c_ptr = c + nn;
    for (int mm = 0; mm < m; ++mm) {
      const float* b_ptr = bt + nn * k;
      std::memset(cv, 0, sizeof(cv));
      for (int kk = 0; kk < k; ++kk) {
        av = _mm256_broadcast_ss(a_ptr);
        std::memcpy(bv, b_ptr, sizeof(bv));
        for (int i = 0; i < sizeof(cv) / sizeof(cv[0]); ++i) {
          cv[i] += av * bv[i];
        }
        ++a_ptr;
        b_ptr += panel_width;
      }
      std::memcpy(c_ptr, cv, sizeof(cv));
      c_ptr += n;
    }
  }

  delete[] bt;
}

auto _mm_panel_40 = mm_panel<40>;
