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
  static_assert(sizeof(av) + sizeof(bv) + sizeof(cv) <= 32 * 32);

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
        for (unsigned i = 0; i < sizeof(cv) / sizeof(cv[0]); ++i) {
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

template <int tile_height, int tile_width>
static void mm_tile(const float* __restrict a, const float* __restrict b,
                    float* __restrict c, int m, int n, int k) {
  //XXX: ignore edge case for now
  static_assert(tile_width % 8 == 0);
  if (m % tile_height || n % tile_width) std::abort();

  // transpose each row panel of a per element
  float *at = new float[m * k];
  float* at_ptr = at;
  for (int mm = 0; mm < m; mm += tile_height) {
    const float* a_ptr = a + mm * k;
    for (int kk = 0; kk < k; ++kk) {
      for (int h = 0; h < tile_height; ++h) {
        *at_ptr++ = a_ptr[h * k + kk];
      }
    }
  }

  // transpose each col panel of b per row
  float *bt = new float[k * n];
  float *bt_ptr = bt;
  for (int nn = 0; nn < n; nn += tile_width) {
    const float* b_ptr = b + nn;
    for (int kk = 0; kk < k; ++kk) {
      std::memcpy(bt_ptr, b_ptr + kk * n, tile_width * sizeof(float));
      bt_ptr += tile_width;
    }
  }

  struct {
    __m256 a_dup[tile_height];
    __m256 b_row[tile_width / 8];
    __m256 c_tile[tile_height][tile_width / 8];
  } v;
  // let the compiler take care of "more than hardware registers"
  // static_assert(sizeof(v) <= 16 * sizeof(__m256));

  for (int nn = 0; nn < n; nn += tile_width) {
    for (int mm = 0; mm < m; mm += tile_height) {
      const float* a_ptr = at + mm * k;
      const float* b_ptr = bt + nn * k;
      float* c_ptr = c + mm * n + nn;

      std::memset(v.c_tile, 0, sizeof(v.c_tile));
      for (int kk = 0; kk < k; ++kk) {
        // load and broadcast a
        for (int h = 0; h < tile_height; ++h) {
          v.a_dup[h] = _mm256_broadcast_ss(a_ptr++);
        }
        // load row b
        std::memcpy(v.b_row, b_ptr, sizeof(v.b_row));
        b_ptr += sizeof(v.b_row) / 4/*sizeof(float)*/;
        // accumulate c tile
        for (int h = 0; h < tile_height; ++h) {
          for (int w = 0; w < tile_width; w += 8) {
            v.c_tile[h][w/8] += v.a_dup[h] * v.b_row[w/8];
          }
        }
      }

      // store tile c
      for (int h = 0; h < tile_height; ++h) {
        std::memcpy(c_ptr + h * n, v.c_tile[h], tile_width * sizeof(float));
      }
    }
  }

  delete[] at;
  delete[] bt;
}

auto _mm_panel_40 = mm_panel<40>;
auto _mm_tile_4x24 = mm_tile<4, 24>;
