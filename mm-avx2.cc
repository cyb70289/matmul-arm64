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

  // transpose b (vector-wise) in panels
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

template <int panel_height>
static void mm_panela(const float* __restrict a, const float* __restrict b,
                      float* __restrict c, int m, int n, int k) {
  // XXX: ignore edge case for now
  if (k % 8 != 0 || n % 8 != 0 || m % panel_height != 0) std::abort();

  // transpose a (element-wise) in panels
  float* at = new float[m * k];
  float* at_ptr = at;
  for (int mm = 0; mm < m; mm += panel_height) {
    const float* a_ptr = a + mm * k;
    for (int kk = 0; kk < k; ++kk) {
      for (int i = 0; i < panel_height; ++i) {
        *at_ptr++ = a_ptr[i * k + kk];
      }
    }
  }

  // store c with transposed (vector-wise) format in panels
  float* ct = new float[m * n]{};

  const float* a_ptr = at;

  for (int mm = 0; mm < m; mm += panel_height) {
    const float* b_ptr = b;

    for (int kk = 0; kk < k; ++kk) {
      float* c_ptr = ct + mm * n;

      __m256 av[panel_height];
      for (int h = 0; h < panel_height; ++h) {
        av[h] = _mm256_broadcast_ss(a_ptr);
        ++a_ptr;
      }

      for (int nn = 0; nn < n; nn += 8) {
        const __m256 bv = _mm256_loadu_ps(b_ptr);
        b_ptr += 8;

        for (int h = 0; h < panel_height; ++h) {
          __m256 cv = _mm256_loadu_ps(c_ptr);
          cv += av[h] * bv;
          _mm256_storeu_ps(c_ptr, cv);
          c_ptr += 8;
        }
      }
    }
  }

  // transpose ct back to c
  const float* ct_ptr = ct;
  for (int mm = 0; mm < m; mm += panel_height) {
    for (int nn = 0; nn < n; nn += 8) {
      float* c_ptr = c + mm * n + nn;
      for (int h = 0; h < panel_height; ++h) {
        std::memcpy(c_ptr + h * n, ct_ptr, 8 * sizeof(float));
        ct_ptr += 8;
      }
    }
  }

  delete[] at;
  delete[] ct;
}

// XXX: bad perfomance, just for reference
#if 0
template <int tile_height, int tile_width>
static void mm_tile(const float* __restrict a, const float* __restrict b,
                    float* __restrict c, int m, int n, int k) {

  //XXX: ignore edge case for now
  static_assert(tile_width % 8 == 0);
  if (m % tile_height || n % tile_width || k % 8) std::abort();

  // transpose each row panel of a for sequential memory access
  float *at = new float[m * k];
  float *bt = new float[k * n];
  {
    float* at_ptr = at;
    for (int mm = 0; mm < m; mm += tile_height) {
      const float* a_ptr = a + mm * k;
      for (int col = 0; col < k; col += 8) {
        for (int row = 0; row < tile_height; ++row) {
          std::memcpy(at_ptr, a_ptr + row * k + col, 8 * sizeof(float));
          at_ptr += 8;
        }
      }
    }

    float *bt_ptr = bt;
    for (int nn = 0; nn < n; nn += tile_width) {
      const float* b_ptr = b + nn;
      for (int row = 0; row < k; ++row) {
        std::memcpy(bt_ptr, b_ptr + row * n, tile_width * sizeof(float));
        bt_ptr += tile_width;
      }
    }
  }

  struct {
    const __m256i index[8] = {
      _mm256_set1_epi32(0), _mm256_set1_epi32(1),
      _mm256_set1_epi32(2), _mm256_set1_epi32(3),
      _mm256_set1_epi32(4), _mm256_set1_epi32(5),
      _mm256_set1_epi32(6), _mm256_set1_epi32(7),
    };
    __m256 row_a_brc[8];
    __m256 tile_b[8][tile_width / 8];
    __m256 tile_c[tile_height][tile_width / 8];
  } v;
  static_assert(sizeof(v) <= 32 * 32);

  for (int nn = 0; nn < n; nn += tile_width) {
    for (int mm = 0; mm < m; mm += tile_height) {
      const float* a_ptr = at + mm * k;
      const float* b_ptr = bt + nn * k;
      float* c_ptr = c + mm * n + nn;

      std::memset(v.tile_c, 0, sizeof(v.tile_c));
      for (int kk = 0; kk < k; kk += 8) {
        // load tile b
        std::memcpy(v.tile_b, b_ptr, sizeof(v.tile_b));
        b_ptr += sizeof(v.tile_b) / 4/*sizeof(float)*/;
        // accumulate tile c
        for (int h = 0; h < tile_height; ++h) {
          // load vector from a, broadcast lanes
          __m256 row_a;
          std::memcpy(&row_a, a_ptr, sizeof(row_a));
          a_ptr += 8;
          for (int i = 0; i < 8; ++i) {
            v.row_a_brc[i] = _mm256_permutevar8x32_ps(row_a, v.index[i]);
          }
          for (int w = 0; w < tile_width; w += 8) {
            v.tile_c[h][w/8] += v.row_a_brc[0] * v.tile_b[0][w/8];
            v.tile_c[h][w/8] += v.row_a_brc[1] * v.tile_b[1][w/8];
            v.tile_c[h][w/8] += v.row_a_brc[2] * v.tile_b[2][w/8];
            v.tile_c[h][w/8] += v.row_a_brc[3] * v.tile_b[3][w/8];
            v.tile_c[h][w/8] += v.row_a_brc[4] * v.tile_b[4][w/8];
            v.tile_c[h][w/8] += v.row_a_brc[5] * v.tile_b[5][w/8];
            v.tile_c[h][w/8] += v.row_a_brc[6] * v.tile_b[6][w/8];
            v.tile_c[h][w/8] += v.row_a_brc[7] * v.tile_b[7][w/8];
          }
        }
      }

      // store tile c
      for (int h = 0; h < tile_height; ++h) {
        std::memcpy(c_ptr + h * n, v.tile_c[h], tile_width * sizeof(float));
      }
    }
  }

  delete[] at;
  delete[] bt;
}
#endif

auto _mm_panel_40 = mm_panel<40>;
auto _mm_panela_8 = mm_panela<8>;
