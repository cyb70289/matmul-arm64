#include <cstdlib>
#include <cstring>
#include <arm_neon.h>

// visit both a and b in rows, cache friendly
// - c[row] = a[row][0]*b[0] + a[row][1]*b[1] + ... + a[row][k-1]*b[k-1]
static void mm_baseline(const float* __restrict a, const float* __restrict b,
                        float* __restrict c, int m, int n, int k) {
  const float *a_ptr = a;
  float *c_ptr = c;
  for (int row = 0; row < m; ++row) {
    const float *b_ptr = b;
    for (int col = 0; col < n; ++col) {
      c_ptr[col] = a_ptr[0] * b_ptr[col];
    }
    for (int b_row = 1; b_row < k; ++b_row) {
      b_ptr += n;
      for (int col = 0; col < n; ++col) {
        c_ptr[col] += a_ptr[b_row] * b_ptr[col];
      }
    }
    a_ptr += k;
    c_ptr += n;
  }
}

// calculate c by column panels, re-use b panel in cache
// - c[0][00~23], c[1][00~23], ...
// - c[0][24~47], c[1][24,27], ...
template <int col_blk_size = 24>
static void mm_panel(const float* __restrict a, const float* __restrict b,
                     float* __restrict c, int m, int n, int k) {
  // XXX: ignore edge case for now
  if (n % col_blk_size != 0 && k % 4 == 0) std::abort();

  for (int col = 0; col < n; col += col_blk_size) {
    const float* a_ptr = a;
    float* c_ptr = c + col;
    for (int row = 0; row < m; ++row) {
      const float* b_ptr = b + col;
      float v[col_blk_size]{};
      for (int i = 0; i < k; i += 4) {
        for (int j = 0; j < col_blk_size; ++j) {
          v[j] += a_ptr[i] * b_ptr[j];
        }
        b_ptr += n;
        for (int j = 0; j < col_blk_size; ++j) {
          v[j] += a_ptr[i+1] * b_ptr[j];
        }
        b_ptr += n;
        for (int j = 0; j < col_blk_size; ++j) {
          v[j] += a_ptr[i+2] * b_ptr[j];
        }
        b_ptr += n;
        for (int j = 0; j < col_blk_size; ++j) {
          v[j] += a_ptr[i+3] * b_ptr[j];
        }
        b_ptr += n;
      }
      std::memcpy(c_ptr, v, sizeof(v));
      a_ptr += k;
      c_ptr += n;
    }
  }
}

// calculate c by tile
// - visit a by row panels, b by column panels
// - reduce memory accesses and total instructions
template <int tile_height = 8, int tile_width = 8,
          bool transpose_a = true, bool transpose_b = true>
static void mm_tile(const float* __restrict a, const float* __restrict b,
                    float* __restrict c, int m, int n, int k) {
  // XXX: ignore edge case for now
  static_assert(tile_height % 4 == 0 && tile_width % 4 == 0);
  if (m % tile_height || n % tile_width || k % 4) std::abort();

  // transpose each row panel of a for sequential memory access
  float* a_tx{};
  if (transpose_a) {
    a_tx = new float[m * k];
    float* a_tx_ptr = a_tx;
    for (int mm = 0; mm < m; mm += tile_height) {
      const float* a_ptr = a + mm * k;
      for (int col = 0; col < k; col += 4) {
        for (int row = 0; row < tile_height; ++row) {
          std::memcpy(a_tx_ptr, a_ptr + row * k + col, 4 * sizeof(float));
          a_tx_ptr += 4;
        }
      }
    }
  }

  // transpose each column panel of b for sequential memory access
  float* b_tx{};
  if (transpose_b) {
    b_tx = new float[k * n];
    float* b_tx_ptr = b_tx;
    for (int nn = 0; nn < n; nn += tile_width) {
      const float* b_ptr = b + nn;
      for (int row = 0; row < k; ++row) { 
        std::memcpy(b_tx_ptr, b_ptr + row * n, tile_width * sizeof(float));
        b_tx_ptr += tile_width;
      }
    }
  }

  // a: tile_height * 4; b: 4 * tile_width; c: tile_height * tile_width
  float32x4_t tile_a[tile_height];
  float32x4_t tile_b[4][tile_width / 4];
  float32x4_t tile_c[tile_height][tile_width / 4];

  // make sure all floating point numbers can be held in neon registers
  const int total_size = sizeof(tile_a) + sizeof(tile_b) + sizeof(tile_c);
  static_assert(total_size <= 32 * 16);

  // nn: start column of matrix c's tile under calculation
  for (int nn = 0; nn < n; nn += tile_width) {
    // mm: start row of matrix c's tile under calculation
    for (int mm = 0; mm < m; mm += tile_height) {
      // calculate c tile starts from [mm, nn]
      //
      // panel a:              panel b:                tile c:
      // tile_height * k       k * tile_width          tile_height * tile_width
      //
      //                                                         nn
      // ......+---------+             nn            ......+-----v-------+
      // ^     |         |     --+-----v-------+     ^     |             |
      // |     |         |     | |     b b     |     |     |             |
      // |  mm > a  a  a |  *  k |     b b     |  =  |  mm >     c c     |
      // m     | a  a  a |     | |     b b     |     m     |     c c     |
      // |     | a  a  a |     --+-------------+     |     |     c c     |
      // |     | a  a  a |       |<---- n ---->|     |     |     c c     |
      // v     |         |                           v     |             |
      // ......+---------+                           ......+-------------+
      //       |<-- k -->|                                 |<---- n ---->|
      const float* a_ptr = (transpose_a ? a_tx : a) + mm * k;
      const float* b_ptr = transpose_b ? (b_tx + nn * k) : (b + nn);
      float *c_ptr = c + mm * n + nn;

      // calculate tile c
      std::memset(tile_c, 0, sizeof(tile_c));
      // kk: iterate panel_a by 4 cols and panel_b by 4 rows (one vector)
      for (int kk = 0; kk < k; kk += 4) {
        // load tile a: rows = tile_height, cols = 4
        if (transpose_a) {
          std::memcpy(tile_a, a_ptr, sizeof(tile_a));
          a_ptr += sizeof(tile_a) / 4/*sizeof(float)*/;
        } else {
          for (int h = 0; h < tile_height; ++h) {
            std::memcpy(&tile_a[h], a_ptr + h * k, 4 * sizeof(float));
          }
          a_ptr += 4;
        }

        // load tile b: rows = 4, cols = tile_width
        if (transpose_b) {
          std::memcpy(tile_b, b_ptr, sizeof(tile_b));
          b_ptr += sizeof(tile_b) / 4/*sizeof(float)*/;
        } else {
          for (int i = 0; i < 4; ++i) {
            std::memcpy(tile_b[i], b_ptr + i * n, tile_width * sizeof(float));
          }
          b_ptr += 4 * n;
        }

        // accumulate c tile, all data are in registers
        // tile_c += tile_a * tile_b
        for (int h = 0; h < tile_height; ++h) {
          for (int w = 0; w < tile_width; w += 4) {
            tile_c[h][w/4] += tile_a[h][0] * tile_b[0][w/4];
            tile_c[h][w/4] += tile_a[h][1] * tile_b[1][w/4];
            tile_c[h][w/4] += tile_a[h][2] * tile_b[2][w/4];
            tile_c[h][w/4] += tile_a[h][3] * tile_b[3][w/4];
          }
        }
      }

      // store to c tile
      for (int h = 0; h < tile_height; ++h) {
        std::memcpy(c_ptr + h * n, tile_c[h], tile_width * sizeof(float));
      }
    }
  }

  delete[] a_tx;
  delete[] b_tx;
}

extern "C" {
void mm_panel_24_asm(const float*, const float*, float*, int, int, int);
void mm_tile_8x8_asm(const float*, const float*, float*, int, int, int);
}

auto _mm_baseline = mm_baseline;
auto _mm_panel_24 = mm_panel<24>;
auto _mm_panel_24_asm = mm_panel_24_asm;
auto _mm_tile_8x8 = mm_tile<8, 8, false, false>;
auto _mm_tile_8x8_asm = mm_tile_8x8_asm;
auto _mm_tile_8x8_T = mm_tile<8, 8, true, true>;
