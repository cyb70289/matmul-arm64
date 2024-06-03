#include <cstdlib>
#include <cstring>
#include <arm_neon.h>

// calculate c by tile
// - visit a by row panels, b by column panels
// - a, b must be pre-reordered
// - reduce memory accesses and total instructions
// - clang16 vectorizes the code quite good: https://godbolt.org/z/MWvefG6ds
template <int tile_height = 8, int tile_width = 8>
static void mm_tile(const float* __restrict a_tx, const float* __restrict b_tx,
                    float* __restrict c, int m, int n, int k) {
  // XXX: ignore edge case for now
  static_assert(tile_height % 4 == 0 && tile_width % 4 == 0);
  if (m % tile_height || n % tile_width || k % 4) std::abort();

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
      const float* a_ptr = a_tx + mm * k;
      const float* b_ptr = b_tx + nn * k;
      float *c_ptr = c + mm * n + nn;

      // calculate tile c
      std::memset(tile_c, 0, sizeof(tile_c));
      // kk: iterate panel_a by 4 cols and panel_b by 4 rows (one vector)
      for (int kk = 0; kk < k; kk += 4) {
        // load tile a: rows = tile_height, cols = 4
        std::memcpy(tile_a, a_ptr, sizeof(tile_a));
        a_ptr += sizeof(tile_a) / 4/*sizeof(float)*/;

        // load tile b: rows = 4, cols = tile_width
        std::memcpy(tile_b, b_ptr, sizeof(tile_b));
        b_ptr += sizeof(tile_b) / 4/*sizeof(float)*/;

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
}

template <int tile_height = 8, int tile_width = 8>
void reorder(const float* __restrict a, const float* __restrict b,
             float* __restrict a_tx, float* __restrict b_tx,
             int m, int n, int k) {
  // transpose each row panel of a for sequential memory access
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

  // transpose each column panel of b for sequential memory access
  float* b_tx_ptr = b_tx;
  for (int nn = 0; nn < n; nn += tile_width) {
    const float* b_ptr = b + nn;
    for (int row = 0; row < k; ++row) {
      std::memcpy(b_tx_ptr, b_ptr + row * n, tile_width * sizeof(float));
      b_tx_ptr += tile_width;
    }
  }
}

auto _mm_tile_8x8= mm_tile<8, 8>;
auto _reorder = reorder<8, 8>;
