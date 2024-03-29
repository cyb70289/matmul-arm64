#include <arm_neon.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static void mm_base(const float* __restrict a, const float* __restrict b,
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

const int tile_height = 8, tile_width = 8;
static void mm_tile(const float* __restrict a, const float* __restrict b,
                    float* __restrict c, int m, int n, int k) {
#ifdef EBM
    const float *a_tx = a;
    const float *b_tx = b;
#else
    float *a_tx = malloc(m*k*sizeof(float));
    float *a_tx_ptr = a_tx;
    for (int mm = 0; mm < m; mm += tile_height) {
        const float *a_ptr = a + mm * k;
        for (int col = 0; col < k; col += 4) {
            for (int row = 0; row < tile_height; ++row) {
                memcpy(a_tx_ptr, a_ptr + row * k + col, 4 * sizeof(float));
                a_tx_ptr += 4;
            }
        }
    }

    float *b_tx = malloc(k*n*sizeof(float));
    float *b_tx_ptr = b_tx;
    for (int nn = 0; nn < n; nn += tile_width) {
        const float *b_ptr = b + nn;
        for (int row = 0; row < k; ++row) { 
            memcpy(b_tx_ptr, b_ptr + row * n, tile_width * sizeof(float));
            b_tx_ptr += tile_width;
        }
    }
#endif

    float32x4_t tile_a[tile_height];
    float32x4_t tile_b[4][tile_width / 4];
    float32x4_t tile_c[tile_height][tile_width / 4];

    for (int nn = 0; nn < n; nn += tile_width) {
        for (int mm = 0; mm < m; mm += tile_height) {
            const float* a_ptr = a_tx + mm * k;
            const float* b_ptr = b_tx + nn * k;
            float *c_ptr = c + mm * n + nn;

            memset(tile_c, 0, sizeof(tile_c));
            for (int kk = 0; kk < k; kk += 4) {
                memcpy(tile_a, a_ptr, sizeof(tile_a));
                a_ptr += sizeof(tile_a) / 4;
                memcpy(tile_b, b_ptr, sizeof(tile_b));
                b_ptr += sizeof(tile_b) / 4;
                for (int h = 0; h < tile_height; ++h) {
                    for (int w = 0; w < tile_width; w += 4) {
                        tile_c[h][w/4] += tile_a[h][0] * tile_b[0][w/4];
                        tile_c[h][w/4] += tile_a[h][1] * tile_b[1][w/4];
                        tile_c[h][w/4] += tile_a[h][2] * tile_b[2][w/4];
                        tile_c[h][w/4] += tile_a[h][3] * tile_b[3][w/4];
                    }
                }
            }

            for (int h = 0; h < tile_height; ++h) {
                memcpy(c_ptr + h * n, tile_c[h], tile_width * sizeof(float));
            }
        }
    }

#ifndef EBM
    free(a_tx);
    free(b_tx);
#endif
}

// tile_height|m, tile_width|n, 4|k 
//#define m 320
//#define n 160
//#define k 80
enum {m = 320, n = 160, k = 80};
static float a[m*k], b[k*n], c[m*n];

int main() {
#ifndef EBM
    for (int i = 0; i < m*k; ++i) a[i] = (float)i;
    for (int i = 0; i < k*n; ++i) b[i] = (float)i;
#endif

    mm_tile(a, b, c, m, n, k);

#ifndef EBM
    float *t = malloc(m*n*sizeof(float));
    mm_base(a, b, t, m, n, k);
    for (long i = 0; i < (long)m*n; ++i) {
        if (fabs(c[i] - t[i]) > FLT_MIN) {
            printf("Failed\n");
            return 1;
        }
    }
    printf("OK\n");
#endif

    return c[m*n/2] > c[m*n/4];
}
