// https://github.com/Mozilla-Ocho/llamafile
// http://justine.lol/matmul/

#include <arm_neon.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define VECTOR_REGISTERS 32

#define KN 4

#define V float32x4_t
#define D float32x4_t
#define TA float
#define TB float
#define TC float

static inline V load(const float *p) {
  return vld1q_f32(p);
}

static inline float hsum(V x) {
  return vaddvq_f32(x);
}

static inline V madd(V a, V b, V c) {
  return a * b + c;
}

#define dontinline __attribute__((noinline))

class SGEMMER {
  public:
    SGEMMER(int k, const TA *A, int lda, const TB *B, int ldb, TC *C, int ldc, int ith, int nth)
      : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int m, int n) {
      mnpack(0, m, 0, n);
    }

  private:
    dontinline void mnpack(int m0, int m, int n0, int n) {
      int mc, nc, mp, np;
      if (m - m0 <= 0 || n - n0 <= 0)
        return;
      if (VECTOR_REGISTERS >= 32 && m - m0 >= 8 && n - n0 >= 2) {
        mc = 8;
        nc = 2;
        gemm<8, 2>(m0, m, n0, n);
      } else if (m - m0 >= 4 && n - n0 >= 2) {
        mc = 4;
        nc = 2;
        gemm<4, 2>(m0, m, n0, n);
      } else if (n - n0 >= 4) {
        mc = 1;
        nc = 4;
        gemm<1, 4>(m0, m, n0, n);
      } else if (m - m0 >= 4) {
        mc = 4;
        nc = 1;
        gemm<4, 1>(m0, m, n0, n);
      } else {
        mc = 1;
        nc = 1;
        gemm<1, 1>(m0, m, n0, n);
      }
      mp = m0 + (m - m0) / mc * mc;
      np = n0 + (n - n0) / nc * nc;
      mnpack(mp, m, n0, np);
      mnpack(m0, mp, np, n);
      mnpack(mp, m, np, n);
    }

    template <int RM, int RN> dontinline void gemm(int m0, int m, int n0, int n) {
      int ytiles = (m - m0) / RM;
      int xtiles = (n - n0) / RN;
      int tiles = xtiles * ytiles;
      int duty = (tiles + nth - 1) / nth;
      int start = duty * ith;
      int end = start + duty;
      if (end > tiles)
        end = tiles;
      for (int job = start; job < end; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;
        D Cv[RN][RM] = {0};
        for (int l = 0; l < k; l += KN)
          for (int j = 0; j < RN; ++j)
            for (int i = 0; i < RM; ++i)
              Cv[j][i] = madd(load(A + lda * (ii + i) + l), //
                  load(B + ldb * (jj + j) + l), //
                  Cv[j][i]);
        TC Cd[RN][RM];
        for (int j = 0; j < RN; ++j)
          for (int i = 0; i < RM; ++i)
            Cd[j][i] = hsum(Cv[j][i]);
        for (int j = 0; j < RN; ++j)
          for (int i = 0; i < RM; ++i)
            C[ldc * (jj + j) + (ii + i)] = Cd[j][i];
      }
    }

    const TA *const __restrict A;
    const TB *const __restrict B;
    TC *const __restrict C;
    const int k;
    const int lda;
    const int ldb;
    const int ldc;
    const int ith;
    const int nth;
};

/**
 * Performs optimized matrix multiplication on CPU.
 *
 * This subroutine may compute C = Aᵀ * B with column major ordering.
 * Despite its name, this isn't a generalized implementation. Work is
 * only performed when a handwritten kernel is written and available.
 * Otherwise the caller should fall back to a general matmul routine.
 *
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param A is first input matrix (always transposed)
 * @param lda is row stride of `A`
 * @param B is second input matrix (never transposed)
 * @param ldb is row stride of `B`
 * @param C is input/output array of output matrices
 * @param ldc is row stride of `C`
 * @param ith is thread id (must be less than `nth`)
 * @param nth is number of threads (must be greater than zero)
 * @param task is GGML task type
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @return true if this function was able to service the matmul request
 */
static void llamafile_sgemm(int m, int n, int k, const float *a, const float *b, float *c) {
  if (k % KN != 0) std::abort();

  SGEMMER tb{k, a, k, b, k, c, m, 0, 1};
  tb.matmul(m, n);
}

int main() {
  const long batch = 512;
  const int m = 1000, n = 240, k = 200;

  float *a = (float*)malloc(batch*m*k*sizeof(float));
  float *b = (float*)malloc(batch*k*n*sizeof(float));
  float *c = (float*)malloc(batch*m*n*sizeof(float));

  for(long i = 0; i < batch*m*k; i++) a[i] = (float)i;
  for(long i = 0; i < batch*k*n; i++) b[i] = (float)i;

  // transpose b
  float *bt = (float*)malloc(batch*k*n*sizeof(float));
  for (long i = 0; i < batch; ++i) {
    const float *bb = b + i*k*n;
    float *bbt = bt + i*k*n;
    for (long row = 0; row < k; ++row) {
      for (long col = 0; col < n; ++col) {
        bbt[col*k + row] = *bb++;
      }
    }
  }
  free(b);
  b = bt;

  // warmup
  for (int i = 0; i < batch; ++i) {
    const float *aa = a + i*m*k;
    const float *bb = b + i*k*n;
    float *cc = c + i*m*n;
    llamafile_sgemm(m, n, k, aa, bb, cc);
  }

  // benchmark
  const clock_t start = clock();
  for (long i = 0; i < batch; ++i) {
    const float *aa = a + i*m*k;
    const float *bb = b + i*k*n;
    float *cc = c + i*m*n;
    llamafile_sgemm(m, n, k, aa, bb, cc);
  }
  const clock_t end = clock();
  const double time_spent = (double)(end-start) / CLOCKS_PER_SEC;
  printf("time: %.2f ms\n", time_spent*1000);

  // transpose c
  float *ct = (float*)malloc(batch*m*n*sizeof(float));
  for (long i = 0; i < batch; ++i) {
    const float *cc = c + i*m*n;
    float *cct = ct + i*m*n;
    for (long row = 0; row < m; ++row) {
      for (long col = 0; col < n; ++col) {
        cct[col*m + row] = *cc++;
      }
    }
  }
  free(c);
  c = ct;

  // print some results for quick debugging
  printf("c[0]    = %e\n", c[0]);
  printf("c[9973] = %e\n", c[9973]);
  printf("c[-1]   = %e\n", c[batch*m*n-1]);

  free(a);
  free(b);
  free(c);
  return 0;
}
