// Benchmark BLIS(https://github.com/flame/blis)
// - multithreading diabled

// tested on bluewhale, clang-16
// - onednn + acl:   1194 ms
// - blis:           1213 ms

#include <stdio.h>
#include <time.h>
#include <blis/blis.h>

int main() {
    const long batch = 512;
    const dim_t m = 1000, n = 240, k = 200;

    float *a = malloc(batch*m*k*sizeof(float));
    float *b = malloc(batch*k*n*sizeof(float));
    float *c = malloc(batch*m*n*sizeof(float));

    for(int i = 0; i < batch*m*k; i++) a[i] = (float)i;
    for(int i = 0; i < batch*k*n; i++) b[i] = (float)i;

    float alpha = 1.0f;
    float beta = 0.0f;

    // warmup
    for (long i = 0; i < batch; ++i) {
        const float *aa = a + i*m*k;
        const float *bb = b + i*k*n;
        float *cc = c + i*m*n;
        bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k,
                  &alpha, aa, k, 1, bb, n, 1, &beta, cc, n, 1);
    }

    // benchmark
    const clock_t start = clock();
    for (long i = 0; i < batch; ++i) {
        const float *aa = a + i*m*k;
        const float *bb = b + i*k*n;
        float *cc = c + i*m*n;
        bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k,
                  &alpha, aa, k, 1, bb, n, 1, &beta, cc, n, 1);
    }
    const clock_t end = clock();
    const double time_spent = (double)(end-start) / CLOCKS_PER_SEC;
    printf("time: %.2f ms\n", time_spent*1000);

    // print some results for quick debugging
    printf("c[0]    = %e\n", c[0]);
    printf("c[9973] = %e\n", c[9973]);
    printf("c[-1]   = %e\n", c[batch*m*n-1]);

    free(a);
    free(b);
    free(c);
    return 0;
}
