/*
 * template <int tile_height = 8, int tile_width = 8>
 * static void mm_tile(const float* __restrict a, const float* __restrict b,
 *                     float* __restrict c, int m, int n, int k) {
 *   // XXX: ignore edge case for now
 *   static_assert(tile_height % 4 == 0 && tile_width % 4 == 0);
 *   if (m % tile_height || n % tile_width || k % 4) std::abort();
 * 
 *   float32x4_t tile_a[tile_height];
 *   float32x4_t tile_b[4][tile_width / 4];
 *   float32x4_t tile_c[tile_height][tile_width / 4];
 * 
 *   for (int nn = 0; nn < n; nn += tile_width) {
 *     for (int mm = 0; mm < m; mm += tile_height) {
 *       const float* a_ptr = a + mm * k;
 *       const float* b_ptr = b + nn;
 *       float *c_ptr = c + mm * n + nn;
 * 
 *       std::memset(tile_c, 0, sizeof(tile_c));
 *       for (int kk = 0; kk < k; kk += 4) {
 *         for (int h = 0; h < tile_height; ++h) {
 *           std::memcpy(&tile_a[h], a_ptr + h * k, 4 * sizeof(float));
 *         }
 *         a_ptr += 4;
 * 
 *         for (int i = 0; i < 4; ++i) {
 *           std::memcpy(tile_b[i], b_ptr + i * n, tile_width * sizeof(float));
 *         }
 *         b_ptr += 4 * n;
 * 
 *         for (int h = 0; h < tile_height; ++h) {
 *           for (int w = 0; w < tile_width; w += 4) {
 *             tile_c[h][w/4] += tile_a[h][0] * tile_b[0][w/4];
 *             tile_c[h][w/4] += tile_a[h][1] * tile_b[1][w/4];
 *             tile_c[h][w/4] += tile_a[h][2] * tile_b[2][w/4];
 *             tile_c[h][w/4] += tile_a[h][3] * tile_b[3][w/4];
 *           }
 *         }
 *       }
 * 
 *       for (int h = 0; h < tile_height; ++h) {
 *         std::memcpy(c_ptr + h * n, tile_c[h], tile_width * sizeof(float));
 *       }
 *     }
 *   }
 * }
 */

        .text
        .arch armv8.2-a

        .global mm_tile_8x8_asm

mm_tile_8x8_asm:

        // general registers
        a     .req x0
        b     .req x1
        c     .req x2
        m     .req x3
        n     .req x4
        k     .req x5
        nn    .req x6
        mm    .req x7
        kk    .req x8
        a_ptr .req x9
        b_ptr .req x10
        c_ptr .req x11
        nx32  .req x13
        kx32  .req x14
        stepa .req x15
        stepc .req x16
        tmp   .req x17
        kx4   .req x18
        kx8   .req x19
        kx12  .req x20
        kx16  .req x21
        kx20  .req x22
        kx24  .req x23
        kx28  .req x24

        // vector registers
        // - tile_a[8]:     v0 ~ v7
        // - tile_b[4][2]:  (v8,  v9 ), (v10, v11), (v12, v13), (v14, v15)
        // - tile_c[8][2] : (v16, v17), (v18, v19), (v20, v21), (v22, v23)
        //                  (v24, v25), (v26, v27), (v28, v29), (v30, v31)

        sub   sp, sp, #144
        stp   d8,  d9,  [sp, #0]
        stp   d10, d11, [sp, #16]
        stp   d12, d13, [sp, #32]
        stp   d14, d15, [sp, #48]
        stp   x19, x20, [sp, #64]
        stp   x21, x22, [sp, #80]
        stp   x23, x24, [sp, #96]
        stp   x25, x26, [sp, #112]
        stp   x27, x28, [sp, #128]

        # a_ptr, c_ptr steps
        lsl   kx32, k, #5
        lsl   nx32, n, #5

        # tile a row offsets
        lsl   kx4, k, #2
        add   kx8,  kx4,  kx4
        add   kx12, kx8,  kx4
        add   kx16, kx12, kx4 
        add   kx20, kx16, kx4
        add   kx24, kx20, kx4
        add   kx28, kx24, kx4

        mov   nn, xzr
.Ln:
        mov   stepa, xzr
        lsl   stepc, nn, #2

        mov   mm, xzr
.Lm:
        add   a_ptr, a, stepa
        add   b_ptr, b, nn, lsl #2
        add   c_ptr, c, stepc
        add   stepa, stepa, kx32
        add   stepc, stepc, nx32

        // clear c tile registers
        movi  v16.4s, #0
        movi  v17.4s, #0
        movi  v18.4s, #0
        movi  v19.4s, #0
        movi  v20.4s, #0
        movi  v21.4s, #0
        movi  v22.4s, #0
        movi  v23.4s, #0
        movi  v24.4s, #0
        movi  v25.4s, #0
        movi  v26.4s, #0
        movi  v27.4s, #0
        movi  v28.4s, #0
        movi  v29.4s, #0
        movi  v30.4s, #0
        movi  v31.4s, #0

        mov   kk, xzr
.Lk:
        // load tile a
        ldr   q0, [a_ptr]
        ldr   q1, [a_ptr, kx4]
        ldr   q2, [a_ptr, kx8]
        ldr   q3, [a_ptr, kx12]
        ldr   q4, [a_ptr, kx16]
        ldr   q5, [a_ptr, kx20]
        ldr   q6, [a_ptr, kx24]
        ldr   q7, [a_ptr, kx28]
        // increment a_ptr
        add   a_ptr, a_ptr, 4 << 2

        // load tile b
        mov   tmp, b_ptr
        ldp   q8,  q9,  [tmp]
        add   tmp, tmp, n, lsl #2
        ldp   q10, q11, [tmp]
        add   tmp, tmp, n, lsl #2
        ldp   q12, q13, [tmp]
        add   tmp, tmp, n, lsl #2
        ldp   q14, q15, [tmp]
        // increment b_ptr
        add   b_ptr, b_ptr, n, lsl #4

        // calculate tile c
        // row 0
        fmla  v16.4s, v8.4s,  v0.s[0]
        fmla  v17.4s, v9.4s,  v0.s[0]
        fmla  v16.4s, v10.4s, v0.s[1]
        fmla  v17.4s, v11.4s, v0.s[1]
        fmla  v16.4s, v12.4s, v0.s[2]
        fmla  v17.4s, v13.4s, v0.s[2]
        fmla  v16.4s, v14.4s, v0.s[3]
        fmla  v17.4s, v15.4s, v0.s[3]
        // row 1
        fmla  v18.4s, v8.4s,  v1.s[0]
        fmla  v19.4s, v9.4s,  v1.s[0]
        fmla  v18.4s, v10.4s, v1.s[1]
        fmla  v19.4s, v11.4s, v1.s[1]
        fmla  v18.4s, v12.4s, v1.s[2]
        fmla  v19.4s, v13.4s, v1.s[2]
        fmla  v18.4s, v14.4s, v1.s[3]
        fmla  v19.4s, v15.4s, v1.s[3]
        // row 2
        fmla  v20.4s, v8.4s,  v2.s[0]
        fmla  v21.4s, v9.4s,  v2.s[0]
        fmla  v20.4s, v10.4s, v2.s[1]
        fmla  v21.4s, v11.4s, v2.s[1]
        fmla  v20.4s, v12.4s, v2.s[2]
        fmla  v21.4s, v13.4s, v2.s[2]
        fmla  v20.4s, v14.4s, v2.s[3]
        fmla  v21.4s, v15.4s, v2.s[3]
        // row 3
        fmla  v22.4s, v8.4s,  v3.s[0]
        fmla  v23.4s, v9.4s,  v3.s[0]
        fmla  v22.4s, v10.4s, v3.s[1]
        fmla  v23.4s, v11.4s, v3.s[1]
        fmla  v22.4s, v12.4s, v3.s[2]
        fmla  v23.4s, v13.4s, v3.s[2]
        fmla  v22.4s, v14.4s, v3.s[3]
        fmla  v23.4s, v15.4s, v3.s[3]
        // row 4
        fmla  v24.4s, v8.4s,  v4.s[0]
        fmla  v25.4s, v9.4s,  v4.s[0]
        fmla  v24.4s, v10.4s, v4.s[1]
        fmla  v25.4s, v11.4s, v4.s[1]
        fmla  v24.4s, v12.4s, v4.s[2]
        fmla  v25.4s, v13.4s, v4.s[2]
        fmla  v24.4s, v14.4s, v4.s[3]
        fmla  v25.4s, v15.4s, v4.s[3]
        // row 5
        fmla  v26.4s, v8.4s,  v5.s[0]
        fmla  v27.4s, v9.4s,  v5.s[0]
        fmla  v26.4s, v10.4s, v5.s[1]
        fmla  v27.4s, v11.4s, v5.s[1]
        fmla  v26.4s, v12.4s, v5.s[2]
        fmla  v27.4s, v13.4s, v5.s[2]
        fmla  v26.4s, v14.4s, v5.s[3]
        fmla  v27.4s, v15.4s, v5.s[3]
        // row 6
        fmla  v28.4s, v8.4s,  v6.s[0]
        fmla  v29.4s, v9.4s,  v6.s[0]
        fmla  v28.4s, v10.4s, v6.s[1]
        fmla  v29.4s, v11.4s, v6.s[1]
        fmla  v28.4s, v12.4s, v6.s[2]
        fmla  v29.4s, v13.4s, v6.s[2]
        fmla  v28.4s, v14.4s, v6.s[3]
        fmla  v29.4s, v15.4s, v6.s[3]
        // row 7
        fmla  v30.4s, v8.4s,  v7.s[0]
        fmla  v31.4s, v9.4s,  v7.s[0]
        fmla  v30.4s, v10.4s, v7.s[1]
        fmla  v31.4s, v11.4s, v7.s[1]
        fmla  v30.4s, v12.4s, v7.s[2]
        fmla  v31.4s, v13.4s, v7.s[2]
        fmla  v30.4s, v14.4s, v7.s[3]
        fmla  v31.4s, v15.4s, v7.s[3]

        add   kk, kk, #4
        cmp   kk, k
        b.lt  .Lk
.Lk_end:

        // populate tile c
        mov   tmp, c_ptr
        stp   q16, q17, [tmp]
        add   tmp, tmp, n, lsl #2
        stp   q18, q19, [tmp]
        add   tmp, tmp, n, lsl #2
        stp   q20, q21, [tmp]
        add   tmp, tmp, n, lsl #2
        stp   q22, q23, [tmp]
        add   tmp, tmp, n, lsl #2
        stp   q24, q25, [tmp]
        add   tmp, tmp, n, lsl #2
        stp   q26, q27, [tmp]
        add   tmp, tmp, n, lsl #2
        stp   q28, q29, [tmp]
        add   tmp, tmp, n, lsl #2
        stp   q30, q31, [tmp]

        add   mm, mm, 8
        cmp   mm, m
        b.lt  .Lm
.Lm_end:

        add   nn, nn, 8
        cmp   nn, n
        b.lt  .Ln
.Ln_end:

        ldp   d8,  d9,  [sp], #16
        ldp   d10, d11, [sp], #16
        ldp   d12, d13, [sp], #16
        ldp   d14, d15, [sp], #16
        ldp   x19, x20, [sp], #16
        ldp   x21, x22, [sp], #16
        ldp   x23, x24, [sp], #16
        ldp   x25, x26, [sp], #16
        ldp   x27, x28, [sp], #16
        ret
