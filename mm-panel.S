/* 
 * template <int col_blk_size = 24>
 * void mm_panel(const float* __restrict a, const float* __restrict b,
 *               float* __restrict c, int m, int n, int k) {
 *   // XXX: ignore edge case for now
 *   if (n % col_blk_size != 0 && k % 4 == 0) std::abort();
 * 
 *   for (int col = 0; col < n; col += col_blk_size) {
 *     const float* a_ptr = a;
 *     float* c_ptr = c + col;
 *     for (int row = 0; row < m; ++row) {
 *       const float* b_ptr = b + col;
 *       float v[col_blk_size]{};
 *       for (int i = 0; i < k; i += 4) {
 *         for (int j = 0; j < col_blk_size; ++j) {
 *           v[j] += a_ptr[i] * b_ptr[j];
 *         }
 *         b_ptr += n;
 *         for (int j = 0; j < col_blk_size; ++j) {
 *           v[j] += a_ptr[i+1] * b_ptr[j];
 *         }
 *         b_ptr += n;
 *         for (int j = 0; j < col_blk_size; ++j) {
 *           v[j] += a_ptr[i+2] * b_ptr[j];
 *         }
 *         b_ptr += n;
 *         for (int j = 0; j < col_blk_size; ++j) {
 *           v[j] += a_ptr[i+3] * b_ptr[j];
 *         }
 *         b_ptr += n;
 *       }
 *       std::memcpy(c_ptr, v, sizeof(v));
 *       a_ptr += k;
 *       c_ptr += n;
 *     }
 *   }
 * }
 */

        .text
        .arch armv8.2-a

        .global mm_panel_24_asm

mm_panel_24_asm:

        a     .req x0
        b     .req x1
        c     .req x2
        m     .req x3
        n     .req x4
        k     .req x5
        col   .req x6
        row   .req x7
        a_ptr .req x8
        b_ptr .req x9
        c_ptr .req x10
        i     .req x11

        sub   sp, sp, #64
        stp   d8,  d9,  [sp, #0]
        stp   d10, d11, [sp, #16]
        stp   d12, d13, [sp, #32]
        stp   d14, d15, [sp, #48]

        mov   col, xzr
.Lcol:  // assume n > 0
        mov   a_ptr, a
        add   c_ptr, c, col, lsl #2   // no penalty for lsl <= 4

        mov   row, xzr
.Lrow:  // assume m > 0
        add   b_ptr, b, col, lsl #2
        movi  v0.4s, #0
        movi  v1.4s, #0
        movi  v2.4s, #0
        movi  v3.4s, #0
        movi  v4.4s, #0
        movi  v5.4s, #0

        mov   i, xzr
.Li:    // assume k > 0
        ldr   q6, [a_ptr], #16

        ldp   q16, q17, [b_ptr, #0]
        ldp   q18, q19, [b_ptr, #32]
        ldp   q20, q21, [b_ptr, #64]
        add   b_ptr, b_ptr, n, lsl #2
        fmla  v0.4s, v16.4s, v6.s[0]
        fmla  v1.4s, v17.4s, v6.s[0]
        fmla  v2.4s, v18.4s, v6.s[0]
        fmla  v3.4s, v19.4s, v6.s[0]
        fmla  v4.4s, v20.4s, v6.s[0]
        fmla  v5.4s, v21.4s, v6.s[0]

        ldp   q16, q17, [b_ptr, #0]
        ldp   q18, q19, [b_ptr, #32]
        ldp   q20, q21, [b_ptr, #64]
        add   b_ptr, b_ptr, n, lsl #2
        fmla  v0.4s, v16.4s, v6.s[1]
        fmla  v1.4s, v17.4s, v6.s[1]
        fmla  v2.4s, v18.4s, v6.s[1]
        fmla  v3.4s, v19.4s, v6.s[1]
        fmla  v4.4s, v20.4s, v6.s[1]
        fmla  v5.4s, v21.4s, v6.s[1]

        ldp   q16, q17, [b_ptr, #0]
        ldp   q18, q19, [b_ptr, #32]
        ldp   q20, q21, [b_ptr, #64]
        add   b_ptr, b_ptr, n, lsl #2
        fmla  v0.4s, v16.4s, v6.s[2]
        fmla  v1.4s, v17.4s, v6.s[2]
        fmla  v2.4s, v18.4s, v6.s[2]
        fmla  v3.4s, v19.4s, v6.s[2]
        fmla  v4.4s, v20.4s, v6.s[2]
        fmla  v5.4s, v21.4s, v6.s[2]

        ldp   q16, q17, [b_ptr, #0]
        ldp   q18, q19, [b_ptr, #32]
        ldp   q20, q21, [b_ptr, #64]
        add   b_ptr, b_ptr, n, lsl #2
        fmla  v0.4s, v16.4s, v6.s[3]
        fmla  v1.4s, v17.4s, v6.s[3]
        fmla  v2.4s, v18.4s, v6.s[3]
        fmla  v3.4s, v19.4s, v6.s[3]
        fmla  v4.4s, v20.4s, v6.s[3]
        fmla  v5.4s, v21.4s, v6.s[3]

        add   i, i, #4
        cmp   i, k
        b.lt  .Li
.Li_end:

        stp   q0, q1, [c_ptr, #0]
        stp   q2, q3, [c_ptr, #32]
        stp   q4, q5, [c_ptr, #64]
        add   c_ptr, c_ptr, n, lsl#2

        add   row, row, #1
        cmp   row, m
        b.lt  .Lrow
.Lrow_end:

        add   col, col, #24   // block size = 24
        cmp   col, n
        b.lt  .Lcol
.Lcol_end:

        ldp   d8,  d9,  [sp], #16
        ldp   d10, d11, [sp], #16
        ldp   d12, d13, [sp], #16
        ldp   d14, d15, [sp], #16
        ret
