#define _POSIX_C_SOURCE 200809L
/*
 * matmul-arena: comparing matrix multiplication implementations
 *
 * Tests naive, tiled (cache-friendly), multithreaded, and SIMD
 * variants against each other. Measures GFLOPS and speedup.
 *
 * The point: there's a 10-100x performance gap between "correct"
 * and "fast" matrix multiplication on the same hardware. Cache
 * locality and parallelism matter more than algorithmic tricks
 * for dense linear algebra at practical sizes.
 *
 * Usage:
 *   ./matmul-arena [size]     # default: 1024x1024
 *   ./matmul-arena 512        # 512x512 matrices
 *
 * author: pulin chhangawala
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#ifdef __SSE__
#include <immintrin.h>
#define HAS_SIMD 1
#else
#define HAS_SIMD 0
#endif

/* ---- matrix helpers ---- */

typedef struct {
    double *data;
    int rows, cols;
} matrix_t;

matrix_t *matrix_create(int rows, int cols) {
    matrix_t *m = malloc(sizeof(matrix_t));
    /* aligned allocation for SIMD */
    posix_memalign((void**)&m->data, 64, rows * cols * sizeof(double));
    m->rows = rows;
    m->cols = cols;
    return m;
}

void matrix_destroy(matrix_t *m) {
    if (m) { free(m->data); free(m); }
}

void matrix_randomize(matrix_t *m, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < m->rows * m->cols; i++)
        m->data[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

void matrix_zero(matrix_t *m) {
    memset(m->data, 0, m->rows * m->cols * sizeof(double));
}

#define MAT(m, i, j) ((m)->data[(i) * (m)->cols + (j)])

/* verify two matrices are approximately equal */
int matrix_verify(matrix_t *a, matrix_t *b, double tol) {
    if (a->rows != b->rows || a->cols != b->cols) return 0;
    for (int i = 0; i < a->rows * a->cols; i++) {
        if (fabs(a->data[i] - b->data[i]) > tol)
            return 0;
    }
    return 1;
}

/* ---- timing ---- */

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double compute_gflops(int n, double seconds) {
    /* matrix multiply is 2*n^3 FLOPs (multiply + add per element) */
    double flops = 2.0 * (double)n * n * n;
    return flops / seconds / 1e9;
}

/* ============================================================
 * Implementation 1: Naive (ijk ordering)
 * This is what you'd write if you just translated the math formula.
 * Terrible cache behavior because B is accessed column-wise.
 * ============================================================ */

void matmul_naive(matrix_t *A, matrix_t *B, matrix_t *C) {
    int n = A->rows;
    matrix_zero(C);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                MAT(C, i, j) += MAT(A, i, k) * MAT(B, k, j);
}

/* ============================================================
 * Implementation 2: Transposed (ikj ordering)
 * Transpose B first so both matrices are accessed row-wise.
 * This alone typically gives 3-5x speedup on large matrices.
 * ============================================================ */

void matmul_transposed(matrix_t *A, matrix_t *B, matrix_t *C) {
    int n = A->rows;
    matrix_zero(C);

    /* transpose B */
    matrix_t *BT = matrix_create(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            MAT(BT, j, i) = MAT(B, i, j);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += MAT(A, i, k) * MAT(BT, j, k);
            MAT(C, i, j) = sum;
        }

    matrix_destroy(BT);
}

/* ============================================================
 * Implementation 3: Tiled / Blocked
 * Process the matrix in tiles that fit in L1 cache (~32KB).
 * Combines cache locality with good spatial access patterns.
 * ============================================================ */

#define TILE_SIZE 64  /* tuned for 32KB L1 with 8-byte doubles */

void matmul_tiled(matrix_t *A, matrix_t *B, matrix_t *C) {
    int n = A->rows;
    matrix_zero(C);

    for (int ii = 0; ii < n; ii += TILE_SIZE)
        for (int jj = 0; jj < n; jj += TILE_SIZE)
            for (int kk = 0; kk < n; kk += TILE_SIZE) {
                int i_end = (ii + TILE_SIZE < n) ? ii + TILE_SIZE : n;
                int j_end = (jj + TILE_SIZE < n) ? jj + TILE_SIZE : n;
                int k_end = (kk + TILE_SIZE < n) ? kk + TILE_SIZE : n;

                for (int i = ii; i < i_end; i++)
                    for (int k = kk; k < k_end; k++) {
                        double a_ik = MAT(A, i, k);
                        for (int j = jj; j < j_end; j++)
                            MAT(C, i, j) += a_ik * MAT(B, k, j);
                    }
            }
}

/* ============================================================
 * Implementation 4: Multithreaded (pthreads)
 * Each thread computes a horizontal strip of the result matrix.
 * Uses the tiled approach within each strip.
 * ============================================================ */

typedef struct {
    matrix_t *A, *B, *C;
    int start_row, end_row;
} thread_arg_t;

void *matmul_thread_worker(void *arg) {
    thread_arg_t *ta = (thread_arg_t *)arg;
    int n = ta->A->cols;

    for (int ii = ta->start_row; ii < ta->end_row; ii += TILE_SIZE)
        for (int jj = 0; jj < n; jj += TILE_SIZE)
            for (int kk = 0; kk < n; kk += TILE_SIZE) {
                int i_end = (ii + TILE_SIZE < ta->end_row) ? ii + TILE_SIZE : ta->end_row;
                int j_end = (jj + TILE_SIZE < n) ? jj + TILE_SIZE : n;
                int k_end = (kk + TILE_SIZE < n) ? kk + TILE_SIZE : n;

                for (int i = ii; i < i_end; i++)
                    for (int k = kk; k < k_end; k++) {
                        double a_ik = MAT(ta->A, i, k);
                        for (int j = jj; j < j_end; j++)
                            MAT(ta->C, i, j) += a_ik * MAT(ta->B, k, j);
                    }
            }

    return NULL;
}

void matmul_threaded(matrix_t *A, matrix_t *B, matrix_t *C, int num_threads) {
    int n = A->rows;
    matrix_zero(C);

    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_arg_t *args = malloc(num_threads * sizeof(thread_arg_t));

    int rows_per_thread = n / num_threads;

    for (int t = 0; t < num_threads; t++) {
        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        args[t].start_row = t * rows_per_thread;
        args[t].end_row = (t == num_threads - 1) ? n : (t + 1) * rows_per_thread;
        pthread_create(&threads[t], NULL, matmul_thread_worker, &args[t]);
    }

    for (int t = 0; t < num_threads; t++)
        pthread_join(threads[t], NULL);

    free(threads);
    free(args);
}

/* ============================================================
 * Implementation 5: SIMD (SSE2 intrinsics)
 * Uses 128-bit vector operations to process 2 doubles at once.
 * Combined with tiling for cache efficiency.
 * ============================================================ */

#if HAS_SIMD
void matmul_simd(matrix_t *A, matrix_t *B, matrix_t *C) {
    int n = A->rows;
    matrix_zero(C);

    for (int ii = 0; ii < n; ii += TILE_SIZE)
        for (int jj = 0; jj < n; jj += TILE_SIZE)
            for (int kk = 0; kk < n; kk += TILE_SIZE) {
                int i_end = (ii + TILE_SIZE < n) ? ii + TILE_SIZE : n;
                int j_end = (jj + TILE_SIZE < n) ? jj + TILE_SIZE : n;
                int k_end = (kk + TILE_SIZE < n) ? kk + TILE_SIZE : n;

                for (int i = ii; i < i_end; i++)
                    for (int k = kk; k < k_end; k++) {
                        __m128d a_vec = _mm_set1_pd(MAT(A, i, k));
                        int j;
                        for (j = jj; j + 2 <= j_end; j += 2) {
                            __m128d b_vec = _mm_loadu_pd(&MAT(B, k, j));
                            __m128d c_vec = _mm_loadu_pd(&MAT(C, i, j));
                            c_vec = _mm_add_pd(c_vec, _mm_mul_pd(a_vec, b_vec));
                            _mm_storeu_pd(&MAT(C, i, j), c_vec);
                        }
                        /* handle remainder */
                        for (; j < j_end; j++)
                            MAT(C, i, j) += MAT(A, i, k) * MAT(B, k, j);
                    }
            }
}
#endif

/* ============================================================
 * Implementation 6: Strassen Algorithm
 * O(n^2.807) divide-and-conquer. Splits each matrix into 4
 * quadrants, does 7 multiplications instead of 8. The constant
 * factor is high, so it only wins for large matrices (n > 256).
 *
 * Falls back to tiled multiplication below the crossover point.
 * ============================================================ */

#define STRASSEN_CROSSOVER 128

/* helper: add/subtract submatrices */
static void mat_add_sub(double *A, double *B, double *C, int n, int stride, int add) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i * n + j] = add ? (A[i * stride + j] + B[i * stride + j])
                               : (A[i * stride + j] - B[i * stride + j]);
}

static void strassen_recurse(double *A, double *B, double *C,
                              int n, int sa, int sb, int sc) {
    if (n <= STRASSEN_CROSSOVER) {
        /* base case: standard multiplication */
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++) {
                double a_ik = A[i * sa + k];
                for (int j = 0; j < n; j++)
                    C[i * sc + j] += a_ik * B[k * sb + j];
            }
        return;
    }

    int half = n / 2;
    int sz = half * half;

    /* allocate 7 product matrices + temporaries */
    double *M1 = calloc(sz, sizeof(double));
    double *M2 = calloc(sz, sizeof(double));
    double *M3 = calloc(sz, sizeof(double));
    double *M4 = calloc(sz, sizeof(double));
    double *M5 = calloc(sz, sizeof(double));
    double *M6 = calloc(sz, sizeof(double));
    double *M7 = calloc(sz, sizeof(double));
    double *T1 = malloc(sz * sizeof(double));
    double *T2 = malloc(sz * sizeof(double));

    /* submatrix pointers */
    double *A11 = A, *A12 = A + half;
    double *A21 = A + half * sa, *A22 = A + half * sa + half;
    double *B11 = B, *B12 = B + half;
    double *B21 = B + half * sb, *B22 = B + half * sb + half;

    /* M1 = (A11 + A22) * (B11 + B22) */
    mat_add_sub(A11, A22, T1, half, sa, 1);
    mat_add_sub(B11, B22, T2, half, sb, 1);
    strassen_recurse(T1, T2, M1, half, half, half, half);

    /* M2 = (A21 + A22) * B11 */
    mat_add_sub(A21, A22, T1, half, sa, 1);
    strassen_recurse(T1, B11, M2, half, half, sb, half);

    /* M3 = A11 * (B12 - B22) */
    mat_add_sub(B12, B22, T1, half, sb, 0);
    strassen_recurse(A11, T1, M3, half, sa, half, half);

    /* M4 = A22 * (B21 - B11) */
    mat_add_sub(B21, B11, T1, half, sb, 0);
    strassen_recurse(A22, T1, M4, half, sa, half, half);

    /* M5 = (A11 + A12) * B22 */
    mat_add_sub(A11, A12, T1, half, sa, 1);
    strassen_recurse(T1, B22, M5, half, half, sb, half);

    /* M6 = (A21 - A11) * (B11 + B12) */
    mat_add_sub(A21, A11, T1, half, sa, 0);
    mat_add_sub(B11, B12, T2, half, sb, 1);
    strassen_recurse(T1, T2, M6, half, half, half, half);

    /* M7 = (A12 - A22) * (B21 + B22) */
    mat_add_sub(A12, A22, T1, half, sa, 0);
    mat_add_sub(B21, B22, T2, half, sb, 1);
    strassen_recurse(T1, T2, M7, half, half, half, half);

    /* C11 = M1 + M4 - M5 + M7 */
    /* C12 = M3 + M5 */
    /* C21 = M2 + M4 */
    /* C22 = M1 - M2 + M3 + M6 */
    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++) {
            int idx = i * half + j;
            C[i * sc + j]            += M1[idx] + M4[idx] - M5[idx] + M7[idx];
            C[i * sc + j + half]     += M3[idx] + M5[idx];
            C[(i+half) * sc + j]     += M2[idx] + M4[idx];
            C[(i+half) * sc + j+half]+= M1[idx] - M2[idx] + M3[idx] + M6[idx];
        }

    free(M1); free(M2); free(M3); free(M4);
    free(M5); free(M6); free(M7);
    free(T1); free(T2);
}

void matmul_strassen(matrix_t *A, matrix_t *B, matrix_t *C) {
    int n = A->rows;
    matrix_zero(C);

    /* Strassen requires power-of-2 sizes. For simplicity, we
     * pad to the next power of 2 if needed. */
    int m = 1;
    while (m < n) m <<= 1;

    if (m == n) {
        strassen_recurse(A->data, B->data, C->data, n, n, n, n);
    } else {
        /* pad matrices to m×m */
        matrix_t *Ap = matrix_create(m, m);
        matrix_t *Bp = matrix_create(m, m);
        matrix_t *Cp = matrix_create(m, m);
        matrix_zero(Ap); matrix_zero(Bp); matrix_zero(Cp);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                MAT(Ap, i, j) = MAT(A, i, j);
                MAT(Bp, i, j) = MAT(B, i, j);
            }

        strassen_recurse(Ap->data, Bp->data, Cp->data, m, m, m, m);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                MAT(C, i, j) = MAT(Cp, i, j);

        matrix_destroy(Ap);
        matrix_destroy(Bp);
        matrix_destroy(Cp);
    }
}

/* ============================================================
 * Benchmark runner
 * ============================================================ */

typedef struct {
    const char *name;
    double seconds;
    double gflops;
    double speedup;
    int correct;
} result_t;

static void export_json(result_t *results, int n_results, int n, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return;

    fprintf(f, "{\n  \"matrix_size\": %d,\n  \"results\": [\n", n);
    for (int i = 0; i < n_results; i++) {
        fprintf(f, "    {\"name\": \"%s\", \"seconds\": %.6f, "
               "\"gflops\": %.4f, \"speedup\": %.2f, "
               "\"correct\": %s, \"matrix_size\": %d}",
               results[i].name, results[i].seconds,
               results[i].gflops, results[i].speedup,
               results[i].correct ? "true" : "false", n);
        fprintf(f, "%s\n", i < n_results - 1 ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    printf("  Results exported to %s\n", path);
}

int main(int argc, char **argv) {
    int n = 1024;
    const char *json_out = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0 && i+1 < argc)
            json_out = argv[++i];
        else
            n = atoi(argv[i]);
    }

    /* clamp to reasonable size */
    if (n < 64) n = 64;
    if (n > 4096) n = 4096;

    int num_threads = 4;  /* adjust based on your machine */

    printf("\n");
    printf("================================================================\n");
    printf("  matmul-arena: %dx%d matrix multiplication benchmark\n", n, n);
    printf("================================================================\n\n");

    matrix_t *A = matrix_create(n, n);
    matrix_t *B = matrix_create(n, n);
    matrix_t *C_ref = matrix_create(n, n);  /* reference result */
    matrix_t *C = matrix_create(n, n);

    matrix_randomize(A, 42);
    matrix_randomize(B, 123);

    /* run reference (tiled, since naive is too slow for large n) */
    printf("  Computing reference result (tiled)...\n");
    double t0 = get_time();
    matmul_tiled(A, B, C_ref);
    double ref_time = get_time() - t0;

    result_t results[8];
    int n_results = 0;

    /* --- naive --- */
    if (n <= 1024) {  /* skip for very large sizes, life is too short */
        printf("  Running naive...\n");
        t0 = get_time();
        matmul_naive(A, B, C);
        double dt = get_time() - t0;
        results[n_results++] = (result_t){
            "Naive (ijk)", dt, compute_gflops(n, dt), 1.0,
            matrix_verify(C, C_ref, 1e-6)
        };
    }

    /* --- transposed --- */
    printf("  Running transposed...\n");
    t0 = get_time();
    matmul_transposed(A, B, C);
    double dt = get_time() - t0;
    double naive_time = n_results > 0 ? results[0].seconds : dt;
    results[n_results++] = (result_t){
        "Transposed", dt, compute_gflops(n, dt),
        naive_time / dt, matrix_verify(C, C_ref, 1e-6)
    };

    /* --- tiled --- */
    printf("  Running tiled (block=%d)...\n", TILE_SIZE);
    t0 = get_time();
    matmul_tiled(A, B, C);
    dt = get_time() - t0;
    results[n_results++] = (result_t){
        "Tiled", dt, compute_gflops(n, dt),
        naive_time / dt, matrix_verify(C, C_ref, 1e-6)
    };

    /* --- threaded --- */
    printf("  Running threaded (%d threads)...\n", num_threads);
    t0 = get_time();
    matmul_threaded(A, B, C, num_threads);
    dt = get_time() - t0;
    results[n_results++] = (result_t){
        "Threaded", dt, compute_gflops(n, dt),
        naive_time / dt, matrix_verify(C, C_ref, 1e-6)
    };

    /* --- SIMD --- */
#if HAS_SIMD
    printf("  Running SIMD (SSE2 + tiled)...\n");
    t0 = get_time();
    matmul_simd(A, B, C);
    dt = get_time() - t0;
    results[n_results++] = (result_t){
        "SIMD+Tiled", dt, compute_gflops(n, dt),
        naive_time / dt, matrix_verify(C, C_ref, 1e-6)
    };
#endif

    /* --- Strassen --- */
    printf("  Running Strassen (crossover=%d)...\n", STRASSEN_CROSSOVER);
    t0 = get_time();
    matmul_strassen(A, B, C);
    dt = get_time() - t0;
    results[n_results++] = (result_t){
        "Strassen", dt, compute_gflops(n, dt),
        naive_time / dt, matrix_verify(C, C_ref, 1e-6)
    };

    /* --- print results --- */
    printf("\n");
    printf("  %-20s %10s %10s %10s %8s\n",
           "Implementation", "Time (s)", "GFLOPS", "Speedup", "Correct");
    printf("  %-20s %10s %10s %10s %8s\n",
           "--------------------", "----------", "----------", "----------", "--------");

    for (int i = 0; i < n_results; i++) {
        printf("  %-20s %10.4f %10.2f %9.1fx %8s\n",
               results[i].name,
               results[i].seconds,
               results[i].gflops,
               results[i].speedup,
               results[i].correct ? "✓" : "✗");
    }

    printf("\n================================================================\n\n");

    if (json_out) {
        export_json(results, n_results, n, json_out);
    }

    matrix_destroy(A);
    matrix_destroy(B);
    matrix_destroy(C_ref);
    matrix_destroy(C);

    return 0;
}
