/**
 * cuda_solver.cu - CUDA GPU implementations of Jacobi and Red-Black Gauss-Seidel
 *
 * Each grid point is mapped to one GPU thread.
 * 2D thread blocks are used for efficient memory access patterns.
 *
 * Usage: ./cuda_solver [grid_size] [max_iterations] [tolerance]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEFAULT_GRID_SIZE  100
#define MAX_ITERATIONS     10000
#define TOLERANCE          1e-6
#define BLOCK_SIZE         16

/* Timer */
static inline double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* CUDA error checking macro */
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/**
 * CUDA Kernel: Jacobi iteration
 *
 * Each thread computes one grid point (i, j).
 * Reads from u_old (previous iteration), writes to u_new.
 * Also computes local difference for convergence checking.
 */
__global__ void jacobi_kernel(const double *u_old, double *u_new,
                               const double *f, double *diff, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        double left  = (j > 0)     ? u_old[i * n + (j - 1)] : 0.0;
        double right = (j < n - 1) ? u_old[i * n + (j + 1)] : 0.0;
        double up    = (i > 0)     ? u_old[(i - 1) * n + j]  : 0.0;
        double down  = (i < n - 1) ? u_old[(i + 1) * n + j]  : 0.0;

        double new_val = (left + right + up + down + f[i * n + j]) / 4.0;
        u_new[i * n + j] = new_val;
        diff[i * n + j] = fabs(new_val - u_old[i * n + j]);
    }
}

/**
 * CUDA Kernel: Red-Black Gauss-Seidel - update one color
 *
 * color = 0: update RED   points where (i+j) % 2 == 0
 * color = 1: update BLACK points where (i+j) % 2 == 1
 *
 * In-place update: reads neighbors (which are the other color, already stable
 * for this phase) and writes to the current point.
 */
__global__ void redblack_kernel(double *u, const double *f, double *diff,
                                 int n, int color) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n && (i + j) % 2 == color) {
        double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;
        double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;
        double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;
        double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;

        double old_val = u[i * n + j];
        double new_val = (left + right + up + down + f[i * n + j]) / 4.0;
        u[i * n + j] = new_val;
        diff[i * n + j] = fabs(new_val - old_val);
    } else if (i < n && j < n) {
        diff[i * n + j] = 0.0;
    }
}

/**
 * CUDA Kernel: Parallel max reduction
 *
 * Reduces an array of n*n doubles to find the maximum value.
 * Uses shared memory within each block.
 */
__global__ void max_reduce_kernel(const double *input, double *output, int total) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double val = 0.0;
    if (idx < total) val = input[idx];
    if (idx + blockDim.x < total) val = fmax(val, input[idx + blockDim.x]);
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

/**
 * Host function: find max of device array using reduction
 */
double gpu_max_reduce(double *d_array, int total) {
    int threads = 256;
    int blocks = (total + threads * 2 - 1) / (threads * 2);

    double *d_partial, *d_result;
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(double)));

    max_reduce_kernel<<<blocks, threads, threads * sizeof(double)>>>(d_array, d_partial, total);

    /* Iteratively reduce until we have one value */
    int current_size = blocks;
    double *d_in = d_partial;

    while (current_size > 1) {
        int new_blocks = (current_size + threads * 2 - 1) / (threads * 2);
        CUDA_CHECK(cudaMalloc(&d_result, new_blocks * sizeof(double)));
        max_reduce_kernel<<<new_blocks, threads, threads * sizeof(double)>>>(d_in, d_result, current_size);
        if (d_in != d_partial) cudaFree(d_in);
        d_in = d_result;
        current_size = new_blocks;
    }

    double max_val;
    CUDA_CHECK(cudaMemcpy(&max_val, d_in, sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_partial);
    if (d_in != d_partial) cudaFree(d_in);

    return max_val;
}

/* Host-side helper functions */
void init_rhs(double *f, int n) {
    double h = 1.0 / (n + 1);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double x = (i + 1) * h;
            double y = (j + 1) * h;
            f[i * n + j] = h * h * 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
}

void exact_solution(double *u_exact, int n) {
    double h = 1.0 / (n + 1);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double x = (i + 1) * h;
            double y = (j + 1) * h;
            u_exact[i * n + j] = sin(M_PI * x) * sin(M_PI * y);
        }
}

double compute_rmse(const double *u, const double *u_ref, int n) {
    double sum = 0.0;
    long total = (long)n * n;
    for (long k = 0; k < total; k++) {
        double d = u[k] - u_ref[k];
        sum += d * d;
    }
    return sqrt(sum / total);
}

double compute_residual(const double *u, const double *f, int n) {
    double max_res = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;
            double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;
            double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;
            double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;
            double res = f[i * n + j] - (4.0 * u[i * n + j] - left - right - up - down);
            if (fabs(res) > max_res) max_res = fabs(res);
        }
    return max_res;
}

void print_results(const char *method, const char *impl,
                   int n, int iters, double time_sec,
                   double residual, double rmse_exact) {
    printf("=============================================================\n");
    printf("  Method : %s (%s)\n", method, impl);
    printf("  Grid   : %d x %d  (%d unknowns)\n", n, n, n * n);
    printf("  Iters  : %d\n", iters);
    printf("  Time   : %.6f seconds\n", time_sec);
    printf("  Residual (inf-norm) : %.2e\n", residual);
    printf("  RMSE vs exact       : %.2e\n", rmse_exact);
    printf("=============================================================\n\n");
}

int main(int argc, char *argv[]) {
    int n        = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;
    int max_iter = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;
    double tol   = (argc > 3) ? atof(argv[3]) : TOLERANCE;

    size_t grid_bytes = (size_t)n * n * sizeof(double);

    /* Print GPU info */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Iterative Linear Solvers - CUDA Implementation\n");
    printf("GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e\n\n", n, n, max_iter, tol);

    /* Host arrays */
    double *h_f     = (double *)malloc(grid_bytes);
    double *h_u     = (double *)calloc((size_t)n * n, sizeof(double));
    double *h_exact = (double *)malloc(grid_bytes);

    init_rhs(h_f, n);
    exact_solution(h_exact, n);

    /* Device arrays */
    double *d_u, *d_u_old, *d_f, *d_diff;
    CUDA_CHECK(cudaMalloc(&d_u, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_old, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_f, grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_diff, grid_bytes));

    CUDA_CHECK(cudaMemcpy(d_f, h_f, grid_bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* ===== JACOBI ===== */
    CUDA_CHECK(cudaMemset(d_u, 0, grid_bytes));
    CUDA_CHECK(cudaMemset(d_u_old, 0, grid_bytes));

    double t_start = get_time();
    int jac_iters = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Swap: u_old = u */
        CUDA_CHECK(cudaMemcpy(d_u_old, d_u, grid_bytes, cudaMemcpyDeviceToDevice));

        /* Compute new values */
        jacobi_kernel<<<grid_dim, block>>>(d_u_old, d_u, d_f, d_diff, n);
        CUDA_CHECK(cudaGetLastError());

        /* Convergence check */
        double max_diff = gpu_max_reduce(d_diff, n * n);

        jac_iters = iter + 1;
        if (max_diff < tol) break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_jac = get_time() - t_start;

    /* Copy result back */
    CUDA_CHECK(cudaMemcpy(h_u, d_u, grid_bytes, cudaMemcpyDeviceToHost));
    double res_jac  = compute_residual(h_u, h_f, n);
    double rmse_jac = compute_rmse(h_u, h_exact, n);
    print_results("Jacobi", "CUDA", n, jac_iters, t_jac, res_jac, rmse_jac);

    /* Save Jacobi result for comparison */
    double *h_u_jac = (double *)malloc(grid_bytes);
    memcpy(h_u_jac, h_u, grid_bytes);

    /* ===== RED-BLACK GAUSS-SEIDEL ===== */
    CUDA_CHECK(cudaMemset(d_u, 0, grid_bytes));

    t_start = get_time();
    int gs_iters = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Phase 1: Update RED points (color=0) */
        redblack_kernel<<<grid_dim, block>>>(d_u, d_f, d_diff, n, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Phase 2: Update BLACK points (color=1) */
        redblack_kernel<<<grid_dim, block>>>(d_u, d_f, d_diff, n, 1);
        CUDA_CHECK(cudaGetLastError());

        /* Convergence (check diff from black phase + red diffs) */
        double max_diff = gpu_max_reduce(d_diff, n * n);

        gs_iters = iter + 1;
        if (max_diff < tol) break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_gs = get_time() - t_start;

    CUDA_CHECK(cudaMemcpy(h_u, d_u, grid_bytes, cudaMemcpyDeviceToHost));
    double res_gs  = compute_residual(h_u, h_f, n);
    double rmse_gs = compute_rmse(h_u, h_exact, n);
    print_results("Red-Black Gauss-Seidel", "CUDA", n, gs_iters, t_gs, res_gs, rmse_gs);

    double rmse_jac_vs_gs = compute_rmse(h_u_jac, h_u, n);
    printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);

    /* Cleanup */
    free(h_f);
    free(h_u);
    free(h_exact);
    free(h_u_jac);
    cudaFree(d_u);
    cudaFree(d_u_old);
    cudaFree(d_f);
    cudaFree(d_diff);

    return 0;
}
