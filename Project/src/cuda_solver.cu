/**
 * cuda_solver.cu - CUDA GPU implementations of Jacobi and Red-Black Gauss-Seidel
 *
 * Each grid point is mapped to one GPU thread.
 * 2D thread blocks are used for efficient memory access patterns.
 *
 * Usage: ./cuda_solver [grid_size] [max_iterations] [tolerance]
 */

#include <stdio.h>      /* Standard I/O: printf, fprintf */
#include <stdlib.h>     /* Standard library: malloc, calloc, free, atoi, atof */
#include <math.h>       /* Math functions: sin, fabs, sqrt, fmax */
#include <string.h>     /* String/memory: memcpy */
#include <sys/time.h>   /* Time functions: gettimeofday */

#ifndef M_PI                        /* Check if pi constant is not already defined */
#define M_PI 3.14159265358979323846  /* Define pi with high precision */
#endif

#define DEFAULT_GRID_SIZE  100   /* Default grid dimension (100x100 interior points) */
#define MAX_ITERATIONS     10000 /* Maximum solver iterations */
#define TOLERANCE          1e-6  /* Convergence threshold */
#define BLOCK_SIZE         16    /* CUDA thread block dimension (16x16 = 256 threads per block) */

/* Timer */
static inline double get_time(void) {      /* Returns current wall-clock time in seconds */
    struct timeval tv;                      /* Struct for seconds and microseconds */
    gettimeofday(&tv, NULL);               /* Get current time */
    return tv.tv_sec + tv.tv_usec * 1e-6;  /* Convert to double seconds */
}

/* CUDA error checking macro */
#define CUDA_CHECK(call) do { \                                     /* Macro to wrap CUDA API calls with error checking */
    cudaError_t err = call; \                                       /* Execute the CUDA call and capture error code */
    if (err != cudaSuccess) { \                                     /* Check if the call failed */
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \  /* Print file name and line number */
                cudaGetErrorString(err)); \                          /* Print human-readable error message */
        exit(1); \                                                   /* Exit program on CUDA error */
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
                               const double *f, double *diff, int n) {  /* GPU kernel: each thread handles one grid point */
    int j = blockIdx.x * blockDim.x + threadIdx.x;  /* Compute column index from block and thread indices */
    int i = blockIdx.y * blockDim.y + threadIdx.y;  /* Compute row index from block and thread indices */

    if (i < n && j < n) {  /* Bounds check: only process valid grid points */
        double left  = (j > 0)     ? u_old[i * n + (j - 1)] : 0.0;  /* Left neighbor (boundary = 0) */
        double right = (j < n - 1) ? u_old[i * n + (j + 1)] : 0.0;  /* Right neighbor (boundary = 0) */
        double up    = (i > 0)     ? u_old[(i - 1) * n + j]  : 0.0;  /* Upper neighbor (boundary = 0) */
        double down  = (i < n - 1) ? u_old[(i + 1) * n + j]  : 0.0;  /* Lower neighbor (boundary = 0) */

        double new_val = (left + right + up + down + f[i * n + j]) / 4.0;  /* Compute new value using Jacobi formula */
        u_new[i * n + j] = new_val;  /* Store new value in output array */
        diff[i * n + j] = fabs(new_val - u_old[i * n + j]);  /* Store absolute difference for convergence check */
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
                                 int n, int color) {  /* GPU kernel: updates one color (red=0 or black=1) */
    int j = blockIdx.x * blockDim.x + threadIdx.x;  /* Compute column index */
    int i = blockIdx.y * blockDim.y + threadIdx.y;  /* Compute row index */

    if (i < n && j < n && (i + j) % 2 == color) {  /* Only process points matching the specified color */
        double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;  /* Left neighbor (other color, stable in this phase) */
        double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;  /* Right neighbor */
        double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;  /* Upper neighbor */
        double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;  /* Lower neighbor */

        double old_val = u[i * n + j];  /* Save current value before update */
        double new_val = (left + right + up + down + f[i * n + j]) / 4.0;  /* Compute new value */
        u[i * n + j] = new_val;  /* Update point in-place */
        diff[i * n + j] = fabs(new_val - old_val);  /* Store change for convergence check */
    } else if (i < n && j < n) {  /* For points of the other color (not being updated) */
        diff[i * n + j] = 0.0;  /* Set diff to 0 so they don't affect convergence check */
    }
}

/**
 * CUDA Kernel: Parallel max reduction
 *
 * Reduces an array of n*n doubles to find the maximum value.
 * Uses shared memory within each block.
 */
__global__ void max_reduce_kernel(const double *input, double *output, int total) {  /* Kernel to find maximum value in an array using parallel reduction */
    extern __shared__ double sdata[];  /* Dynamically allocated shared memory (size specified at launch) */

    int tid = threadIdx.x;  /* Thread index within the block */
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;  /* Global index (each thread loads 2 elements for first reduction) */

    double val = 0.0;  /* Initialize local value */
    if (idx < total) val = input[idx];  /* Load first element if within bounds */
    if (idx + blockDim.x < total) val = fmax(val, input[idx + blockDim.x]);  /* Load and compare second element (stride by blockDim) */
    sdata[tid] = val;  /* Store in shared memory */
    __syncthreads();  /* Synchronize all threads in the block before reduction */

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {  /* Tree-based reduction: halve active threads each step */
        if (tid < s) {  /* Only first s threads participate */
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);  /* Compare with element s positions away */
        }
        __syncthreads();  /* Synchronize after each reduction step */
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];  /* Thread 0 writes block's maximum to output array */
}

/**
 * Host function: find max of device array using reduction
 */
double gpu_max_reduce(double *d_array, int total) {  /* Host function to find max value of a GPU array */
    int threads = 256;  /* Threads per block for reduction */
    int blocks = (total + threads * 2 - 1) / (threads * 2);  /* Number of blocks needed (each thread handles 2 elements) */

    double *d_partial, *d_result;  /* Device pointers for intermediate reduction results */
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(double)));  /* Allocate space for partial maximums (one per block) */

    max_reduce_kernel<<<blocks, threads, threads * sizeof(double)>>>(d_array, d_partial, total);  /* Launch first reduction pass (shared mem = threads * sizeof(double)) */

    /* Iteratively reduce until we have one value */
    int current_size = blocks;  /* Number of values remaining to reduce */
    double *d_in = d_partial;   /* Input pointer for next reduction pass */

    while (current_size > 1) {  /* Continue reducing until only one value remains */
        int new_blocks = (current_size + threads * 2 - 1) / (threads * 2);  /* Blocks needed for this pass */
        CUDA_CHECK(cudaMalloc(&d_result, new_blocks * sizeof(double)));  /* Allocate output for this pass */
        max_reduce_kernel<<<new_blocks, threads, threads * sizeof(double)>>>(d_in, d_result, current_size);  /* Launch reduction kernel */
        if (d_in != d_partial) cudaFree(d_in);  /* Free intermediate buffer (but not the original d_partial) */
        d_in = d_result;  /* Set input for next pass */
        current_size = new_blocks;  /* Update remaining size */
    }

    double max_val;  /* Host variable for the result */
    CUDA_CHECK(cudaMemcpy(&max_val, d_in, sizeof(double), cudaMemcpyDeviceToHost));  /* Copy single max value from GPU to CPU */

    cudaFree(d_partial);  /* Free the first partial results buffer */
    if (d_in != d_partial) cudaFree(d_in);  /* Free last intermediate buffer if different from d_partial */

    return max_val;  /* Return the maximum value */
}

/* Host-side helper functions */
void init_rhs(double *f, int n) {  /* Initialize RHS vector on CPU (same as common.h version but non-inline for CUDA) */
    double h = 1.0 / (n + 1);  /* Grid spacing */
    for (int i = 0; i < n; i++)  /* Loop over rows */
        for (int j = 0; j < n; j++) {  /* Loop over columns */
            double x = (i + 1) * h;  /* x-coordinate */
            double y = (j + 1) * h;  /* y-coordinate */
            f[i * n + j] = h * h * 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);  /* h^2 * f(x,y) */
        }
}

void exact_solution(double *u_exact, int n) {  /* Compute exact analytical solution on CPU */
    double h = 1.0 / (n + 1);  /* Grid spacing */
    for (int i = 0; i < n; i++)  /* Loop over rows */
        for (int j = 0; j < n; j++) {  /* Loop over columns */
            double x = (i + 1) * h;  /* x-coordinate */
            double y = (j + 1) * h;  /* y-coordinate */
            u_exact[i * n + j] = sin(M_PI * x) * sin(M_PI * y);  /* Exact solution: sin(pi*x)*sin(pi*y) */
        }
}

double compute_rmse(const double *u, const double *u_ref, int n) {  /* Compute RMSE between two arrays on CPU */
    double sum = 0.0;  /* Accumulator */
    long total = (long)n * n;  /* Total grid points */
    for (long k = 0; k < total; k++) {  /* Loop over all points */
        double d = u[k] - u_ref[k];  /* Difference at point k */
        sum += d * d;  /* Accumulate squared difference */
    }
    return sqrt(sum / total);  /* Return RMSE */
}

double compute_residual(const double *u, const double *f, int n) {  /* Compute infinity-norm residual on CPU */
    double max_res = 0.0;  /* Track maximum residual */
    for (int i = 0; i < n; i++)  /* Loop over rows */
        for (int j = 0; j < n; j++) {  /* Loop over columns */
            double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;  /* Left neighbor */
            double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;  /* Right neighbor */
            double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;  /* Upper neighbor */
            double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;  /* Lower neighbor */
            double res = f[i * n + j] - (4.0 * u[i * n + j] - left - right - up - down);  /* Residual = f - A*u */
            if (fabs(res) > max_res) max_res = fabs(res);  /* Update max residual */
        }
    return max_res;  /* Return maximum absolute residual */
}

void print_results(const char *method, const char *impl,
                   int n, int iters, double time_sec,
                   double residual, double rmse_exact) {  /* Print formatted solver results */
    printf("=============================================================\n");      /* Separator */
    printf("  Method : %s (%s)\n", method, impl);                                   /* Method and implementation */
    printf("  Grid   : %d x %d  (%d unknowns)\n", n, n, n * n);                    /* Grid dimensions */
    printf("  Iters  : %d\n", iters);                                                /* Iteration count */
    printf("  Time   : %.6f seconds\n", time_sec);                                   /* Elapsed time */
    printf("  Residual (inf-norm) : %.2e\n", residual);                              /* Max residual */
    printf("  RMSE vs exact       : %.2e\n", rmse_exact);                            /* RMSE vs analytical solution */
    printf("=============================================================\n\n");     /* Closing separator */
}

int main(int argc, char *argv[]) {  /* Entry point */
    int n        = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;  /* Parse grid size or default */
    int max_iter = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;     /* Parse max iterations or default */
    double tol   = (argc > 3) ? atof(argv[3]) : TOLERANCE;          /* Parse tolerance or default */

    size_t grid_bytes = (size_t)n * n * sizeof(double);  /* Total bytes needed for one n*n grid array */

    /* Print GPU info */
    cudaDeviceProp prop;  /* Structure to hold GPU device properties */
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));  /* Query properties of GPU device 0 */
    printf("Iterative Linear Solvers - CUDA Implementation\n");  /* Print header */
    printf("GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);  /* Print GPU name and compute capability */
    printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e\n\n", n, n, max_iter, tol);  /* Print configuration */

    /* Host arrays */
    double *h_f     = (double *)malloc(grid_bytes);               /* Allocate host RHS vector */
    double *h_u     = (double *)calloc((size_t)n * n, sizeof(double));  /* Allocate and zero host solution array */
    double *h_exact = (double *)malloc(grid_bytes);               /* Allocate host exact solution */

    init_rhs(h_f, n);           /* Initialize RHS on host */
    exact_solution(h_exact, n);  /* Compute exact solution on host */

    /* Device arrays */
    double *d_u, *d_u_old, *d_f, *d_diff;  /* GPU memory pointers */
    CUDA_CHECK(cudaMalloc(&d_u, grid_bytes));      /* Allocate GPU memory for current solution */
    CUDA_CHECK(cudaMalloc(&d_u_old, grid_bytes));  /* Allocate GPU memory for old solution (Jacobi) */
    CUDA_CHECK(cudaMalloc(&d_f, grid_bytes));      /* Allocate GPU memory for RHS */
    CUDA_CHECK(cudaMalloc(&d_diff, grid_bytes));   /* Allocate GPU memory for per-point differences */

    CUDA_CHECK(cudaMemcpy(d_f, h_f, grid_bytes, cudaMemcpyHostToDevice));  /* Copy RHS from host to GPU */

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);  /* Define 2D thread block dimensions (16x16 threads) */
    dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);  /* Compute grid dimensions to cover entire n*n domain (ceiling division) */

    /* ===== JACOBI ===== */
    CUDA_CHECK(cudaMemset(d_u, 0, grid_bytes));      /* Zero-initialize current solution on GPU */
    CUDA_CHECK(cudaMemset(d_u_old, 0, grid_bytes));  /* Zero-initialize old solution on GPU */

    double t_start = get_time();  /* Record start time */
    int jac_iters = 0;  /* Initialize iteration counter */

    for (int iter = 0; iter < max_iter; iter++) {  /* Host-side iteration loop */
        /* Swap: u_old = u */
        CUDA_CHECK(cudaMemcpy(d_u_old, d_u, grid_bytes, cudaMemcpyDeviceToDevice));  /* Copy current solution to old on GPU (device-to-device) */

        /* Compute new values */
        jacobi_kernel<<<grid_dim, block>>>(d_u_old, d_u, d_f, d_diff, n);  /* Launch Jacobi kernel: each thread computes one grid point */
        CUDA_CHECK(cudaGetLastError());  /* Check for kernel launch errors */

        /* Convergence check */
        double max_diff = gpu_max_reduce(d_diff, n * n);  /* Find maximum difference across all grid points using GPU reduction */

        jac_iters = iter + 1;  /* Update iteration count */
        if (max_diff < tol) break;  /* Exit if converged */
    }
    CUDA_CHECK(cudaDeviceSynchronize());  /* Ensure all GPU operations are complete */
    double t_jac = get_time() - t_start;  /* Compute Jacobi elapsed time */

    /* Copy result back */
    CUDA_CHECK(cudaMemcpy(h_u, d_u, grid_bytes, cudaMemcpyDeviceToHost));  /* Copy Jacobi solution from GPU to host */
    double res_jac  = compute_residual(h_u, h_f, n);   /* Compute residual on host */
    double rmse_jac = compute_rmse(h_u, h_exact, n);   /* Compute RMSE vs exact on host */
    print_results("Jacobi", "CUDA", n, jac_iters, t_jac, res_jac, rmse_jac);  /* Print Jacobi results */

    /* Save Jacobi result for comparison */
    double *h_u_jac = (double *)malloc(grid_bytes);  /* Allocate host array to save Jacobi result */
    memcpy(h_u_jac, h_u, grid_bytes);  /* Copy Jacobi result for later comparison with GS */

    /* ===== RED-BLACK GAUSS-SEIDEL ===== */
    CUDA_CHECK(cudaMemset(d_u, 0, grid_bytes));  /* Zero-initialize solution on GPU for GS */

    t_start = get_time();  /* Record GS start time */
    int gs_iters = 0;  /* Initialize GS iteration counter */

    for (int iter = 0; iter < max_iter; iter++) {  /* Host-side iteration loop for GS */
        /* Phase 1: Update RED points (color=0) */
        redblack_kernel<<<grid_dim, block>>>(d_u, d_f, d_diff, n, 0);  /* Launch kernel to update all red points */
        CUDA_CHECK(cudaGetLastError());  /* Check for launch errors */
        CUDA_CHECK(cudaDeviceSynchronize());  /* Wait for red phase to complete before black phase */

        /* Phase 2: Update BLACK points (color=1) */
        redblack_kernel<<<grid_dim, block>>>(d_u, d_f, d_diff, n, 1);  /* Launch kernel to update all black points */
        CUDA_CHECK(cudaGetLastError());  /* Check for launch errors */

        /* Convergence (check diff from black phase + red diffs) */
        double max_diff = gpu_max_reduce(d_diff, n * n);  /* Find max change using GPU reduction */

        gs_iters = iter + 1;  /* Update iteration count */
        if (max_diff < tol) break;  /* Exit if converged */
    }
    CUDA_CHECK(cudaDeviceSynchronize());  /* Ensure all GPU operations complete */
    double t_gs = get_time() - t_start;  /* Compute GS elapsed time */

    CUDA_CHECK(cudaMemcpy(h_u, d_u, grid_bytes, cudaMemcpyDeviceToHost));  /* Copy GS solution to host */
    double res_gs  = compute_residual(h_u, h_f, n);    /* Compute GS residual */
    double rmse_gs = compute_rmse(h_u, h_exact, n);    /* Compute GS RMSE */
    print_results("Red-Black Gauss-Seidel", "CUDA", n, gs_iters, t_gs, res_gs, rmse_gs);  /* Print GS results */

    double rmse_jac_vs_gs = compute_rmse(h_u_jac, h_u, n);  /* Compare Jacobi and GS solutions */
    printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);  /* Print comparison */

    /* Cleanup */
    free(h_f);       /* Free host RHS */
    free(h_u);       /* Free host solution */
    free(h_exact);   /* Free host exact solution */
    free(h_u_jac);   /* Free saved Jacobi result */
    cudaFree(d_u);       /* Free GPU current solution */
    cudaFree(d_u_old);   /* Free GPU old solution */
    cudaFree(d_f);       /* Free GPU RHS */
    cudaFree(d_diff);    /* Free GPU difference array */

    return 0;  /* Exit successfully */
}
