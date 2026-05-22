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

/* CUDA error checking macro - Clean and stripped of problematic internal comments */
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/**
 * CUDA Kernel: Jacobi iteration
 * 
 * Implements one iteration of the Jacobi method. Each thread computes the new value
 * at grid point (i, j) based on the 4-point stencil from the previous iteration.
 * The Jacobi method uses values from the old solution for all 4 neighbors.
 * 
 * Parameters:
 *   u_old : previous solution values (read-only)
 *   u_new : storage for new solution values
 *   f     : right-hand side of the system
 *   diff  : array to store convergence differences at each point
 *   n     : grid dimension (n x n interior points)
 */
__global__ void jacobi_kernel(const double *u_old, double *u_new, const double *f, double *diff, int n) {
    // Map thread to grid point: x -> column (j), y -> row (i)
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: only process interior grid points
    if (i < n && j < n) {
        // Load neighbors from old solution with zero boundary conditions
        double left  = (j > 0)     ? u_old[i * n + (j - 1)] : 0.0;
        double right = (j < n - 1) ? u_old[i * n + (j + 1)] : 0.0;
        double up    = (i > 0)     ? u_old[(i - 1) * n + j]  : 0.0;
        double down  = (i < n - 1) ? u_old[(i + 1) * n + j]  : 0.0;

        // Compute new value using discrete Laplace operator: (u_left + u_right + u_up + u_down + f) / 4
        double new_val = (left + right + up + down + f[i * n + j]) / 4.0;
        u_new[i * n + j] = new_val;
        
        // Store absolute difference for convergence check
        diff[i * n + j] = fabs(new_val - u_old[i * n + j]);
    }
}

/**
 * CUDA Kernel: Red-Black Gauss-Seidel - update one color
 * 
 * Implements one color sweep of the Red-Black Gauss-Seidel method.
 * The grid is colored in a checkerboard pattern (red/black), where each color
 * represents grid points (i,j) with (i+j) % 2 == color.
 * Each color can be updated in parallel since points of the same color are non-adjacent.
 * Red-Black GS uses updated values when available (from the same or previous color).
 * 
 * Parameters:
 *   u     : solution array (read-write, updated in-place)
 *   f     : right-hand side of the system
 *   diff  : array to store convergence differences at each point
 *   n     : grid dimension (n x n interior points)
 *   color : 0 for red (i+j even), 1 for black (i+j odd)
 */
__global__ void redblack_kernel(double *u, const double *f, double *diff, int n, int color) {
    // Map thread to grid point
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // Only process grid points that match the current color
    if (i < n && j < n && (i + j) % 2 == color) {
        // Load neighbors (may be old or newly updated values depending on order)
        double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;
        double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;
        double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;
        double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;

        // Store old value before update for convergence tracking
        double old_val = u[i * n + j];
        
        // Compute and immediately apply new value (in-place update)
        double new_val = (left + right + up + down + f[i * n + j]) / 4.0;
        u[i * n + j] = new_val;
        
        // Track convergence difference
        diff[i * n + j] = fabs(new_val - old_val);
    } else if (i < n && j < n) {
        // Non-active colors have zero difference in this sweep
        diff[i * n + j] = 0.0;
    }
}

/**
 * CUDA Kernel: Parallel max reduction
 * 
 * Reduces an array to find its maximum element using parallel reduction.
 * This kernel processes 2*blockDim.x elements per block in the first load,
 * then reduces within shared memory using a binary tree pattern.
 * Output: one maximum value per block (requires subsequent reductions).
 * 
 * Algorithm:
 *   1. Load 2 input values per thread and compute local max
 *   2. Store in shared memory and synchronize
 *   3. Tree reduction: each thread compares with distance s = blockDim.x/2, s/2, ..., 1
 *   4. First thread writes block result
 * 
 * Parameters:
 *   input  : input array to find maximum of
 *   output : output array (one max per block)
 *   total  : total number of elements in input
 */
__global__ void max_reduce_kernel(const double *input, double *output, int total) {
    // Shared memory for intra-block reduction
    extern __shared__ double sdata[];

    int tid = threadIdx.x;  // Local thread index within block
    // Global index: each thread loads 2 elements
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load phase: each thread reads up to 2 input values and computes local max
    double val = 0.0;
    if (idx < total) val = input[idx];
    if (idx + blockDim.x < total) val = fmax(val, input[idx + blockDim.x]);
    sdata[tid] = val;
    __syncthreads();

    // Tree reduction in shared memory: stride-based binary reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // Compare with thread at distance s and keep maximum
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write block result: thread 0 has the max for this block
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

/**
 * Host function: find max of device array using multi-level reduction
 * 
 * Performs complete reduction by iteratively calling the max_reduce_kernel
 * until a single maximum value remains. Each iteration reduces the array size.
 * Memory is dynamically allocated for intermediate results.
 * 
 * Parameters:
 *   d_array : device array to find maximum of
 *   total   : number of elements in the array
 * 
 * Returns: maximum value in the array
 */
double gpu_max_reduce(double *d_array, int total) {
    int threads = 256;  // Threads per block for reduction kernel
    // First level: process 2*threads elements per block
    int blocks = (total + threads * 2 - 1) / (threads * 2);

    double *d_partial, *d_result;
    // Allocate space for first reduction level
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(double)));

    // Initial reduction: input array -> partial results (one per block)
    max_reduce_kernel<<<blocks, threads, threads * sizeof(double)>>>(d_array, d_partial, total);

    // Multi-level reduction loop: reduce partial results until single value remains
    int current_size = blocks;
    double *d_in = d_partial;

    while (current_size > 1) {
        int new_blocks = (current_size + threads * 2 - 1) / (threads * 2);
        CUDA_CHECK(cudaMalloc(&d_result, new_blocks * sizeof(double)));
        max_reduce_kernel<<<new_blocks, threads, threads * sizeof(double)>>>(d_in, d_result, current_size);
        
        // Free intermediate arrays except the initial partial results
        if (d_in != d_partial) cudaFree(d_in);
        d_in = d_result;
        current_size = new_blocks;
    }

    // Copy final maximum value from device to host
    double max_val;
    CUDA_CHECK(cudaMemcpy(&max_val, d_in, sizeof(double), cudaMemcpyDeviceToHost));

    // Cleanup: free all intermediate device memory
    cudaFree(d_partial);
    if (d_in != d_partial) cudaFree(d_in);

    return max_val;
}

/* ============ Host-side helper functions ============ */

/**
 * Initialize the right-hand side (RHS) array f
 * Using the test problem: f(x,y) = 2π² sin(πx)sin(πy)
 * This gives the exact solution u(x,y) = sin(πx)sin(πy)
 */
void init_rhs(double *f, int n) {
    double h = 1.0 / (n + 1);  // Grid spacing
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Map grid indices to physical coordinates in [0,1]×[0,1]
            double x = (i + 1) * h;
            double y = (j + 1) * h;
            // Compute RHS value: h² * 2π² sin(πx)sin(πy)
            f[i * n + j] = h * h * 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }
}

/**
 * Compute exact solution: u(x,y) = sin(πx)sin(πy)
 * Used to compare solver results and compute RMSE error
 */
void exact_solution(double *u_exact, int n) {
    double h = 1.0 / (n + 1);  // Grid spacing
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Map grid indices to physical coordinates in [0,1]×[0,1]
            double x = (i + 1) * h;
            double y = (j + 1) * h;
            // Exact solution value at this point
            u_exact[i * n + j] = sin(M_PI * x) * sin(M_PI * y);
        }
    }
}

/**
 * Compute Root Mean Square Error (RMSE) between two solutions
 * RMSE = sqrt(sum((u[k] - u_ref[k])²) / n²)
 */
double compute_rmse(const double *u, const double *u_ref, int n) {
    double sum = 0.0;
    long total = (long)n * n;
    
    // Sum squared differences across all grid points
    for (long k = 0; k < total; k++) {
        double d = u[k] - u_ref[k];
        sum += d * d;
    }
    
    // Return RMS error
    return sqrt(sum / total);
}

/**
 * Compute infinity-norm residual
 * residual = max |f[i,j] - (4*u[i,j] - u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1])|
 * Measures how well the solution satisfies the discrete Laplace equation
 */
double compute_residual(const double *u, const double *f, int n) {
    double max_res = 0.0;
    
    // Check residual at each interior grid point
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Load neighbors with zero boundary conditions
            double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;
            double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;
            double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;
            double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;
            
            // Compute residual: RHS - discrete Laplacian
            double res = f[i * n + j] - (4.0 * u[i * n + j] - left - right - up - down);
            if (fabs(res) > max_res) max_res = fabs(res);
        }
    }
    
    return max_res;
}

/**
 * Print formatted solver results for comparison
 */
void print_results(const char *method, const char *impl, int n, int iters, double time_sec, double residual, double rmse_exact) {
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
    // Parse command-line arguments with defaults
    int n        = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;
    int max_iter = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;
    double tol   = (argc > 3) ? atof(argv[3]) : TOLERANCE;

    // Calculate total memory needed for an n×n grid
    size_t grid_bytes = (size_t)n * n * sizeof(double);

    // Display GPU information
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Iterative Linear Solvers - CUDA Implementation\n");
    printf("GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e\n\n", n, n, max_iter, tol);

    // Allocate host memory for problem arrays
    double *h_f     = (double *)malloc(grid_bytes);           // RHS array
    double *h_u     = (double *)calloc((size_t)n * n, sizeof(double));  // Solution (zero-initialized)
    double *h_exact = (double *)malloc(grid_bytes);           // Exact solution for comparison

    // Initialize problem data on host
    init_rhs(h_f, n);
    exact_solution(h_exact, n);

    // Allocate device (GPU) memory for solver arrays
    double *d_u, *d_u_old, *d_f, *d_diff;
    CUDA_CHECK(cudaMalloc(&d_u, grid_bytes));        // Current solution
    CUDA_CHECK(cudaMalloc(&d_u_old, grid_bytes));    // Previous solution (Jacobi only)
    CUDA_CHECK(cudaMalloc(&d_f, grid_bytes));        // RHS (constant during iterations)
    CUDA_CHECK(cudaMalloc(&d_diff, grid_bytes));     // Convergence differences

    // Copy RHS from host to device
    CUDA_CHECK(cudaMemcpy(d_f, h_f, grid_bytes, cudaMemcpyHostToDevice));

    // Configure CUDA grid/block dimensions for 2D execution
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);  // 16×16 = 256 threads per block
    // Calculate number of blocks needed to cover the n×n grid
    dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* ===== JACOBI METHOD ===== */
    // Initialize solution arrays to zero
    CUDA_CHECK(cudaMemset(d_u, 0, grid_bytes));
    CUDA_CHECK(cudaMemset(d_u_old, 0, grid_bytes));

    // Time the Jacobi solver
    double t_start = get_time();
    int jac_iters = 0;

    // Jacobi iteration loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Copy current solution to old solution (Jacobi requires values from previous iteration)
        CUDA_CHECK(cudaMemcpy(d_u_old, d_u, grid_bytes, cudaMemcpyDeviceToDevice));

        // Perform one Jacobi iteration on all grid points
        jacobi_kernel<<<grid_dim, block>>>(d_u_old, d_u, d_f, d_diff, n);
        CUDA_CHECK(cudaGetLastError());

        // Find maximum convergence difference across all grid points
        double max_diff = gpu_max_reduce(d_diff, n * n);

        jac_iters = iter + 1;
        // Check convergence: if max_diff < tolerance, solution has converged
        if (max_diff < tol) break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_jac = get_time() - t_start;

    // Copy Jacobi solution from device to host
    CUDA_CHECK(cudaMemcpy(h_u, d_u, grid_bytes, cudaMemcpyDeviceToHost));
    
    // Compute error metrics for Jacobi
    double res_jac  = compute_residual(h_u, h_f, n);  // Residual of discrete system
    double rmse_jac = compute_rmse(h_u, h_exact, n);  // Error vs exact solution
    print_results("Jacobi", "CUDA", n, jac_iters, t_jac, res_jac, rmse_jac);

    // Save Jacobi solution for later comparison
    double *h_u_jac = (double *)malloc(grid_bytes);
    memcpy(h_u_jac, h_u, grid_bytes);

    /* ===== RED-BLACK GAUSS-SEIDEL METHOD ===== */
    // Initialize solution array to zero
    CUDA_CHECK(cudaMemset(d_u, 0, grid_bytes));

    // Time the Red-Black GS solver
    t_start = get_time();
    int gs_iters = 0;

    // Red-Black Gauss-Seidel iteration loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Update all red points (color=0, i+j even)
        redblack_kernel<<<grid_dim, block>>>(d_u, d_f, d_diff, n, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());  // Synchronize before black phase

        // Update all black points (color=1, i+j odd)
        // Black points now use updated values from red points
        redblack_kernel<<<grid_dim, block>>>(d_u, d_f, d_diff, n, 1);
        CUDA_CHECK(cudaGetLastError());

        // Check convergence using maximum difference from both color phases
        double max_diff = gpu_max_reduce(d_diff, n * n);

        gs_iters = iter + 1;
        // Check convergence: if max_diff < tolerance, solution has converged
        if (max_diff < tol) break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_gs = get_time() - t_start;

    // Copy Red-Black GS solution from device to host
    CUDA_CHECK(cudaMemcpy(h_u, d_u, grid_bytes, cudaMemcpyDeviceToHost));
    
    // Compute error metrics for Red-Black GS
    double res_gs  = compute_residual(h_u, h_f, n);   // Residual of discrete system
    double rmse_gs = compute_rmse(h_u, h_exact, n);   // Error vs exact solution
    print_results("Red-Black Gauss-Seidel", "CUDA", n, gs_iters, t_gs, res_gs, rmse_gs);

    // Compare solutions from both methods
    double rmse_jac_vs_gs = compute_rmse(h_u_jac, h_u, n);
    printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);

    /* ===== CLEANUP ===== */
    // Free host memory
    free(h_f);
    free(h_u);
    free(h_exact);
    free(h_u_jac);
    
    // Free device (GPU) memory
    cudaFree(d_u);
    cudaFree(d_u_old);
    cudaFree(d_f);
    cudaFree(d_diff);

    return 0;
}