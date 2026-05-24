/**
 * hybrid_cuda_solver.cu - Hybrid MPI + CUDA implementations
 *
 * Distributed memory parallelization using MPI across multiple nodes/processes,
 * with each process offloading its local domain computation to a GPU using CUDA.
 *
 * Usage: mpirun -np <procs> ./hybrid_cuda_solver [grid_size] [max_iterations] [tolerance]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEFAULT_GRID_SIZE  100
#define MAX_ITERATIONS     10000
#define TOLERANCE          1e-6
#define BLOCK_SIZE         16

/* CUDA error checking macro */
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/* ============ CUDA Kernels ============ */

__global__ void jacobi_kernel(const double *u_old, double *u_new, const double *f, double *diff, int local_rows, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    /* i is from 1 to local_rows (inclusive) because 0 and local_rows+1 are ghost rows */
    if (i >= 1 && i <= local_rows && j < n) {
        double left  = (j > 0)     ? u_old[i * n + (j - 1)] : 0.0;
        double right = (j < n - 1) ? u_old[i * n + (j + 1)] : 0.0;
        double up    = u_old[(i - 1) * n + j];
        double down  = u_old[(i + 1) * n + j];

        int fi = i - 1; /* f array doesn't have ghost rows */
        double new_val = (left + right + up + down + f[fi * n + j]) / 4.0;
        u_new[i * n + j] = new_val;
        diff[i * n + j] = fabs(new_val - u_old[i * n + j]);
    } else if (i <= local_rows + 1 && j < n) {
        diff[i * n + j] = 0.0;
    }
}

__global__ void redblack_kernel(double *u, const double *f, double *diff, int local_rows, int n, int global_row_start, int color) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i <= local_rows && j < n) {
        int gi = global_row_start + (i - 1);
        if ((gi + j) % 2 == color) {
            double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;
            double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;
            double up    = u[(i - 1) * n + j];
            double down  = u[(i + 1) * n + j];

            int fi = i - 1;
            double old_val = u[i * n + j];
            double new_val = (left + right + up + down + f[fi * n + j]) / 4.0;
            u[i * n + j] = new_val;
            diff[i * n + j] = fabs(new_val - old_val);
        } else {
            diff[i * n + j] = 0.0;
        }
    } else if (i <= local_rows + 1 && j < n) {
        diff[i * n + j] = 0.0;
    }
}

extern __shared__ double sdata[];

__global__ void max_reduce_kernel(const double *input, double *output, int total) {
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

double gpu_max_reduce(double *d_array, int total) {
    int threads = 256;
    int blocks = (total + threads * 2 - 1) / (threads * 2);

    double *d_partial, *d_result;
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(double)));

    max_reduce_kernel<<<blocks, threads, threads * sizeof(double)>>>(d_array, d_partial, total);

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

/* ============ Host Helpers ============ */

void save_solution(const char *filename, const double *u, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp, "%.12e ", u[i * n + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("Saved solution to %s\n", filename);
}

void init_rhs(double *f, int n) {
    double h = 1.0 / (n + 1);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double x = (i + 1) * h;
            double y = (j + 1) * h;
            f[i * n + j] = h * h * 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }
}

void exact_solution(double *u_exact, int n) {
    double h = 1.0 / (n + 1);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double x = (i + 1) * h;
            double y = (j + 1) * h;
            u_exact[i * n + j] = sin(M_PI * x) * sin(M_PI * y);
        }
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
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;
            double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;
            double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;
            double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;
            double res = f[i * n + j] - (4.0 * u[i * n + j] - left - right - up - down);
            if (fabs(res) > max_res) max_res = fabs(res);
        }
    }
    return max_res;
}

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

/* ============ Main MPI+CUDA Routine ============ */

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n        = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;
    int max_iter = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;
    double tol   = (argc > 3) ? atof(argv[3]) : TOLERANCE;

    // Set GPU device (assign GPUs to MPI ranks round-robin if multiple are available)
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(rank % num_devices);

    if (rank == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Iterative Linear Solvers - Hybrid MPI+CUDA Implementation\n");
        printf("GPU: %s (Compute %d.%d) - found %d devices\n", prop.name, prop.major, prop.minor, num_devices);
        printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e, Processes: %d\n\n",
               n, n, max_iter, tol, size);
    }

    int rows_per_proc = n / size;
    int extra = n % size;
    int local_rows = rows_per_proc + (rank < extra ? 1 : 0);
    int global_row_start = rank * rows_per_proc + (rank < extra ? rank : extra);

    size_t local_bytes = (size_t)local_rows * n * sizeof(double);
    size_t ghost_bytes = (size_t)(local_rows + 2) * n * sizeof(double);
    size_t row_bytes = n * sizeof(double);

    // Host memory
    double *h_f_local = (double *)malloc(local_bytes);
    double *h_u_jac   = (double *)calloc((local_rows + 2) * n, sizeof(double));
    double *h_u_gs    = (double *)calloc((local_rows + 2) * n, sizeof(double));
    
    // Ghost row buffers for host-side MPI exchanges
    double *h_send_top = (double *)malloc(row_bytes);
    double *h_send_bot = (double *)malloc(row_bytes);
    double *h_recv_top = (double *)malloc(row_bytes);
    double *h_recv_bot = (double *)malloc(row_bytes);

    double *f_global = NULL, *u_global_jac = NULL, *u_global_gs = NULL, *u_exact = NULL;

    if (rank == 0) {
        f_global     = (double *)malloc((size_t)n * n * sizeof(double));
        u_global_jac = (double *)malloc((size_t)n * n * sizeof(double));
        u_global_gs  = (double *)malloc((size_t)n * n * sizeof(double));
        u_exact      = (double *)malloc((size_t)n * n * sizeof(double));
        init_rhs(f_global, n);
        exact_solution(u_exact, n);
    }

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));
    int offset = 0;
    for (int p = 0; p < size; p++) {
        int lr = rows_per_proc + (p < extra ? 1 : 0);
        sendcounts[p] = lr * n;
        displs[p] = offset;
        offset += lr * n;
    }

    MPI_Scatterv(f_global, sendcounts, displs, MPI_DOUBLE,
                 h_f_local, local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Device memory
    double *d_f_local, *d_u, *d_u_old, *d_diff;
    CUDA_CHECK(cudaMalloc(&d_f_local, local_bytes));
    CUDA_CHECK(cudaMalloc(&d_u, ghost_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_old, ghost_bytes));
    CUDA_CHECK(cudaMalloc(&d_diff, ghost_bytes));

    CUDA_CHECK(cudaMemcpy(d_f_local, h_f_local, local_bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (local_rows + 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* ==== JACOBI ==== */
    CUDA_CHECK(cudaMemset(d_u, 0, ghost_bytes));
    CUDA_CHECK(cudaMemset(d_u_old, 0, ghost_bytes));

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();
    int jac_iters = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        CUDA_CHECK(cudaMemcpy(d_u_old, d_u, ghost_bytes, cudaMemcpyDeviceToDevice));

        // Copy boundary rows to host for MPI exchange
        CUDA_CHECK(cudaMemcpy(h_send_top, d_u_old + 1 * n, row_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_send_bot, d_u_old + local_rows * n, row_bytes, cudaMemcpyDeviceToHost));

        MPI_Sendrecv(h_send_top, n, MPI_DOUBLE, prev, 0,
                     h_recv_bot, n, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(h_send_bot, n, MPI_DOUBLE, next, 1,
                     h_recv_top, n, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Copy received ghost rows to device
        CUDA_CHECK(cudaMemcpy(d_u_old + (local_rows + 1) * n, h_recv_bot, row_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u_old + 0 * n, h_recv_top, row_bytes, cudaMemcpyHostToDevice));

        jacobi_kernel<<<grid_dim, block>>>(d_u_old, d_u, d_f_local, d_diff, local_rows, n);
        
        double local_max_diff = gpu_max_reduce(d_diff, (local_rows + 2) * n);
        double global_max_diff;
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        jac_iters = iter + 1;
        if (global_max_diff < tol) break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_jac = MPI_Wtime() - t_start;

    CUDA_CHECK(cudaMemcpy(h_u_jac, d_u, ghost_bytes, cudaMemcpyDeviceToHost));
    MPI_Gatherv(&h_u_jac[1 * n], local_rows * n, MPI_DOUBLE,
                u_global_jac, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ==== RED-BLACK GS ==== */
    CUDA_CHECK(cudaMemset(d_u, 0, ghost_bytes));

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();
    int gs_iters = 0;

    for (int iter = 0; iter < max_iter; iter++) {
        // Phase 1: RED
        CUDA_CHECK(cudaMemcpy(h_send_top, d_u + 1 * n, row_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_send_bot, d_u + local_rows * n, row_bytes, cudaMemcpyDeviceToHost));

        MPI_Sendrecv(h_send_top, n, MPI_DOUBLE, prev, 0,
                     h_recv_bot, n, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(h_send_bot, n, MPI_DOUBLE, next, 1,
                     h_recv_top, n, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        CUDA_CHECK(cudaMemcpy(d_u + (local_rows + 1) * n, h_recv_bot, row_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u + 0 * n, h_recv_top, row_bytes, cudaMemcpyHostToDevice));

        redblack_kernel<<<grid_dim, block>>>(d_u, d_f_local, d_diff, local_rows, n, global_row_start, 0);

        // Phase 2: BLACK
        CUDA_CHECK(cudaMemcpy(h_send_top, d_u + 1 * n, row_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_send_bot, d_u + local_rows * n, row_bytes, cudaMemcpyDeviceToHost));

        MPI_Sendrecv(h_send_top, n, MPI_DOUBLE, prev, 2,
                     h_recv_bot, n, MPI_DOUBLE, next, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(h_send_bot, n, MPI_DOUBLE, next, 3,
                     h_recv_top, n, MPI_DOUBLE, prev, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        CUDA_CHECK(cudaMemcpy(d_u + (local_rows + 1) * n, h_recv_bot, row_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u + 0 * n, h_recv_top, row_bytes, cudaMemcpyHostToDevice));

        redblack_kernel<<<grid_dim, block>>>(d_u, d_f_local, d_diff, local_rows, n, global_row_start, 1);

        double local_max_diff = gpu_max_reduce(d_diff, (local_rows + 2) * n);
        double global_max_diff;
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        gs_iters = iter + 1;
        if (global_max_diff < tol) break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_gs = MPI_Wtime() - t_start;

    CUDA_CHECK(cudaMemcpy(h_u_gs, d_u, ghost_bytes, cudaMemcpyDeviceToHost));
    MPI_Gatherv(&h_u_gs[1 * n], local_rows * n, MPI_DOUBLE,
                u_global_gs, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double max_t_jac, max_t_gs;
    MPI_Reduce(&t_jac, &max_t_jac, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_gs, &max_t_gs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double res_jac  = compute_residual(u_global_jac, f_global, n);
        double rmse_jac = compute_rmse(u_global_jac, u_exact, n);
        print_results("Jacobi", "MPI+CUDA", n, jac_iters, max_t_jac, res_jac, rmse_jac);

        double res_gs  = compute_residual(u_global_gs, f_global, n);
        double rmse_gs = compute_rmse(u_global_gs, u_exact, n);
        print_results("Red-Black Gauss-Seidel", "MPI+CUDA", n, gs_iters, max_t_gs, res_gs, rmse_gs);

        double rmse_jac_vs_gs = compute_rmse(u_global_jac, u_global_gs, n);
        printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);

        save_solution("hybrid_cuda_jacobi.txt", u_global_jac, n);
        save_solution("hybrid_cuda_rbgs.txt", u_global_gs, n);

        free(f_global);
        free(u_global_jac);
        free(u_global_gs);
        free(u_exact);
    }

    free(h_f_local);
    free(h_u_jac);
    free(h_u_gs);
    free(h_send_top); free(h_send_bot);
    free(h_recv_top); free(h_recv_bot);
    free(sendcounts); free(displs);
    
    cudaFree(d_f_local); cudaFree(d_u); cudaFree(d_u_old); cudaFree(d_diff);

    MPI_Finalize();
    return 0;
}
