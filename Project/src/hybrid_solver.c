/**
 * hybrid_solver.c - Hybrid MPI + OpenMP implementation
 *
 * Combines distributed memory (MPI row decomposition) with shared memory
 * (OpenMP loop parallelization within each MPI process).
 *
 * Usage: mpirun -np <procs> ./hybrid_solver [grid_size] [max_iter] [tol] [omp_threads]
 */

#include "../include/common.h"
#include <mpi.h>
#include <omp.h>

/**
 * Hybrid Jacobi Method
 *
 * MPI: Row-wise domain decomposition with ghost row exchange.
 * OpenMP: Parallel for over local rows within each MPI process.
 */
int jacobi_hybrid(double *u_local, double *f_local, int local_rows, int n,
                  int max_iter, double tol, int rank, int size, int omp_threads) {
    int total_local = (local_rows + 2) * n;
    double *u_old = (double *)calloc(total_local, sizeof(double));
    if (!u_old) { fprintf(stderr, "Rank %d: allocation failed\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }

    omp_set_num_threads(omp_threads);

    int prev = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        memcpy(u_old, u_local, total_local * sizeof(double));

        /* Exchange ghost rows (MPI communication - done by master thread) */
        MPI_Sendrecv(&u_old[1 * n], n, MPI_DOUBLE, prev, 0,
                     &u_old[(local_rows + 1) * n], n, MPI_DOUBLE, next, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u_old[local_rows * n], n, MPI_DOUBLE, next, 1,
                     &u_old[0], n, MPI_DOUBLE, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* OpenMP parallel computation over local rows */
        double local_max_diff = 0.0;

        #pragma omp parallel for collapse(2) reduction(max:local_max_diff) schedule(static)
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 0; j < n; j++) {
                double left  = (j > 0)     ? u_old[i * n + (j - 1)] : 0.0;
                double right = (j < n - 1) ? u_old[i * n + (j + 1)] : 0.0;
                double up    = u_old[(i - 1) * n + j];
                double down  = u_old[(i + 1) * n + j];

                int fi = i - 1;
                u_local[i * n + j] = (left + right + up + down + f_local[fi * n + j]) / 4.0;

                double diff = fabs(u_local[i * n + j] - u_old[i * n + j]);
                if (diff > local_max_diff) local_max_diff = diff;
            }
        }

        /* Global convergence check (MPI) */
        double global_max_diff;
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (global_max_diff < tol) {
            iter++;
            break;
        }
    }

    free(u_old);
    return iter;
}

/**
 * Hybrid Red-Black Gauss-Seidel Method
 *
 * MPI: Row decomposition + ghost row exchange between phases.
 * OpenMP: Parallel for within each color phase.
 */
int redblack_gs_hybrid(double *u_local, double *f_local, int local_rows, int n,
                       int max_iter, double tol, int rank, int size,
                       int global_row_start, int omp_threads) {
    omp_set_num_threads(omp_threads);

    int prev = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        double local_max_diff = 0.0;

        /* Exchange ghost rows before RED phase */
        MPI_Sendrecv(&u_local[1 * n], n, MPI_DOUBLE, prev, 0,
                     &u_local[(local_rows + 1) * n], n, MPI_DOUBLE, next, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u_local[local_rows * n], n, MPI_DOUBLE, next, 1,
                     &u_local[0], n, MPI_DOUBLE, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Phase 1: RED points - OpenMP parallel */
        #pragma omp parallel for collapse(2) reduction(max:local_max_diff) schedule(static)
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 0; j < n; j++) {
                int gi = global_row_start + (i - 1);
                if ((gi + j) % 2 != 0) continue;

                double left  = (j > 0)     ? u_local[i * n + (j - 1)] : 0.0;
                double right = (j < n - 1) ? u_local[i * n + (j + 1)] : 0.0;
                double up    = u_local[(i - 1) * n + j];
                double down  = u_local[(i + 1) * n + j];

                int fi = i - 1;
                double old_val = u_local[i * n + j];
                u_local[i * n + j] = (left + right + up + down + f_local[fi * n + j]) / 4.0;

                double diff = fabs(u_local[i * n + j] - old_val);
                if (diff > local_max_diff) local_max_diff = diff;
            }
        }

        /* Exchange ghost rows before BLACK phase */
        MPI_Sendrecv(&u_local[1 * n], n, MPI_DOUBLE, prev, 2,
                     &u_local[(local_rows + 1) * n], n, MPI_DOUBLE, next, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u_local[local_rows * n], n, MPI_DOUBLE, next, 3,
                     &u_local[0], n, MPI_DOUBLE, prev, 3,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Phase 2: BLACK points - OpenMP parallel */
        #pragma omp parallel for collapse(2) reduction(max:local_max_diff) schedule(static)
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 0; j < n; j++) {
                int gi = global_row_start + (i - 1);
                if ((gi + j) % 2 != 1) continue;

                double left  = (j > 0)     ? u_local[i * n + (j - 1)] : 0.0;
                double right = (j < n - 1) ? u_local[i * n + (j + 1)] : 0.0;
                double up    = u_local[(i - 1) * n + j];
                double down  = u_local[(i + 1) * n + j];

                int fi = i - 1;
                double old_val = u_local[i * n + j];
                u_local[i * n + j] = (left + right + up + down + f_local[fi * n + j]) / 4.0;

                double diff = fabs(u_local[i * n + j] - old_val);
                if (diff > local_max_diff) local_max_diff = diff;
            }
        }

        double global_max_diff;
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (global_max_diff < tol) {
            iter++;
            break;
        }
    }

    return iter;
}

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n           = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;
    int max_iter    = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;
    double tol      = (argc > 3) ? atof(argv[3]) : TOLERANCE;
    int omp_threads = (argc > 4) ? atoi(argv[4]) : 2;

    if (rank == 0) {
        printf("Iterative Linear Solvers - Hybrid MPI+OpenMP Implementation\n");
        printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e\n",
               n, n, max_iter, tol);
        printf("MPI Processes: %d, OpenMP Threads/Process: %d (Total cores: %d)\n\n",
               size, omp_threads, size * omp_threads);
    }

    /* Row distribution */
    int rows_per_proc = n / size;
    int extra = n % size;
    int local_rows = rows_per_proc + (rank < extra ? 1 : 0);
    int global_row_start = rank * rows_per_proc + (rank < extra ? rank : extra);

    /* Allocate local arrays */
    double *f_local = (double *)malloc((size_t)local_rows * n * sizeof(double));
    double *u_jac   = (double *)calloc((size_t)(local_rows + 2) * n, sizeof(double));
    double *u_gs    = (double *)calloc((size_t)(local_rows + 2) * n, sizeof(double));

    double *f_global     = NULL;
    double *u_global_jac = NULL;
    double *u_global_gs  = NULL;
    double *u_exact      = NULL;

    if (rank == 0) {
        f_global     = (double *)malloc((size_t)n * n * sizeof(double));
        u_global_jac = (double *)malloc((size_t)n * n * sizeof(double));
        u_global_gs  = (double *)malloc((size_t)n * n * sizeof(double));
        u_exact      = (double *)malloc((size_t)n * n * sizeof(double));
        init_rhs(f_global, n);
        exact_solution(u_exact, n);
    }

    /* Scatter f */
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
                 f_local, local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ---- Jacobi ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();
    int jac_iters = jacobi_hybrid(u_jac, f_local, local_rows, n,
                                   max_iter, tol, rank, size, omp_threads);
    double t_jac = MPI_Wtime() - t_start;

    MPI_Gatherv(&u_jac[1 * n], local_rows * n, MPI_DOUBLE,
                u_global_jac, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ---- Red-Black GS ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();
    int gs_iters = redblack_gs_hybrid(u_gs, f_local, local_rows, n,
                                       max_iter, tol, rank, size,
                                       global_row_start, omp_threads);
    double t_gs = MPI_Wtime() - t_start;

    MPI_Gatherv(&u_gs[1 * n], local_rows * n, MPI_DOUBLE,
                u_global_gs, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double max_t_jac, max_t_gs;
    MPI_Reduce(&t_jac, &max_t_jac, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_gs, &max_t_gs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double res_jac  = compute_residual(u_global_jac, f_global, n);
        double rmse_jac = compute_rmse(u_global_jac, u_exact, n);
        print_results("Jacobi", "Hybrid MPI+OpenMP", n, jac_iters, max_t_jac, res_jac, rmse_jac);

        double res_gs  = compute_residual(u_global_gs, f_global, n);
        double rmse_gs = compute_rmse(u_global_gs, u_exact, n);
        print_results("Red-Black Gauss-Seidel", "Hybrid MPI+OpenMP", n, gs_iters, max_t_gs, res_gs, rmse_gs);

        double rmse_jac_vs_gs = compute_rmse(u_global_jac, u_global_gs, n);
        printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);

        free(f_global);
        free(u_global_jac);
        free(u_global_gs);
        free(u_exact);
    }

    free(f_local);
    free(u_jac);
    free(u_gs);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
