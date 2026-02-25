/**
 * mpi_solver.c - MPI implementations of Jacobi and Red-Black Gauss-Seidel
 *
 * Distributed memory parallelization using MPI.
 * The grid is decomposed by rows: each process owns a contiguous band of rows.
 * Halo (ghost) rows are exchanged between neighbors using MPI_Sendrecv.
 *
 * Usage: mpirun -np <procs> ./mpi_solver [grid_size] [max_iterations] [tolerance]
 */

#include "../include/common.h"
#include <mpi.h>

/**
 * MPI Jacobi Method
 *
 * Domain decomposition: process p owns rows [row_start, row_end).
 * Ghost rows: each process maintains one extra row above and below its domain
 * for values received from neighboring processes.
 *
 * Each iteration:
 *   1. Exchange ghost rows with neighbors (MPI_Sendrecv)
 *   2. Compute new values using only old values (Jacobi property)
 *   3. MPI_Allreduce to check global convergence
 */
int jacobi_mpi(double *u_local, double *f_local, int local_rows, int n,
               int max_iter, double tol, int rank, int size) {
    /* u_local has (local_rows + 2) rows: [ghost_top | local_rows | ghost_bottom] */
    /* f_local has local_rows rows */
    int total_local = (local_rows + 2) * n;
    double *u_old = (double *)calloc(total_local, sizeof(double));
    if (!u_old) { fprintf(stderr, "Rank %d: allocation failed\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }

    int prev = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        memcpy(u_old, u_local, total_local * sizeof(double));

        /* Exchange ghost rows */
        /* Send my first real row up, receive into bottom ghost from below */
        MPI_Sendrecv(&u_old[1 * n], n, MPI_DOUBLE, prev, 0,
                     &u_old[(local_rows + 1) * n], n, MPI_DOUBLE, next, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* Send my last real row down, receive into top ghost from above */
        MPI_Sendrecv(&u_old[local_rows * n], n, MPI_DOUBLE, next, 1,
                     &u_old[0], n, MPI_DOUBLE, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double local_max_diff = 0.0;

        for (int i = 1; i <= local_rows; i++) {
            for (int j = 0; j < n; j++) {
                double left  = (j > 0)     ? u_old[i * n + (j - 1)] : 0.0;
                double right = (j < n - 1) ? u_old[i * n + (j + 1)] : 0.0;
                double up    = u_old[(i - 1) * n + j];
                double down  = u_old[(i + 1) * n + j];

                int fi = i - 1; /* index into f_local */
                u_local[i * n + j] = (left + right + up + down + f_local[fi * n + j]) / 4.0;

                double diff = fabs(u_local[i * n + j] - u_old[i * n + j]);
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

    free(u_old);
    return iter;
}

/**
 * MPI Red-Black Gauss-Seidel Method
 *
 * Same domain decomposition as Jacobi.
 * Phase 1: Update red points, exchange ghost rows.
 * Phase 2: Update black points, exchange ghost rows.
 * Global convergence check via MPI_Allreduce.
 */
int redblack_gs_mpi(double *u_local, double *f_local, int local_rows, int n,
                    int max_iter, double tol, int rank, int size, int global_row_start) {
    int prev = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        double local_max_diff = 0.0;

        /* --- Phase 1: RED points --- */
        /* Exchange ghost rows before red update */
        MPI_Sendrecv(&u_local[1 * n], n, MPI_DOUBLE, prev, 0,
                     &u_local[(local_rows + 1) * n], n, MPI_DOUBLE, next, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u_local[local_rows * n], n, MPI_DOUBLE, next, 1,
                     &u_local[0], n, MPI_DOUBLE, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 1; i <= local_rows; i++) {
            int gi = global_row_start + (i - 1); /* global row index */
            for (int j = 0; j < n; j++) {
                if ((gi + j) % 2 != 0) continue; /* skip black */

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

        /* --- Phase 2: BLACK points --- */
        /* Exchange ghost rows before black update (red values have been updated) */
        MPI_Sendrecv(&u_local[1 * n], n, MPI_DOUBLE, prev, 2,
                     &u_local[(local_rows + 1) * n], n, MPI_DOUBLE, next, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u_local[local_rows * n], n, MPI_DOUBLE, next, 3,
                     &u_local[0], n, MPI_DOUBLE, prev, 3,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 1; i <= local_rows; i++) {
            int gi = global_row_start + (i - 1);
            for (int j = 0; j < n; j++) {
                if ((gi + j) % 2 != 1) continue; /* skip red */

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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n        = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;
    int max_iter = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;
    double tol   = (argc > 3) ? atof(argv[3]) : TOLERANCE;

    if (rank == 0) {
        printf("Iterative Linear Solvers - MPI Implementation\n");
        printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e, Processes: %d\n\n",
               n, n, max_iter, tol, size);
    }

    /* Distribute rows across processes */
    int rows_per_proc = n / size;
    int extra = n % size;
    int local_rows = rows_per_proc + (rank < extra ? 1 : 0);
    int global_row_start = rank * rows_per_proc + (rank < extra ? rank : extra);

    /* Allocate local arrays with ghost rows */
    double *f_local   = (double *)malloc((size_t)local_rows * n * sizeof(double));
    double *u_jac     = (double *)calloc((size_t)(local_rows + 2) * n, sizeof(double));
    double *u_gs      = (double *)calloc((size_t)(local_rows + 2) * n, sizeof(double));

    /* Global arrays on rank 0 for gathering results */
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

    /* Scatter f to all processes */
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
    int jac_iters = jacobi_mpi(u_jac, f_local, local_rows, n, max_iter, tol, rank, size);
    double t_jac = MPI_Wtime() - t_start;

    /* Gather Jacobi results (real rows only, offset by 1 for ghost) */
    double *u_jac_real = &u_jac[1 * n];
    MPI_Gatherv(u_jac_real, local_rows * n, MPI_DOUBLE,
                u_global_jac, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ---- Red-Black GS ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();
    int gs_iters = redblack_gs_mpi(u_gs, f_local, local_rows, n, max_iter, tol,
                                    rank, size, global_row_start);
    double t_gs = MPI_Wtime() - t_start;

    double *u_gs_real = &u_gs[1 * n];
    MPI_Gatherv(u_gs_real, local_rows * n, MPI_DOUBLE,
                u_global_gs, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Get max time across processes */
    double max_t_jac, max_t_gs;
    MPI_Reduce(&t_jac, &max_t_jac, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_gs, &max_t_gs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Print results on rank 0 */
    if (rank == 0) {
        double res_jac  = compute_residual(u_global_jac, f_global, n);
        double rmse_jac = compute_rmse(u_global_jac, u_exact, n);
        print_results("Jacobi", "MPI", n, jac_iters, max_t_jac, res_jac, rmse_jac);

        double res_gs  = compute_residual(u_global_gs, f_global, n);
        double rmse_gs = compute_rmse(u_global_gs, u_exact, n);
        print_results("Red-Black Gauss-Seidel", "MPI", n, gs_iters, max_t_gs, res_gs, rmse_gs);

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
