/**
 * hybrid_solver.c - Hybrid MPI + OpenMP implementation
 *
 * Combines distributed memory (MPI row decomposition) with shared memory
 * (OpenMP loop parallelization within each MPI process).
 *
 * Usage: mpirun -np <procs> ./hybrid_solver [grid_size] [max_iter] [tol] [omp_threads]
 */

#include "../include/common.h"  /* Include shared utilities: grid setup, timing, error computation */
#include <mpi.h>                /* MPI header: distributed memory communication */
#include <omp.h>                /* OpenMP header: shared memory parallelization directives */

/**
 * Hybrid Jacobi Method
 *
 * MPI: Row-wise domain decomposition with ghost row exchange.
 * OpenMP: Parallel for over local rows within each MPI process.
 */
int jacobi_hybrid(double *u_local, double *f_local, int local_rows, int n,
                  int max_iter, double tol, int rank, int size, int omp_threads) { /* Hybrid Jacobi: MPI between processes, OpenMP within each process */
    int total_local = (local_rows + 2) * n;  /* Total elements including top and bottom ghost rows */
    double *u_old = (double *)calloc(total_local, sizeof(double));  /* Allocate old values array with ghost rows */
    if (!u_old) { fprintf(stderr, "Rank %d: allocation failed\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }  /* Abort all if allocation fails */

    omp_set_num_threads(omp_threads);  /* Set number of OpenMP threads for this process */

    int prev = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;  /* Previous MPI process (null if first) */
    int next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;  /* Next MPI process (null if last) */

    int iter;  /* Iteration counter */
    for (iter = 0; iter < max_iter; iter++) {  /* Main iteration loop */
        memcpy(u_old, u_local, total_local * sizeof(double));  /* Copy current solution to old array */

        /* Exchange ghost rows (MPI communication - done by master thread) */
        MPI_Sendrecv(&u_old[1 * n], n, MPI_DOUBLE, prev, 0,            /* Send first real row upward to previous process */
                     &u_old[(local_rows + 1) * n], n, MPI_DOUBLE, next, 0,  /* Receive bottom ghost row from next process */
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);                /* Default communicator */
        MPI_Sendrecv(&u_old[local_rows * n], n, MPI_DOUBLE, next, 1,   /* Send last real row downward to next process */
                     &u_old[0], n, MPI_DOUBLE, prev, 1,                 /* Receive top ghost row from previous process */
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);                /* Default communicator */

        /* OpenMP parallel computation over local rows */
        double local_max_diff = 0.0;  /* Track maximum change within this process */

        #pragma omp parallel for collapse(2) reduction(max:local_max_diff) schedule(static)  /* Parallelize loops with OpenMP; reduce max diff; static load balancing */
        for (int i = 1; i <= local_rows; i++) {  /* Loop over real rows (skip ghost rows 0 and local_rows+1) */
            for (int j = 0; j < n; j++) {  /* Loop over columns */
                double left  = (j > 0)     ? u_old[i * n + (j - 1)] : 0.0;  /* Left neighbor from old values */
                double right = (j < n - 1) ? u_old[i * n + (j + 1)] : 0.0;  /* Right neighbor from old values */
                double up    = u_old[(i - 1) * n + j];  /* Upper neighbor (may be ghost from previous process) */
                double down  = u_old[(i + 1) * n + j];  /* Lower neighbor (may be ghost from next process) */

                int fi = i - 1;  /* Index into f_local (which has no ghost rows) */
                u_local[i * n + j] = (left + right + up + down + f_local[fi * n + j]) / 4.0;  /* Jacobi update */

                double diff = fabs(u_local[i * n + j] - u_old[i * n + j]);  /* Change at this point */
                if (diff > local_max_diff) local_max_diff = diff;  /* Update thread-local max (reduced by OpenMP) */
            }
        }

        /* Global convergence check (MPI) */
        double global_max_diff;  /* Will hold the maximum change across all processes */
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);  /* Find global max across all MPI processes */

        if (global_max_diff < tol) {  /* Check convergence */
            iter++;  /* Count completed iteration */
            break;   /* Exit: converged */
        }
    }

    free(u_old);   /* Free old values array */
    return iter;   /* Return iteration count */
}

/**
 * Hybrid Red-Black Gauss-Seidel Method
 *
 * MPI: Row decomposition + ghost row exchange between phases.
 * OpenMP: Parallel for within each color phase.
 */
int redblack_gs_hybrid(double *u_local, double *f_local, int local_rows, int n,
                       int max_iter, double tol, int rank, int size,
                       int global_row_start, int omp_threads) { /* Hybrid Red-Black GS: MPI + OpenMP */
    omp_set_num_threads(omp_threads);  /* Set OpenMP thread count */

    int prev = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;  /* Previous MPI process */
    int next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;  /* Next MPI process */

    int iter;  /* Iteration counter */
    for (iter = 0; iter < max_iter; iter++) {  /* Main iteration loop */
        double local_max_diff = 0.0;  /* Track max change in this process */

        /* Exchange ghost rows before RED phase */
        MPI_Sendrecv(&u_local[1 * n], n, MPI_DOUBLE, prev, 0,            /* Send first real row to previous process */
                     &u_local[(local_rows + 1) * n], n, MPI_DOUBLE, next, 0,  /* Receive bottom ghost from next process */
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);                  /* Default communicator */
        MPI_Sendrecv(&u_local[local_rows * n], n, MPI_DOUBLE, next, 1,   /* Send last real row to next process */
                     &u_local[0], n, MPI_DOUBLE, prev, 1,                 /* Receive top ghost from previous process */
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);                  /* Default communicator */

        /* Phase 1: RED points - OpenMP parallel */
        #pragma omp parallel for collapse(2) reduction(max:local_max_diff) schedule(static)  /* Parallelize red update with OpenMP */
        for (int i = 1; i <= local_rows; i++) {  /* Loop over real rows */
            for (int j = 0; j < n; j++) {  /* Loop over columns */
                int gi = global_row_start + (i - 1);  /* Compute global row index for color determination */
                if ((gi + j) % 2 != 0) continue;  /* Skip black points */

                double left  = (j > 0)     ? u_local[i * n + (j - 1)] : 0.0;  /* Left neighbor */
                double right = (j < n - 1) ? u_local[i * n + (j + 1)] : 0.0;  /* Right neighbor */
                double up    = u_local[(i - 1) * n + j];  /* Upper neighbor (possibly ghost) */
                double down  = u_local[(i + 1) * n + j];  /* Lower neighbor (possibly ghost) */

                int fi = i - 1;  /* Index into f_local */
                double old_val = u_local[i * n + j];  /* Save current value */
                u_local[i * n + j] = (left + right + up + down + f_local[fi * n + j]) / 4.0;  /* Update red point */

                double diff = fabs(u_local[i * n + j] - old_val);  /* Compute change */
                if (diff > local_max_diff) local_max_diff = diff;  /* Update max change */
            }
        }

        /* Exchange ghost rows before BLACK phase */
        MPI_Sendrecv(&u_local[1 * n], n, MPI_DOUBLE, prev, 2,            /* Send updated first row to previous process */
                     &u_local[(local_rows + 1) * n], n, MPI_DOUBLE, next, 2,  /* Receive updated bottom ghost */
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);                  /* Default communicator */
        MPI_Sendrecv(&u_local[local_rows * n], n, MPI_DOUBLE, next, 3,   /* Send updated last row to next process */
                     &u_local[0], n, MPI_DOUBLE, prev, 3,                 /* Receive updated top ghost */
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);                  /* Default communicator */

        /* Phase 2: BLACK points - OpenMP parallel */
        #pragma omp parallel for collapse(2) reduction(max:local_max_diff) schedule(static)  /* Parallelize black update with OpenMP */
        for (int i = 1; i <= local_rows; i++) {  /* Loop over real rows */
            for (int j = 0; j < n; j++) {  /* Loop over columns */
                int gi = global_row_start + (i - 1);  /* Compute global row index */
                if ((gi + j) % 2 != 1) continue;  /* Skip red points */

                double left  = (j > 0)     ? u_local[i * n + (j - 1)] : 0.0;  /* Left neighbor (updated red) */
                double right = (j < n - 1) ? u_local[i * n + (j + 1)] : 0.0;  /* Right neighbor (updated red) */
                double up    = u_local[(i - 1) * n + j];  /* Upper neighbor (updated red from ghost) */
                double down  = u_local[(i + 1) * n + j];  /* Lower neighbor (updated red from ghost) */

                int fi = i - 1;  /* Index into f_local */
                double old_val = u_local[i * n + j];  /* Save current value */
                u_local[i * n + j] = (left + right + up + down + f_local[fi * n + j]) / 4.0;  /* Update black point */

                double diff = fabs(u_local[i * n + j] - old_val);  /* Compute change */
                if (diff > local_max_diff) local_max_diff = diff;  /* Update max change */
            }
        }

        double global_max_diff;  /* Global max change across all processes */
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);  /* MPI reduce to find global max */

        if (global_max_diff < tol) {  /* Check convergence */
            iter++;  /* Count completed iteration */
            break;   /* Exit: converged */
        }
    }

    return iter;  /* Return iteration count */
}

int main(int argc, char *argv[]) {  /* Entry point */
    int provided;  /* Will hold the level of MPI thread support actually provided */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);  /* Initialize MPI with thread support (FUNNELED = only master thread makes MPI calls) */

    int rank, size;  /* rank = process ID, size = total processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  /* Get this process's rank */
    MPI_Comm_size(MPI_COMM_WORLD, &size);  /* Get total number of processes */

    int n           = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;  /* Parse grid size or default */
    int max_iter    = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;     /* Parse max iterations or default */
    double tol      = (argc > 3) ? atof(argv[3]) : TOLERANCE;          /* Parse tolerance or default */
    int omp_threads = (argc > 4) ? atoi(argv[4]) : 2;                  /* Parse OpenMP threads per process or default (2) */

    if (rank == 0) {  /* Only master process prints header */
        printf("Iterative Linear Solvers - Hybrid MPI+OpenMP Implementation\n");  /* Print header */
        printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e\n",
               n, n, max_iter, tol);  /* Print grid and solver parameters */
        printf("MPI Processes: %d, OpenMP Threads/Process: %d (Total cores: %d)\n\n",
               size, omp_threads, size * omp_threads);  /* Print parallelism configuration */
    }

    /* Row distribution */
    int rows_per_proc = n / size;  /* Base rows per process */
    int extra = n % size;  /* Extra rows for first few processes */
    int local_rows = rows_per_proc + (rank < extra ? 1 : 0);  /* This process's row count */
    int global_row_start = rank * rows_per_proc + (rank < extra ? rank : extra);  /* Starting global row index */

    /* Allocate local arrays */
    double *f_local = (double *)malloc((size_t)local_rows * n * sizeof(double));       /* Local RHS (no ghost rows) */
    double *u_jac   = (double *)calloc((size_t)(local_rows + 2) * n, sizeof(double));  /* Local Jacobi solution with ghost rows */
    double *u_gs    = (double *)calloc((size_t)(local_rows + 2) * n, sizeof(double));  /* Local GS solution with ghost rows */

    double *f_global     = NULL;  /* Full RHS (rank 0 only) */
    double *u_global_jac = NULL;  /* Full Jacobi result (rank 0 only) */
    double *u_global_gs  = NULL;  /* Full GS result (rank 0 only) */
    double *u_exact      = NULL;  /* Exact solution (rank 0 only) */

    if (rank == 0) {  /* Master process allocates global arrays */
        f_global     = (double *)malloc((size_t)n * n * sizeof(double));  /* Allocate full RHS */
        u_global_jac = (double *)malloc((size_t)n * n * sizeof(double));  /* Allocate full Jacobi result */
        u_global_gs  = (double *)malloc((size_t)n * n * sizeof(double));  /* Allocate full GS result */
        u_exact      = (double *)malloc((size_t)n * n * sizeof(double));  /* Allocate exact solution */
        init_rhs(f_global, n);         /* Initialize full RHS */
        exact_solution(u_exact, n);    /* Compute exact solution */
    }

    /* Scatter f */
    int *sendcounts = (int *)malloc(size * sizeof(int));  /* Element counts for each process */
    int *displs     = (int *)malloc(size * sizeof(int));  /* Displacements in global array */
    int offset = 0;  /* Running offset */
    for (int p = 0; p < size; p++) {  /* Compute counts and displacements */
        int lr = rows_per_proc + (p < extra ? 1 : 0);  /* Rows for process p */
        sendcounts[p] = lr * n;  /* Elements for process p */
        displs[p] = offset;  /* Offset for process p */
        offset += lr * n;  /* Advance offset */
    }

    MPI_Scatterv(f_global, sendcounts, displs, MPI_DOUBLE,  /* Distribute RHS from rank 0 */
                 f_local, local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);  /* Each process receives its portion */

    /* ---- Jacobi ---- */
    MPI_Barrier(MPI_COMM_WORLD);  /* Synchronize before timing */
    double t_start = MPI_Wtime();  /* Record start time */
    int jac_iters = jacobi_hybrid(u_jac, f_local, local_rows, n,
                                   max_iter, tol, rank, size, omp_threads);  /* Run hybrid Jacobi */
    double t_jac = MPI_Wtime() - t_start;  /* Compute elapsed time */

    MPI_Gatherv(&u_jac[1 * n], local_rows * n, MPI_DOUBLE,  /* Gather real rows (skip ghost) */
                u_global_jac, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);  /* Assemble on rank 0 */

    /* ---- Red-Black GS ---- */
    MPI_Barrier(MPI_COMM_WORLD);  /* Synchronize before timing */
    t_start = MPI_Wtime();  /* Record start time */
    int gs_iters = redblack_gs_hybrid(u_gs, f_local, local_rows, n,
                                       max_iter, tol, rank, size,
                                       global_row_start, omp_threads);  /* Run hybrid Red-Black GS */
    double t_gs = MPI_Wtime() - t_start;  /* Compute elapsed time */

    MPI_Gatherv(&u_gs[1 * n], local_rows * n, MPI_DOUBLE,  /* Gather GS real rows */
                u_global_gs, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);  /* Assemble on rank 0 */

    double max_t_jac, max_t_gs;  /* Maximum time across all processes (bottleneck) */
    MPI_Reduce(&t_jac, &max_t_jac, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);  /* Find slowest Jacobi time */
    MPI_Reduce(&t_gs, &max_t_gs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);    /* Find slowest GS time */

    if (rank == 0) {  /* Only master prints results */
        double res_jac  = compute_residual(u_global_jac, f_global, n);    /* Compute Jacobi residual */
        double rmse_jac = compute_rmse(u_global_jac, u_exact, n);         /* Compute Jacobi RMSE */
        print_results("Jacobi", "Hybrid MPI+OpenMP", n, jac_iters, max_t_jac, res_jac, rmse_jac);  /* Print Jacobi results */

        double res_gs  = compute_residual(u_global_gs, f_global, n);    /* Compute GS residual */
        double rmse_gs = compute_rmse(u_global_gs, u_exact, n);         /* Compute GS RMSE */
        print_results("Red-Black Gauss-Seidel", "Hybrid MPI+OpenMP", n, gs_iters, max_t_gs, res_gs, rmse_gs);  /* Print GS results */

        double rmse_jac_vs_gs = compute_rmse(u_global_jac, u_global_gs, n);  /* Compare the two solutions */
        printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);    /* Print comparison */

        free(f_global);      /* Free global RHS */
        free(u_global_jac);  /* Free global Jacobi result */
        free(u_global_gs);   /* Free global GS result */
        free(u_exact);       /* Free exact solution */
    }

    free(f_local);    /* Free local RHS */
    free(u_jac);      /* Free local Jacobi array */
    free(u_gs);       /* Free local GS array */
    free(sendcounts); /* Free send counts */
    free(displs);     /* Free displacements */

    MPI_Finalize();  /* Finalize MPI (must be called before exit) */
    return 0;        /* Exit successfully */
}
