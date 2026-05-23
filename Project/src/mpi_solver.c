/**
 * mpi_solver.c - MPI implementations of Jacobi and Red-Black Gauss-Seidel
 *
 * Distributed memory parallelization using MPI.
 * The grid is decomposed by rows: each process owns a contiguous band of rows.
 * Halo (ghost) rows are exchanged between neighbors using MPI_Sendrecv.
 *
 * Usage: mpirun -np <procs> ./mpi_solver [grid_size] [max_iterations] [tolerance]
 */

#include "../include/common.h"  /* Include shared utilities: grid setup, timing, error computation */
#include <mpi.h>                /* MPI header: provides distributed memory communication functions */

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
               int max_iter, double tol, int rank, int size) { /* MPI Jacobi: u_local includes ghost rows, f_local has only owned rows */
    /* u_local has (local_rows + 2) rows: [ghost_top | local_rows | ghost_bottom] */
    /* f_local has local_rows rows */
    int total_local = (local_rows + 2) * n;  /* Total elements in u_local including 2 ghost rows */
    double *u_old = (double *)calloc(total_local, sizeof(double));  /* Allocate old values array (same size as u_local with ghost rows) */
    if (!u_old) { fprintf(stderr, "Rank %d: allocation failed\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }  /* Abort all processes if allocation fails */

    int prev = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;  /* Previous process rank (MPI_PROC_NULL if this is the first process) */
    int next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;  /* Next process rank (MPI_PROC_NULL if this is the last process) */

    int iter;  /* Iteration counter */
    for (iter = 0; iter < max_iter; iter++) {  /* Main iteration loop */
        memcpy(u_old, u_local, total_local * sizeof(double));  /* Copy current solution (with ghosts) to old array */

        /* Exchange ghost rows */
        /* Send my first real row up, receive into bottom ghost from below */
        MPI_Sendrecv(&u_old[1 * n], n, MPI_DOUBLE, prev, 0,            /* Send first real row (row 1) to previous process with tag 0 */
                     &u_old[(local_rows + 1) * n], n, MPI_DOUBLE, next, 0,  /* Receive into bottom ghost row from next process with tag 0 */
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);                /* Use default communicator, ignore status */
        /* Send my last real row down, receive into top ghost from above */
        MPI_Sendrecv(&u_old[local_rows * n], n, MPI_DOUBLE, next, 1,   /* Send last real row to next process with tag 1 */
                     &u_old[0], n, MPI_DOUBLE, prev, 1,                 /* Receive into top ghost row (row 0) from previous process */
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);                /* Use default communicator, ignore status */

        double local_max_diff = 0.0;  /* Track maximum change within this process's domain */

        for (int i = 1; i <= local_rows; i++) {  /* Loop over local real rows (rows 1 to local_rows, skipping ghost rows 0 and local_rows+1) */
            for (int j = 0; j < n; j++) {  /* Loop over each column */
                double left  = (j > 0)     ? u_old[i * n + (j - 1)] : 0.0;  /* Left neighbor from old values (0 at left boundary) */
                double right = (j < n - 1) ? u_old[i * n + (j + 1)] : 0.0;  /* Right neighbor from old values (0 at right boundary) */
                double up    = u_old[(i - 1) * n + j];   /* Upper neighbor (may be ghost row from previous process) */
                double down  = u_old[(i + 1) * n + j];   /* Lower neighbor (may be ghost row from next process) */

                int fi = i - 1; /* index into f_local: offset by 1 because f_local has no ghost rows */
                u_local[i * n + j] = (left + right + up + down + f_local[fi * n + j]) / 4.0;  /* Compute new value using Jacobi formula */

                double diff = fabs(u_local[i * n + j] - u_old[i * n + j]);  /* Compute change at this point */
                if (diff > local_max_diff) local_max_diff = diff;  /* Update local maximum change */
            }
        }

        double global_max_diff;  /* Will hold the global maximum change across all processes */
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);  /* Reduce local max to global max across all processes (all processes get the result) */

        if (global_max_diff < tol) {  /* Check if global convergence criterion is met */
            iter++;  /* Count this completed iteration */
            break;   /* Exit loop: convergence achieved */
        }
    }

    free(u_old);   /* Free temporary old values array */
    return iter;   /* Return number of iterations performed */
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
                    int max_iter, double tol, int rank, int size, int global_row_start) { /* MPI Red-Black GS: global_row_start maps local rows to global indices */
    int prev = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;  /* Previous process (MPI_PROC_NULL if first) */
    int next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;  /* Next process (MPI_PROC_NULL if last) */

    int iter;  /* Iteration counter */
    for (iter = 0; iter < max_iter; iter++) {  /* Main iteration loop */
        double local_max_diff = 0.0;  /* Track maximum change within this process */

        /* --- Phase 1: RED points --- */
        /* Exchange ghost rows before red update */
        MPI_Sendrecv(&u_local[1 * n], n, MPI_DOUBLE, prev, 0,            /* Send first real row to previous process */
                     &u_local[(local_rows + 1) * n], n, MPI_DOUBLE, next, 0,  /* Receive bottom ghost from next process */
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);                  /* Default communicator */
        MPI_Sendrecv(&u_local[local_rows * n], n, MPI_DOUBLE, next, 1,   /* Send last real row to next process */
                     &u_local[0], n, MPI_DOUBLE, prev, 1,                 /* Receive top ghost from previous process */
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);                  /* Default communicator */

        for (int i = 1; i <= local_rows; i++) {  /* Loop over local real rows */
            int gi = global_row_start + (i - 1); /* global row index: maps local row i to its position in the full grid */
            for (int j = 0; j < n; j++) {  /* Loop over columns */
                if ((gi + j) % 2 != 0) continue; /* skip black: only update red points where (global_row + col) is even */

                double left  = (j > 0)     ? u_local[i * n + (j - 1)] : 0.0;  /* Left neighbor (0 at left boundary) */
                double right = (j < n - 1) ? u_local[i * n + (j + 1)] : 0.0;  /* Right neighbor (0 at right boundary) */
                double up    = u_local[(i - 1) * n + j];  /* Upper neighbor (possibly from ghost row) */
                double down  = u_local[(i + 1) * n + j];  /* Lower neighbor (possibly from ghost row) */

                int fi = i - 1;  /* Index into f_local (no ghost row offset) */
                double old_val = u_local[i * n + j];  /* Save current value before update */
                u_local[i * n + j] = (left + right + up + down + f_local[fi * n + j]) / 4.0;  /* Update red point in-place */

                double diff = fabs(u_local[i * n + j] - old_val);  /* Compute change at this point */
                if (diff > local_max_diff) local_max_diff = diff;  /* Update local max change */
            }
        }

        /* --- Phase 2: BLACK points --- */
        /* Exchange ghost rows before black update (red values have been updated) */
        MPI_Sendrecv(&u_local[1 * n], n, MPI_DOUBLE, prev, 2,            /* Send updated first real row to previous process (tag 2) */
                     &u_local[(local_rows + 1) * n], n, MPI_DOUBLE, next, 2,  /* Receive updated bottom ghost from next process */
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);                  /* Default communicator */
        MPI_Sendrecv(&u_local[local_rows * n], n, MPI_DOUBLE, next, 3,   /* Send updated last real row to next process (tag 3) */
                     &u_local[0], n, MPI_DOUBLE, prev, 3,                 /* Receive updated top ghost from previous process */
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);                  /* Default communicator */

        for (int i = 1; i <= local_rows; i++) {  /* Loop over local real rows */
            int gi = global_row_start + (i - 1);  /* Compute global row index for color determination */
            for (int j = 0; j < n; j++) {  /* Loop over columns */
                if ((gi + j) % 2 != 1) continue; /* skip red: only update black points where (global_row + col) is odd */

                double left  = (j > 0)     ? u_local[i * n + (j - 1)] : 0.0;  /* Left neighbor (uses updated red values) */
                double right = (j < n - 1) ? u_local[i * n + (j + 1)] : 0.0;  /* Right neighbor (uses updated red values) */
                double up    = u_local[(i - 1) * n + j];  /* Upper neighbor (uses updated red values from ghost) */
                double down  = u_local[(i + 1) * n + j];  /* Lower neighbor (uses updated red values from ghost) */

                int fi = i - 1;  /* Index into f_local */
                double old_val = u_local[i * n + j];  /* Save current black point value */
                u_local[i * n + j] = (left + right + up + down + f_local[fi * n + j]) / 4.0;  /* Update black point using updated red neighbors */

                double diff = fabs(u_local[i * n + j] - old_val);  /* Compute change */
                if (diff > local_max_diff) local_max_diff = diff;  /* Update local max change */
            }
        }

        double global_max_diff;  /* Will store global maximum change */
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);  /* Compute global max change across all processes */

        if (global_max_diff < tol) {  /* Check global convergence */
            iter++;  /* Count completed iteration */
            break;   /* Exit: converged */
        }
    }

    return iter;  /* Return iteration count */
}

int main(int argc, char *argv[]) {  /* Entry point */
    MPI_Init(&argc, &argv);  /* Initialize MPI runtime (must be called before any other MPI function) */

    int rank, size;  /* rank = this process's ID, size = total number of processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  /* Get this process's rank (0 to size-1) */
    MPI_Comm_size(MPI_COMM_WORLD, &size);  /* Get total number of processes */

    int n        = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;  /* Parse grid size or use default */
    int max_iter = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;     /* Parse max iterations or use default */
    double tol   = (argc > 3) ? atof(argv[3]) : TOLERANCE;          /* Parse tolerance or use default */

    if (rank == 0) {  /* Only rank 0 (master) prints the header */
        printf("Iterative Linear Solvers - MPI Implementation\n");  /* Print program header */
        printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e, Processes: %d\n\n",
               n, n, max_iter, tol, size);  /* Print configuration */
    }

    /* Distribute rows across processes */
    int rows_per_proc = n / size;  /* Base number of rows per process */
    int extra = n % size;  /* Number of processes that get one extra row (for uneven division) */
    int local_rows = rows_per_proc + (rank < extra ? 1 : 0);  /* This process's row count (first 'extra' processes get one more) */
    int global_row_start = rank * rows_per_proc + (rank < extra ? rank : extra);  /* Starting global row index for this process */

    /* Allocate local arrays with ghost rows */
    double *f_local   = (double *)malloc((size_t)local_rows * n * sizeof(double));       /* Local RHS (no ghost rows needed) */
    double *u_jac     = (double *)calloc((size_t)(local_rows + 2) * n, sizeof(double));  /* Local Jacobi solution with top and bottom ghost rows */
    double *u_gs      = (double *)calloc((size_t)(local_rows + 2) * n, sizeof(double));  /* Local GS solution with ghost rows */

    /* Global arrays on rank 0 for gathering results */
    double *f_global     = NULL;  /* Full RHS vector (only allocated on rank 0) */
    double *u_global_jac = NULL;  /* Full Jacobi solution (only on rank 0) */
    double *u_global_gs  = NULL;  /* Full GS solution (only on rank 0) */
    double *u_exact      = NULL;  /* Exact solution (only on rank 0) */

    if (rank == 0) {  /* Rank 0 allocates and initializes global arrays */
        f_global     = (double *)malloc((size_t)n * n * sizeof(double));  /* Allocate full RHS */
        u_global_jac = (double *)malloc((size_t)n * n * sizeof(double));  /* Allocate full Jacobi result */
        u_global_gs  = (double *)malloc((size_t)n * n * sizeof(double));  /* Allocate full GS result */
        u_exact      = (double *)malloc((size_t)n * n * sizeof(double));  /* Allocate exact solution */
        init_rhs(f_global, n);         /* Initialize the full RHS vector */
        exact_solution(u_exact, n);    /* Compute the exact analytical solution */
    }

    /* Scatter f to all processes */
    int *sendcounts = (int *)malloc(size * sizeof(int));  /* Array of element counts to send to each process */
    int *displs     = (int *)malloc(size * sizeof(int));  /* Array of displacements (offsets) for each process */
    int offset = 0;  /* Running offset for displacement calculation */
    for (int p = 0; p < size; p++) {  /* Calculate send counts and displacements for each process */
        int lr = rows_per_proc + (p < extra ? 1 : 0);  /* Number of rows for process p */
        sendcounts[p] = lr * n;  /* Number of elements (rows * columns) for process p */
        displs[p] = offset;  /* Starting position in the global array */
        offset += lr * n;  /* Advance offset by this process's element count */
    }

    MPI_Scatterv(f_global, sendcounts, displs, MPI_DOUBLE,  /* Scatter RHS from rank 0 to all processes */
                 f_local, local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);  /* Each process receives its portion of f */

    /* ---- Jacobi ---- */
    MPI_Barrier(MPI_COMM_WORLD);  /* Synchronize all processes before timing */
    double t_start = MPI_Wtime();  /* Record start time using MPI's wall clock timer */
    int jac_iters = jacobi_mpi(u_jac, f_local, local_rows, n, max_iter, tol, rank, size);  /* Run MPI Jacobi solver */
    double t_jac = MPI_Wtime() - t_start;  /* Compute elapsed time */

    /* Gather Jacobi results (real rows only, offset by 1 for ghost) */
    double *u_jac_real = &u_jac[1 * n];  /* Pointer to first real row (skip top ghost row) */
    MPI_Gatherv(u_jac_real, local_rows * n, MPI_DOUBLE,  /* Gather local real rows */
                u_global_jac, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);  /* Assemble into global array on rank 0 */

    /* ---- Red-Black GS ---- */
    MPI_Barrier(MPI_COMM_WORLD);  /* Synchronize before timing GS */
    t_start = MPI_Wtime();  /* Record start time */
    int gs_iters = redblack_gs_mpi(u_gs, f_local, local_rows, n, max_iter, tol,
                                    rank, size, global_row_start);  /* Run MPI Red-Black GS solver */
    double t_gs = MPI_Wtime() - t_start;  /* Compute elapsed time */

    double *u_gs_real = &u_gs[1 * n];  /* Pointer to first real row of GS solution */
    MPI_Gatherv(u_gs_real, local_rows * n, MPI_DOUBLE,  /* Gather GS local results */
                u_global_gs, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);  /* Assemble on rank 0 */

    /* Get max time across processes */
    double max_t_jac, max_t_gs;  /* Will hold the slowest process's time (bottleneck) */
    MPI_Reduce(&t_jac, &max_t_jac, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);  /* Find max Jacobi time across all processes */
    MPI_Reduce(&t_gs, &max_t_gs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);    /* Find max GS time across all processes */

    /* Print results on rank 0 */
    if (rank == 0) {  /* Only master process prints results */
        double res_jac  = compute_residual(u_global_jac, f_global, n);    /* Compute Jacobi residual on full solution */
        double rmse_jac = compute_rmse(u_global_jac, u_exact, n);         /* Compute Jacobi RMSE vs exact */
        print_results("Jacobi", "MPI", n, jac_iters, max_t_jac, res_jac, rmse_jac);  /* Print Jacobi results */

        double res_gs  = compute_residual(u_global_gs, f_global, n);    /* Compute GS residual */
        double rmse_gs = compute_rmse(u_global_gs, u_exact, n);         /* Compute GS RMSE vs exact */
        print_results("Red-Black Gauss-Seidel", "MPI", n, gs_iters, max_t_gs, res_gs, rmse_gs);  /* Print GS results */

        double rmse_jac_vs_gs = compute_rmse(u_global_jac, u_global_gs, n);  /* Compare the two solutions */
        printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);    /* Print comparison */

        save_solution("mpi_jacobi.txt", u_global_jac, n);
        save_solution("mpi_rbgs.txt", u_global_gs, n);

        free(f_global);      /* Free global RHS */
        free(u_global_jac);  /* Free global Jacobi result */
        free(u_global_gs);   /* Free global GS result */
        free(u_exact);       /* Free exact solution */
    }

    free(f_local);    /* Free local RHS array */
    free(u_jac);      /* Free local Jacobi array (including ghost rows) */
    free(u_gs);       /* Free local GS array (including ghost rows) */
    free(sendcounts); /* Free send counts array */
    free(displs);     /* Free displacements array */

    MPI_Finalize();  /* Finalize MPI runtime (must be called before exit) */
    return 0;        /* Exit successfully */
}
