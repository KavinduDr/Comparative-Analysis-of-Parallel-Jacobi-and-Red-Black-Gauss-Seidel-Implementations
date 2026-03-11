/**
 * openmp_solver.c
 *
 * Parallel iterative PDE solvers using OpenMP for shared-memory systems.
 * Implements both the Jacobi method and the Red-Black Gauss-Seidel method
 * for solving the 2D Poisson equation on a uniform grid.
 *
 * Compilation requires OpenMP support (e.g., gcc -fopenmp).
 * Usage: ./openmp_solver [grid_size] [max_iterations] [tolerance] [num_threads]
 */

#include "../include/common.h"  /* Include shared utilities: grid setup, timing, error computation */
#include <omp.h>                /* OpenMP header: provides parallel directives and runtime functions */

/**
 * OpenMP Jacobi Method
 *
 * The outer loop over grid rows is parallelized with OpenMP.
 * Each thread computes a subset of rows independently since Jacobi
 * only reads from the old array (no data dependencies within an iteration).
 * A reduction is used to compute the global maximum change.
 */
int jacobi_openmp(double *u, const double *f, int n, int max_iter, double tol, int num_threads) { /* OpenMP Jacobi solver with configurable thread count */
    double *u_old = (double *)calloc((size_t)n * n, sizeof(double));  /* Allocate and zero-initialize array to store previous iteration values */
    if (!u_old) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }  /* Exit if allocation fails */

    omp_set_num_threads(num_threads);  /* Set the number of OpenMP threads to use */

    int iter;  /* Iteration counter */
    for (iter = 0; iter < max_iter; iter++) {  /* Main iteration loop */
        memcpy(u_old, u, (size_t)n * n * sizeof(double));  /* Copy current solution to old (sequential - needed before parallel region) */

        double max_diff = 0.0;  /* Track the maximum change across all grid points */

        #pragma omp parallel for collapse(2) reduction(max:max_diff) schedule(static)  /* Parallelize both i and j loops; reduce max_diff across threads; static schedule divides work evenly */
        for (int i = 0; i < n; i++) {  /* Loop over rows (distributed across threads) */
            for (int j = 0; j < n; j++) {  /* Loop over columns (also distributed due to collapse(2)) */
                double left  = (j > 0)     ? u_old[i * n + (j - 1)] : 0.0;  /* Left neighbor from old values (boundary = 0) */
                double right = (j < n - 1) ? u_old[i * n + (j + 1)] : 0.0;  /* Right neighbor from old values (boundary = 0) */
                double up    = (i > 0)     ? u_old[(i - 1) * n + j]  : 0.0;  /* Upper neighbor from old values (boundary = 0) */
                double down  = (i < n - 1) ? u_old[(i + 1) * n + j]  : 0.0;  /* Lower neighbor from old values (boundary = 0) */

                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;  /* Compute new value using Jacobi formula */

                double diff = fabs(u[i * n + j] - u_old[i * n + j]);  /* Compute absolute change at this point */
                if (diff > max_diff) max_diff = diff;  /* Update thread-local max (reduced globally after loop) */
            }
        }

        if (max_diff < tol) {  /* Check global convergence (max_diff is the reduced maximum) */
            iter++;  /* Count this completed iteration */
            break;   /* Convergence achieved: exit loop */
        }
    }

    free(u_old);   /* Free the temporary old values array */
    return iter;   /* Return number of iterations performed */
}

/**
 * OpenMP Red-Black Gauss-Seidel Method
 *
 * Phase 1 (Red): All points where (i+j) is even are updated in parallel.
 *   Within this phase there are NO data dependencies (red neighbors are all black).
 * Phase 2 (Black): All points where (i+j) is odd are updated in parallel.
 *   Black neighbors are all red (already updated).
 *
 * An implicit barrier between phases ensures correctness.
 */
int redblack_gs_openmp(double *u, const double *f, int n, int max_iter, double tol, int num_threads) { /* OpenMP Red-Black GS solver */
    omp_set_num_threads(num_threads);  /* Set the number of OpenMP threads */

    int iter;  /* Iteration counter */
    for (iter = 0; iter < max_iter; iter++) {  /* Main iteration loop */
        double max_diff = 0.0;  /* Track maximum change across both phases */

        /* Phase 1: Update RED points */
        #pragma omp parallel for collapse(2) reduction(max:max_diff) schedule(static)  /* Parallelize red point updates; reduce max_diff; static scheduling */
        for (int i = 0; i < n; i++) {  /* Loop over rows (distributed across threads) */
            for (int j = 0; j < n; j++) {  /* Loop over columns */
                if ((i + j) % 2 != 0) continue;  /* Skip black points: only update red points where (i+j) is even */

                double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;  /* Left neighbor (black point, not yet updated this iteration) */
                double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;  /* Right neighbor (black point) */
                double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;  /* Upper neighbor (black point) */
                double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;  /* Lower neighbor (black point) */

                double old_val = u[i * n + j];  /* Save current value before updating */
                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;  /* Update red point in-place */

                double diff = fabs(u[i * n + j] - old_val);  /* Compute change at this red point */
                if (diff > max_diff) max_diff = diff;  /* Update max change (thread-safe via reduction) */
            }
        }
        /* Implicit barrier here ensures all red points are updated before black phase */

        /* Phase 2: Update BLACK points */
        #pragma omp parallel for collapse(2) reduction(max:max_diff) schedule(static)  /* Parallelize black point updates; reduce max_diff */
        for (int i = 0; i < n; i++) {  /* Loop over rows */
            for (int j = 0; j < n; j++) {  /* Loop over columns */
                if ((i + j) % 2 != 1) continue;  /* Skip red points: only update black points where (i+j) is odd */

                double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;  /* Left neighbor (red point, freshly updated) */
                double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;  /* Right neighbor (red point, freshly updated) */
                double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;  /* Upper neighbor (red point, freshly updated) */
                double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;  /* Lower neighbor (red point, freshly updated) */

                double old_val = u[i * n + j];  /* Save current value before updating */
                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;  /* Update black point using new red neighbor values */

                double diff = fabs(u[i * n + j] - old_val);  /* Compute change at this black point */
                if (diff > max_diff) max_diff = diff;  /* Update max change */
            }
        }

        if (max_diff < tol) {  /* Check convergence after both phases */
            iter++;  /* Count this completed iteration */
            break;   /* Exit loop: convergence achieved */
        }
    }

    return iter;  /* Return number of iterations performed */
}

int main(int argc, char *argv[]) {  /* Entry point */
    int n           = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;  /* Parse grid size or use default (100) */
    int max_iter    = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;     /* Parse max iterations or use default (10000) */
    double tol      = (argc > 3) ? atof(argv[3]) : TOLERANCE;          /* Parse tolerance or use default (1e-6) */
    int num_threads = (argc > 4) ? atoi(argv[4]) : 4;                  /* Parse number of threads or use default (4) */

    printf("Iterative Linear Solvers - OpenMP Implementation\n");  /* Print program header */
    printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e, Threads: %d\n\n",
           n, n, max_iter, tol, num_threads);  /* Print configuration including thread count */

    double *f       = (double *)malloc((size_t)n * n * sizeof(double));   /* Allocate RHS vector */
    double *u_jac   = (double *)calloc((size_t)n * n, sizeof(double));   /* Allocate and zero-initialize Jacobi solution */
    double *u_gs    = (double *)calloc((size_t)n * n, sizeof(double));   /* Allocate and zero-initialize GS solution */
    double *u_exact = (double *)malloc((size_t)n * n * sizeof(double));  /* Allocate exact solution array */

    if (!f || !u_jac || !u_gs || !u_exact) {  /* Check if any allocation failed */
        fprintf(stderr, "Memory allocation failed\n");  /* Print error */
        return 1;  /* Exit with error */
    }

    init_rhs(f, n);              /* Initialize RHS vector with problem-specific values */
    exact_solution(u_exact, n);  /* Compute exact analytical solution */

    /* ---- Jacobi ---- */
    double t_start = get_time();  /* Record start time */
    int jac_iters = jacobi_openmp(u_jac, f, n, max_iter, tol, num_threads);  /* Run OpenMP Jacobi solver */
    double t_jac = get_time() - t_start;  /* Compute elapsed time */

    double res_jac  = compute_residual(u_jac, f, n);    /* Compute residual of Jacobi solution */
    double rmse_jac = compute_rmse(u_jac, u_exact, n);  /* Compute RMSE vs exact solution */
    print_results("Jacobi", "OpenMP", n, jac_iters, t_jac, res_jac, rmse_jac);  /* Print Jacobi results */

    /* ---- Red-Black Gauss-Seidel ---- */
    t_start = get_time();  /* Record start time for GS */
    int gs_iters = redblack_gs_openmp(u_gs, f, n, max_iter, tol, num_threads);  /* Run OpenMP Red-Black GS solver */
    double t_gs = get_time() - t_start;  /* Compute elapsed time */

    double res_gs  = compute_residual(u_gs, f, n);    /* Compute residual of GS solution */
    double rmse_gs = compute_rmse(u_gs, u_exact, n);  /* Compute RMSE vs exact solution */
    print_results("Red-Black Gauss-Seidel", "OpenMP", n, gs_iters, t_gs, res_gs, rmse_gs);  /* Print GS results */

    double rmse_jac_vs_gs = compute_rmse(u_jac, u_gs, n);  /* Compute RMSE between the two solutions */
    printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);  /* Print comparison metric */

    free(f);        /* Free RHS vector */
    free(u_jac);    /* Free Jacobi solution */
    free(u_gs);     /* Free GS solution */
    free(u_exact);  /* Free exact solution */

    return 0;  /* Exit successfully */
}
