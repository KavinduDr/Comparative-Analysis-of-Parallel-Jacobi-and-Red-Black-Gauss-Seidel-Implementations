/**
 * serial.c - Serial implementations of Jacobi and Red-Black Gauss-Seidel
 *
 * Solves the 2D Poisson equation on a unit square using the 5-point stencil.
 * This serves as the baseline reference for all parallel implementations.
 *
 * Usage: ./serial [grid_size] [max_iterations] [tolerance]
 */

#include "../include/common.h"  /* Include shared utilities: grid setup, timing, error computation */

/**
 * Serial Jacobi Method
 *
 * Updates all grid points simultaneously using values from the previous
 * iteration only. Requires two arrays (old and new).
 *
 * For each interior point (i,j):
 *   u_new(i,j) = (u_old(i-1,j) + u_old(i+1,j) +
 *                  u_old(i,j-1) + u_old(i,j+1) + f(i,j)) / 4
 */
int jacobi_serial(double *u, const double *f, int n, int max_iter, double tol) { /* Jacobi solver: u=solution array, f=RHS, n=grid size, max_iter=iteration limit, tol=convergence tolerance */
    double *u_old = (double *)calloc((size_t)n * n, sizeof(double));  /* Allocate and zero-initialize array to store previous iteration values */
    if (!u_old) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }  /* Exit if memory allocation fails */

    int iter;  /* Iteration counter */
    for (iter = 0; iter < max_iter; iter++) {  /* Main iteration loop: repeat until convergence or max iterations */
        /* Copy current solution to old */
        memcpy(u_old, u, (size_t)n * n * sizeof(double));  /* Save current solution as "old" for this iteration */

        double max_diff = 0.0;  /* Track the maximum change in any grid point this iteration */

        for (int i = 0; i < n; i++) {  /* Loop over each row of the grid */
            for (int j = 0; j < n; j++) {  /* Loop over each column of the grid */
                double left  = (j > 0)     ? u_old[i * n + (j - 1)] : 0.0;  /* Get left neighbor from old values (0 at boundary) */
                double right = (j < n - 1) ? u_old[i * n + (j + 1)] : 0.0;  /* Get right neighbor from old values (0 at boundary) */
                double up    = (i > 0)     ? u_old[(i - 1) * n + j]  : 0.0;  /* Get upper neighbor from old values (0 at boundary) */
                double down  = (i < n - 1) ? u_old[(i + 1) * n + j]  : 0.0;  /* Get lower neighbor from old values (0 at boundary) */

                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;  /* Compute new value: average of 4 neighbors plus RHS contribution */

                double diff = fabs(u[i * n + j] - u_old[i * n + j]);  /* Compute absolute change from previous iteration */
                if (diff > max_diff) max_diff = diff;  /* Update maximum change if this point changed more */
            }
        }

        if (max_diff < tol) {  /* Check if the solution has converged (max change below tolerance) */
            iter++;  /* Increment iter to reflect actual number of completed iterations */
            break;   /* Exit the loop early since convergence is achieved */
        }
    }

    free(u_old);   /* Free the temporary array used for old iteration values */
    return iter;   /* Return the number of iterations performed */
}

/**
 * Serial Red-Black Gauss-Seidel Method
 *
 * Uses checkerboard (red-black) ordering to expose parallelism.
 * Red points: (i+j) is even  |  Black points: (i+j) is odd
 *
 * Phase 1: Update all RED points using current (possibly old) black values.
 * Phase 2: Update all BLACK points using the newly computed red values.
 *
 * This two-phase approach gives faster convergence than Jacobi while
 * enabling parallelism within each phase.
 */
int redblack_gs_serial(double *u, const double *f, int n, int max_iter, double tol) { /* Red-Black Gauss-Seidel solver: in-place updates on u */
    int iter;  /* Iteration counter */
    for (iter = 0; iter < max_iter; iter++) {  /* Main iteration loop */
        double max_diff = 0.0;  /* Track maximum change across both phases */

        /* Phase 1: Update RED points ((i+j) even) */
        for (int i = 0; i < n; i++) {  /* Loop over each row */
            for (int j = 0; j < n; j++) {  /* Loop over each column */
                if ((i + j) % 2 != 0) continue; /* skip black points: only process red points where (i+j) is even */

                double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;  /* Left neighbor (0 at boundary) */
                double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;  /* Right neighbor (0 at boundary) */
                double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;  /* Upper neighbor (0 at boundary) */
                double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;  /* Lower neighbor (0 at boundary) */

                double old_val = u[i * n + j];  /* Save the current value before updating */
                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;  /* Update red point in-place using Gauss-Seidel formula */

                double diff = fabs(u[i * n + j] - old_val);  /* Compute absolute change from previous value */
                if (diff > max_diff) max_diff = diff;  /* Update maximum change tracker */
            }
        }

        /* Phase 2: Update BLACK points ((i+j) odd) */
        for (int i = 0; i < n; i++) {  /* Loop over each row */
            for (int j = 0; j < n; j++) {  /* Loop over each column */
                if ((i + j) % 2 != 1) continue; /* skip red points: only process black points where (i+j) is odd */

                double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;  /* Left neighbor (now uses updated red values) */
                double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;  /* Right neighbor (now uses updated red values) */
                double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;  /* Upper neighbor (now uses updated red values) */
                double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;  /* Lower neighbor (now uses updated red values) */

                double old_val = u[i * n + j];  /* Save current black point value */
                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;  /* Update black point using freshly computed red neighbors */

                double diff = fabs(u[i * n + j] - old_val);  /* Compute change at this black point */
                if (diff > max_diff) max_diff = diff;  /* Update maximum change tracker */
            }
        }

        if (max_diff < tol) {  /* Check convergence after both red and black phases */
            iter++;  /* Count this completed iteration */
            break;   /* Stop iterating: solution has converged */
        }
    }

    return iter;  /* Return total number of iterations performed */
}

int main(int argc, char *argv[]) {  /* Entry point: argc=argument count, argv=argument values */
    int n        = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;  /* Parse grid size from command line or use default (100) */
    int max_iter = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;     /* Parse max iterations from command line or use default (10000) */
    double tol   = (argc > 3) ? atof(argv[3]) : TOLERANCE;          /* Parse convergence tolerance from command line or use default (1e-6) */

    printf("Iterative Linear Solvers - Serial Implementation\n");                      /* Print program header */
    printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e\n\n", n, n, max_iter, tol);  /* Print configuration parameters */

    /* Allocate arrays */
    double *f       = (double *)malloc((size_t)n * n * sizeof(double));   /* Allocate RHS vector f (uninitialized) */
    double *u_jac   = (double *)calloc((size_t)n * n, sizeof(double));   /* Allocate and zero-initialize Jacobi solution array */
    double *u_gs    = (double *)calloc((size_t)n * n, sizeof(double));   /* Allocate and zero-initialize Gauss-Seidel solution array */
    double *u_exact = (double *)malloc((size_t)n * n * sizeof(double));  /* Allocate array for exact analytical solution */

    if (!f || !u_jac || !u_gs || !u_exact) {  /* Check if any allocation failed */
        fprintf(stderr, "Memory allocation failed\n");  /* Print error to stderr */
        return 1;  /* Exit with error code */
    }

    /* Initialize */
    init_rhs(f, n);           /* Fill RHS vector with h^2 * f(x,y) values */
    exact_solution(u_exact, n);  /* Compute exact solution for error comparison */

    /* ---- Jacobi Method ---- */
    double t_start = get_time();  /* Record start time for Jacobi solver */
    int jac_iters = jacobi_serial(u_jac, f, n, max_iter, tol);  /* Run serial Jacobi solver and get iteration count */
    double t_jac = get_time() - t_start;  /* Compute elapsed time for Jacobi */

    double res_jac  = compute_residual(u_jac, f, n);    /* Compute infinity-norm residual of Jacobi solution */
    double rmse_jac = compute_rmse(u_jac, u_exact, n);  /* Compute RMSE of Jacobi solution vs exact */
    print_results("Jacobi", "Serial", n, jac_iters, t_jac, res_jac, rmse_jac);  /* Print Jacobi results summary */

    /* ---- Red-Black Gauss-Seidel ---- */
    t_start = get_time();  /* Record start time for Gauss-Seidel solver */
    int gs_iters = redblack_gs_serial(u_gs, f, n, max_iter, tol);  /* Run serial Red-Black GS solver and get iteration count */
    double t_gs = get_time() - t_start;  /* Compute elapsed time for Gauss-Seidel */

    double res_gs  = compute_residual(u_gs, f, n);    /* Compute infinity-norm residual of GS solution */
    double rmse_gs = compute_rmse(u_gs, u_exact, n);  /* Compute RMSE of GS solution vs exact */
    print_results("Red-Black Gauss-Seidel", "Serial", n, gs_iters, t_gs, res_gs, rmse_gs);  /* Print GS results summary */

    /* ---- Comparison ---- */
    double rmse_jac_vs_gs = compute_rmse(u_jac, u_gs, n);  /* Compute RMSE between Jacobi and GS solutions */
    printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);  /* Print how closely the two methods agree */

    /* Cleanup */
    free(f);        /* Free the RHS vector */
    free(u_jac);    /* Free the Jacobi solution array */
    free(u_gs);     /* Free the Gauss-Seidel solution array */
    free(u_exact);  /* Free the exact solution array */

    return 0;  /* Exit successfully */
}
