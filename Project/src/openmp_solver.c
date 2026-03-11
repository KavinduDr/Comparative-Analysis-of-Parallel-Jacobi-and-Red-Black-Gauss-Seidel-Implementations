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

#include "../include/common.h"  /* Shared utilities: grid initialization, timing helpers, residual/RMSE computation */
#include <omp.h>                /* OpenMP API: parallel regions, thread management, synchronization primitives */

/**
 * jacobi_openmp - Parallel Jacobi iterative solver
 *
 * Distributes row computations across OpenMP threads using a collapse(2)
 * directive. Because Jacobi reads exclusively from the previous iteration's
 * values (u_old), there are no write-write or read-write conflicts between
 * threads. A max-reduction determines the global convergence metric.
 *
 * @param u          Solution vector (n*n), updated in place
 * @param f          Right-hand side vector (n*n)
 * @param n          Grid dimension (n x n interior points)
 * @param max_iter   Maximum number of iterations allowed
 * @param tol        Convergence tolerance (max absolute change)
 * @param num_threads Number of OpenMP threads to spawn
 * @return           Number of iterations actually performed
 */
int jacobi_openmp(double *u, const double *f, int n, int max_iter, double tol, int num_threads) { /* Entry: parallel Jacobi solver */
    double *u_old = (double *)calloc((size_t)n * n, sizeof(double));  /* Temporary buffer holding the previous iteration's solution */
    if (!u_old) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }  /* Guard: abort on allocation failure */

    omp_set_num_threads(num_threads);  /* Configure the thread pool size for subsequent parallel regions */

    int iter;  /* Counts completed sweeps */
    for (iter = 0; iter < max_iter; iter++) {  /* Outer sweep loop: one full grid update per iteration */
        memcpy(u_old, u, (size_t)n * n * sizeof(double));  /* Snapshot current solution; must complete before the parallel stencil pass */

        double max_diff = 0.0;  /* Convergence metric: largest pointwise change in this sweep */

        #pragma omp parallel for collapse(2) reduction(max:max_diff) schedule(static)  /* Flatten 2D loop; each thread accumulates a local max_diff, merged after the region */
        for (int i = 0; i < n; i++) {  /* Row index — work is split evenly (static schedule) */
            for (int j = 0; j < n; j++) {  /* Column index — also distributed thanks to collapse(2) */
                double left  = (j > 0)     ? u_old[i * n + (j - 1)] : 0.0;  /* West neighbor (Dirichlet zero at boundary) */
                double right = (j < n - 1) ? u_old[i * n + (j + 1)] : 0.0;  /* East neighbor (Dirichlet zero at boundary) */
                double up    = (i > 0)     ? u_old[(i - 1) * n + j]  : 0.0;  /* North neighbor (Dirichlet zero at boundary) */
                double down  = (i < n - 1) ? u_old[(i + 1) * n + j]  : 0.0;  /* South neighbor (Dirichlet zero at boundary) */

                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;  /* 5-point stencil average: standard Jacobi update */

                double diff = fabs(u[i * n + j] - u_old[i * n + j]);  /* Absolute pointwise change between iterations */
                if (diff > max_diff) max_diff = diff;  /* Thread-local max; OpenMP reduces all locals after the region */
            }
        }

        if (max_diff < tol) {  /* Convergence test: solution has stabilized within the specified tolerance */
            iter++;  /* Account for the iteration that just completed */
            break;   /* Early exit: no further sweeps needed */
        }
    }

    free(u_old);   /* Release scratch buffer */
    return iter;   /* Total sweeps executed (may be < max_iter if converged) */
}

/**
 * redblack_gs_openmp - Parallel Red-Black Gauss-Seidel iterative solver
 *
 * Splits each iteration into two half-sweeps that can be parallelized:
 *   Red sweep  – updates points where (i+j) is even; all their neighbors
 *                are black (untouched), so no data races occur.
 *   Black sweep – updates points where (i+j) is odd; all their neighbors
 *                 are the freshly-updated red points.
 *
 * An implicit OpenMP barrier between the two parallel regions guarantees
 * that every red point is written before any black point reads it.
 *
 * @param u          Solution vector (n*n), updated in place
 * @param f          Right-hand side vector (n*n)
 * @param n          Grid dimension (n x n interior points)
 * @param max_iter   Maximum number of iterations allowed
 * @param tol        Convergence tolerance (max absolute change)
 * @param num_threads Number of OpenMP threads to spawn
 * @return           Number of iterations actually performed
 */
int redblack_gs_openmp(double *u, const double *f, int n, int max_iter, double tol, int num_threads) { /* Parallel RB Gauss-Seidel entry point */
    omp_set_num_threads(num_threads);  /* Configure thread pool for both red and black parallel regions */

    int iter;  /* Counts completed full (red + black) sweeps */
    for (iter = 0; iter < max_iter; iter++) {  /* Outer iteration loop */
        double max_diff = 0.0;  /* Convergence metric: accumulated across both color phases */

        /* --- Red half-sweep: update all (i+j)-even points --- */
        #pragma omp parallel for collapse(2) reduction(max:max_diff) schedule(static)  /* Distribute red updates; local max_diff values merged after region */
        for (int i = 0; i < n; i++) {  /* Row index */
            for (int j = 0; j < n; j++) {  /* Column index */
                if ((i + j) % 2 != 0) continue;  /* Skip black-colored cells; only reds are touched here */

                double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;  /* West neighbor — a black point, still at previous-iteration value */
                double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;  /* East neighbor — black, unchanged so far */
                double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;  /* North neighbor — black */
                double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;  /* South neighbor — black */

                double old_val = u[i * n + j];  /* Snapshot before in-place write */
                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;  /* Gauss-Seidel stencil update for this red cell */

                double diff = fabs(u[i * n + j] - old_val);  /* Magnitude of change at this cell */
                if (diff > max_diff) max_diff = diff;  /* Feed into the OpenMP max-reduction */
            }
        }
        /* OpenMP implicit barrier: all threads finish red updates before black sweep begins */

        /* --- Black half-sweep: update all (i+j)-odd points --- */
        #pragma omp parallel for collapse(2) reduction(max:max_diff) schedule(static)  /* Distribute black updates; max_diff continues accumulating */
        for (int i = 0; i < n; i++) {  /* Row index */
            for (int j = 0; j < n; j++) {  /* Column index */
                if ((i + j) % 2 != 1) continue;  /* Skip red-colored cells; only blacks are touched here */

                double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;  /* West neighbor — a red point, freshly updated in the preceding sweep */
                double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;  /* East neighbor — red, already holds this iteration's value */
                double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;  /* North neighbor — red, current */
                double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;  /* South neighbor — red, current */

                double old_val = u[i * n + j];  /* Snapshot before in-place write */
                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;  /* Gauss-Seidel stencil update using newest red values */

                double diff = fabs(u[i * n + j] - old_val);  /* Magnitude of change at this cell */
                if (diff > max_diff) max_diff = diff;  /* Continue accumulating into the max-reduction */
            }
        }

        if (max_diff < tol) {  /* Convergence test after both color sweeps complete */
            iter++;  /* Account for this completed iteration */
            break;   /* Early exit: solution is within tolerance */
        }
    }

    return iter;  /* Total full iterations executed */
}

int main(int argc, char *argv[]) {  /* Program entry: parse CLI args, run both solvers, report results */
    int n           = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;  /* Grid dimension; defaults to 100 if not supplied */
    int max_iter    = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;     /* Upper bound on iterations; defaults to 10000 */
    double tol      = (argc > 3) ? atof(argv[3]) : TOLERANCE;          /* Stopping criterion; defaults to 1e-6 */
    int num_threads = (argc > 4) ? atoi(argv[4]) : 4;                  /* OpenMP thread count; defaults to 4 */

    printf("Iterative Linear Solvers - OpenMP Implementation\n");  /* Banner line */
    printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e, Threads: %d\n\n",
           n, n, max_iter, tol, num_threads);  /* Display run parameters */

    double *f       = (double *)malloc((size_t)n * n * sizeof(double));   /* Right-hand side of the discretized PDE */
    double *u_jac   = (double *)calloc((size_t)n * n, sizeof(double));   /* Jacobi solution vector, zero-initialized */
    double *u_gs    = (double *)calloc((size_t)n * n, sizeof(double));   /* Red-Black GS solution vector, zero-initialized */
    double *u_exact = (double *)malloc((size_t)n * n * sizeof(double));  /* Reference analytical solution */

    if (!f || !u_jac || !u_gs || !u_exact) {  /* Safety check: any allocation failure is fatal */
        fprintf(stderr, "Memory allocation failed\n");  /* Report to stderr */
        return 1;  /* Non-zero exit signals error */
    }

    init_rhs(f, n);              /* Populate RHS from the chosen test problem */
    exact_solution(u_exact, n);  /* Build the closed-form reference solution */

    /* ---- Jacobi benchmark ---- */
    double t_start = get_time();  /* Wall-clock start for Jacobi */
    int jac_iters = jacobi_openmp(u_jac, f, n, max_iter, tol, num_threads);  /* Execute parallel Jacobi solver */
    double t_jac = get_time() - t_start;  /* Elapsed wall-clock time */

    double res_jac  = compute_residual(u_jac, f, n);    /* ||Au - f||: residual norm for Jacobi */
    double rmse_jac = compute_rmse(u_jac, u_exact, n);  /* Root-mean-square error against the analytical solution */
    print_results("Jacobi", "OpenMP", n, jac_iters, t_jac, res_jac, rmse_jac);  /* Formatted output for Jacobi run */

    /* ---- Red-Black Gauss-Seidel benchmark ---- */
    t_start = get_time();  /* Wall-clock start for RB-GS */
    int gs_iters = redblack_gs_openmp(u_gs, f, n, max_iter, tol, num_threads);  /* Execute parallel RB-GS solver */
    double t_gs = get_time() - t_start;  /* Elapsed wall-clock time */

    double res_gs  = compute_residual(u_gs, f, n);    /* ||Au - f||: residual norm for RB-GS */
    double rmse_gs = compute_rmse(u_gs, u_exact, n);  /* RMSE against the analytical solution */
    print_results("Red-Black Gauss-Seidel", "OpenMP", n, gs_iters, t_gs, res_gs, rmse_gs);  /* Formatted output for RB-GS run */

    double rmse_jac_vs_gs = compute_rmse(u_jac, u_gs, n);  /* Cross-compare the two numerical solutions */
    printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);  /* Quantify inter-solver agreement */

    free(f);        /* Release RHS memory */
    free(u_jac);    /* Release Jacobi solution memory */
    free(u_gs);     /* Release GS solution memory */
    free(u_exact);  /* Release reference solution memory */

    return 0;  /* Clean exit */
}
