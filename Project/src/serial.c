/**
 * serial.c - Serial implementations of Jacobi and Red-Black Gauss-Seidel
 *
 * Solves the 2D Poisson equation on a unit square using the 5-point stencil.
 * This serves as the baseline reference for all parallel implementations.
 *
 * Usage: ./serial [grid_size] [max_iterations] [tolerance]
 */

#include "../include/common.h"

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
int jacobi_serial(double *u, const double *f, int n, int max_iter, double tol) {
    double *u_old = (double *)calloc((size_t)n * n, sizeof(double));
    if (!u_old) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        /* Copy current solution to old */
        memcpy(u_old, u, (size_t)n * n * sizeof(double));

        double max_diff = 0.0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double left  = (j > 0)     ? u_old[i * n + (j - 1)] : 0.0;
                double right = (j < n - 1) ? u_old[i * n + (j + 1)] : 0.0;
                double up    = (i > 0)     ? u_old[(i - 1) * n + j]  : 0.0;
                double down  = (i < n - 1) ? u_old[(i + 1) * n + j]  : 0.0;

                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;

                double diff = fabs(u[i * n + j] - u_old[i * n + j]);
                if (diff > max_diff) max_diff = diff;
            }
        }

        if (max_diff < tol) {
            iter++;
            break;
        }
    }

    free(u_old);
    return iter;
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
int redblack_gs_serial(double *u, const double *f, int n, int max_iter, double tol) {
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        double max_diff = 0.0;

        /* Phase 1: Update RED points ((i+j) even) */
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((i + j) % 2 != 0) continue; /* skip black */

                double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;
                double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;
                double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;
                double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;

                double old_val = u[i * n + j];
                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;

                double diff = fabs(u[i * n + j] - old_val);
                if (diff > max_diff) max_diff = diff;
            }
        }

        /* Phase 2: Update BLACK points ((i+j) odd) */
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((i + j) % 2 != 1) continue; /* skip red */

                double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;
                double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;
                double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;
                double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;

                double old_val = u[i * n + j];
                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;

                double diff = fabs(u[i * n + j] - old_val);
                if (diff > max_diff) max_diff = diff;
            }
        }

        if (max_diff < tol) {
            iter++;
            break;
        }
    }

    return iter;
}

int main(int argc, char *argv[]) {
    int n        = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;
    int max_iter = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;
    double tol   = (argc > 3) ? atof(argv[3]) : TOLERANCE;

    printf("Iterative Linear Solvers - Serial Implementation\n");
    printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e\n\n", n, n, max_iter, tol);

    /* Allocate arrays */
    double *f       = (double *)malloc((size_t)n * n * sizeof(double));
    double *u_jac   = (double *)calloc((size_t)n * n, sizeof(double));
    double *u_gs    = (double *)calloc((size_t)n * n, sizeof(double));
    double *u_exact = (double *)malloc((size_t)n * n * sizeof(double));

    if (!f || !u_jac || !u_gs || !u_exact) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    /* Initialize */
    init_rhs(f, n);
    exact_solution(u_exact, n);

    /* ---- Jacobi Method ---- */
    double t_start = get_time();
    int jac_iters = jacobi_serial(u_jac, f, n, max_iter, tol);
    double t_jac = get_time() - t_start;

    double res_jac  = compute_residual(u_jac, f, n);
    double rmse_jac = compute_rmse(u_jac, u_exact, n);
    print_results("Jacobi", "Serial", n, jac_iters, t_jac, res_jac, rmse_jac);

    /* ---- Red-Black Gauss-Seidel ---- */
    t_start = get_time();
    int gs_iters = redblack_gs_serial(u_gs, f, n, max_iter, tol);
    double t_gs = get_time() - t_start;

    double res_gs  = compute_residual(u_gs, f, n);
    double rmse_gs = compute_rmse(u_gs, u_exact, n);
    print_results("Red-Black Gauss-Seidel", "Serial", n, gs_iters, t_gs, res_gs, rmse_gs);

    /* ---- Comparison ---- */
    double rmse_jac_vs_gs = compute_rmse(u_jac, u_gs, n);
    printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);

    /* Cleanup */
    free(f);
    free(u_jac);
    free(u_gs);
    free(u_exact);

    return 0;
}
