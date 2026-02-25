/**
 * openmp_solver.c - OpenMP implementations of Jacobi and Red-Black Gauss-Seidel
 *
 * Shared memory parallelization using OpenMP directives.
 * Parallelizes the loop iterations across threads.
 *
 * Usage: ./openmp_solver [grid_size] [max_iterations] [tolerance] [num_threads]
 */

#include "../include/common.h"
#include <omp.h>

/**
 * OpenMP Jacobi Method
 *
 * The outer loop over grid rows is parallelized with OpenMP.
 * Each thread computes a subset of rows independently since Jacobi
 * only reads from the old array (no data dependencies within an iteration).
 * A reduction is used to compute the global maximum change.
 */
int jacobi_openmp(double *u, const double *f, int n, int max_iter, double tol, int num_threads) {
    double *u_old = (double *)calloc((size_t)n * n, sizeof(double));
    if (!u_old) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }

    omp_set_num_threads(num_threads);

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        memcpy(u_old, u, (size_t)n * n * sizeof(double));

        double max_diff = 0.0;

        #pragma omp parallel for collapse(2) reduction(max:max_diff) schedule(static)
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
 * OpenMP Red-Black Gauss-Seidel Method
 *
 * Phase 1 (Red): All points where (i+j) is even are updated in parallel.
 *   Within this phase there are NO data dependencies (red neighbors are all black).
 * Phase 2 (Black): All points where (i+j) is odd are updated in parallel.
 *   Black neighbors are all red (already updated).
 *
 * An implicit barrier between phases ensures correctness.
 */
int redblack_gs_openmp(double *u, const double *f, int n, int max_iter, double tol, int num_threads) {
    omp_set_num_threads(num_threads);

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        double max_diff = 0.0;

        /* Phase 1: Update RED points */
        #pragma omp parallel for collapse(2) reduction(max:max_diff) schedule(static)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((i + j) % 2 != 0) continue;

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
        /* Implicit barrier here ensures all red points are updated */

        /* Phase 2: Update BLACK points */
        #pragma omp parallel for collapse(2) reduction(max:max_diff) schedule(static)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((i + j) % 2 != 1) continue;

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
    int n           = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;
    int max_iter    = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;
    double tol      = (argc > 3) ? atof(argv[3]) : TOLERANCE;
    int num_threads = (argc > 4) ? atoi(argv[4]) : 4;

    printf("Iterative Linear Solvers - OpenMP Implementation\n");
    printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e, Threads: %d\n\n",
           n, n, max_iter, tol, num_threads);

    double *f       = (double *)malloc((size_t)n * n * sizeof(double));
    double *u_jac   = (double *)calloc((size_t)n * n, sizeof(double));
    double *u_gs    = (double *)calloc((size_t)n * n, sizeof(double));
    double *u_exact = (double *)malloc((size_t)n * n * sizeof(double));

    if (!f || !u_jac || !u_gs || !u_exact) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    init_rhs(f, n);
    exact_solution(u_exact, n);

    /* ---- Jacobi ---- */
    double t_start = get_time();
    int jac_iters = jacobi_openmp(u_jac, f, n, max_iter, tol, num_threads);
    double t_jac = get_time() - t_start;

    double res_jac  = compute_residual(u_jac, f, n);
    double rmse_jac = compute_rmse(u_jac, u_exact, n);
    print_results("Jacobi", "OpenMP", n, jac_iters, t_jac, res_jac, rmse_jac);

    /* ---- Red-Black Gauss-Seidel ---- */
    t_start = get_time();
    int gs_iters = redblack_gs_openmp(u_gs, f, n, max_iter, tol, num_threads);
    double t_gs = get_time() - t_start;

    double res_gs  = compute_residual(u_gs, f, n);
    double rmse_gs = compute_rmse(u_gs, u_exact, n);
    print_results("Red-Black Gauss-Seidel", "OpenMP", n, gs_iters, t_gs, res_gs, rmse_gs);

    double rmse_jac_vs_gs = compute_rmse(u_jac, u_gs, n);
    printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);

    free(f);
    free(u_jac);
    free(u_gs);
    free(u_exact);

    return 0;
}
