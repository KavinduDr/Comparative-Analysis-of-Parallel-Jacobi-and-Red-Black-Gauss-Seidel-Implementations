#include "../include/common.h"

/* Serial Jacobi: updates all points using previous iteration values */
int jacobi_serial(double *u, const double *f, int n, int max_iter, double tol) {
    double *u_old = (double *)calloc((size_t)n * n, sizeof(double));
    if (!u_old) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        memcpy(u_old, u, (size_t)n * n * sizeof(double)); /* save current solution */

        double max_diff = 0.0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                /* get 4 neighbors from old values, 0 at boundaries */
                double left  = (j > 0)     ? u_old[i * n + (j - 1)] : 0.0;
                double right = (j < n - 1) ? u_old[i * n + (j + 1)] : 0.0;
                double up    = (i > 0)     ? u_old[(i - 1) * n + j]  : 0.0;
                double down  = (i < n - 1) ? u_old[(i + 1) * n + j]  : 0.0;

                /* 5-point stencil update */
                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;

                /* track max change for convergence check */
                double diff = fabs(u[i * n + j] - u_old[i * n + j]);
                if (diff > max_diff) max_diff = diff;
            }
        }

        if (max_diff < tol) { /* converged */
            iter++;
            break;
        }
    }

    free(u_old);
    return iter;
}

/* Serial Red-Black Gauss-Seidel: two-phase checkerboard update */
int redblack_gs_serial(double *u, const double *f, int n, int max_iter, double tol) {
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        double max_diff = 0.0;

        /* Phase 1: update red points where (i+j) is even */
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

        /* Phase 2: update black points where (i+j) is odd */
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

        if (max_diff < tol) { /* converged */
            iter++;
            break;
        }
    }

    return iter;
}

int main(int argc, char *argv[]) {
    /* parse command-line args or use defaults */
    int n        = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;
    int max_iter = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;
    double tol   = (argc > 3) ? atof(argv[3]) : TOLERANCE;

    printf("Iterative Linear Solvers - Serial Implementation\n");
    printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e\n\n", n, n, max_iter, tol);

    /* allocate arrays */
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

    /* run Jacobi solver */
    double t_start = get_time();
    int jac_iters = jacobi_serial(u_jac, f, n, max_iter, tol);
    double t_jac = get_time() - t_start;

    double res_jac  = compute_residual(u_jac, f, n);
    double rmse_jac = compute_rmse(u_jac, u_exact, n);
    print_results("Jacobi", "Serial", n, jac_iters, t_jac, res_jac, rmse_jac);

    /* run Red-Black Gauss-Seidel solver */
    t_start = get_time();
    int gs_iters = redblack_gs_serial(u_gs, f, n, max_iter, tol);
    double t_gs = get_time() - t_start;

    double res_gs  = compute_residual(u_gs, f, n);
    double rmse_gs = compute_rmse(u_gs, u_exact, n);
    print_results("Red-Black Gauss-Seidel", "Serial", n, gs_iters, t_gs, res_gs, rmse_gs);

    /* compare the two solutions */
    double rmse_jac_vs_gs = compute_rmse(u_jac, u_gs, n);
    printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);

    /* cleanup */
    free(f);
    free(u_jac);
    free(u_gs);
    free(u_exact);

    return 0;
}
