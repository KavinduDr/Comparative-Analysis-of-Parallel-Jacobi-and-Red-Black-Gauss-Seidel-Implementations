/**
 * common.h - Shared utilities for iterative linear solver project
 *
 * Provides:
 *   - 2D Poisson problem setup (5-point stencil, diagonally dominant)
 *   - Timing utilities
 *   - Error computation (RMSE)
 *   - Solution verification
 *
 * Grid layout: n x n interior points on the unit square [0,1]^2
 * Boundary conditions: Dirichlet (u = 0 on boundary)
 * RHS: f(x,y) = 2*pi^2 * sin(pi*x) * sin(pi*y)
 * Exact solution: u(x,y) = sin(pi*x) * sin(pi*y)
 */

#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEFAULT_GRID_SIZE  100
#define MAX_ITERATIONS     10000
#define TOLERANCE          1e-6

/* High-resolution wall clock timer */
static inline double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/**
 * Initialize the RHS vector for -Laplacian u = f
 * f(x,y) = 2*pi^2 * sin(pi*x) * sin(pi*y)
 * Stored as h^2 * f for the discretized system.
 */
static inline void init_rhs(double *f, int n) {
    double h = 1.0 / (n + 1);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double x = (i + 1) * h;
            double y = (j + 1) * h;
            f[i * n + j] = h * h * 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }
}

/**
 * Compute the exact solution: u(x,y) = sin(pi*x) * sin(pi*y)
 */
static inline void exact_solution(double *u_exact, int n) {
    double h = 1.0 / (n + 1);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double x = (i + 1) * h;
            double y = (j + 1) * h;
            u_exact[i * n + j] = sin(M_PI * x) * sin(M_PI * y);
        }
    }
}

/**
 * Initialize solution vector to zero
 */
static inline void init_zero(double *u, int n) {
    memset(u, 0, (size_t)n * n * sizeof(double));
}

/**
 * Compute RMSE between two solution vectors
 */
static inline double compute_rmse(const double *u, const double *u_ref, int n) {
    double sum = 0.0;
    long total = (long)n * n;
    for (long k = 0; k < total; k++) {
        double diff = u[k] - u_ref[k];
        sum += diff * diff;
    }
    return sqrt(sum / total);
}

/**
 * Compute the maximum absolute residual (infinity norm of residual)
 * for the 2D Poisson equation: r = f - (4*u(i,j) - neighbors)
 */
static inline double compute_residual(const double *u, const double *f, int n) {
    double max_res = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;
            double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;
            double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;
            double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;

            double residual = f[i * n + j] - (4.0 * u[i * n + j] - left - right - up - down);
            if (fabs(residual) > max_res) max_res = fabs(residual);
        }
    }
    return max_res;
}

/**
 * Print solver results summary
 */
static inline void print_results(const char *method, const char *impl,
                                  int n, int iters, double time_sec,
                                  double residual, double rmse_exact) {
    printf("=============================================================\n");
    printf("  Method : %s (%s)\n", method, impl);
    printf("  Grid   : %d x %d  (%d unknowns)\n", n, n, n * n);
    printf("  Iters  : %d\n", iters);
    printf("  Time   : %.6f seconds\n", time_sec);
    printf("  Residual (inf-norm) : %.2e\n", residual);
    printf("  RMSE vs exact       : %.2e\n", rmse_exact);
    printf("=============================================================\n\n");
}

#endif /* COMMON_H */
