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

#ifndef COMMON_H   /* Include guard: prevents multiple inclusions of this header */
#define COMMON_H   /* Define the guard macro */

#include <stdio.h>     /* Standard I/O functions: printf, fprintf, etc. */
#include <stdlib.h>    /* Standard library: malloc, calloc, free, atoi, atof, exit */
#include <math.h>      /* Math functions: sin, fabs, sqrt, fmax */
#include <string.h>    /* String/memory functions: memset, memcpy */
#include <sys/time.h>  /* Time functions: gettimeofday for high-resolution timing */

#ifndef M_PI                       /* Check if M_PI (pi constant) is not already defined */
#define M_PI 3.14159265358979323846 /* Define pi with high precision */
#endif

#define DEFAULT_GRID_SIZE  100   /* Default number of interior grid points per dimension */
#define MAX_ITERATIONS     10000 /* Maximum number of solver iterations before stopping */
#define TOLERANCE          1e-6  /* Convergence threshold: stop when max change < this value */

/* High-resolution wall clock timer */
static inline double get_time(void) {    /* Returns current wall-clock time in seconds */
    struct timeval tv;                    /* Struct to hold seconds and microseconds */
    gettimeofday(&tv, NULL);             /* Get current time (NULL = no timezone info needed) */
    return tv.tv_sec + tv.tv_usec * 1e-6; /* Convert to double: seconds + microseconds as fraction */
}

/**
 * Initialize the RHS vector for -Laplacian u = f
 * f(x,y) = 2*pi^2 * sin(pi*x) * sin(pi*y)
 * Stored as h^2 * f for the discretized system.
 */
static inline void init_rhs(double *f, int n) {  /* Fills RHS array f for an n x n grid */
    double h = 1.0 / (n + 1);                     /* Grid spacing: divide unit interval by (n+1) points */
    for (int i = 0; i < n; i++) {                  /* Loop over each row of the grid */
        for (int j = 0; j < n; j++) {              /* Loop over each column of the grid */
            double x = (i + 1) * h;                /* Compute x-coordinate of grid point (i,j) */
            double y = (j + 1) * h;                /* Compute y-coordinate of grid point (i,j) */
            f[i * n + j] = h * h * 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y); /* Compute h^2 * f(x,y) and store in 1D array using row-major indexing */
        }
    }
}

/**
 * Compute the exact solution: u(x,y) = sin(pi*x) * sin(pi*y)
 */
static inline void exact_solution(double *u_exact, int n) { /* Fills array with exact analytical solution */
    double h = 1.0 / (n + 1);                                /* Grid spacing on the unit square */
    for (int i = 0; i < n; i++) {                             /* Loop over each row */
        for (int j = 0; j < n; j++) {                         /* Loop over each column */
            double x = (i + 1) * h;                           /* x-coordinate of interior point (i,j) */
            double y = (j + 1) * h;                           /* y-coordinate of interior point (i,j) */
            u_exact[i * n + j] = sin(M_PI * x) * sin(M_PI * y); /* Exact solution value at (x,y) */
        }
    }
}

/**
 * Initialize solution vector to zero
 */
static inline void init_zero(double *u, int n) {              /* Sets all n*n elements of u to zero */
    memset(u, 0, (size_t)n * n * sizeof(double));              /* Fill memory block with zeros (fast bulk zeroing) */
}

/**
 * Compute RMSE between two solution vectors
 */
static inline double compute_rmse(const double *u, const double *u_ref, int n) { /* Computes Root Mean Square Error between u and u_ref */
    double sum = 0.0;                    /* Accumulator for sum of squared differences */
    long total = (long)n * n;            /* Total number of grid points (cast to long to avoid overflow) */
    for (long k = 0; k < total; k++) {   /* Loop over all grid points linearly */
        double diff = u[k] - u_ref[k];   /* Compute difference at point k */
        sum += diff * diff;               /* Accumulate squared difference */
    }
    return sqrt(sum / total);             /* Return sqrt of mean squared error = RMSE */
}

/**
 * Compute the maximum absolute residual (infinity norm of residual)
 * for the 2D Poisson equation: r = f - (4*u(i,j) - neighbors)
 */
static inline double compute_residual(const double *u, const double *f, int n) { /* Computes infinity-norm of the residual vector */
    double max_res = 0.0;                                     /* Track the maximum absolute residual */
    for (int i = 0; i < n; i++) {                             /* Loop over each row */
        for (int j = 0; j < n; j++) {                         /* Loop over each column */
            double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;  /* Left neighbor value (0 if at left boundary) */
            double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;  /* Right neighbor value (0 if at right boundary) */
            double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;  /* Upper neighbor value (0 if at top boundary) */
            double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;  /* Lower neighbor value (0 if at bottom boundary) */

            double residual = f[i * n + j] - (4.0 * u[i * n + j] - left - right - up - down); /* Residual = f - A*u, where A is the 5-point stencil matrix */
            if (fabs(residual) > max_res) max_res = fabs(residual); /* Update max if this residual is larger */
        }
    }
    return max_res;  /* Return the maximum absolute residual (infinity norm) */
}

/**
 * Print solver results summary
 */
static inline void print_results(const char *method, const char *impl,
                                  int n, int iters, double time_sec,
                                  double residual, double rmse_exact) { /* Prints a formatted summary of solver results */
    printf("=============================================================\n");      /* Print separator line */
    printf("  Method : %s (%s)\n", method, impl);                                   /* Print solver method name and implementation type */
    printf("  Grid   : %d x %d  (%d unknowns)\n", n, n, n * n);                    /* Print grid dimensions and total unknowns */
    printf("  Iters  : %d\n", iters);                                                /* Print number of iterations performed */
    printf("  Time   : %.6f seconds\n", time_sec);                                   /* Print elapsed wall-clock time */
    printf("  Residual (inf-norm) : %.2e\n", residual);                              /* Print maximum absolute residual */
    printf("  RMSE vs exact       : %.2e\n", rmse_exact);                            /* Print RMSE compared to exact analytical solution */
    printf("=============================================================\n\n");     /* Print closing separator line */
}

/**
 * Save the final solution grid to a text file
 * Can be used to compare results across different implementations (Serial vs MPI, etc.)
 */
static inline void save_solution(const char *filename, const double *u, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp, "%.12e ", u[i * n + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("Saved solution to %s\n", filename);
}

#endif /* COMMON_H */  /* End of include guard */
