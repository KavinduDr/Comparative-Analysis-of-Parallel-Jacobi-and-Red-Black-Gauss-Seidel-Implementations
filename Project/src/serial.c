/**
 * serial.c - Serial implementations of Jacobi and Red-Black Gauss-Seidel
 *
 * This program solves the 2D Poisson equation:
 *     -u_xx - u_yy = f(x, y)
 * on a unit square domain [0,1] x [0,1].
 *
 * Boundary conditions: Dirichlet (u = 0 on all four edges).
 *
 * Discretization:
 *   The domain is divided into an n x n grid of equally spaced interior points.
 *   The grid spacing is h = 1/(n+1). Each interior point (i,j) maps to the
 *   physical coordinate (x, y) = (j*h, i*h).
 *
 *   The Laplacian is approximated using the 5-point finite difference stencil:
 *       -u(i-1,j) - u(i+1,j) - u(i,j-1) - u(i,j+1) + 4*u(i,j) = h^2 * f(i,j)
 *   Rearranging to solve for u(i,j):
 *       u(i,j) = ( u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h^2*f(i,j) ) / 4
 *
 * Two iterative methods are implemented to solve this system:
 *
 *   1. Jacobi Method:
 *      - Updates ALL grid points simultaneously using ONLY values from the
 *        previous iteration. Requires two arrays (u and u_old).
 *      - Every update is independent -> naturally parallel.
 *      - Convergence is slower because newly computed values are not used
 *        until the next iteration.
 *      - Convergence rate depends on the spectral radius of the iteration
 *        matrix, which for the 2D Poisson problem is ~1 - O(h^2).
 *
 *   2. Red-Black Gauss-Seidel Method:
 *      - Grid points are colored in a checkerboard pattern:
 *          RED   = (i+j) even    BLACK = (i+j) odd
 *      - Phase 1: update all RED points (their neighbors are all BLACK, still old).
 *      - Phase 2: update all BLACK points (their neighbors are all RED, now new).
 *      - Updates are done IN-PLACE (only one array needed).
 *      - Converges roughly 2x faster than Jacobi because it immediately
 *        uses freshly computed values.
 *      - Within each phase, all updates are independent -> parallelizable.
 *      - Standard (lexicographic) Gauss-Seidel is inherently sequential,
 *        but the red-black ordering breaks this dependency to allow parallelism.
 *
 * This serial version serves as the baseline reference for comparing
 * performance and correctness against parallel implementations (OpenMP, MPI, CUDA).
 *
 * Usage: ./serial [grid_size] [max_iterations] [tolerance]
 *   Examples:
 *     ./serial              -> uses defaults (100x100 grid, 10000 iters, 1e-6 tol)
 *     ./serial 200          -> 200x200 grid
 *     ./serial 500 50000    -> 500x500 grid, up to 50000 iterations
 */

#include "../include/common.h"  /* shared utilities: grid init, timing, error norms */

/**
 * jacobi_serial - Solve the discrete Poisson equation using the Jacobi method
 *
 * @param u        Solution array of size n*n, stored in row-major order.
 *                 On entry: initial guess (typically all zeros).
 *                 On exit:  the computed approximate solution.
 * @param f        Right-hand side array of size n*n. Contains h^2 * f(x,y)
 *                 at each interior grid point. This array is NOT modified.
 * @param n        Number of grid points in each dimension (grid is n x n).
 * @param max_iter Maximum number of iterations. The solver stops after this
 *                 many iterations even if convergence has not been reached.
 * @param tol      Convergence tolerance. The solver stops when the maximum
 *                 absolute change at any grid point between two consecutive
 *                 iterations is less than tol (infinity-norm of the update).
 * @return         The number of iterations actually performed.
 *
 * Algorithm:
 *   1. Copy u into u_old (snapshot of the previous iteration).
 *   2. For every grid point (i,j), compute:
 *        u_new(i,j) = (u_old(i-1,j) + u_old(i+1,j) +
 *                       u_old(i,j-1) + u_old(i,j+1) + f(i,j)) / 4
 *      All reads come from u_old, all writes go to u. This guarantees that
 *      the order in which points are processed does not affect the result.
 *   3. Compute max_diff = max |u_new(i,j) - u_old(i,j)| over all (i,j).
 *   4. If max_diff < tol, declare convergence and stop.
 *   5. Otherwise, go back to step 1 for the next iteration.
 *
 * Memory: O(n^2) extra for the u_old array.
 * Time:   O(max_iter * n^2) in the worst case (no early convergence).
 */
int jacobi_serial(double *u, const double *f, int n, int max_iter, double tol) {
    /*
     * Allocate u_old: a temporary array to store the previous iteration's values.
     *
     * Why is this needed?
     *   In Jacobi, the new value at point (i,j) depends on the OLD values of
     *   its 4 neighbors. If we updated u in-place, then by the time we reach
     *   point (i,j), some of its neighbors might already have been overwritten
     *   with new values, breaking the Jacobi property. So we keep a separate
     *   copy of the old values.
     *
     * calloc zeros the memory, which serves as the initial "previous iteration"
     * for the very first iteration (matching the zero initial guess).
     */
    double *u_old = (double *)calloc((size_t)n * n, sizeof(double));
    if (!u_old) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }

    int iter;  /* declared outside the loop so we can return it after the loop ends */
    for (iter = 0; iter < max_iter; iter++) {

        /*
         * Step 1: Snapshot the current solution into u_old.
         * After this, u_old[k] holds the value from iteration (iter),
         * and u[k] will be overwritten with values for iteration (iter+1).
         */
        memcpy(u_old, u, (size_t)n * n * sizeof(double));

        /*
         * max_diff tracks the infinity-norm of the update vector:
         *   max_diff = max over all (i,j) of |u_new(i,j) - u_old(i,j)|
         * This tells us how much the solution changed in this iteration.
         * A small max_diff means the solution is barely changing -> converged.
         */
        double max_diff = 0.0;

        /*
         * Step 2: Loop over every interior grid point.
         * The grid is stored as a 1D array in row-major order:
         *   element (i, j) is at index [i * n + j]
         * where i is the row (0 = top, n-1 = bottom) and j is the column.
         */
        for (int i = 0; i < n; i++) {        /* iterate over rows */
            for (int j = 0; j < n; j++) {    /* iterate over columns */

                /*
                 * Fetch the 4 neighbors of point (i,j) from the OLD array.
                 *
                 * Boundary handling:
                 *   If (i,j) is on the edge of the grid, one or more neighbors
                 *   would fall outside the domain. Those neighbors are boundary
                 *   points with value 0.0 (Dirichlet BC: u = 0 on all edges).
                 *
                 *   - left:  exists if j > 0         (not on the left edge)
                 *   - right: exists if j < n-1       (not on the right edge)
                 *   - up:    exists if i > 0         (not on the top edge)
                 *   - down:  exists if i < n-1       (not on the bottom edge)
                 *
                 * The ternary operator ? : handles this: if the neighbor exists,
                 * read it from u_old; otherwise, use 0.0.
                 */
                double left  = (j > 0)     ? u_old[i * n + (j - 1)] : 0.0;  /* neighbor to the left  (j-1) */
                double right = (j < n - 1) ? u_old[i * n + (j + 1)] : 0.0;  /* neighbor to the right (j+1) */
                double up    = (i > 0)     ? u_old[(i - 1) * n + j]  : 0.0;  /* neighbor above        (i-1) */
                double down  = (i < n - 1) ? u_old[(i + 1) * n + j]  : 0.0;  /* neighbor below        (i+1) */

                /*
                 * Apply the 5-point stencil update formula:
                 *   u_new(i,j) = (left + right + up + down + f(i,j)) / 4.0
                 *
                 * This comes from rearranging the discretized Poisson equation:
                 *   4*u(i,j) - u(i-1,j) - u(i+1,j) - u(i,j-1) - u(i,j+1) = h^2*f(i,j)
                 * to isolate u(i,j) on the left-hand side.
                 *
                 * Note: f already contains h^2 * f(x,y), so no extra h^2 factor is needed.
                 */
                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;

                /*
                 * Step 3: Track convergence.
                 * Compute how much this grid point changed from the old value.
                 * Keep the maximum change seen so far across all grid points.
                 */
                double diff = fabs(u[i * n + j] - u_old[i * n + j]);
                if (diff > max_diff) max_diff = diff;  /* update worst-case change */
            }
        }

        /*
         * Step 4: Convergence test (checked after processing ALL grid points).
         * If the largest change at any point is below the tolerance,
         * the solution is no longer changing significantly -> we're done.
         *
         * We increment iter before breaking so the returned count reflects
         * the actual number of completed iterations (this iteration counts).
         */
        if (max_diff < tol) {
            iter++;  /* count this completed iteration before exiting */
            break;   /* exit the for loop early - convergence achieved */
        }
    }

    free(u_old);    /* release the temporary array (no longer needed) */
    return iter;    /* return the total number of iterations performed */
}

/**
 * redblack_gs_serial - Solve using the Red-Black Gauss-Seidel iterative method
 *
 * @param u        Solution array of size n*n, stored in row-major order.
 *                 On entry: initial guess (typically all zeros).
 *                 On exit:  the computed approximate solution.
 *                 Updated IN-PLACE (no second array needed).
 * @param f        Right-hand side array of size n*n (not modified).
 * @param n        Number of grid points in each dimension.
 * @param max_iter Maximum number of iterations allowed.
 * @param tol      Convergence tolerance (infinity-norm of the update).
 * @return         The number of iterations actually performed.
 *
 * Algorithm:
 *   The grid is partitioned into two independent sets using a checkerboard coloring:
 *
 *       R B R B R B        R = Red point  (i+j is even)
 *       B R B R B R        B = Black point (i+j is odd)
 *       R B R B R B
 *       B R B R B R        Key property: every RED neighbor is BLACK
 *                                         every BLACK neighbor is RED
 *
 *   Each iteration consists of two sweeps:
 *
 *   Phase 1 (Red sweep):
 *     - Update every RED point using the 5-point stencil.
 *     - All neighbors of a red point are black, and no black points have
 *       been modified yet in this iteration, so all red updates read the
 *       same "old" black values. This means red updates are independent.
 *
 *   Phase 2 (Black sweep):
 *     - Update every BLACK point using the 5-point stencil.
 *     - All neighbors of a black point are red, and they were JUST updated
 *       in Phase 1. So black points benefit from the newest red values.
 *     - Black updates are also independent of each other.
 *
 *   Why Red-Black instead of standard Gauss-Seidel?
 *     - Standard (lexicographic) Gauss-Seidel processes points in row order.
 *       Each update depends on the just-updated neighbor above and to the left,
 *       creating a sequential dependency chain that prevents parallelism.
 *     - Red-Black ordering eliminates these dependencies WITHIN each phase.
 *       All red points can be updated simultaneously, then all black points.
 *       This makes it ideal for OpenMP, CUDA, or MPI parallelization.
 *
 *   Why is it faster than Jacobi?
 *     - Gauss-Seidel uses the LATEST available values (not previous iteration).
 *       Information propagates faster through the grid, so it converges in
 *       roughly half the number of iterations compared to Jacobi.
 *     - It also needs less memory (no u_old array), which improves cache usage.
 *
 * Memory: No extra arrays needed (in-place update).
 * Time:   O(max_iter * n^2) worst case, but typically fewer iterations than Jacobi.
 */
int redblack_gs_serial(double *u, const double *f, int n, int max_iter, double tol) {
    int iter;  /* declared outside loop so we can return it */
    for (iter = 0; iter < max_iter; iter++) {
        /*
         * max_diff tracks the largest change across ALL grid points
         * (both red and black) in this iteration, used for convergence check.
         */
        double max_diff = 0.0;

        /* ================================================================
         * Phase 1: Update RED points where (i + j) is even
         * ================================================================
         * On a checkerboard, red points look like this (R = updated, . = skipped):
         *
         *     R . R . R .
         *     . R . R . R
         *     R . R . R .
         *
         * Every neighbor of a red point is black. Since we haven't touched
         * any black points yet, all neighbor values are from the previous
         * iteration. This means all red updates are independent and could
         * be parallelized (e.g., with OpenMP parallel for).
         */
        for (int i = 0; i < n; i++) {        /* iterate over rows */
            for (int j = 0; j < n; j++) {    /* iterate over columns */
                if ((i + j) % 2 != 0) continue;  /* skip black points (odd sum) */

                /*
                 * Fetch 4 neighbors directly from u (not from a copy).
                 * At this point in Phase 1, all neighbors are BLACK and
                 * still hold their values from the previous iteration.
                 * Boundary points outside the grid are 0.0 (Dirichlet BC).
                 */
                double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;  /* left neighbor  (black) */
                double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;  /* right neighbor (black) */
                double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;  /* upper neighbor (black) */
                double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;  /* lower neighbor (black) */

                double old_val = u[i * n + j];  /* save current value to measure change */

                /*
                 * In-place update: overwrite u(i,j) with the new value.
                 * Unlike Jacobi, we don't need u_old because the neighbors
                 * (all black) haven't been modified in this iteration yet.
                 */
                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;

                /* Track the maximum change for convergence detection */
                double diff = fabs(u[i * n + j] - old_val);
                if (diff > max_diff) max_diff = diff;
            }
        }

        /* ================================================================
         * Phase 2: Update BLACK points where (i + j) is odd
         * ================================================================
         * Now we update the black points (. = skipped, B = updated):
         *
         *     . B . B . B
         *     B . B . B .
         *     . B . B . B
         *
         * Every neighbor of a black point is red. All red points were JUST
         * updated in Phase 1 above. So black points automatically use the
         * latest red values — this is the Gauss-Seidel property that gives
         * faster convergence than Jacobi.
         *
         * Again, all black updates are independent of each other (no black
         * point is a neighbor of another black point), so this phase is
         * also parallelizable.
         */
        for (int i = 0; i < n; i++) {        /* iterate over rows */
            for (int j = 0; j < n; j++) {    /* iterate over columns */
                if ((i + j) % 2 != 1) continue;  /* skip red points (even sum) */

                /*
                 * Fetch 4 neighbors from u. These are all RED points that
                 * were updated in Phase 1, so we're reading the NEWEST values.
                 * This is what makes Gauss-Seidel converge faster than Jacobi.
                 */
                double left  = (j > 0)     ? u[i * n + (j - 1)] : 0.0;  /* left neighbor  (red, NEW) */
                double right = (j < n - 1) ? u[i * n + (j + 1)] : 0.0;  /* right neighbor (red, NEW) */
                double up    = (i > 0)     ? u[(i - 1) * n + j]  : 0.0;  /* upper neighbor (red, NEW) */
                double down  = (i < n - 1) ? u[(i + 1) * n + j]  : 0.0;  /* lower neighbor (red, NEW) */

                double old_val = u[i * n + j];  /* save current value to measure change */

                /* In-place update for this black point */
                u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;

                /* Track convergence across black points too */
                double diff = fabs(u[i * n + j] - old_val);
                if (diff > max_diff) max_diff = diff;
            }
        }

        /*
         * Convergence check: performed after BOTH phases are complete.
         * If the largest change at any grid point (red or black) is below
         * the tolerance, the solution has stabilized and we can stop.
         */
        if (max_diff < tol) {
            iter++;  /* count this completed iteration */
            break;   /* solution has converged - exit the loop */
        }
    }

    return iter;  /* return total number of iterations performed */
}

/**
 * main - Entry point of the serial solver program
 *
 * This function:
 *   1. Parses command-line arguments (grid size, max iterations, tolerance).
 *   2. Allocates memory for the grid arrays.
 *   3. Initializes the right-hand side (RHS) and exact solution.
 *   4. Runs the Jacobi solver and measures its time, residual, and error.
 *   5. Runs the Red-Black GS solver and measures its time, residual, and error.
 *   6. Compares the two solutions against each other.
 *   7. Frees all allocated memory.
 *
 * Usage: ./serial [grid_size] [max_iterations] [tolerance]
 *   - grid_size:      number of interior points per dimension (default: DEFAULT_GRID_SIZE)
 *   - max_iterations: upper limit on solver iterations (default: MAX_ITERATIONS)
 *   - tolerance:      convergence threshold for stopping (default: TOLERANCE)
 */
int main(int argc, char *argv[]) {
    /*
     * Parse command-line arguments.
     * argc counts the number of arguments (including the program name).
     * argv[0] = program name, argv[1] = grid size, argv[2] = max_iter, argv[3] = tolerance.
     * If an argument is not provided, the corresponding default from common.h is used.
     */
    int n        = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;  /* grid size (n x n interior points) */
    int max_iter = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;     /* maximum iterations before giving up */
    double tol   = (argc > 3) ? atof(argv[3]) : TOLERANCE;          /* convergence tolerance (e.g. 1e-6) */

    /* Print configuration so the user knows what parameters are being used */
    printf("Iterative Linear Solvers - Serial Implementation\n");
    printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e\n\n", n, n, max_iter, tol);

    /*
     * Allocate the four arrays needed:
     *
     *   f       - Right-hand side of the equation. Stores h^2 * f(x,y) at each
     *             grid point. Allocated with malloc (will be filled by init_rhs).
     *
     *   u_jac   - Solution array for the Jacobi solver. Allocated with calloc
     *             (zero-initialized) because the initial guess is u = 0 everywhere.
     *
     *   u_gs    - Solution array for the Red-Black GS solver. Also zero-initialized.
     *             This is a SEPARATE array so both solvers start from the same
     *             initial guess and their results can be compared fairly.
     *
     *   u_exact - The known analytical (exact) solution. Used to compute the
     *             error (RMSE) of each solver's result, measuring accuracy.
     */
    double *f       = (double *)malloc((size_t)n * n * sizeof(double));   /* RHS vector */
    double *u_jac   = (double *)calloc((size_t)n * n, sizeof(double));   /* Jacobi solution (starts at 0) */
    double *u_gs    = (double *)calloc((size_t)n * n, sizeof(double));   /* GS solution (starts at 0) */
    double *u_exact = (double *)malloc((size_t)n * n * sizeof(double));  /* exact solution for error check */

    /* Verify all memory allocations succeeded before proceeding */
    if (!f || !u_jac || !u_gs || !u_exact) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;  /* exit with error code */
    }

    /*
     * Initialize the problem:
     *   init_rhs(f, n)       - Fills the RHS array with h^2 * f(x,y) values.
     *                          The source function f(x,y) is defined in common.h.
     *   exact_solution(u_exact, n) - Computes the known analytical solution at
     *                          each grid point for error measurement later.
     */
    init_rhs(f, n);
    exact_solution(u_exact, n);

    /* ======================================================================
     * Run Jacobi Solver
     * ======================================================================
     * Time the solver using get_time() (wall-clock time).
     * After solving, compute two quality metrics:
     *   - Residual: ||Au - f||_inf measures how well the computed solution
     *     satisfies the original equation. Smaller = better.
     *   - RMSE: root-mean-square error vs the exact analytical solution.
     *     Measures the actual accuracy of the numerical solution.
     */
    double t_start = get_time();                                   /* record start time */
    int jac_iters = jacobi_serial(u_jac, f, n, max_iter, tol);    /* run the Jacobi solver */
    double t_jac = get_time() - t_start;                           /* compute elapsed wall-clock time */

    double res_jac  = compute_residual(u_jac, f, n);               /* residual: how well Au = f is satisfied */
    double rmse_jac = compute_rmse(u_jac, u_exact, n);             /* RMSE: accuracy vs exact solution */
    print_results("Jacobi", "Serial", n, jac_iters, t_jac, res_jac, rmse_jac);  /* print summary */

    /* ======================================================================
     * Run Red-Black Gauss-Seidel Solver
     * ======================================================================
     * Same procedure: time the solver, then compute residual and RMSE.
     * We expect GS to converge in fewer iterations than Jacobi, and
     * potentially faster wall-clock time for the same grid size.
     */
    t_start = get_time();                                          /* record start time */
    int gs_iters = redblack_gs_serial(u_gs, f, n, max_iter, tol); /* run the GS solver */
    double t_gs = get_time() - t_start;                            /* compute elapsed wall-clock time */

    double res_gs  = compute_residual(u_gs, f, n);                 /* residual norm */
    double rmse_gs = compute_rmse(u_gs, u_exact, n);               /* RMSE vs exact solution */
    print_results("Red-Black Gauss-Seidel", "Serial", n, gs_iters, t_gs, res_gs, rmse_gs);  /* print summary */

    /* ======================================================================
     * Compare Jacobi vs Gauss-Seidel Solutions
     * ======================================================================
     * Both methods solve the same equation, so their solutions should be
     * very close (especially if both converged). This RMSE measures the
     * difference between the two numerical solutions, providing a sanity
     * check that both solvers are working correctly. A small value (e.g. 1e-6)
     * indicates both methods converged to essentially the same answer.
     */
    double rmse_jac_vs_gs = compute_rmse(u_jac, u_gs, n);
    printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);

    /*
     * Save the final numerical solutions to text files.
     * These files can be used later for plotting, visual inspection, or
     * comparing the serial baseline against OpenMP/MPI/CUDA implementations.
     */
    save_solution("serial_jacobi.txt", u_jac, n);  /* write Jacobi solution grid */
    save_solution("serial_rbgs.txt", u_gs, n);     /* write Red-Black GS solution grid */

    /*
     * Free all dynamically allocated memory.
     * Good practice to avoid memory leaks, even though the OS would
     * reclaim memory when the process exits.
     */
    free(f);        /* free the RHS vector */
    free(u_jac);    /* free the Jacobi solution array */
    free(u_gs);     /* free the Gauss-Seidel solution array */
    free(u_exact);  /* free the exact solution array */

    return 0;  /* exit successfully */
}
