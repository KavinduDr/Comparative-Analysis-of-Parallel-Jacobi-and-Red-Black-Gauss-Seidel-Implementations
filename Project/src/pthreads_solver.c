/**
 * pthreads_solver.c - POSIX Threads implementations of Jacobi and Red-Black GS
 *
 * Core Multi-threading Barrier Shim & Thread Design by:
 *   Rathnayaka I.G.T.A. (EG/2021/4754 — Tharanga Anuradha)
 *
 * Shared memory parallelization using POSIX threads with barrier synchronization.
 * Each thread is assigned a contiguous block of rows to process.
 *
 * Usage: ./pthreads_solver [grid_size] [max_iterations] [tolerance] [num_threads]
 */

#include "../include/common.h"  /* Include shared utilities: grid setup, timing, error computation */
#include <pthread.h>            /* POSIX threads: pthread_create, pthread_join, barriers, mutexes */

#ifdef __APPLE__
#include <errno.h>
#ifndef PTHREAD_BARRIER_H_
#define PTHREAD_BARRIER_H_

typedef int pthread_barrierattr_t;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int tripCount;
} pthread_barrier_t;

static inline int pthread_barrier_init(pthread_barrier_t *barrier, const pthread_barrierattr_t *attr, unsigned int count) {
    if (count == 0) {
        errno = EINVAL;
        return -1;
    }
    if (pthread_mutex_init(&barrier->mutex, 0) < 0) return -1;
    if (pthread_cond_init(&barrier->cond, 0) < 0) {
        pthread_mutex_destroy(&barrier->mutex);
        return -1;
    }
    barrier->tripCount = count;
    barrier->count = 0;
    return 0;
}

static inline int pthread_barrier_destroy(pthread_barrier_t *barrier) {
    pthread_cond_destroy(&barrier->cond);
    pthread_mutex_destroy(&barrier->mutex);
    return 0;
}

static inline int pthread_barrier_wait(pthread_barrier_t *barrier) {
    pthread_mutex_lock(&barrier->mutex);
    ++(barrier->count);
    if (barrier->count >= barrier->tripCount) {
        barrier->count = 0;
        pthread_cond_broadcast(&barrier->cond);
        pthread_mutex_unlock(&barrier->mutex);
        return 1;
    } else {
        pthread_cond_wait(&barrier->cond, &barrier->mutex);
        pthread_mutex_unlock(&barrier->mutex);
        return 0;
    }
}
#endif // PTHREAD_BARRIER_H_
#endif // __APPLE__

/* Shared data accessible by all threads */
typedef struct {                    /* Structure holding data shared across all threads */
    double *u;                      /* Current solution array (shared, written by all threads) */
    double *u_old;                  /* Previous iteration values (shared, read-only during compute) */
    const double *f;                /* RHS vector (shared, read-only) */
    int n;                          /* Grid dimension (n x n interior points) */
    int max_iter;                   /* Maximum number of iterations */
    double tol;                     /* Convergence tolerance */
    int num_threads;                /* Total number of threads */
    int total_iters;                /* Output: total iterations completed (set by thread 0) */
    int converged;                  /* Flag: 1 if solution converged, 0 otherwise */
    double global_max_diff;         /* Global maximum change across all threads (reduced via mutex) */
    pthread_barrier_t barrier;      /* Barrier for synchronizing all threads between phases */
    pthread_mutex_t mutex;          /* Mutex for thread-safe update of global_max_diff */
} shared_data_t;

/* Per-thread arguments */
typedef struct {          /* Structure holding arguments specific to each thread */
    int tid;              /* Thread ID (0 to num_threads-1) */
    int row_start;        /* First row this thread is responsible for */
    int row_end;          /* One past the last row (exclusive upper bound) */
    shared_data_t *shared; /* Pointer to the shared data structure */
} thread_arg_t;

/**
 * Jacobi worker thread
 *
 * Each thread processes rows [row_start, row_end).
 * Barrier synchronization between: copy, compute, convergence check.
 */
void *jacobi_worker(void *arg) {  /* Thread function for Jacobi iteration */
    thread_arg_t *targ = (thread_arg_t *)arg;  /* Cast void pointer to thread argument struct */
    shared_data_t *s = targ->shared;           /* Get pointer to shared data */
    int n = s->n;                              /* Local copy of grid size for convenience */

    for (int iter = 0; iter < s->max_iter; iter++) {  /* Iterate until convergence or max iterations */
        /* Step 1: Copy u to u_old (each thread copies its own rows) */
        for (int i = targ->row_start; i < targ->row_end; i++) {  /* Loop over this thread's assigned rows */
            memcpy(&s->u_old[i * n], &s->u[i * n], n * sizeof(double));  /* Copy one row from u to u_old */
        }
        pthread_barrier_wait(&s->barrier);  /* Wait until ALL threads finish copying (ensures u_old is complete) */

        /* Step 2: Compute new values */
        double local_max_diff = 0.0;  /* Track maximum change within this thread's rows */
        for (int i = targ->row_start; i < targ->row_end; i++) {  /* Loop over assigned rows */
            for (int j = 0; j < n; j++) {  /* Loop over columns */
                double left  = (j > 0)     ? s->u_old[i * n + (j - 1)] : 0.0;  /* Left neighbor from old values */
                double right = (j < n - 1) ? s->u_old[i * n + (j + 1)] : 0.0;  /* Right neighbor from old values */
                double up    = (i > 0)     ? s->u_old[(i - 1) * n + j]  : 0.0;  /* Upper neighbor from old values */
                double down  = (i < n - 1) ? s->u_old[(i + 1) * n + j]  : 0.0;  /* Lower neighbor from old values */

                s->u[i * n + j] = (left + right + up + down + s->f[i * n + j]) / 4.0;  /* Compute new value using Jacobi formula */

                double diff = fabs(s->u[i * n + j] - s->u_old[i * n + j]);  /* Compute change from previous iteration */
                if (diff > local_max_diff) local_max_diff = diff;  /* Update thread-local maximum change */
            }
        }

        /* Step 3: Reduce max_diff across threads */
        pthread_mutex_lock(&s->mutex);  /* Acquire mutex for thread-safe access to global_max_diff */
        if (local_max_diff > s->global_max_diff)
            s->global_max_diff = local_max_diff;  /* Update global max if this thread's max is larger */
        pthread_mutex_unlock(&s->mutex);  /* Release mutex */

        pthread_barrier_wait(&s->barrier);  /* Wait until all threads have contributed their local max */

        /* Thread 0 checks convergence and resets global_max_diff */
        if (targ->tid == 0) {  /* Only thread 0 performs convergence check */
            if (s->global_max_diff < s->tol) {  /* Check if global max change is below tolerance */
                s->converged = 1;           /* Set converged flag */
                s->total_iters = iter + 1;  /* Record iteration count */
            }
            s->global_max_diff = 0.0;  /* Reset global max for next iteration */
        }
        pthread_barrier_wait(&s->barrier);  /* Wait until thread 0 finishes convergence check */

        if (s->converged) break;  /* All threads exit if convergence is achieved */
    }

    /* If not converged, thread 0 records max iterations */
    if (targ->tid == 0 && !s->converged) {  /* If max iterations reached without convergence */
        s->total_iters = s->max_iter;  /* Set total iterations to max */
    }

    return NULL;  /* Thread exits */
}

/**
 * Red-Black GS worker thread
 *
 * Each thread processes rows [row_start, row_end).
 * Two barriers per iteration: one after red phase, one after black phase.
 */
void *redblack_gs_worker(void *arg) {  /* Thread function for Red-Black Gauss-Seidel */
    thread_arg_t *targ = (thread_arg_t *)arg;  /* Cast to thread argument struct */
    shared_data_t *s = targ->shared;           /* Get shared data pointer */
    int n = s->n;                              /* Local copy of grid size */

    for (int iter = 0; iter < s->max_iter; iter++) {  /* Main iteration loop */
        double local_max_diff = 0.0;  /* Track max change within this thread's rows */

        /* Phase 1: Update RED points */
        for (int i = targ->row_start; i < targ->row_end; i++) {  /* Loop over assigned rows */
            for (int j = 0; j < n; j++) {  /* Loop over columns */
                if ((i + j) % 2 != 0) continue;  /* Skip black points: only process red where (i+j) is even */

                double left  = (j > 0)     ? s->u[i * n + (j - 1)] : 0.0;  /* Left neighbor */
                double right = (j < n - 1) ? s->u[i * n + (j + 1)] : 0.0;  /* Right neighbor */
                double up    = (i > 0)     ? s->u[(i - 1) * n + j]  : 0.0;  /* Upper neighbor */
                double down  = (i < n - 1) ? s->u[(i + 1) * n + j]  : 0.0;  /* Lower neighbor */

                double old_val = s->u[i * n + j];  /* Save current value */
                s->u[i * n + j] = (left + right + up + down + s->f[i * n + j]) / 4.0;  /* Update red point in-place */

                double diff = fabs(s->u[i * n + j] - old_val);  /* Compute change */
                if (diff > local_max_diff) local_max_diff = diff;  /* Update local max */
            }
        }
        pthread_barrier_wait(&s->barrier); /* Ensure all red points are updated before black phase */

        /* Phase 2: Update BLACK points */
        for (int i = targ->row_start; i < targ->row_end; i++) {  /* Loop over assigned rows */
            for (int j = 0; j < n; j++) {  /* Loop over columns */
                if ((i + j) % 2 != 1) continue;  /* Skip red points: only process black where (i+j) is odd */

                double left  = (j > 0)     ? s->u[i * n + (j - 1)] : 0.0;  /* Left neighbor (red, freshly updated) */
                double right = (j < n - 1) ? s->u[i * n + (j + 1)] : 0.0;  /* Right neighbor (red, freshly updated) */
                double up    = (i > 0)     ? s->u[(i - 1) * n + j]  : 0.0;  /* Upper neighbor (red, freshly updated) */
                double down  = (i < n - 1) ? s->u[(i + 1) * n + j]  : 0.0;  /* Lower neighbor (red, freshly updated) */

                double old_val = s->u[i * n + j];  /* Save current black value */
                s->u[i * n + j] = (left + right + up + down + s->f[i * n + j]) / 4.0;  /* Update black point */

                double diff = fabs(s->u[i * n + j] - old_val);  /* Compute change */
                if (diff > local_max_diff) local_max_diff = diff;  /* Update local max */
            }
        }

        /* Reduce max_diff */
        pthread_mutex_lock(&s->mutex);  /* Lock mutex for thread-safe global max update */
        if (local_max_diff > s->global_max_diff)
            s->global_max_diff = local_max_diff;  /* Update global max if this thread's is larger */
        pthread_mutex_unlock(&s->mutex);  /* Unlock mutex */

        pthread_barrier_wait(&s->barrier);  /* Wait for all threads to finish and contribute max_diff */

        if (targ->tid == 0) {  /* Thread 0 checks convergence */
            if (s->global_max_diff < s->tol) {  /* Converged if max change < tolerance */
                s->converged = 1;           /* Set flag */
                s->total_iters = iter + 1;  /* Record iteration count */
            }
            s->global_max_diff = 0.0;  /* Reset for next iteration */
        }
        pthread_barrier_wait(&s->barrier);  /* Ensure convergence decision is visible to all threads */

        if (s->converged) break;  /* Exit if converged */
    }

    if (targ->tid == 0 && !s->converged) {  /* If max iterations reached */
        s->total_iters = s->max_iter;  /* Record max iterations */
    }

    return NULL;  /* Thread exits */
}

/**
 * Launch a solver using pthreads
 * worker_func: either jacobi_worker or redblack_gs_worker
 */
int run_pthreads_solver(double *u, const double *f, int n,
                        int max_iter, double tol, int num_threads,
                        void *(*worker_func)(void *)) {  /* Generic launcher: takes a worker function pointer */
    shared_data_t shared;  /* Create shared data structure on the stack */
    shared.u = u;  /* Set solution array pointer */
    shared.u_old = (double *)calloc((size_t)n * n, sizeof(double));  /* Allocate old values array */
    shared.f = f;  /* Set RHS pointer */
    shared.n = n;  /* Set grid size */
    shared.max_iter = max_iter;  /* Set max iterations */
    shared.tol = tol;  /* Set convergence tolerance */
    shared.num_threads = num_threads;  /* Set thread count */
    shared.total_iters = max_iter;  /* Initialize to max (updated if converges earlier) */
    shared.converged = 0;  /* Not yet converged */
    shared.global_max_diff = 0.0;  /* Initialize global max difference */

    if (!shared.u_old) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }  /* Check allocation */

    pthread_barrier_init(&shared.barrier, NULL, num_threads);  /* Initialize barrier for num_threads participants */
    pthread_mutex_init(&shared.mutex, NULL);  /* Initialize mutex with default attributes */

    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));  /* Allocate array of thread handles */
    thread_arg_t *args = (thread_arg_t *)malloc(num_threads * sizeof(thread_arg_t));  /* Allocate array of per-thread arguments */

    /* Distribute rows evenly across threads */
    int rows_per_thread = n / num_threads;  /* Base number of rows per thread */
    int extra_rows = n % num_threads;  /* Remainder rows to distribute */

    int row = 0;  /* Current row counter for distribution */
    for (int t = 0; t < num_threads; t++) {  /* Set up arguments for each thread */
        args[t].tid = t;  /* Assign thread ID */
        args[t].shared = &shared;  /* Point to shared data */
        args[t].row_start = row;  /* Set starting row for this thread */
        args[t].row_end = row + rows_per_thread + (t < extra_rows ? 1 : 0);  /* Set ending row (first 'extra_rows' threads get one more) */
        row = args[t].row_end;  /* Advance row counter */
        pthread_create(&threads[t], NULL, worker_func, &args[t]);  /* Create thread with the specified worker function */
    }

    for (int t = 0; t < num_threads; t++) {  /* Wait for all threads to finish */
        pthread_join(threads[t], NULL);  /* Block until thread t completes */
    }

    int iters = shared.total_iters;  /* Retrieve total iterations from shared data */

    pthread_barrier_destroy(&shared.barrier);  /* Destroy the barrier (free resources) */
    pthread_mutex_destroy(&shared.mutex);      /* Destroy the mutex */
    free(shared.u_old);  /* Free old values array */
    free(threads);       /* Free thread handles array */
    free(args);          /* Free thread arguments array */

    return iters;  /* Return number of iterations performed */
}

int main(int argc, char *argv[]) {  /* Entry point */
    int n           = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;  /* Parse grid size or default */
    int max_iter    = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;     /* Parse max iterations or default */
    double tol      = (argc > 3) ? atof(argv[3]) : TOLERANCE;          /* Parse tolerance or default */
    int num_threads = (argc > 4) ? atoi(argv[4]) : 4;                  /* Parse thread count or default (4) */

    printf("Iterative Linear Solvers - POSIX Threads Implementation\n");  /* Print header */
    printf("Grid: %d x %d, Max Iterations: %d, Tolerance: %.2e, Threads: %d\n\n",
           n, n, max_iter, tol, num_threads);  /* Print configuration */

    double *f       = (double *)malloc((size_t)n * n * sizeof(double));   /* Allocate RHS vector */
    double *u_jac   = (double *)calloc((size_t)n * n, sizeof(double));   /* Allocate Jacobi solution (zero-initialized) */
    double *u_gs    = (double *)calloc((size_t)n * n, sizeof(double));   /* Allocate GS solution (zero-initialized) */
    double *u_exact = (double *)malloc((size_t)n * n * sizeof(double));  /* Allocate exact solution */

    if (!f || !u_jac || !u_gs || !u_exact) {  /* Check all allocations */
        fprintf(stderr, "Memory allocation failed\n");  /* Print error */
        return 1;  /* Exit with error */
    }

    init_rhs(f, n);              /* Initialize RHS vector */
    exact_solution(u_exact, n);  /* Compute exact solution */

    /* ---- Jacobi ---- */
    double t_start = get_time();  /* Record start time */
    int jac_iters = run_pthreads_solver(u_jac, f, n, max_iter, tol, num_threads, jacobi_worker);  /* Run Jacobi with pthreads */
    double t_jac = get_time() - t_start;  /* Compute elapsed time */

    double res_jac  = compute_residual(u_jac, f, n);    /* Compute Jacobi residual */
    double rmse_jac = compute_rmse(u_jac, u_exact, n);  /* Compute Jacobi RMSE */
    print_results("Jacobi", "Pthreads", n, jac_iters, t_jac, res_jac, rmse_jac);  /* Print results */

    /* ---- Red-Black Gauss-Seidel ---- */
    t_start = get_time();  /* Record start time */
    int gs_iters = run_pthreads_solver(u_gs, f, n, max_iter, tol, num_threads, redblack_gs_worker);  /* Run Red-Black GS with pthreads */
    double t_gs = get_time() - t_start;  /* Compute elapsed time */

    double res_gs  = compute_residual(u_gs, f, n);    /* Compute GS residual */
    double rmse_gs = compute_rmse(u_gs, u_exact, n);  /* Compute GS RMSE */
    print_results("Red-Black Gauss-Seidel", "Pthreads", n, gs_iters, t_gs, res_gs, rmse_gs);  /* Print results */

    double rmse_jac_vs_gs = compute_rmse(u_jac, u_gs, n);  /* Compare the two solutions */
    printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);  /* Print comparison */

    save_solution("pthreads_jacobi.txt", u_jac, n);
    save_solution("pthreads_rbgs.txt", u_gs, n);

    free(f);        /* Free RHS */
    free(u_jac);    /* Free Jacobi solution */
    free(u_gs);     /* Free GS solution */
    free(u_exact);  /* Free exact solution */

    return 0;  /* Exit successfully */
}
