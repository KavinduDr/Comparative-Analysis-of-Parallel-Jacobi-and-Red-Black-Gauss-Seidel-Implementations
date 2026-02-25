/**
 * pthreads_solver.c - POSIX Threads implementations of Jacobi and Red-Black GS
 *
 * Shared memory parallelization using POSIX threads with barrier synchronization.
 * Each thread is assigned a contiguous block of rows to process.
 *
 * Usage: ./pthreads_solver [grid_size] [max_iterations] [tolerance] [num_threads]
 */

#include "../include/common.h"
#include <pthread.h>

/* Shared data accessible by all threads */
typedef struct {
    double *u;
    double *u_old;
    const double *f;
    int n;
    int max_iter;
    double tol;
    int num_threads;
    int total_iters;
    int converged;
    double global_max_diff;
    pthread_barrier_t barrier;
    pthread_mutex_t mutex;
} shared_data_t;

/* Per-thread arguments */
typedef struct {
    int tid;
    int row_start;
    int row_end;
    shared_data_t *shared;
} thread_arg_t;

/**
 * Jacobi worker thread
 *
 * Each thread processes rows [row_start, row_end).
 * Barrier synchronization between: copy, compute, convergence check.
 */
void *jacobi_worker(void *arg) {
    thread_arg_t *targ = (thread_arg_t *)arg;
    shared_data_t *s = targ->shared;
    int n = s->n;

    for (int iter = 0; iter < s->max_iter; iter++) {
        /* Step 1: Copy u to u_old (each thread copies its own rows) */
        for (int i = targ->row_start; i < targ->row_end; i++) {
            memcpy(&s->u_old[i * n], &s->u[i * n], n * sizeof(double));
        }
        pthread_barrier_wait(&s->barrier);

        /* Step 2: Compute new values */
        double local_max_diff = 0.0;
        for (int i = targ->row_start; i < targ->row_end; i++) {
            for (int j = 0; j < n; j++) {
                double left  = (j > 0)     ? s->u_old[i * n + (j - 1)] : 0.0;
                double right = (j < n - 1) ? s->u_old[i * n + (j + 1)] : 0.0;
                double up    = (i > 0)     ? s->u_old[(i - 1) * n + j]  : 0.0;
                double down  = (i < n - 1) ? s->u_old[(i + 1) * n + j]  : 0.0;

                s->u[i * n + j] = (left + right + up + down + s->f[i * n + j]) / 4.0;

                double diff = fabs(s->u[i * n + j] - s->u_old[i * n + j]);
                if (diff > local_max_diff) local_max_diff = diff;
            }
        }

        /* Step 3: Reduce max_diff across threads */
        pthread_mutex_lock(&s->mutex);
        if (local_max_diff > s->global_max_diff)
            s->global_max_diff = local_max_diff;
        pthread_mutex_unlock(&s->mutex);

        pthread_barrier_wait(&s->barrier);

        /* Thread 0 checks convergence and resets global_max_diff */
        if (targ->tid == 0) {
            if (s->global_max_diff < s->tol) {
                s->converged = 1;
                s->total_iters = iter + 1;
            }
            s->global_max_diff = 0.0;
        }
        pthread_barrier_wait(&s->barrier);

        if (s->converged) break;
    }

    /* If not converged, thread 0 records max iterations */
    if (targ->tid == 0 && !s->converged) {
        s->total_iters = s->max_iter;
    }

    return NULL;
}

/**
 * Red-Black GS worker thread
 *
 * Each thread processes rows [row_start, row_end).
 * Two barriers per iteration: one after red phase, one after black phase.
 */
void *redblack_gs_worker(void *arg) {
    thread_arg_t *targ = (thread_arg_t *)arg;
    shared_data_t *s = targ->shared;
    int n = s->n;

    for (int iter = 0; iter < s->max_iter; iter++) {
        double local_max_diff = 0.0;

        /* Phase 1: Update RED points */
        for (int i = targ->row_start; i < targ->row_end; i++) {
            for (int j = 0; j < n; j++) {
                if ((i + j) % 2 != 0) continue;

                double left  = (j > 0)     ? s->u[i * n + (j - 1)] : 0.0;
                double right = (j < n - 1) ? s->u[i * n + (j + 1)] : 0.0;
                double up    = (i > 0)     ? s->u[(i - 1) * n + j]  : 0.0;
                double down  = (i < n - 1) ? s->u[(i + 1) * n + j]  : 0.0;

                double old_val = s->u[i * n + j];
                s->u[i * n + j] = (left + right + up + down + s->f[i * n + j]) / 4.0;

                double diff = fabs(s->u[i * n + j] - old_val);
                if (diff > local_max_diff) local_max_diff = diff;
            }
        }
        pthread_barrier_wait(&s->barrier); /* Ensure all red points updated */

        /* Phase 2: Update BLACK points */
        for (int i = targ->row_start; i < targ->row_end; i++) {
            for (int j = 0; j < n; j++) {
                if ((i + j) % 2 != 1) continue;

                double left  = (j > 0)     ? s->u[i * n + (j - 1)] : 0.0;
                double right = (j < n - 1) ? s->u[i * n + (j + 1)] : 0.0;
                double up    = (i > 0)     ? s->u[(i - 1) * n + j]  : 0.0;
                double down  = (i < n - 1) ? s->u[(i + 1) * n + j]  : 0.0;

                double old_val = s->u[i * n + j];
                s->u[i * n + j] = (left + right + up + down + s->f[i * n + j]) / 4.0;

                double diff = fabs(s->u[i * n + j] - old_val);
                if (diff > local_max_diff) local_max_diff = diff;
            }
        }

        /* Reduce max_diff */
        pthread_mutex_lock(&s->mutex);
        if (local_max_diff > s->global_max_diff)
            s->global_max_diff = local_max_diff;
        pthread_mutex_unlock(&s->mutex);

        pthread_barrier_wait(&s->barrier);

        if (targ->tid == 0) {
            if (s->global_max_diff < s->tol) {
                s->converged = 1;
                s->total_iters = iter + 1;
            }
            s->global_max_diff = 0.0;
        }
        pthread_barrier_wait(&s->barrier);

        if (s->converged) break;
    }

    if (targ->tid == 0 && !s->converged) {
        s->total_iters = s->max_iter;
    }

    return NULL;
}

/**
 * Launch a solver using pthreads
 * worker_func: either jacobi_worker or redblack_gs_worker
 */
int run_pthreads_solver(double *u, const double *f, int n,
                        int max_iter, double tol, int num_threads,
                        void *(*worker_func)(void *)) {
    shared_data_t shared;
    shared.u = u;
    shared.u_old = (double *)calloc((size_t)n * n, sizeof(double));
    shared.f = f;
    shared.n = n;
    shared.max_iter = max_iter;
    shared.tol = tol;
    shared.num_threads = num_threads;
    shared.total_iters = max_iter;
    shared.converged = 0;
    shared.global_max_diff = 0.0;

    if (!shared.u_old) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }

    pthread_barrier_init(&shared.barrier, NULL, num_threads);
    pthread_mutex_init(&shared.mutex, NULL);

    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    thread_arg_t *args = (thread_arg_t *)malloc(num_threads * sizeof(thread_arg_t));

    /* Distribute rows evenly across threads */
    int rows_per_thread = n / num_threads;
    int extra_rows = n % num_threads;

    int row = 0;
    for (int t = 0; t < num_threads; t++) {
        args[t].tid = t;
        args[t].shared = &shared;
        args[t].row_start = row;
        args[t].row_end = row + rows_per_thread + (t < extra_rows ? 1 : 0);
        row = args[t].row_end;
        pthread_create(&threads[t], NULL, worker_func, &args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    int iters = shared.total_iters;

    pthread_barrier_destroy(&shared.barrier);
    pthread_mutex_destroy(&shared.mutex);
    free(shared.u_old);
    free(threads);
    free(args);

    return iters;
}

int main(int argc, char *argv[]) {
    int n           = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;
    int max_iter    = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;
    double tol      = (argc > 3) ? atof(argv[3]) : TOLERANCE;
    int num_threads = (argc > 4) ? atoi(argv[4]) : 4;

    printf("Iterative Linear Solvers - POSIX Threads Implementation\n");
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
    int jac_iters = run_pthreads_solver(u_jac, f, n, max_iter, tol, num_threads, jacobi_worker);
    double t_jac = get_time() - t_start;

    double res_jac  = compute_residual(u_jac, f, n);
    double rmse_jac = compute_rmse(u_jac, u_exact, n);
    print_results("Jacobi", "Pthreads", n, jac_iters, t_jac, res_jac, rmse_jac);

    /* ---- Red-Black Gauss-Seidel ---- */
    t_start = get_time();
    int gs_iters = run_pthreads_solver(u_gs, f, n, max_iter, tol, num_threads, redblack_gs_worker);
    double t_gs = get_time() - t_start;

    double res_gs  = compute_residual(u_gs, f, n);
    double rmse_gs = compute_rmse(u_gs, u_exact, n);
    print_results("Red-Black Gauss-Seidel", "Pthreads", n, gs_iters, t_gs, res_gs, rmse_gs);

    double rmse_jac_vs_gs = compute_rmse(u_jac, u_gs, n);
    printf("RMSE (Jacobi vs Red-Black GS): %.2e\n", rmse_jac_vs_gs);

    free(f);
    free(u_jac);
    free(u_gs);
    free(u_exact);

    return 0;
}
