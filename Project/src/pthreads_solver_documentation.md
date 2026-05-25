# Understanding `pthreads_solver.c`

This document explains the code in [`src/pthreads_solver.c`](pthreads_solver.c).
The file implements parallel versions of the Jacobi method and Red-Black
Gauss-Seidel method using POSIX Threads, usually called Pthreads.

The mathematical problem is the same as the serial version:

```text
-u_xx - u_yy = f(x, y)
```

on the unit square with zero boundary values:

```text
u = 0 on all four edges
```

The difference is that this file splits the grid rows across multiple CPU
threads so several parts of the grid can be updated at the same time.

---

## 1. Purpose of This File

`pthreads_solver.c` is the shared-memory parallel implementation.

It contains:

- A Pthreads Jacobi solver
- A Pthreads Red-Black Gauss-Seidel solver
- A generic thread launcher used by both methods
- Timing, error checking, and output code in `main`

This implementation is useful for comparing:

- Serial execution versus thread-parallel execution
- Jacobi versus Red-Black Gauss-Seidel
- Pthreads versus OpenMP, MPI, CUDA, and hybrid versions

---

## 2. Header Files

The file includes:

```c
#include "../include/common.h"
#include <pthread.h>
```

`common.h` provides project helper functions:

| Function or Constant | Purpose |
|---|---|
| `DEFAULT_GRID_SIZE` | Default grid size |
| `MAX_ITERATIONS` | Default maximum iteration count |
| `TOLERANCE` | Default stopping tolerance |
| `get_time()` | Measures wall-clock time |
| `init_rhs()` | Initializes the right-hand side array |
| `exact_solution()` | Computes the known exact solution |
| `compute_residual()` | Computes the residual norm |
| `compute_rmse()` | Computes root-mean-square error |
| `print_results()` | Prints solver performance and accuracy |
| `save_solution()` | Saves the final solution grid |

`pthread.h` provides the Pthreads API:

| Pthreads Item | Purpose |
|---|---|
| `pthread_create` | Starts a new worker thread |
| `pthread_join` | Waits for a thread to finish |
| `pthread_barrier_t` | Synchronizes all threads at a checkpoint |
| `pthread_mutex_t` | Protects shared data from simultaneous writes |

---

## 3. Grid Storage

The mathematical grid is 2D, but the C program stores it as a 1D array.

For an `n x n` grid:

```text
2D point (i, j) -> 1D index i * n + j
```

In code:

```c
u[i * n + j]
```

means row `i`, column `j`.

Each update uses the 5-point stencil:

```text
u(i,j) = (left + right + up + down + f(i,j)) / 4
```

If a neighbor is outside the grid, the code uses `0.0` because the boundary
condition is zero.

---

## 4. Parallelization Strategy

The code uses row decomposition.

Each thread receives a contiguous block of rows:

```text
Thread 0 -> rows 0 to a
Thread 1 -> rows a to b
Thread 2 -> rows b to c
...
```

The row range for each thread is stored as:

```c
row_start
row_end
```

The thread processes rows:

```text
[row_start, row_end)
```

This means `row_start` is included and `row_end` is excluded.

### Why Split by Rows?

Splitting by rows is simple and efficient:

- Each thread works on a different part of the grid.
- Threads do not write to the same grid points.
- Memory access stays mostly contiguous.
- It is easy to distribute extra rows when `n` is not divisible by the number of
  threads.

---

## 5. Shared Data Structure

The shared structure is:

```c
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
```

This structure stores values that all threads need.

| Field | Meaning |
|---|---|
| `u` | Current solution array |
| `u_old` | Previous iteration array, used by Jacobi |
| `f` | Right-hand side array, read-only |
| `n` | Grid size |
| `max_iter` | Maximum number of iterations |
| `tol` | Convergence tolerance |
| `num_threads` | Number of Pthreads workers |
| `total_iters` | Final iteration count |
| `converged` | Shared flag telling threads to stop |
| `global_max_diff` | Largest update difference across all threads |
| `barrier` | Synchronization point for all threads |
| `mutex` | Lock used when updating `global_max_diff` |

### Why Shared Data Is Needed

Each thread needs access to the same arrays, tolerance, iteration limit, and
synchronization objects. Passing one shared structure avoids duplicating that
information.

---

## 6. Per-Thread Argument Structure

Each thread also receives a small private argument structure:

```c
typedef struct {
    int tid;
    int row_start;
    int row_end;
    shared_data_t *shared;
} thread_arg_t;
```

| Field | Meaning |
|---|---|
| `tid` | Thread ID, from `0` to `num_threads - 1` |
| `row_start` | First row handled by this thread |
| `row_end` | One past the final row handled by this thread |
| `shared` | Pointer to the shared data structure |

The shared structure contains global problem data. The thread argument structure
tells each thread which rows it owns.

---

## 7. Function: `jacobi_worker`

Function signature:

```c
void *jacobi_worker(void *arg)
```

This is the worker function used by each Pthreads thread for the Jacobi method.

### Jacobi Reminder

Jacobi updates each grid point using only values from the previous iteration.
That is why it needs two arrays:

- `u_old` for previous values
- `u` for new values

### Jacobi Worker Flow

Each thread performs the same loop, but only for its assigned rows.

```text
for each iteration:
    copy this thread's rows from u to u_old
    wait until all threads finish copying

    compute new values for this thread's rows
    track local maximum difference

    lock mutex
    update global maximum difference
    unlock mutex

    wait until all threads finish reduction

    thread 0 checks convergence
    thread 0 resets global_max_diff

    wait until convergence decision is visible to all threads

    if converged:
        exit loop
```

### Step 1: Copy Rows

```c
for (int i = targ->row_start; i < targ->row_end; i++) {
    memcpy(&s->u_old[i * n], &s->u[i * n], n * sizeof(double));
}
```

Each thread copies only its own rows.

Then all threads wait:

```c
pthread_barrier_wait(&s->barrier);
```

This barrier is required. A thread must not start computing with `u_old` until
every thread has finished copying its portion of `u_old`.

### Step 2: Compute New Values

Each thread loops over its assigned rows:

```c
for (int i = targ->row_start; i < targ->row_end; i++) {
    for (int j = 0; j < n; j++) {
        ...
    }
}
```

Neighbor values are read from `u_old`:

```c
double left  = (j > 0)     ? s->u_old[i * n + (j - 1)] : 0.0;
double right = (j < n - 1) ? s->u_old[i * n + (j + 1)] : 0.0;
double up    = (i > 0)     ? s->u_old[(i - 1) * n + j] : 0.0;
double down  = (i < n - 1) ? s->u_old[(i + 1) * n + j] : 0.0;
```

The new value is written to `u`:

```c
s->u[i * n + j] =
    (left + right + up + down + s->f[i * n + j]) / 4.0;
```

This is safe because each thread writes only to its own rows.

### Step 3: Local Maximum Difference

Each thread computes its own maximum update change:

```c
double diff = fabs(s->u[i * n + j] - s->u_old[i * n + j]);
if (diff > local_max_diff) local_max_diff = diff;
```

This avoids using a lock inside every grid-point update.

### Step 4: Global Maximum Difference

After finishing its rows, each thread updates the shared global maximum:

```c
pthread_mutex_lock(&s->mutex);
if (local_max_diff > s->global_max_diff)
    s->global_max_diff = local_max_diff;
pthread_mutex_unlock(&s->mutex);
```

The mutex is needed because multiple threads could otherwise write
`global_max_diff` at the same time.

### Step 5: Convergence Check

Only thread `0` checks convergence:

```c
if (targ->tid == 0) {
    if (s->global_max_diff < s->tol) {
        s->converged = 1;
        s->total_iters = iter + 1;
    }
    s->global_max_diff = 0.0;
}
```

Then all threads wait at a barrier before reading `s->converged`.

---

## 8. Function: `redblack_gs_worker`

Function signature:

```c
void *redblack_gs_worker(void *arg)
```

This is the worker function used by each Pthreads thread for Red-Black
Gauss-Seidel.

### Red-Black Reminder

The grid is split like a checkerboard:

```text
R B R B
B R B R
R B R B
B R B R
```

Color rule:

```text
red   -> (i + j) is even
black -> (i + j) is odd
```

Each iteration has two phases:

1. Update red points.
2. Update black points.

### Why Barriers Are Important

All red points must be finished before any thread starts updating black points.

That is why the code uses:

```c
pthread_barrier_wait(&s->barrier);
```

between the red phase and black phase.

Without this barrier, one thread could start updating black points while another
thread is still updating red points. That would break the Red-Black
Gauss-Seidel ordering.

### Red Phase

The red phase skips black points:

```c
if ((i + j) % 2 != 0) continue;
```

Only points where `(i + j)` is even are updated.

### Black Phase

The black phase skips red points:

```c
if ((i + j) % 2 != 1) continue;
```

Only points where `(i + j)` is odd are updated.

### Red-Black Worker Flow

```text
for each iteration:
    local_max_diff = 0

    red phase:
        update red points in this thread's rows
        track local maximum difference

    wait until all red updates are complete

    black phase:
        update black points in this thread's rows
        track local maximum difference

    lock mutex
    update global maximum difference
    unlock mutex

    wait until all threads finish reduction

    thread 0 checks convergence
    thread 0 resets global_max_diff

    wait until convergence decision is visible to all threads

    if converged:
        exit loop
```

### In-Place Update

Unlike Jacobi, Red-Black Gauss-Seidel updates directly in `u`:

```c
s->u[i * n + j] =
    (left + right + up + down + s->f[i * n + j]) / 4.0;
```

It does not need `u_old` for the algorithm. The generic launcher still allocates
`u_old` because it is shared by both solver launch paths, but the Red-Black
worker does not use it.

---

## 9. Function: `run_pthreads_solver`

Function signature:

```c
int run_pthreads_solver(
    double *u,
    const double *f,
    int n,
    int max_iter,
    double tol,
    int num_threads,
    void *(*worker_func)(void *)
)
```

This function launches either Pthreads solver. The final parameter is a function
pointer:

```c
void *(*worker_func)(void *)
```

That means the caller can pass either:

```c
jacobi_worker
```

or:

```c
redblack_gs_worker
```

### What This Function Does

```text
create shared data
allocate u_old
initialize barrier and mutex
allocate thread handles
allocate per-thread argument structures
split rows across threads
create all worker threads
join all worker threads
destroy barrier and mutex
free temporary memory
return iteration count
```

### Row Distribution

The code computes:

```c
int rows_per_thread = n / num_threads;
int extra_rows = n % num_threads;
```

If the rows do not divide evenly, the first `extra_rows` threads get one extra
row:

```c
args[t].row_end =
    row + rows_per_thread + (t < extra_rows ? 1 : 0);
```

Example with `n = 10` and `num_threads = 3`:

| Thread | Rows |
|---|---|
| 0 | 0, 1, 2, 3 |
| 1 | 4, 5, 6 |
| 2 | 7, 8, 9 |

### Creating Threads

```c
pthread_create(&threads[t], NULL, worker_func, &args[t]);
```

Each thread receives:

- Its thread ID
- Its assigned row range
- A pointer to the shared data
- The selected worker function

### Joining Threads

```c
pthread_join(threads[t], NULL);
```

The main thread waits until each worker thread finishes.

### Cleanup

The function destroys synchronization objects:

```c
pthread_barrier_destroy(&shared.barrier);
pthread_mutex_destroy(&shared.mutex);
```

Then it frees allocated arrays:

```c
free(shared.u_old);
free(threads);
free(args);
```

---

## 10. Function: `main`

The `main` function runs the full experiment.

### Step 1: Parse Arguments

```c
int n           = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;
int max_iter    = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;
double tol      = (argc > 3) ? atof(argv[3]) : TOLERANCE;
int num_threads = (argc > 4) ? atoi(argv[4]) : 4;
```

Program usage:

```bash
./bin/pthreads_solver [grid_size] [max_iterations] [tolerance] [num_threads]
```

Example:

```bash
./bin/pthreads_solver 200 5000 1e-6 4
```

This means:

- Grid size: `200 x 200`
- Maximum iterations: `5000`
- Tolerance: `1e-6`
- Number of threads: `4`

### Step 2: Allocate Arrays

```c
double *f       = malloc(n * n * sizeof(double));
double *u_jac   = calloc(n * n, sizeof(double));
double *u_gs    = calloc(n * n, sizeof(double));
double *u_exact = malloc(n * n * sizeof(double));
```

| Array | Purpose |
|---|---|
| `f` | Right-hand side values |
| `u_jac` | Jacobi solution array |
| `u_gs` | Red-Black Gauss-Seidel solution array |
| `u_exact` | Known exact solution |

`u_jac` and `u_gs` are zero-initialized so both methods start from the same
initial guess.

### Step 3: Initialize the Problem

```c
init_rhs(f, n);
exact_solution(u_exact, n);
```

### Step 4: Run Jacobi with Pthreads

```c
int jac_iters = run_pthreads_solver(
    u_jac, f, n, max_iter, tol, num_threads, jacobi_worker
);
```

The program measures runtime, residual, and RMSE.

### Step 5: Run Red-Black Gauss-Seidel with Pthreads

```c
int gs_iters = run_pthreads_solver(
    u_gs, f, n, max_iter, tol, num_threads, redblack_gs_worker
);
```

Again, the program measures runtime, residual, and RMSE.

### Step 6: Compare Both Solutions

```c
double rmse_jac_vs_gs = compute_rmse(u_jac, u_gs, n);
```

Both solvers should produce similar results if they converge correctly.

### Step 7: Save Output Files

```c
save_solution("pthreads_jacobi.txt", u_jac, n);
save_solution("pthreads_rbgs.txt", u_gs, n);
```

These files store the final solution grids.

### Step 8: Free Memory

```c
free(f);
free(u_jac);
free(u_gs);
free(u_exact);
```

---

## 11. Synchronization Summary

| Synchronization Object | Used For |
|---|---|
| Barrier after Jacobi copy | Ensures `u_old` is fully copied before computation |
| Barrier after red phase | Ensures all red points are updated before black points |
| Barrier after max reduction | Ensures all threads updated `global_max_diff` |
| Barrier after convergence check | Ensures all threads see the convergence decision |
| Mutex | Protects writes to `global_max_diff` |

The barrier makes threads wait for each other. The mutex protects one shared
variable from simultaneous writes.

---

## 12. Build and Run

From the project root:

```bash
make pthreads
```

Run with default values:

```bash
./bin/pthreads_solver
```

Run with custom values:

```bash
./bin/pthreads_solver 200 5000 1e-6 4
```

Argument meaning:

| Position | Example | Meaning |
|---|---:|---|
| 1 | `200` | Grid size, so the grid is `200 x 200` |
| 2 | `5000` | Maximum number of iterations |
| 3 | `1e-6` | Convergence tolerance |
| 4 | `4` | Number of Pthreads worker threads |

---

## 13. Output Meaning

The program prints one result block for Jacobi and one result block for
Red-Black Gauss-Seidel.

Important fields:

| Field | Meaning |
|---|---|
| `Method` | Solver method |
| `Grid` | Grid size and number of unknowns |
| `Iters` | Number of iterations completed |
| `Time` | Runtime in seconds |
| `Residual` | How well the solution satisfies the equation |
| `RMSE vs exact` | Error compared with the exact analytical solution |

It also prints:

```text
RMSE (Jacobi vs Red-Black GS)
```

This compares the two numerical solutions against each other.

---

## 14. High-Level Program Flow

```text
start
  |
  v
read grid size, max iterations, tolerance, and thread count
  |
  v
allocate f, u_jac, u_gs, and u_exact
  |
  v
initialize RHS and exact solution
  |
  v
run Pthreads Jacobi solver
  |
  v
print Jacobi timing and accuracy
  |
  v
run Pthreads Red-Black Gauss-Seidel solver
  |
  v
print Red-Black timing and accuracy
  |
  v
compare Jacobi and Red-Black solutions
  |
  v
save output text files
  |
  v
free memory
  |
  v
end
```

---

## 15. Key Points to Remember

- `pthreads_solver.c` parallelizes the grid by assigning row blocks to threads.
- `shared_data_t` stores global data used by all threads.
- `thread_arg_t` stores each thread's ID and assigned rows.
- Jacobi uses `u_old` and requires a barrier after copying.
- Red-Black Gauss-Seidel updates in-place using red and black phases.
- A barrier is required between the red and black phases.
- Each thread computes a `local_max_diff`.
- A mutex is used when updating the shared `global_max_diff`.
- Thread `0` performs the convergence check.
- `run_pthreads_solver` is generic and can launch either worker function.
- The output files are `pthreads_jacobi.txt` and `pthreads_rbgs.txt`.

