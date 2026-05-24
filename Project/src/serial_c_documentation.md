# Understanding `serial.c`

This document explains the code in [`src/serial.c`](serial.c). The file is the
serial baseline implementation for the project. It solves the 2D Poisson
equation using two iterative methods:

- Jacobi method
- Red-Black Gauss-Seidel method

The serial version is important because the OpenMP, Pthreads, MPI, hybrid, and
CUDA versions can be checked against it for correctness and performance.

---

## 1. What Problem Does This Code Solve?

The program solves the 2D Poisson equation on the unit square:

```text
-u_xx - u_yy = f(x, y)
```

The boundary condition is:

```text
u = 0 on all four edges
```

This is called a Dirichlet boundary condition.

The project uses a known exact solution:

```text
u(x, y) = sin(pi*x) * sin(pi*y)
```

Because the exact answer is known, the program can measure how accurate the
computed numerical solution is.

---

## 2. Important Files Used by `serial.c`

`serial.c` includes:

```c
#include "../include/common.h"
```

`common.h` provides shared helper functions and constants:

| Item | Purpose |
|---|---|
| `DEFAULT_GRID_SIZE` | Default grid size if the user does not pass one |
| `MAX_ITERATIONS` | Default maximum number of solver iterations |
| `TOLERANCE` | Default convergence tolerance |
| `get_time()` | Measures wall-clock time |
| `init_rhs()` | Initializes the right-hand side vector `f` |
| `exact_solution()` | Computes the known exact answer |
| `compute_residual()` | Checks how well the solution satisfies the equation |
| `compute_rmse()` | Measures error against another solution |
| `print_results()` | Prints solver results |
| `save_solution()` | Saves a solution grid to a text file |

---

## 3. Grid and Memory Layout

The mathematical grid is 2D, but the C code stores it in a 1D array.

For an `n x n` grid:

```text
2D point (i, j) -> 1D index i * n + j
```

Example for a `4 x 4` grid:

```text
(0,0) (0,1) (0,2) (0,3)      index 0   1   2   3
(1,0) (1,1) (1,2) (1,3)      index 4   5   6   7
(2,0) (2,1) (2,2) (2,3)      index 8   9  10  11
(3,0) (3,1) (3,2) (3,3)      index 12 13  14  15
```

So in the code:

```c
u[i * n + j]
```

means the value of `u` at row `i`, column `j`.

---

## 4. The 5-Point Stencil

Each grid point is updated using its four direct neighbors:

```text
             up
              |
left ---- center ---- right
              |
            down
```

The update formula is:

```text
u(i,j) = (left + right + up + down + f(i,j)) / 4
```

In C, this appears as:

```c
u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;
```

If a point is on the edge of the interior grid, one of its neighbors is outside
the grid. That outside value is treated as `0.0` because the boundary condition
is `u = 0`.

Example:

```c
double left = (j > 0) ? u_old[i * n + (j - 1)] : 0.0;
```

This means:

- If there is a left neighbor, use it.
- Otherwise, use the boundary value `0.0`.

---

## 5. Function: `jacobi_serial`

Function signature:

```c
int jacobi_serial(double *u, const double *f, int n, int max_iter, double tol)
```

### Parameters

| Parameter | Meaning |
|---|---|
| `u` | Solution array. It starts as the initial guess and ends as the computed solution. |
| `f` | Right-hand side array. It is read-only inside this function. |
| `n` | Grid size. The grid has `n * n` interior points. |
| `max_iter` | Maximum number of iterations allowed. |
| `tol` | Convergence tolerance. |

### Return Value

The function returns the number of iterations actually performed.

### Main Idea

The Jacobi method updates every point using only values from the previous
iteration. That is why the code creates a second array:

```c
double *u_old = (double *)calloc((size_t)n * n, sizeof(double));
```

At the start of every iteration:

```c
memcpy(u_old, u, (size_t)n * n * sizeof(double));
```

This copies the current solution into `u_old`. Then the new values are written
into `u`, while all neighbor reads come from `u_old`.

### Why Jacobi Needs `u_old`

If Jacobi updated `u` in-place, some points would read old neighbor values and
some would read already-updated neighbor values. That would no longer be the
Jacobi method.

Using `u_old` guarantees that every point in the current iteration uses the
same previous-iteration data.

### Jacobi Algorithm

```text
allocate u_old

for each iteration:
    copy u into u_old
    max_diff = 0

    for every grid point (i, j):
        read left, right, up, down from u_old
        compute new u(i,j)
        compute how much u(i,j) changed
        update max_diff

    if max_diff < tolerance:
        stop early

free u_old
return iteration count
```

### Convergence Check

The code checks the largest change in any grid point:

```c
double diff = fabs(u[i * n + j] - u_old[i * n + j]);
if (diff > max_diff) max_diff = diff;
```

If the biggest change is smaller than `tol`, the solution is considered
converged:

```c
if (max_diff < tol) {
    iter++;
    break;
}
```

`iter++` is used before `break` so the returned iteration count includes the
iteration that just finished.

---

## 6. Function: `redblack_gs_serial`

Function signature:

```c
int redblack_gs_serial(double *u, const double *f, int n, int max_iter, double tol)
```

### Parameters

The parameters mean the same thing as in `jacobi_serial`.

### Main Idea

Red-Black Gauss-Seidel updates the grid in-place using a checkerboard pattern.

Grid points are split into two colors:

```text
red   : (i + j) is even
black : (i + j) is odd
```

Checkerboard view:

```text
R B R B
B R B R
R B R B
B R B R
```

Each iteration has two phases:

1. Update all red points.
2. Update all black points.

### Why This Works

Every red point has only black neighbors. Every black point has only red
neighbors.

During the red phase, black points have not been changed yet, so all red points
can be updated independently.

During the black phase, red points have already been updated, so black points use
the newest red values. This is the Gauss-Seidel advantage.

### Red Phase in the Code

```c
if ((i + j) % 2 != 0) continue;
```

This skips black points. Only red points are updated.

### Black Phase in the Code

```c
if ((i + j) % 2 != 1) continue;
```

This skips red points. Only black points are updated.

### Red-Black Gauss-Seidel Algorithm

```text
for each iteration:
    max_diff = 0

    red phase:
        for every grid point (i, j):
            if point is not red, skip it
            read neighbors from u
            save old value
            update u(i,j) in-place
            update max_diff

    black phase:
        for every grid point (i, j):
            if point is not black, skip it
            read neighbors from u
            save old value
            update u(i,j) in-place
            update max_diff

    if max_diff < tolerance:
        stop early

return iteration count
```

### Why It Uses Less Memory Than Jacobi

Red-Black Gauss-Seidel does not allocate `u_old`. It updates directly in `u`:

```c
u[i * n + j] = (left + right + up + down + f[i * n + j]) / 4.0;
```

This saves memory and usually improves convergence speed.

---

## 7. Jacobi vs Red-Black Gauss-Seidel

| Feature | Jacobi | Red-Black Gauss-Seidel |
|---|---|---|
| Update style | Uses previous iteration only | Uses newest available values |
| Extra array needed | Yes, `u_old` | No |
| Memory use | Higher | Lower |
| Convergence speed | Slower | Usually faster |
| Parallel potential | Very easy | Parallel inside each red/black phase |
| Code structure | One update sweep | Two update sweeps |

Jacobi is simpler and naturally parallel because all points are independent.

Red-Black Gauss-Seidel is a little more complex, but it normally converges in
fewer iterations.

---

## 8. Function: `main`

The `main` function controls the complete program.

### Step 1: Read Command-Line Arguments

```c
int n        = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;
int max_iter = (argc > 2) ? atoi(argv[2]) : MAX_ITERATIONS;
double tol   = (argc > 3) ? atof(argv[3]) : TOLERANCE;
```

Arguments are optional.

Usage:

```bash
./bin/serial [grid_size] [max_iterations] [tolerance]
```

Examples:

```bash
./bin/serial
./bin/serial 200
./bin/serial 200 5000 1e-6
```

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
| `u_jac` | Solution computed by Jacobi |
| `u_gs` | Solution computed by Red-Black Gauss-Seidel |
| `u_exact` | Exact analytical solution |

`u_jac` and `u_gs` use `calloc` so they start from zero. This gives both solvers
the same initial guess.

### Step 3: Initialize the Problem

```c
init_rhs(f, n);
exact_solution(u_exact, n);
```

`init_rhs` fills the source term values.

`exact_solution` fills the exact answer so the program can measure numerical
error.

### Step 4: Run Jacobi

```c
double t_start = get_time();
int jac_iters = jacobi_serial(u_jac, f, n, max_iter, tol);
double t_jac = get_time() - t_start;
```

The program measures execution time, then computes:

```c
double res_jac  = compute_residual(u_jac, f, n);
double rmse_jac = compute_rmse(u_jac, u_exact, n);
```

### Step 5: Run Red-Black Gauss-Seidel

```c
t_start = get_time();
int gs_iters = redblack_gs_serial(u_gs, f, n, max_iter, tol);
double t_gs = get_time() - t_start;
```

Then it computes residual and RMSE for this method too.

### Step 6: Compare Both Numerical Solutions

```c
double rmse_jac_vs_gs = compute_rmse(u_jac, u_gs, n);
```

Both methods solve the same equation. If both converge correctly, this value
should be small.

### Step 7: Save Output Files

```c
save_solution("serial_jacobi.txt", u_jac, n);
save_solution("serial_rbgs.txt", u_gs, n);
```

These files contain the final solution grids.

### Step 8: Free Memory

```c
free(f);
free(u_jac);
free(u_gs);
free(u_exact);
```

This releases memory allocated with `malloc` and `calloc`.

---

## 9. Build and Run

From the project root:

```bash
make serial
```

Run with default parameters:

```bash
./bin/serial
```

Run with custom parameters:

```bash
./bin/serial 100 10000 1e-6
```

Parameter meaning:

| Position | Example | Meaning |
|---|---:|---|
| 1 | `100` | Grid size, so the grid is `100 x 100` |
| 2 | `10000` | Maximum number of iterations |
| 3 | `1e-6` | Stopping tolerance |

---

## 10. Output Meaning

The program prints results for each method.

Important fields:

| Field | Meaning |
|---|---|
| `Method` | Solver method used |
| `Grid` | Grid size and number of unknowns |
| `Iters` | Number of iterations used |
| `Time` | Runtime in seconds |
| `Residual` | How well the computed solution satisfies the discrete equation |
| `RMSE vs exact` | Error compared with the known exact solution |

Lower residual and lower RMSE mean better numerical accuracy.

---

## 11. High-Level Program Flow

```text
start
  |
  v
read command-line arguments
  |
  v
allocate arrays
  |
  v
initialize RHS and exact solution
  |
  v
run Jacobi solver
  |
  v
print Jacobi time, residual, and RMSE
  |
  v
run Red-Black Gauss-Seidel solver
  |
  v
print GS time, residual, and RMSE
  |
  v
compare Jacobi and GS solutions
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

## 12. Key Points to Remember

- `serial.c` solves the same mathematical problem using two iterative methods.
- The grid is stored as a 1D array even though it represents a 2D domain.
- `i * n + j` converts 2D coordinates into a 1D array index.
- The update formula uses the four neighboring values and the source term.
- Jacobi uses a copy array, `u_old`, to keep previous-iteration values.
- Red-Black Gauss-Seidel updates in-place using red and black checkerboard phases.
- `max_diff < tol` is the stopping condition for both solvers.
- `compute_residual` checks equation satisfaction.
- `compute_rmse` checks error against the exact solution.
- The serial implementation is the correctness and performance baseline for the
  parallel implementations.

