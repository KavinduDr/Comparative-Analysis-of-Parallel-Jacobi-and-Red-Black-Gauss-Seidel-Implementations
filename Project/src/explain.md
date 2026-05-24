# Serial Implementation — Explanation of Code and Concepts

## 1. The Problem: 2D Poisson Equation

We are solving the **2D Poisson equation** on a unit square domain `[0,1] x [0,1]`:

```
-∂²u/∂x² - ∂²u/∂y² = f(x, y)
```

Where:
- `u(x, y)` is the **unknown function** we want to find (e.g., temperature, potential).
- `f(x, y)` is the **source term** (a known function that drives the system).
- The equation says: the negative sum of second derivatives equals the source.

**Boundary conditions**: Dirichlet — `u = 0` on all four edges of the square.

**Our specific problem**:
- Source function: `f(x, y) = 2π² sin(πx) sin(πy)`
- Exact (analytical) solution: `u(x, y) = sin(πx) sin(πy)`

Having a known exact solution lets us verify our numerical methods by measuring the error.

---

## 2. Discretization: Turning Calculus into Algebra

Since computers cannot work with continuous functions, we **discretize** the domain into a grid of `n × n` equally spaced interior points.

### Grid Setup

```
  0   h   2h  3h  ...  nh  1
  |---|---|---|---|------|---|
  BC  1   2   3   ...  n   BC    (BC = boundary condition, u = 0)
```

- Grid spacing: `h = 1 / (n + 1)`
- Point `(i, j)` maps to physical coordinates `(x, y) = ((i+1)*h, (j+1)*h)`
- Total unknowns: `n × n` (only interior points; boundaries are known to be 0)

### The 5-Point Stencil

The second derivatives are approximated using **finite differences**:

```
∂²u/∂x² ≈ [u(i,j-1) - 2u(i,j) + u(i,j+1)] / h²
∂²u/∂y² ≈ [u(i-1,j) - 2u(i,j) + u(i+1,j)] / h²
```

Plugging into the Poisson equation and multiplying by `h²`:

```
4·u(i,j) - u(i-1,j) - u(i+1,j) - u(i,j-1) - u(i,j+1) = h²·f(i,j)
```

This is the **5-point stencil** — each point depends on its 4 direct neighbors:

```
         u(i-1,j)         (up)
            |
u(i,j-1) — u(i,j) — u(i,j+1)    (left — center — right)
            |
         u(i+1,j)         (down)
```

Rearranging to solve for the center point:

```
u(i,j) = [ u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + h²·f(i,j) ] / 4
```

This creates a **system of n² linear equations** (one per grid point) that we solve iteratively.

---

## 3. Memory Layout: Row-Major Order

The 2D grid is stored as a **1D array** in row-major order:

```
Grid point (i, j)  →  Array index [i * n + j]
```

Example for a 4×4 grid:

```
Grid:                     Array:
(0,0) (0,1) (0,2) (0,3)  [0]  [1]  [2]  [3]
(1,0) (1,1) (1,2) (1,3)  [4]  [5]  [6]  [7]
(2,0) (2,1) (2,2) (2,3)  [8]  [9]  [10] [11]
(3,0) (3,1) (3,2) (3,3)  [12] [13] [14] [15]
```

Neighbors of point `(i,j)` at index `[i*n + j]`:
- Left:  `[i*n + (j-1)]`
- Right: `[i*n + (j+1)]`
- Up:    `[(i-1)*n + j]`
- Down:  `[(i+1)*n + j]`

If a neighbor falls outside the grid (e.g., `j-1 < 0`), we use `0.0` (the Dirichlet boundary value).

---

## 4. Iterative Solvers: Why Not Solve Directly?

The system `Au = b` has `n²` unknowns. For `n = 1000`, that's 1,000,000 unknowns.

- **Direct solvers** (like Gaussian elimination) would need O(n⁶) operations and O(n⁴) memory — far too expensive for large grids.
- **Iterative solvers** start with an initial guess (u = 0) and repeatedly improve it until it converges to the true solution. They need only O(n²) memory and O(n²) work per iteration.

### Convergence

Both methods check convergence using the **infinity norm** of the update:

```
max_diff = max |u_new(i,j) - u_old(i,j)|   over all (i,j)
```

If `max_diff < tolerance`, the solution is barely changing, meaning it has **converged**.

---

## 5. Jacobi Method

### Concept

The Jacobi method is the simplest iterative solver. At each iteration:

1. **Copy** the current solution `u` into a temporary array `u_old`.
2. **Update** every grid point using ONLY values from `u_old`:
   ```
   u_new(i,j) = [ u_old(i-1,j) + u_old(i+1,j) + u_old(i,j-1) + u_old(i,j+1) + f(i,j) ] / 4
   ```
3. **Check convergence**: if the maximum change is below the tolerance, stop.

### Key Properties

| Property | Detail |
|----------|--------|
| **Two arrays needed** | Must keep `u_old` because the new value at `(i,j)` depends on OLD neighbor values. If we updated in-place, some neighbors would already be overwritten. |
| **All updates independent** | Since every point reads from `u_old` (which doesn't change during the iteration), the order of processing doesn't matter. Every point can be computed independently. |
| **Naturally parallel** | Independence means we can split the grid across threads/processes and compute in parallel with no data conflicts. |
| **Slower convergence** | New values aren't used until the NEXT iteration. Information propagates slowly through the grid (one cell per iteration). |
| **Memory** | O(n²) extra for `u_old` |

### Code Walkthrough (`jacobi_serial`)

```c
// 1. Allocate u_old (previous iteration values)
double *u_old = calloc(n * n, sizeof(double));

for (iter = 0; iter < max_iter; iter++) {
    // 2. Snapshot: copy current u into u_old
    memcpy(u_old, u, n * n * sizeof(double));

    double max_diff = 0.0;

    // 3. Update every grid point
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            // Read 4 neighbors from u_old (boundary = 0.0)
            left  = (j > 0)     ? u_old[i*n + (j-1)] : 0.0;
            right = (j < n-1)   ? u_old[i*n + (j+1)] : 0.0;
            up    = (i > 0)     ? u_old[(i-1)*n + j]  : 0.0;
            down  = (i < n-1)   ? u_old[(i+1)*n + j]  : 0.0;

            // 5-point stencil update
            u[i*n + j] = (left + right + up + down + f[i*n + j]) / 4.0;

            // Track max change for convergence
            diff = fabs(u[i*n + j] - u_old[i*n + j]);
            if (diff > max_diff) max_diff = diff;
        }
    }

    // 4. Converged?
    if (max_diff < tol) { iter++; break; }
}
free(u_old);
return iter;
```

**Step-by-step**:
1. `calloc` allocates and zeros `u_old` — matching the zero initial guess.
2. `memcpy` copies all `n²` values from `u` to `u_old` at the start of each iteration.
3. The double loop visits every point `(i,j)`. The ternary operators `(j > 0) ? ... : 0.0` handle boundaries — if a neighbor would be outside the grid, use 0 (Dirichlet BC).
4. After updating all points, we check if `max_diff < tol`. If yes, we're converged. We increment `iter` before breaking so the returned count includes this final iteration.

---

## 6. Red-Black Gauss-Seidel Method

### Concept

Gauss-Seidel improves on Jacobi by using **the most recent values** as soon as they're computed (in-place update, no `u_old`). This makes information propagate faster and converges in roughly **half the iterations**.

However, standard (lexicographic) Gauss-Seidel processes points row by row, left to right. Each update depends on the **just-updated** neighbor above and to the left, creating a **sequential dependency chain** that prevents parallelism.

**Red-Black ordering** solves this by splitting the grid into two independent sets using a checkerboard pattern:

```
R B R B R B        R = Red  (i+j is even)
B R B R B R        B = Black (i+j is odd)
R B R B R B
B R B R B R
```

**Key insight**: On a checkerboard, every RED point's 4 neighbors are ALL BLACK, and vice versa. So:

- **Phase 1 (Red sweep)**: Update all RED points. Their neighbors (all black) haven't changed yet, so all red updates are **independent** of each other.
- **Phase 2 (Black sweep)**: Update all BLACK points. Their neighbors (all red) were just updated in Phase 1, so black points use the **freshest** values — this is the Gauss-Seidel advantage.

### Key Properties

| Property | Detail |
|----------|--------|
| **One array (in-place)** | No `u_old` needed. Updates write directly to `u`. |
| **Two phases per iteration** | First all red, then all black. Order within each phase doesn't matter. |
| **Independent within phases** | All red updates are independent; all black updates are independent. |
| **Parallelizable** | Each phase can be parallelized (OpenMP, MPI, CUDA) — same as Jacobi within each phase. |
| **Faster convergence** | Uses newest values immediately. ~2x fewer iterations than Jacobi. |
| **Less memory** | No extra `u_old` array — better cache utilization. |

### Why It Works: The Checkerboard Argument

Consider updating red point `(2,2)` in Phase 1:

```
        (1,2) B ← still old
          |
(2,1) B — (2,2) R — (2,3) B ← still old
          |
        (3,2) B ← still old
```

All 4 neighbors are BLACK and haven't been touched yet in this iteration. So `(2,2)` reads consistent "old" values.

Now consider updating black point `(2,3)` in Phase 2:

```
        (1,3) R ← JUST updated in Phase 1
          |
(2,2) R — (2,3) B — (2,4) R ← JUST updated in Phase 1
          |
        (3,3) R ← JUST updated in Phase 1
```

All 4 neighbors are RED and were just updated. This is the Gauss-Seidel property — using the latest information.

### Code Walkthrough (`redblack_gs_serial`)

```c
for (iter = 0; iter < max_iter; iter++) {
    double max_diff = 0.0;

    // Phase 1: Update RED points (i+j even)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if ((i + j) % 2 != 0) continue;  // skip black

            left  = (j > 0)     ? u[i*n + (j-1)] : 0.0;  // black, old
            right = (j < n-1)   ? u[i*n + (j+1)] : 0.0;  // black, old
            up    = (i > 0)     ? u[(i-1)*n + j]  : 0.0;  // black, old
            down  = (i < n-1)   ? u[(i+1)*n + j]  : 0.0;  // black, old

            old_val = u[i*n + j];
            u[i*n + j] = (left + right + up + down + f[i*n + j]) / 4.0;

            diff = fabs(u[i*n + j] - old_val);
            if (diff > max_diff) max_diff = diff;
        }
    }

    // Phase 2: Update BLACK points (i+j odd)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if ((i + j) % 2 != 1) continue;  // skip red

            left  = (j > 0)     ? u[i*n + (j-1)] : 0.0;  // red, NEW
            right = (j < n-1)   ? u[i*n + (j+1)] : 0.0;  // red, NEW
            up    = (i > 0)     ? u[(i-1)*n + j]  : 0.0;  // red, NEW
            down  = (i < n-1)   ? u[(i+1)*n + j]  : 0.0;  // red, NEW

            old_val = u[i*n + j];
            u[i*n + j] = (left + right + up + down + f[i*n + j]) / 4.0;

            diff = fabs(u[i*n + j] - old_val);
            if (diff > max_diff) max_diff = diff;
        }
    }

    if (max_diff < tol) { iter++; break; }
}
return iter;
```

**Step-by-step**:
1. Phase 1 loops over all points but `continue` skips any point where `(i+j)` is odd (black). Only red points are updated. Reads come from `u` directly (neighbors are black = untouched).
2. Phase 2 does the opposite: skips red, updates black. Now neighbors are red = freshly updated in Phase 1.
3. `old_val` saves the value before overwriting so we can compute the change.
4. Convergence is checked after both phases using the max change from either phase.

---

## 7. The `main` Function: Putting It All Together

### Flow

```
Parse arguments → Allocate arrays → Initialize problem → Run Jacobi → Run GS → Compare → Cleanup
```

### Step-by-step

1. **Parse command-line arguments**: Grid size `n`, max iterations, and tolerance. Uses defaults from `common.h` if not provided.

2. **Allocate four arrays**:
   - `f` — RHS vector (`malloc`, filled by `init_rhs`)
   - `u_jac` — Jacobi solution (`calloc` = zero initial guess)
   - `u_gs` — GS solution (`calloc` = zero initial guess, separate array for fair comparison)
   - `u_exact` — Exact analytical solution for error measurement

3. **Initialize the problem**:
   - `init_rhs(f, n)` fills `f` with `h² × 2π² sin(πx) sin(πy)` at each grid point
   - `exact_solution(u_exact, n)` computes `sin(πx) sin(πy)` at each grid point

4. **Run Jacobi solver**: Timed with `get_time()`. After solving, compute:
   - **Residual** (`||Au - f||∞`): How well the solution satisfies the equation (should be tiny)
   - **RMSE vs exact**: How close the numerical answer is to the true analytical solution

5. **Run Red-Black GS solver**: Same procedure. Expected to converge in fewer iterations.

6. **Compare solutions**: Compute RMSE between Jacobi and GS results. Both solve the same equation, so they should agree closely — this is a sanity check.

7. **Free memory**: Release all `malloc`/`calloc` allocations.

---

## 8. Comparison: Jacobi vs Red-Black Gauss-Seidel

| Aspect | Jacobi | Red-Black Gauss-Seidel |
|--------|--------|------------------------|
| Update rule | Uses old values only | Uses latest available values |
| Arrays needed | 2 (u + u_old) | 1 (u, in-place) |
| Updates per iteration | All n² at once | n²/2 red, then n²/2 black |
| Independence | All n² updates independent | All updates independent within each phase |
| Convergence speed | Slower (~2x more iterations) | Faster (~half the iterations) |
| Memory usage | Higher (extra u_old array) | Lower (no extra array) |
| Parallelizability | Trivially parallel | Parallel within each phase (barrier between phases) |
| Implementation complexity | Simpler | Slightly more complex (two-phase logic) |

### Why Gauss-Seidel Converges Faster

In Jacobi, information travels **one grid cell per iteration** because new values aren't used until the next pass. In Gauss-Seidel, updated values are used **immediately**, so information propagates faster across the grid.

Mathematically, the **spectral radius** (largest eigenvalue) of the Gauss-Seidel iteration matrix is the **square** of Jacobi's spectral radius. For the 2D Poisson problem:
- Jacobi spectral radius: `ρ_J = cos(πh) ≈ 1 - π²h²/2`
- GS spectral radius: `ρ_GS = ρ_J² ≈ 1 - π²h²`

A smaller spectral radius means faster convergence.

---

## 9. Quality Metrics

### Residual (Infinity Norm)

Measures how well the computed solution satisfies the original equation:

```
residual = max |f(i,j) - [4·u(i,j) - u(i-1,j) - u(i+1,j) - u(i,j-1) - u(i,j+1)]|
```

This is `||Au - b||∞`. A small residual means the solution nearly satisfies the discretized equation.

### RMSE (Root Mean Square Error)

Measures accuracy compared to the known exact solution:

```
RMSE = sqrt( (1/n²) × Σ [u_numerical(i,j) - u_exact(i,j)]² )
```

A small RMSE means the numerical solution is close to the true answer.

### Jacobi-vs-GS Comparison

```
RMSE(Jacobi vs GS) = sqrt( (1/n²) × Σ [u_jac(i,j) - u_gs(i,j)]² )
```

Both methods solve the same equation, so this should be very small if both converged — a sanity check that the implementations are correct.

---

## 10. Why This Serial Version Matters

This serial implementation serves as the **baseline** for the entire project:

1. **Correctness reference**: Parallel versions must produce the same (or very close) results.
2. **Performance baseline**: Speedup = serial time / parallel time. Without a serial baseline, we can't measure how much parallelism helps.
3. **Algorithm understanding**: The serial code clearly shows the logic of each method without the complexity of thread synchronization, message passing, or GPU kernels.
4. **Debugging aid**: If a parallel version gives wrong results, comparing against the serial output helps locate bugs.
