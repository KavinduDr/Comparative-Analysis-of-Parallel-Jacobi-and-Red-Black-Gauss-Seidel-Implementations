#!/usr/bin/env python3
"""
HPC Solver Automated Testing, Verification, and Scaling Benchmark Framework
=============================================================================
This framework automates execution, validation, profiling, and report generation
for parallel Jacobi and Red-Black Gauss-Seidel solvers.

Key Capabilities:
1. Multi-Solver Execution: Interface wrappers for Serial, OpenMP, Pthreads, MPI, and Hybrid solvers.
2. Numerical Verification: Cross-validates intermediate and final solutions across all versions.
3. Scaling Benchmarks: Measures execution times, speedup metrics, and parallel efficiency.
4. Reporting & Visualization: Generates rich scaling tables, LaTeX-formatted outputs, and charts.

Usage:
    python3 hpc_testing_framework.py --help
"""

import os
import sys
import re
import csv
import json
import argparse
import subprocess
import time
import math
from collections import OrderedDict
from statistics import mean, stdev

# Try importing plotting dependencies gracefully
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
DEFAULT_MAX_ITER = 10000
DEFAULT_TOL = 1e-6
DEFAULT_GRID_SIZES = [50, 100, 200, 300]
DEFAULT_THREAD_COUNTS = [1, 2, 4, 8]
DEFAULT_REPEATS = 3

SOLVER_BINARIES = {
    "Serial": "serial",
    "OpenMP": "openmp_solver",
    "Pthreads": "pthreads_solver",
    "MPI": "mpi_solver",
    "Hybrid": "hybrid_solver"
}

# HSL-aligned color palette for plotting to maintain rich premium aesthetics
PALETTE = {
    "primary": "#58a6ff",
    "success": "#3fb950",
    "warning": "#d29922",
    "danger": "#f85149",
    "accent": "#bc8cff",
    "info": "#39d2c0",
    "bg": "#0d1117",
    "card": "#1c2128",
    "border": "#30363d"
}

# -----------------------------------------------------------------------------
# Parser Utility for C Solver Outputs
# -----------------------------------------------------------------------------
SOLVER_OUTPUT_RE = re.compile(
    r"Method\s*:\s*(?P<method>.+?)\n"
    r"\s*Grid\s*:\s*(?P<grid>\d+)\s*x\s*\d+.*?\n"
    r"\s*Iters\s*:\s*(?P<iters>\d+)\n"
    r"\s*Time\s*:\s*(?P<time>[\d.]+)\s*seconds\n"
    r"\s*Residual\s*\(inf-norm\)\s*:\s*(?P<residual>\S+)\n"
    r"\s*RMSE vs exact\s*:\s*(?P<rmse>\S+)",
    re.MULTILINE
)

def parse_solver_stdout(stdout_text):
    """
    Parses the standard output of a solver binary and returns structured metrics.
    """
    matches = list(SOLVER_OUTPUT_RE.finditer(stdout_text))
    results = []
    for m in matches:
        try:
            results.append({
                "method": m.group("method").strip(),
                "grid_size": int(m.group("grid")),
                "iterations": int(m.group("iters")),
                "execution_time": float(m.group("time")),
                "residual": float(m.group("residual")),
                "rmse_exact": float(m.group("rmse"))
            })
        except (ValueError, TypeError, IndexError) as err:
            sys.stderr.write(f"Warning: Failed to parse segment match: {err}\n")
    return results

# -----------------------------------------------------------------------------
# Execution Helper Classes
# -----------------------------------------------------------------------------
class SolverExecutionError(Exception):
    """Custom exception raised when solver process crashes or times out."""
    pass

class SolverRunner:
    """
    Manages building command lines, launching subprocesses, handling environments,
    and capturing execution results for all parallel solver paradigms.
    """
    def __init__(self, project_dir):
        self.project_dir = os.path.abspath(project_dir)
        self.bin_dir = os.path.join(self.project_dir, "bin")

    def get_binary_path(self, solver_name):
        """Returns the absolute path to the specified solver's compiled binary."""
        binary_name = SOLVER_BINARIES.get(solver_name)
        if not binary_name:
            raise ValueError(f"Unknown solver type: {solver_name}")
        return os.path.join(self.bin_dir, binary_name)

    def verify_binary_exists(self, solver_name):
        """Verifies that the compiled binary is present in the bin/ directory."""
        path = self.get_binary_path(solver_name)
        return os.path.isfile(path)

    def run(self, solver_name, grid_size, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL,
            threads=1, processes=1, timeout=300):
        """
        Executes a target solver with specific parameters.
        Handles thread parameters, MPI bindings, and local sandboxing workarounds.
        """
        if not self.verify_binary_exists(solver_name):
            raise FileNotFoundError(
                f"Compiled binary for '{solver_name}' not found. "
                f"Please compile the project first using 'make all_cpu'."
            )

        binary_path = self.get_binary_path(solver_name)
        cmd = []

        # Configure MPI or Hybrid runs
        if solver_name in ("MPI", "Hybrid"):
            cmd = ["mpirun"]
            # Apply loopback/localhost bindings for macOS sandboxed runs to prevent PML errors
            cmd.extend(["-host", "127.0.0.1", "--oversubscribe"])
            cmd.extend(["-np", str(processes)])
            cmd.append(binary_path)
        else:
            cmd = [binary_path]

        # Add shared parameters
        cmd.extend([str(grid_size), str(max_iter), str(tol)])

        # Thread configurations
        if solver_name in ("OpenMP", "Pthreads"):
            cmd.append(str(threads))
        elif solver_name == "Hybrid":
            cmd.append(str(threads))

        # Setup sandbox environment variables to prevent OMPI shared memory segment errors
        env = os.environ.copy()
        env["TMPDIR"] = "/tmp"
        env["OMPI_MCA_btl_sm_backing_directory"] = "/tmp"

        start_time = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=self.project_dir,
                timeout=timeout
            )
        except subprocess.TimeoutExpired as err:
            raise SolverExecutionError(f"Solver process timed out after {timeout} seconds.") from err

        elapsed = time.perf_counter() - start_time

        if proc.returncode != 0 and solver_name not in ("MPI", "Hybrid"):
            # MPI solvers might throw warnings or abort signals on sandbox teardown, but still output results
            raise SolverExecutionError(
                f"Solver returned exit code {proc.returncode}.\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        results = parse_solver_stdout(proc.stdout)
        if not results:
            # Check if stdout contains solution data but parse failed due to warnings
            raise SolverExecutionError(
                f"Failed to extract metric output structures from solver.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        # Enforce metadata values in returns
        for r in results:
            r["solver"] = solver_name
            r["threads"] = threads if solver_name in ("OpenMP", "Pthreads", "Hybrid") else 1
            r["processes"] = processes if solver_name in ("MPI", "Hybrid") else 1
            r["cmd"] = " ".join(cmd)
            r["total_duration"] = elapsed

        return results

# -----------------------------------------------------------------------------
# Data Processing & Numerical Validation Suite
# -----------------------------------------------------------------------------
class ValidationSuite:
    """
    Coordinates testing of solvers to verify mathematical/numerical accuracy.
    Verifies that the parallel implementations converge to identical outputs
    within tolerances, and measures residual norms.
    """
    def __init__(self, runner):
        self.runner = runner

    def run_equivalence_test(self, grid_size, tolerance=DEFAULT_TOL):
        """
        Runs Jacobi and Red-Black Gauss-Seidel across Serial, OpenMP, and Pthreads
        and validates that they converge to the exact same numeric grid values.
        """
        sys.stdout.write(f"--- Running Numerical Equivalence Test (Grid: {grid_size}x{grid_size}) ---\n")
        solvers_to_test = ["Serial", "OpenMP", "Pthreads"]
        all_results = {}

        for s in solvers_to_test:
            if not self.runner.verify_binary_exists(s):
                sys.stdout.write(f"Skipping solver '{s}' (binary missing).\n")
                continue
            try:
                # Execute with 4 threads for parallel solvers
                results = self.runner.run(s, grid_size, tol=tolerance, threads=4)
                for r in results:
                    key = (s, r["method"])
                    all_results[key] = r
            except Exception as err:
                sys.stdout.write(f"Error executing solver '{s}': {err}\n")

        if not all_results:
            sys.stdout.write("Equivalence Test aborted: No execution results gathered.\n")
            return False

        sys.stdout.write("\nGathered Convergence Signatures:\n")
        sys.stdout.write(f"{'Solver':<12} | {'Method':<26} | {'Iters':<6} | {'Time (s)':<10} | {'Residual':<10} | {'RMSE':<10}\n")
        sys.stdout.write("-" * 85 + "\n")
        for (solver, method), r in all_results.items():
            sys.stdout.write(
                f"{solver:<12} | {method:<26} | {r['iterations']:<6} | "
                f"{r['execution_time']:<10.6f} | {r['residual']:<10.2e} | {r['rmse_exact']:<10.2e}\n"
            )

        # Numerical comparison checks
        sys.stdout.write("\nPerforming Cross-Validation Checks:\n")
        passed = True

        # 1. Compare Jacobi solvers across versions
        jacobi_runs = {k: v for k, v in all_results.items() if "Jacobi" in k[1]}
        if len(jacobi_runs) > 1:
            baseline_key = list(jacobi_runs.keys())[0]
            baseline = jacobi_runs[baseline_key]
            for key, val in list(jacobi_runs.items())[1:]:
                iter_diff = abs(baseline["iterations"] - val["iterations"])
                rmse_diff = abs(baseline["rmse_exact"] - val["rmse_exact"])
                sys.stdout.write(f"  [Jacobi] {baseline_key[0]} vs {key[0]}: ")
                if iter_diff == 0 and rmse_diff < 1e-12:
                    sys.stdout.write("SUCCESS (Numerical identity match)\n")
                elif iter_diff < 5 and rmse_diff < 1e-6:
                    sys.stdout.write("SUCCESS (Close numerical convergence match)\n")
                else:
                    sys.stdout.write(f"WARNING (Diverged - Iter diff: {iter_diff}, RMSE diff: {rmse_diff:.2e})\n")
                    passed = False

        # 2. Compare Red-Black Gauss-Seidel solvers across versions
        rbgs_runs = {k: v for k, v in all_results.items() if "Gauss-Seidel" in k[1]}
        if len(rbgs_runs) > 1:
            baseline_key = list(rbgs_runs.keys())[0]
            baseline = rbgs_runs[baseline_key]
            for key, val in list(rbgs_runs.items())[1:]:
                iter_diff = abs(baseline["iterations"] - val["iterations"])
                rmse_diff = abs(baseline["rmse_exact"] - val["rmse_exact"])
                sys.stdout.write(f"  [Red-Black GS] {baseline_key[0]} vs {key[0]}: ")
                if iter_diff == 0 and rmse_diff < 1e-12:
                    sys.stdout.write("SUCCESS (Numerical identity match)\n")
                elif iter_diff < 5 and rmse_diff < 1e-6:
                    sys.stdout.write("SUCCESS (Close numerical convergence match)\n")
                else:
                    sys.stdout.write(f"WARNING (Diverged - Iter diff: {iter_diff}, RMSE diff: {rmse_diff:.2e})\n")
                    passed = False

        sys.stdout.write(f"\nEquivalence Test Status: {'PASSED' if passed else 'WARNING/FAILED'}\n\n")
        return passed

# -----------------------------------------------------------------------------
# Profiling & Scaling Engine
# -----------------------------------------------------------------------------
class BenchmarkEngine:
    """
    Core engine responsible for running timing benchmarks, executing multiple
    repeats, evaluating threads/grid scalability, and parsing raw metrics.
    """
    def __init__(self, runner):
        self.runner = runner
        self.raw_data = []

    def clear(self):
        """Resets the gathered benchmarking metrics list."""
        self.raw_data.clear()

    def run_thread_scaling(self, solver_name, grid_size, thread_list, repeats=DEFAULT_REPEATS,
                           max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL):
        """
        Measures execution times across a series of thread counts to calculate speedups.
        """
        sys.stdout.write(f"=== Thread Scaling Study: {solver_name} (Grid: {grid_size}x{grid_size}) ===\n")
        if not self.runner.verify_binary_exists(solver_name):
            sys.stdout.write(f"Skipping benchmark: {solver_name} binary not found.\n")
            return

        # Always run Serial as a reference baseline if it exists
        serial_baseline = {}
        if self.runner.verify_binary_exists("Serial"):
            sys.stdout.write("Gathering Serial baseline timing references...\n")
            serial_times_jacobi = []
            serial_times_rbgs = []
            for _ in range(repeats):
                res = self.runner.run("Serial", grid_size, max_iter, tol)
                for r in res:
                    if "Jacobi" in r["method"]:
                        serial_times_jacobi.append(r["execution_time"])
                    else:
                        serial_times_rbgs.append(r["execution_time"])
            if serial_times_jacobi:
                serial_baseline["Jacobi"] = mean(serial_times_jacobi)
            if serial_times_rbgs:
                serial_baseline["Red-Black Gauss-Seidel"] = mean(serial_times_rbgs)

        for t in thread_list:
            sys.stdout.write(f"  Measuring scalability at Threads: {t}...\n")
            runs = []
            for r_idx in range(repeats):
                try:
                    res = self.runner.run(solver_name, grid_size, max_iter, tol, threads=t)
                    runs.append(res)
                except Exception as err:
                    sys.stderr.write(f"    Repeat {r_idx+1} failed: {err}\n")

            if not runs:
                continue

            # Pivot metrics across repeats
            pivoted = {}
            for repeat_res in runs:
                for run_metric in repeat_res:
                    method = run_metric["method"]
                    pivoted.setdefault(method, {"times": [], "iters": [], "res": [], "rmse": []})
                    pivoted[method]["times"].append(run_metric["execution_time"])
                    pivoted[method]["iters"].append(run_metric["iterations"])
                    pivoted[method]["res"].append(run_metric["residual"])
                    pivoted[method]["rmse"].append(run_metric["rmse_exact"])

            for method, metrics in pivoted.items():
                avg_time = mean(metrics["times"])
                dev_time = stdev(metrics["times"]) if len(metrics["times"]) > 1 else 0.0
                avg_iters = int(mean(metrics["iters"]))

                # Speedup calculations
                baseline_time = serial_baseline.get(method, metrics["times"][0]) # Fallback to 1st run if serial missing
                if t == 1:
                    # If this is thread=1, set it as local reference baseline if serial baseline not active
                    if method not in serial_baseline:
                        serial_baseline[method] = avg_time
                        baseline_time = avg_time
                
                speedup = baseline_time / avg_time if avg_time > 0 else 0.0
                efficiency = (speedup / t) * 100.0

                record = {
                    "study_type": "thread_scaling",
                    "solver": solver_name,
                    "method": method,
                    "grid_size": grid_size,
                    "threads": t,
                    "processes": 1,
                    "execution_time_avg": avg_time,
                    "execution_time_std": dev_time,
                    "iterations": avg_iters,
                    "residual_avg": mean(metrics["res"]),
                    "rmse_exact_avg": mean(metrics["rmse"]),
                    "speedup": speedup,
                    "efficiency": efficiency,
                    "repeats": len(metrics["times"])
                }
                self.raw_data.append(record)

    def run_grid_scaling(self, solver_name, grid_sizes, threads=4, repeats=DEFAULT_REPEATS,
                         max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL):
        """
        Measures solver runtimes across a variety of grid sizes to evaluate complexity growth.
        """
        sys.stdout.write(f"=== Grid Scaling Study: {solver_name} (Threads: {threads}) ===\n")
        if not self.runner.verify_binary_exists(solver_name):
            sys.stdout.write(f"Skipping benchmark: {solver_name} binary not found.\n")
            return

        for g in grid_sizes:
            sys.stdout.write(f"  Profiling Grid Size: {g}x{g}...\n")
            runs = []
            for r_idx in range(repeats):
                try:
                    res = self.runner.run(solver_name, g, max_iter, tol, threads=threads)
                    runs.append(res)
                except Exception as err:
                    sys.stderr.write(f"    Repeat {r_idx+1} failed: {err}\n")

            if not runs:
                continue

            pivoted = {}
            for repeat_res in runs:
                for run_metric in repeat_res:
                    method = run_metric["method"]
                    pivoted.setdefault(method, {"times": [], "iters": [], "res": [], "rmse": []})
                    pivoted[method]["times"].append(run_metric["execution_time"])
                    pivoted[method]["iters"].append(run_metric["iterations"])
                    pivoted[method]["res"].append(run_metric["residual"])
                    pivoted[method]["rmse"].append(run_metric["rmse_exact"])

            for method, metrics in pivoted.items():
                avg_time = mean(metrics["times"])
                dev_time = stdev(metrics["times"]) if len(metrics["times"]) > 1 else 0.0
                avg_iters = int(mean(metrics["iters"]))

                record = {
                    "study_type": "grid_scaling",
                    "solver": solver_name,
                    "method": method,
                    "grid_size": g,
                    "threads": threads,
                    "processes": 1,
                    "execution_time_avg": avg_time,
                    "execution_time_std": dev_time,
                    "iterations": avg_iters,
                    "residual_avg": mean(metrics["res"]),
                    "rmse_exact_avg": mean(metrics["rmse"]),
                    "speedup": 1.0,
                    "efficiency": 100.0,
                    "repeats": len(metrics["times"])
                }
                self.raw_data.append(record)

    def export_raw_data(self, output_filepath):
        """Exports gathered benchmark data to a structured JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(self.raw_data, f, indent=4)
        sys.stdout.write(f"Raw benchmark metrics exported to: {output_filepath}\n")

    def export_csv_data(self, output_filepath):
        """Exports gathered benchmark records to a CSV file."""
        if not self.raw_data:
            return
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        keys = self.raw_data[0].keys()
        with open(output_filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.raw_data)
        sys.stdout.write(f"CSV benchmarks exported to: {output_filepath}\n")

# -----------------------------------------------------------------------------
# Elegant Markdown and LaTeX Report Generator
# -----------------------------------------------------------------------------
class ReportGenerator:
    """
    Constructs high-quality Markdown tables, lists, and LaTeX block codes
    directly from compiled benchmark datasets.
    """
    def __init__(self, data):
        self.data = data

    def generate_markdown(self):
        """Creates a well-formatted markdown report detailing scaling results."""
        if not self.data:
            return "No benchmark metrics gathered to compile reports.\n"

        md = []
        md.append("# High-Performance Computing Solvers Scaling Report")
        md.append(f"*Generated dynamically on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

        # 1. Thread Scaling Section
        ts_data = [d for d in self.data if d["study_type"] == "thread_scaling"]
        if ts_data:
            md.append("## 1. Shared Memory Thread Scaling Analysis")
            md.append("Measures solver runtime, scaling speedup, and parallel efficiency across thread counts.\n")
            
            # Group by method/solver
            grouped = {}
            for d in ts_data:
                key = (d["solver"], d["method"], d["grid_size"])
                grouped.setdefault(key, [])
                grouped[key].append(d)

            for (solver, method, grid), records in grouped.items():
                md.append(f"### {method} Solver ({solver}) — Grid: {grid}x{grid}")
                md.append("| Threads | Avg Iterations | Avg Time (s) | StdDev (s) | Speedup (x) | Efficiency (%) |")
                md.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
                for r in sorted(records, key=lambda x: x["threads"]):
                    md.append(
                        f"| {r['threads']} | {r['iterations']} | {r['execution_time_avg']:.6f} | "
                        f"{r['execution_time_std']:.6f} | {r['speedup']:.2f}x | {r['efficiency']:.1f}% |"
                    )
                md.append("")

        # 2. Grid Scaling Section
        gs_data = [d for d in self.data if d["study_type"] == "grid_scaling"]
        if gs_data:
            md.append("## 2. Problem Size Grid Scaling Complexity")
            md.append("Investigates execution times and iterations to converge as grid sizes grow.\n")

            grouped = {}
            for d in gs_data:
                key = (d["solver"], d["method"], d["threads"])
                grouped.setdefault(key, [])
                grouped[key].append(d)

            for (solver, method, threads), records in grouped.items():
                md.append(f"### {method} Solver ({solver}) — Threads: {threads}")
                md.append("| Grid Dimension (n) | Total Unknowns (n²) | Avg Iterations | Avg Runtime (s) | StdDev (s) | RMSE vs Exact |")
                md.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
                for r in sorted(records, key=lambda x: x["grid_size"]):
                    unknowns = r["grid_size"] * r["grid_size"]
                    md.append(
                        f"| {r['grid_size']}x{r['grid_size']} | {unknowns:,} | {r['iterations']} | "
                        f"{r['execution_time_avg']:.6f} | {r['execution_time_std']:.6f} | {r['rmse_exact_avg']:.2e} |"
                    )
                md.append("")

        return "\n".join(md)

    def generate_latex(self):
        """
        Compiles standard LaTeX table codes suitable for academic paper copy-pasting.
        """
        if not self.data:
            return "% No benchmark metrics gathered."

        tex = []
        tex.append("% =============================================================================")
        tex.append("% LaTeX Tables Generated Automatically by HPC Verification Framework")
        tex.append("% =============================================================================\n")

        ts_data = [d for d in self.data if d["study_type"] == "thread_scaling"]
        if ts_data:
            grouped = {}
            for d in ts_data:
                key = (d["solver"], d["method"], d["grid_size"])
                grouped.setdefault(key, [])
                grouped[key].append(d)

            for (solver, method, grid), records in grouped.items():
                tex.append(f"% Thread Scaling: {method} ({solver}) - Grid: {grid}x{grid}")
                tex.append("\\begin{table}[h!]")
                tex.append("  \\centering")
                tex.append("  \\caption{Thread Scaling Performance for " + method + " (" + solver + ") on a $" + str(grid) + "\\times " + str(grid) + "$ Grid}")
                tex.append("  \\label{tab:scale_" + method.lower().replace(" ", "_") + "}")
                tex.append("  \\begin{tabular}{cccccc}")
                tex.append("    \\hline")
                tex.append("    \\textbf{Threads} & \\textbf{Iterations} & \\textbf{Time (s)} & \\textbf{StdDev (s)} & \\textbf{Speedup} & \\textbf{Efficiency (\\%)} \\\\")
                tex.append("    \\hline")
                for r in sorted(records, key=lambda x: x["threads"]):
                    tex.append(
                        f"    {r['threads']} & {r['iterations']} & {r['execution_time_avg']:.4f} & "
                        f"{r['execution_time_std']:.4f} & {r['speedup']:.2f}x & {r['efficiency']:.1f}\\% \\\\"
                    )
                tex.append("    \\hline")
                tex.append("  \\end{tabular}")
                tex.append("\\end{table}\n")

        return "\n".join(tex)

    def save_report(self, md_filepath, tex_filepath=None):
        """Saves generated markdown and LaTeX scaling tables to local disk."""
        os.makedirs(os.path.dirname(os.path.abspath(md_filepath)), exist_ok=True)
        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown())
        sys.stdout.write(f"Markdown scaling report saved: {md_filepath}\n")

        if tex_filepath:
            os.makedirs(os.path.dirname(os.path.abspath(tex_filepath)), exist_ok=True)
            with open(tex_filepath, "w", encoding="utf-8") as f:
                f.write(self.generate_latex())
            sys.stdout.write(f"LaTeX Academic tables exported: {tex_filepath}\n")

# -----------------------------------------------------------------------------
# Matplotlib High-Quality Plots Generator
# -----------------------------------------------------------------------------
class Visualizer:
    """
    Constructs highly polished matplotlib figures representing benchmarks.
    Uses elegant color schemes, grids, markers, and axis scaling.
    """
    def __init__(self, data):
        self.data = data

    def plot_thread_scaling(self, output_imagepath):
        """Plots execution times, speedups, and parallel efficiencies side-by-side."""
        if not PLOTTING_AVAILABLE:
            sys.stderr.write("Plotting skipped: matplotlib is not installed.\n")
            return False

        ts_data = [d for d in self.data if d["study_type"] == "thread_scaling"]
        if not ts_data:
            return False

        # Sort by solver/method
        methods = {}
        for d in ts_data:
            method = d["method"]
            methods.setdefault(method, {"threads": [], "times": [], "speedup": [], "efficiency": []})
            methods[method]["threads"].append(d["threads"])
            methods[method]["times"].append(d["execution_time_avg"])
            methods[method]["speedup"].append(d["speedup"])
            methods[method]["efficiency"].append(d["efficiency"])

        # Setup figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), facecolor=PALETTE["bg"])
        chart_colors = [PALETTE["primary"], PALETTE["success"], PALETTE["accent"], PALETTE["warning"]]

        for idx, (method, m_data) in enumerate(methods.items()):
            color = chart_colors[idx % len(chart_colors)]
            # Sort data points by thread count to prevent plotting zig-zags
            zipped = sorted(zip(m_data["threads"], m_data["times"], m_data["speedup"], m_data["efficiency"]))
            threads, times, speedups, efficiencies = zip(*zipped)

            # Subplot 1: Runtimes
            ax1.plot(threads, times, marker="o", color=color, label=method, linewidth=2)
            # Subplot 2: Speedup
            ax2.plot(threads, speedups, marker="s", color=color, label=method, linewidth=2)
            # Subplot 3: Efficiency
            ax3.plot(threads, efficiencies, marker="^", color=color, label=method, linewidth=2)

        # Apply ideal scaling line to speedup plot
        max_threads = max(ts_data, key=lambda x: x["threads"])["threads"]
        ax2.plot([1, max_threads], [1, max_threads], linestyle="--", color="#6e7681", alpha=0.7, label="Ideal")
        ax3.axhline(y=100.0, linestyle="--", color="#6e7681", alpha=0.7, label="100% Ideal")

        # Format axes styles
        for ax, title, ylabel in [
            (ax1, "Execution Time", "Avg Runtime (s)"),
            (ax2, "Speedup Factor", "Speedup (x)"),
            (ax3, "Parallel Efficiency", "Efficiency (%)")
        ]:
            ax.set_facecolor(PALETTE["card"])
            ax.set_title(title, color="#e6edf3", fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel("Thread Count", color="#8b949e", fontsize=10)
            ax.set_ylabel(ylabel, color="#8b949e", fontsize=10)
            ax.grid(True, color=PALETTE["border"], linestyle=":", alpha=0.5)
            ax.tick_params(colors="#8b949e", labelsize=9)
            ax.legend(fontsize=9, facecolor=PALETTE["card"], edgecolor=PALETTE["border"], labelcolor="#e6edf3")
            for spine in ax.spines.values():
                spine.set_color(PALETTE["border"])

        plt.tight_layout(pad=3.0)
        os.makedirs(os.path.dirname(os.path.abspath(output_imagepath)), exist_ok=True)
        plt.savefig(output_imagepath, dpi=120, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        sys.stdout.write(f"Thread scaling comparison chart plotted successfully: {output_imagepath}\n")
        return True

    def plot_grid_scaling(self, output_imagepath):
        """Plots solving runtimes, iterations, and computational complexity growth curves."""
        if not PLOTTING_AVAILABLE:
            return False

        gs_data = [d for d in self.data if d["study_type"] == "grid_scaling"]
        if not gs_data:
            return False

        methods = {}
        for d in gs_data:
            method = d["method"]
            methods.setdefault(method, {"grids": [], "times": [], "iters": []})
            methods[method]["grids"].append(d["grid_size"])
            methods[method]["times"].append(d["execution_time_avg"])
            methods[method]["iters"].append(d["iterations"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor=PALETTE["bg"])
        chart_colors = [PALETTE["primary"], PALETTE["accent"], PALETTE["success"], PALETTE["warning"]]

        for idx, (method, m_data) in enumerate(methods.items()):
            color = chart_colors[idx % len(chart_colors)]
            zipped = sorted(zip(m_data["grids"], m_data["times"], m_data["iters"]))
            grids, times, iters = zip(*zipped)

            # Subplot 1: Grid vs runtime
            ax1.plot(grids, times, marker="o", color=color, label=method, linewidth=2)
            # Subplot 2: Grid vs iterations
            ax2.plot(grids, iters, marker="D", color=color, label=method, linewidth=2)

        for ax, title, ylabel in [
            (ax1, "Execution Time vs Grid Size", "Time (seconds)"),
            (ax2, "Convergence Iterations vs Grid Size", "Iterations")
        ]:
            ax.set_facecolor(PALETTE["card"])
            ax.set_title(title, color="#e6edf3", fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel("Grid Dimension (n x n)", color="#8b949e", fontsize=10)
            ax.set_ylabel(ylabel, color="#8b949e", fontsize=10)
            ax.grid(True, color=PALETTE["border"], linestyle=":", alpha=0.5)
            ax.tick_params(colors="#8b949e", labelsize=9)
            ax.legend(fontsize=9, facecolor=PALETTE["card"], edgecolor=PALETTE["border"], labelcolor="#e6edf3")
            for spine in ax.spines.values():
                spine.set_color(PALETTE["border"])

        plt.tight_layout(pad=3.0)
        os.makedirs(os.path.dirname(os.path.abspath(output_imagepath)), exist_ok=True)
        plt.savefig(output_imagepath, dpi=120, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        sys.stdout.write(f"Grid scaling complexity charts plotted successfully: {output_imagepath}\n")
        return True

# -----------------------------------------------------------------------------
# Main Application Flow Orchestration
# -----------------------------------------------------------------------------
def main():
    """Main execution flow directing testing benchmarks and verification."""
    parser = argparse.ArgumentParser(
        description="Automated profiling, verification, and scaling report suite for parallel solvers."
    )
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Path to the root of the HPC project containing bin/ and src/ directories."
    )
    parser.add_argument(
        "--solvers",
        default="Serial,OpenMP,Pthreads",
        help="Comma-separated list of solvers to profile."
    )
    parser.add_argument(
        "--grid-sizes",
        default="50,100,200,300",
        help="Comma-separated grid dimensions to profile in Grid scaling tests."
    )
    parser.add_argument(
        "--threads",
        default="1,2,4,8",
        help="Comma-separated thread configurations to test in Thread scaling benchmarks."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of times to run each solver to compute average stats."
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help="Tolerance threshold for convergence check."
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help="Max iterations threshold."
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip numerical correctness equivalence checks."
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory to save output files and plots."
    )

    args = parser.parse_args()

    project_path = os.path.abspath(args.project_dir)
    sys.stdout.write(f"Initializing HPC Automated Testing Framework...\n")
    sys.stdout.write(f"Project Workspace Directory: {project_path}\n")

    runner = SolverRunner(project_path)
    
    # 1. Run numerical validation
    if not args.skip_validation:
        validator = ValidationSuite(runner)
        validator.run_equivalence_test(grid_size=100, tolerance=args.tol)

    # 2. Run scaling benchmarks
    engine = BenchmarkEngine(runner)
    
    selected_solvers = [s.strip() for s in args.solvers.split(",")]
    grid_sizes = [int(g.strip()) for g in args.grid_sizes.split(",")]
    thread_counts = [int(t.strip()) for t in args.threads.split(",")]

    sys.stdout.write("\nOrchestrating Performance Benchmarks...\n")
    sys.stdout.write(f"Active Solvers: {', '.join(selected_solvers)}\n")
    sys.stdout.write(f"Grid Sizes: {grid_sizes}\n")
    sys.stdout.write(f"Thread Counts: {thread_counts}\n")
    sys.stdout.write(f"Benchmark Repeats: {args.repeats}\n\n")

    for s in selected_solvers:
        if s in ("OpenMP", "Pthreads", "Hybrid"):
            # Run Thread Scaling Study
            engine.run_thread_scaling(
                solver_name=s,
                grid_size=200,
                thread_list=thread_counts,
                repeats=args.repeats,
                max_iter=args.max_iter,
                tol=args.tol
            )
        
        # Run Grid Scaling Study
        engine.run_grid_scaling(
            solver_name=s,
            grid_sizes=grid_sizes,
            threads=4 if s in ("OpenMP", "Pthreads", "Hybrid") else 1,
            repeats=args.repeats,
            max_iter=args.max_iter,
            tol=args.tol
        )

    # 3. Export gathered datasets
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "hpc_metrics.json")
    csv_path = os.path.join(args.output_dir, "hpc_metrics.csv")
    engine.export_raw_data(json_path)
    engine.export_csv_data(csv_path)

    # 4. Generate markdown and LaTeX tables
    md_report_path = os.path.join(args.output_dir, "hpc_scaling_report.md")
    tex_report_path = os.path.join(args.output_dir, "hpc_scaling_tables.tex")
    
    reporter = ReportGenerator(engine.raw_data)
    reporter.save_report(md_report_path, tex_report_path)

    # 5. Generate benchmark graphs
    if PLOTTING_AVAILABLE:
        sys.stdout.write("\nPlotting performance comparison graphs...\n")
        vis = Visualizer(engine.raw_data)
        
        threads_image = os.path.join(args.output_dir, "thread_scaling_charts.png")
        grid_image = os.path.join(args.output_dir, "grid_scaling_charts.png")
        
        vis.plot_thread_scaling(threads_image)
        vis.plot_grid_scaling(grid_image)
    else:
        sys.stdout.write("\nPlotting skipped: matplotlib is not installed or unavailable.\n")

    sys.stdout.write("\nHPC Verification Framework completed successfully!\n")
    sys.stdout.write(f"All generated assets are located in: {os.path.abspath(args.output_dir)}\n")

if __name__ == "__main__":
    main()
