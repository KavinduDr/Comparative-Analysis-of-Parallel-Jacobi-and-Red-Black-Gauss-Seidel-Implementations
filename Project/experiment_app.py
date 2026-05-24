#!/usr/bin/env python3
"""
HPC Solver Experiment Desktop Application
==========================================
Interactive Tkinter + Matplotlib GUI for running and comparing
parallel Jacobi and Red-Black Gauss-Seidel solvers.

Developer: Rathnayaka I.G.T.A. (EG/2021/4754 — Tharanga Anuradha)
Coursework Project: EE7218/EC7207 High Performance Computing

Usage:  python3 experiment_app.py
"""

import os
import re
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from collections import OrderedDict

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ── Paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR    = os.path.join(SCRIPT_DIR, "bin")

SOLVERS = OrderedDict([
    ("Serial",   os.path.join(BIN_DIR, "serial")),
    ("OpenMP",   os.path.join(BIN_DIR, "openmp_solver")),
    ("Pthreads", os.path.join(BIN_DIR, "pthreads_solver")),
])

# ── Colour Palette ─────────────────────────────────────────────────────
C = {
    "bg":        "#0d1117",
    "bg2":       "#161b22",
    "card":      "#1c2128",
    "card_hi":   "#21262d",
    "input":     "#0d1117",
    "border":    "#30363d",
    "fg":        "#e6edf3",
    "fg2":       "#8b949e",
    "fg3":       "#6e7681",
    "accent":    "#58a6ff",
    "accent2":   "#3fb950",
    "purple":    "#bc8cff",
    "pink":      "#f778ba",
    "orange":    "#d29922",
    "red":       "#f85149",
    "cyan":      "#39d2c0",
}
CHART_COLORS = [C["accent"], C["accent2"], C["pink"], C["orange"], C["purple"], C["cyan"]]

# ── Output Parser ──────────────────────────────────────────────────────
_RE = re.compile(
    r"Method\s*:\s*(?P<method>.+?)\s*\n"
    r"\s*Grid\s*:\s*(?P<grid>\d+)\s*x\s*\d+.*?\n"
    r"\s*Iters\s*:\s*(?P<iters>\d+)\s*\n"
    r"\s*Time\s*:\s*(?P<time>[\d.]+)\s*seconds\s*\n"
    r"\s*Residual\s*\(inf-norm\)\s*:\s*(?P<residual>\S+)\s*\n"
    r"\s*RMSE vs exact\s*:\s*(?P<rmse>\S+)", re.MULTILINE)


def parse_output(text):
    return [{
        "method":   m.group("method").strip(),
        "grid":     int(m.group("grid")),
        "iters":    int(m.group("iters")),
        "time":     float(m.group("time")),
        "residual": float(m.group("residual")),
        "rmse":     float(m.group("rmse")),
    } for m in _RE.finditer(text)]


def run_solver(solver_key, grid, max_iter, tol, threads=4):
    binary = SOLVERS[solver_key]
    if not os.path.isfile(binary):
        raise FileNotFoundError(f"Binary not found: {binary}\nRun 'make all_cpu' first.")
    cmd = [binary, str(grid), str(max_iter), str(tol)]
    if solver_key in ("OpenMP", "Pthreads"):
        cmd.append(str(threads))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"Solver failed:\n{proc.stderr}")
    results = parse_output(proc.stdout)
    if not results:
        raise RuntimeError(f"Could not parse output:\n{proc.stdout}")
    for r in results:
        r["solver"] = solver_key
        r["threads"] = threads if solver_key != "Serial" else 1
    return results


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Custom Widgets                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _round_rect(canvas, x1, y1, x2, y2, radius=12, **kw):
    """Draw a rounded rectangle on a canvas using arc + line segments."""
    d = radius * 2
    canvas.create_arc(x1, y1, x1 + d, y1 + d, start=90, extent=90, style="pieslice", **kw)
    canvas.create_arc(x2 - d, y1, x2, y1 + d, start=0, extent=90, style="pieslice", **kw)
    canvas.create_arc(x2 - d, y2 - d, x2, y2, start=270, extent=90, style="pieslice", **kw)
    canvas.create_arc(x1, y2 - d, x1 + d, y2, start=180, extent=90, style="pieslice", **kw)
    canvas.create_rectangle(x1 + radius, y1, x2 - radius, y2, **kw)
    canvas.create_rectangle(x1, y1 + radius, x1 + radius, y2 - radius, **kw)
    canvas.create_rectangle(x2 - radius, y1 + radius, x2, y2 - radius, **kw)


class RoundedCard(tk.Canvas):
    """A card with canvas-drawn rounded corners containing an inner Frame."""
    def __init__(self, parent, bg_color=None, corner=12, pad=14, **kw):
        bg_color = bg_color or C["card"]
        kw.setdefault("highlightthickness", 0)
        super().__init__(parent, bg=C["bg"], **kw)
        self._bg = bg_color
        self._corner = corner
        self._pad = pad
        self.inner = tk.Frame(self, bg=bg_color)
        self._win_id = self.create_window(pad, pad, window=self.inner, anchor="nw")
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, _evt=None):
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2 or h < 2:
            return
        self.delete("bg")
        _round_rect(self, 0, 0, w, h, radius=self._corner,
                    fill=self._bg, outline=C["border"], width=1, tags="bg")
        self.tag_lower("bg")
        p = self._pad
        self.itemconfigure(self._win_id, width=max(1, w - p * 2),
                           height=max(1, h - p * 2))
        self.coords(self._win_id, p, p)


class MetricCard(tk.Canvas):
    """Metric card with rounded corners, accent dot, value, and label."""
    def __init__(self, parent, label, value="—", accent_color=None, **kw):
        accent_color = accent_color or C["accent"]
        kw.setdefault("highlightthickness", 0)
        kw.setdefault("height", 100)
        super().__init__(parent, bg=C["bg"], **kw)
        self._accent = accent_color
        self._label = label.upper()
        self._value = value
        self._corner = 12
        self.bind("<Configure>", self._paint)
        self.after(10, self._paint)

    def _paint(self, _evt=None):
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return
        self.delete("all")
        # Card background
        _round_rect(self, 0, 0, w, h, radius=self._corner,
                    fill=C["card"], outline=C["border"], width=1)
        # Accent dot
        self.create_oval(16, 16, 24, 24, fill=self._accent, outline="")
        # Label
        self.create_text(32, 20, text=self._label, anchor="w",
                         font=("Inter", 9, "bold"), fill=C["fg3"])
        # Value
        self.create_text(18, 58, text=self._value, anchor="w",
                         font=("SF Mono", 24, "bold"), fill=self._accent)

    def set_value(self, v):
        self._value = str(v)
        self._paint()


class RoundedInputBox(tk.Canvas):
    """A rounded outline box that holds an Entry or Combobox."""
    def __init__(self, parent, width=120, height=36, corner=8, **kw):
        kw.setdefault("highlightthickness", 0)
        super().__init__(parent, width=width, height=height, bg=C["card"], **kw)
        self._corner = corner
        self._req_width = width
        self._req_height = height
        self._focus = False
        self.bind("<Configure>", self._paint)
        self.after(10, self._paint)

    def _paint(self, _evt=None):
        w = self.winfo_width() or self._req_width
        h = self.winfo_height() or self._req_height
        self.delete("all")
        outline = C["accent"] if self._focus else C["border"]
        _round_rect(self, 1, 1, w - 2, h - 2, radius=self._corner,
                    fill=C["input"], outline=outline, width=1)

    def set_focus(self, focused):
        self._focus = focused
        self._paint()


class FormGroup(tk.Frame):
    """Vertical form field: label on top, input inside a rounded box below."""
    def __init__(self, parent, label, default="", width=12, values=None, **kw):
        # We handle width manually for the inner widget, don't pass it to Frame
        kw.pop("width", None) 
        super().__init__(parent, bg=C["card"], **kw)

        tk.Label(self, text=label, font=("Inter", 10), fg=C["fg2"],
                 bg=C["card"]).pack(anchor="w", pady=(0, 4))

        self.var = tk.StringVar(value=str(default))
        
        # Fixed width calculation: 12 chars ~ 120px
        px_width = width * 10
        self.box = RoundedInputBox(self, width=px_width, height=36)
        self.box.pack(fill="x", expand=True)
        self.box.pack_propagate(False)

        if values:  # Combobox
            self.widget = ttk.Combobox(self.box, textvariable=self.var, values=values,
                                       state="readonly", width=width,
                                       font=("Inter", 12))
        else:  # Entry
            self.widget = tk.Entry(self.box, textvariable=self.var, width=width,
                                   font=("SF Mono", 12), bg=C["input"],
                                   fg=C["fg"], insertbackground=C["fg"],
                                   relief="flat", highlightthickness=0)
            
        # Use .place() over the Canvas instead of create_window() to avoid macOS Combobox clipping bugs!
        self.widget.place(relx=0, rely=0.5, x=8, anchor="w", relwidth=1.0, width=-16)
        
        # Focus styling
        self.widget.bind("<FocusIn>", lambda e: self.box.set_focus(True))
        self.widget.bind("<FocusOut>", lambda e: self.box.set_focus(False))
        
    def get(self):
        return self.var.get()


class RunButton(tk.Canvas):
    """Rounded run button drawn on canvas with hover effect."""
    def __init__(self, parent, text="▶  Run", command=None, **kw):
        w = kw.pop("width", 140)
        h = kw.pop("height", 42)
        kw.setdefault("highlightthickness", 0)
        super().__init__(parent, width=w, height=h,
                         bg=C["card"], cursor="hand2", **kw)
        self._text = text
        self._cmd = command
        self._req_width = w
        self._req_height = h
        self._hover = False
        self.bind("<Configure>", self._paint)
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<Button-1>", lambda e: self._click())
        self.after(10, self._paint)

    def _set_hover(self, val):
        self._hover = val
        self._paint()

    def _paint(self, _evt=None):
        w = self.winfo_width() or self._req_width
        h = self.winfo_height() or self._req_height
        if w < 10 or h < 10:
            return
        self.delete("all")
        bg = "#2ea043" if self._hover else C["accent2"]
        _round_rect(self, 1, 1, w - 2, h - 2, radius=8,
                    fill=bg, outline="")
        self.create_text(w // 2, h // 2, text=self._text,
                         font=("Inter", 13, "bold"), fill="#fff")

    def _click(self):
        if self._cmd:
            self._cmd()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Main Application                                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HPC Solver Experiment Lab")
        self.geometry("1280x860")
        self.minsize(1060, 720)
        self.configure(bg=C["bg"])
        self.option_add("*TCombobox*Listbox*Background", C["input"])
        self.option_add("*TCombobox*Listbox*Foreground", C["fg"])

        self._setup_styles()
        self._build_header()
        self._build_tabs()
        self._build_status_bar()

        # Fix for macOS: Force the window to the foreground
        self.lift()
        self.attributes('-topmost', True)
        self.after_idle(self.attributes, '-topmost', False)

    def _setup_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=C["bg"], foreground=C["fg"], borderwidth=0,
                     font=("Inter", 12))
        s.configure("TFrame", background=C["bg"])
        s.configure("TNotebook", background=C["bg"], borderwidth=0, padding=0)
        s.configure("TNotebook.Tab", background=C["bg2"], foreground=C["fg3"],
                     font=("Inter", 11, "bold"), padding=(22, 10))
        s.map("TNotebook.Tab",
              background=[("selected", C["card"])],
              foreground=[("selected", C["accent"])])
        s.configure("TCombobox", fieldbackground=C["input"], foreground=C["fg"],
                     selectbackground=C["accent"], font=("Inter", 12))
        s.map("TCombobox", fieldbackground=[("readonly", C["input"])])
        s.configure("Treeview", background=C["card"], foreground=C["fg"],
                     fieldbackground=C["card"], font=("SF Mono", 11),
                     rowheight=34, borderwidth=0)
        s.configure("Treeview.Heading", background=C["bg2"], foreground=C["accent"],
                     font=("Inter", 10, "bold"), borderwidth=0)
        s.map("Treeview",
              background=[("selected", C["accent"])],
              foreground=[("selected", "#fff")])

    # ── Header ─────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=C["bg"])
        hdr.pack(fill="x", padx=30, pady=(22, 0))

        tk.Label(hdr, text="⚡", font=("Inter", 24), fg=C["accent"],
                 bg=C["bg"]).pack(side="left")
        title_frame = tk.Frame(hdr, bg=C["bg"])
        title_frame.pack(side="left", padx=(8, 0))
        tk.Label(title_frame, text="HPC Solver Experiment Lab",
                 font=("Inter", 20, "bold"), fg=C["fg"],
                 bg=C["bg"]).pack(anchor="w")
        tk.Label(title_frame, text="Parallel Jacobi  ·  Red-Black Gauss-Seidel  ·  Performance Analysis",
                 font=("Inter", 11), fg=C["fg3"],
                 bg=C["bg"]).pack(anchor="w")

        # Separator
        sep = tk.Frame(self, bg=C["border"], height=1)
        sep.pack(fill="x", padx=30, pady=(14, 0))

    # ── Tabs ───────────────────────────────────────────────────────────
    def _build_tabs(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=30, pady=(8, 14))

        self.tab_single = tk.Frame(self.nb, bg=C["bg"])
        self.tab_thread = tk.Frame(self.nb, bg=C["bg"])
        self.tab_grid   = tk.Frame(self.nb, bg=C["bg"])

        self.nb.add(self.tab_single, text="   ● Single Run   ")
        self.nb.add(self.tab_thread, text="   ◆ Thread Scaling   ")
        self.nb.add(self.tab_grid,   text="   ▲ Grid Scaling   ")

        self._build_single_tab()
        self._build_thread_tab()
        self._build_grid_tab()

    # ── Status Bar ─────────────────────────────────────────────────────
    def _build_status_bar(self):
        bar = tk.Frame(self, bg=C["bg2"], height=32)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="Ready")
        self.status_lbl = tk.Label(bar, textvariable=self.status_var,
                                   font=("SF Mono", 10), fg=C["fg3"],
                                   bg=C["bg2"], padx=30)
        self.status_lbl.pack(side="left", fill="y")

    # ── Async runner ───────────────────────────────────────────────────
    def _run_async(self, func, callback):
        self.status_var.set("⏳  Running solver…")
        self.status_lbl.config(fg=C["orange"])

        def worker():
            try:
                result = func()
                self.after(0, lambda: callback(result))
            except Exception as e:
                self.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_error(self, msg):
        self.status_var.set("✖  Error")
        self.status_lbl.config(fg=C["red"])
        messagebox.showerror("Solver Error", msg)

    def _status_ok(self, msg):
        self.status_var.set(f"✔  {msg}")
        self.status_lbl.config(fg=C["accent2"])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1 — Single Run
    # ══════════════════════════════════════════════════════════════════
    def _build_single_tab(self):
        tab = self.tab_single

        # ── Controls panel (rounded) ──
        ctrl_card = RoundedCard(tab, pad=16)
        ctrl_card.pack(fill="x", padx=4, pady=(12, 0))

        ctrl_header = tk.Frame(ctrl_card.inner, bg=C["card"])
        ctrl_header.pack(fill="x", pady=(0, 8))
        tk.Label(ctrl_header, text="Configuration", font=("Inter", 13, "bold"),
                 fg=C["fg"], bg=C["card"]).pack(side="left")

        ctrl = tk.Frame(ctrl_card.inner, bg=C["card"])
        ctrl.pack(fill="x")

        self.sr_solver  = FormGroup(ctrl, "Solver", "Serial", width=14,
                                    values=list(SOLVERS.keys()))
        self.sr_grid    = FormGroup(ctrl, "Grid Size", "200", width=10)
        self.sr_iter    = FormGroup(ctrl, "Max Iterations", "10000", width=10)
        self.sr_tol     = FormGroup(ctrl, "Tolerance", "1e-6", width=10)
        self.sr_threads = FormGroup(ctrl, "Threads", "4", width=6)

        for i, w in enumerate([self.sr_solver, self.sr_grid, self.sr_iter,
                                self.sr_tol, self.sr_threads]):
            w.grid(row=0, column=i, padx=(0, 16), sticky="ew")
        ctrl.columnconfigure(list(range(5)), weight=1)

        btn_frame = tk.Frame(ctrl, bg=C["card"])
        btn_frame.grid(row=0, column=5, padx=(8, 0), sticky="se")
        RunButton(btn_frame, text="▶  Run", command=self._single_run).pack()

        # ── Metric cards (rounded) ──
        cards_frame = tk.Frame(tab, bg=C["bg"])
        cards_frame.pack(fill="x", padx=4, pady=(10, 0))
        self.sr_cards = []
        card_defs = [
            ("Exec Time", C["accent"]),
            ("Iterations", C["accent2"]),
            ("Residual", C["purple"]),
            ("RMSE vs Exact", C["pink"]),
        ]
        for i, (label, color) in enumerate(card_defs):
            mc = MetricCard(cards_frame, label, accent_color=color)
            mc.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)
            self.sr_cards.append(mc)
            cards_frame.columnconfigure(i, weight=1)

        # ── Results table ──
        table_header = tk.Frame(tab, bg=C["bg"])
        table_header.pack(fill="x", padx=4, pady=(14, 4))
        tk.Label(table_header, text="Run History", font=("Inter", 13, "bold"),
                 fg=C["fg"], bg=C["bg"]).pack(side="left")

        clear_btn = tk.Label(table_header, text="Clear", font=("Inter", 10),
                             fg=C["accent"], bg=C["bg"], cursor="hand2")
        clear_btn.pack(side="right", padx=4)
        clear_btn.bind("<Button-1>", lambda e: self._clear_history())

        tree_wrap = tk.Frame(tab, bg=C["border"], padx=1, pady=1)
        tree_wrap.pack(fill="both", expand=True, padx=4, pady=(0, 6))

        cols = ("solver", "method", "grid", "threads", "iters", "time", "residual", "rmse")
        self.sr_tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", height=8)
        hdrs = {"solver": "Solver", "method": "Method", "grid": "Grid",
                "threads": "Thr", "iters": "Iterations", "time": "Time (s)",
                "residual": "Residual", "rmse": "RMSE"}
        widths = {"solver": 80, "method": 200, "grid": 60, "threads": 50,
                  "iters": 90, "time": 100, "residual": 100, "rmse": 100}
        for c in cols:
            self.sr_tree.heading(c, text=hdrs[c])
            self.sr_tree.column(c, width=widths[c], anchor="center", minwidth=40)
        self.sr_tree.pack(fill="both", expand=True)

    def _clear_history(self):
        for item in self.sr_tree.get_children():
            self.sr_tree.delete(item)
        for mc in self.sr_cards:
            mc.set_value("—")

    def _single_run(self):
        solver  = self.sr_solver.get()
        grid    = int(self.sr_grid.get())
        mi      = int(self.sr_iter.get())
        tol     = self.sr_tol.get()
        threads = int(self.sr_threads.get())

        def task():
            return run_solver(solver, grid, mi, tol, threads)

        def done(results):
            self._status_ok(f"{solver} finished — {len(results)} result(s)")
            if results:
                r = results[0]
                vals = [f"{r['time']:.4f}s", str(r["iters"]),
                        f"{r['residual']:.2e}", f"{r['rmse']:.2e}"]
                for mc, v in zip(self.sr_cards, vals):
                    mc.set_value(v)
            for r in results:
                self.sr_tree.insert("", 0, values=(
                    r["solver"], r["method"], r["grid"], r["threads"],
                    r["iters"], f"{r['time']:.6f}", f"{r['residual']:.2e}",
                    f"{r['rmse']:.2e}"))

        self._run_async(task, done)

    # ══════════════════════════════════════════════════════════════════
    # TAB 2 — Thread Scaling
    # ══════════════════════════════════════════════════════════════════
    def _build_thread_tab(self):
        tab = self.tab_thread

        ctrl_card = RoundedCard(tab, pad=16)
        ctrl_card.pack(fill="x", padx=4, pady=(12, 0))

        ctrl_header = tk.Frame(ctrl_card.inner, bg=C["card"])
        ctrl_header.pack(fill="x", pady=(0, 8))
        tk.Label(ctrl_header, text="Thread Scaling Study",
                 font=("Inter", 13, "bold"), fg=C["fg"], bg=C["card"]).pack(side="left")
        tk.Label(ctrl_header, text="Measures how performance scales with thread count",
                 font=("Inter", 10), fg=C["fg3"], bg=C["card"]).pack(side="left", padx=(12, 0))

        ctrl = tk.Frame(ctrl_card.inner, bg=C["card"])
        ctrl.pack(fill="x")

        parallel_solvers = [k for k in SOLVERS if k != "Serial"]
        self.ts_solver  = FormGroup(ctrl, "Solver", parallel_solvers[0],
                                    width=14, values=parallel_solvers)
        self.ts_grid    = FormGroup(ctrl, "Grid Size", "300", width=10)
        self.ts_iter    = FormGroup(ctrl, "Max Iterations", "10000", width=10)
        self.ts_tol     = FormGroup(ctrl, "Tolerance", "1e-6", width=10)
        self.ts_threads = FormGroup(ctrl, "Thread List (comma-sep)", "1,2,4,8", width=14)

        for i, w in enumerate([self.ts_solver, self.ts_grid, self.ts_iter,
                                self.ts_tol, self.ts_threads]):
            w.grid(row=0, column=i, padx=(0, 16), sticky="ew")
        ctrl.columnconfigure(list(range(5)), weight=1)

        btn_frame = tk.Frame(ctrl, bg=C["card"])
        btn_frame.grid(row=0, column=5, padx=(8, 0), sticky="se")
        RunButton(btn_frame, text="▶  Run Study", command=self._thread_run).pack()

        self.ts_chart_frame = tk.Frame(tab, bg=C["bg"])
        self.ts_chart_frame.pack(fill="both", expand=True, padx=4, pady=(10, 6))

        tk.Label(self.ts_chart_frame, text="Charts will appear here after running a study",
                 font=("Inter", 12), fg=C["fg3"], bg=C["bg"]).pack(expand=True)

    def _thread_run(self):
        solver  = self.ts_solver.get()
        grid    = int(self.ts_grid.get())
        mi      = int(self.ts_iter.get())
        tol     = self.ts_tol.get()
        threads = [int(t.strip()) for t in self.ts_threads.get().split(",")]

        def task():
            return [(t, run_solver(solver, grid, mi, tol, t)) for t in threads]

        def done(all_results):
            self._status_ok(f"Thread scaling complete — {len(all_results)} data points")
            self._draw_thread_chart(all_results)

        self._run_async(task, done)

    def _draw_thread_chart(self, all_results):
        for w in self.ts_chart_frame.winfo_children():
            w.destroy()

        methods = {}
        for t, results in all_results:
            for r in results:
                methods.setdefault(r["method"], {"threads": [], "times": [], "iters": []})
                methods[r["method"]]["threads"].append(t)
                methods[r["method"]]["times"].append(r["time"])
                methods[r["method"]]["iters"].append(r["iters"])

        fig = Figure(figsize=(12, 4.2), dpi=100, facecolor=C["bg"])

        for idx, (title, ylabel, data_fn) in enumerate([
            ("Execution Time", "Time (s)",
             lambda d: d["times"]),
            ("Speedup", "Speedup (×)",
             lambda d: [d["times"][0] / t if t > 0 else 0 for t in d["times"]]),
            ("Parallel Efficiency", "Efficiency (%)",
             lambda d: [(d["times"][0] / t / th * 100) if t > 0 else 0
                        for t, th in zip(d["times"], d["threads"])]),
        ]):
            ax = fig.add_subplot(1, 3, idx + 1, facecolor=C["card"])
            for i, (meth, data) in enumerate(methods.items()):
                y = data_fn(data)
                marker = ["o", "s", "^"][i % 3]
                ax.plot(data["threads"], y, f"{marker}-",
                        color=CHART_COLORS[i % len(CHART_COLORS)],
                        label=meth, linewidth=2.2, markersize=7,
                        markeredgecolor="#fff", markeredgewidth=0.5)

            if idx == 1:  # Ideal line for speedup
                mx = max(t for t, _ in all_results)
                ax.plot([1, mx], [1, mx], "--", color=C["fg3"], alpha=0.5, label="Ideal")
            if idx == 2:  # Ideal line for efficiency
                ax.axhline(y=100, color=C["fg3"], linestyle="--", alpha=0.5, label="100%")

            ax.set_xlabel("Threads", color=C["fg2"], fontsize=10)
            ax.set_ylabel(ylabel, color=C["fg2"], fontsize=10)
            ax.set_title(title, color=C["fg"], fontsize=12, fontweight="bold", pad=10)
            ax.legend(fontsize=8, facecolor=C["card"], edgecolor=C["border"],
                      labelcolor=C["fg"])
            ax.tick_params(colors=C["fg3"], labelsize=9)
            for sp in ax.spines.values():
                sp.set_color(C["border"])
            ax.grid(True, color=C["border"], alpha=0.3, linewidth=0.5)

        fig.tight_layout(pad=2.5)
        canvas = FigureCanvasTkAgg(fig, master=self.ts_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 3 — Grid Scaling
    # ══════════════════════════════════════════════════════════════════
    def _build_grid_tab(self):
        tab = self.tab_grid

        ctrl_card = RoundedCard(tab, pad=16)
        ctrl_card.pack(fill="x", padx=4, pady=(12, 0))

        ctrl_header = tk.Frame(ctrl_card.inner, bg=C["card"])
        ctrl_header.pack(fill="x", pady=(0, 8))
        tk.Label(ctrl_header, text="Grid Size Scaling Study",
                 font=("Inter", 13, "bold"), fg=C["fg"], bg=C["card"]).pack(side="left")
        tk.Label(ctrl_header, text="Measures how performance changes with problem size",
                 font=("Inter", 10), fg=C["fg3"], bg=C["card"]).pack(side="left", padx=(12, 0))

        ctrl = tk.Frame(ctrl_card.inner, bg=C["card"])
        ctrl.pack(fill="x")

        self.gs_solver  = FormGroup(ctrl, "Solver", "Serial", width=14,
                                    values=list(SOLVERS.keys()))
        self.gs_threads = FormGroup(ctrl, "Threads", "4", width=6)
        self.gs_iter    = FormGroup(ctrl, "Max Iterations", "10000", width=10)
        self.gs_tol     = FormGroup(ctrl, "Tolerance", "1e-6", width=10)
        self.gs_grids   = FormGroup(ctrl, "Grid Sizes (comma-sep)", "50,100,200,300", width=18)

        for i, w in enumerate([self.gs_solver, self.gs_threads, self.gs_iter,
                                self.gs_tol, self.gs_grids]):
            w.grid(row=0, column=i, padx=(0, 16), sticky="ew")
        ctrl.columnconfigure(list(range(5)), weight=1)

        btn_frame = tk.Frame(ctrl, bg=C["card"])
        btn_frame.grid(row=0, column=5, padx=(8, 0), sticky="se")
        RunButton(btn_frame, text="▶  Run Study", command=self._grid_run).pack()

        self.gs_chart_frame = tk.Frame(tab, bg=C["bg"])
        self.gs_chart_frame.pack(fill="both", expand=True, padx=4, pady=(10, 6))

        tk.Label(self.gs_chart_frame, text="Charts will appear here after running a study",
                 font=("Inter", 12), fg=C["fg3"], bg=C["bg"]).pack(expand=True)

    def _grid_run(self):
        solver  = self.gs_solver.get()
        threads = int(self.gs_threads.get())
        mi      = int(self.gs_iter.get())
        tol     = self.gs_tol.get()
        grids   = [int(g.strip()) for g in self.gs_grids.get().split(",")]

        def task():
            return [(g, run_solver(solver, g, mi, tol, threads)) for g in grids]

        def done(all_results):
            self._status_ok(f"Grid scaling complete — {len(all_results)} data points")
            self._draw_grid_chart(all_results)

        self._run_async(task, done)

    def _draw_grid_chart(self, all_results):
        for w in self.gs_chart_frame.winfo_children():
            w.destroy()

        methods = {}
        for g, results in all_results:
            for r in results:
                methods.setdefault(r["method"], {"grids": [], "times": [], "iters": []})
                methods[r["method"]]["grids"].append(g)
                methods[r["method"]]["times"].append(r["time"])
                methods[r["method"]]["iters"].append(r["iters"])

        fig = Figure(figsize=(12, 4.2), dpi=100, facecolor=C["bg"])

        for idx, (title, xlabel, ylabel, xfn, yfn) in enumerate([
            ("Execution Time", "Grid Size (n)", "Time (s)",
             lambda d: d["grids"], lambda d: d["times"]),
            ("Iterations to Converge", "Grid Size (n)", "Iterations",
             lambda d: d["grids"], lambda d: d["iters"]),
            ("Complexity Growth", "Total Unknowns (n²)", "Time (s)",
             lambda d: [g*g for g in d["grids"]], lambda d: d["times"]),
        ]):
            ax = fig.add_subplot(1, 3, idx + 1, facecolor=C["card"])
            for i, (meth, data) in enumerate(methods.items()):
                marker = ["o", "s", "^"][i % 3]
                ax.plot(xfn(data), yfn(data), f"{marker}-",
                        color=CHART_COLORS[i % len(CHART_COLORS)],
                        label=meth, linewidth=2.2, markersize=7,
                        markeredgecolor="#fff", markeredgewidth=0.5)
            ax.set_xlabel(xlabel, color=C["fg2"], fontsize=10)
            ax.set_ylabel(ylabel, color=C["fg2"], fontsize=10)
            ax.set_title(title, color=C["fg"], fontsize=12, fontweight="bold", pad=10)
            ax.legend(fontsize=8, facecolor=C["card"], edgecolor=C["border"],
                      labelcolor=C["fg"])
            ax.tick_params(colors=C["fg3"], labelsize=9)
            for sp in ax.spines.values():
                sp.set_color(C["border"])
            ax.grid(True, color=C["border"], alpha=0.3, linewidth=0.5)

        fig.tight_layout(pad=2.5)
        canvas = FigureCanvasTkAgg(fig, master=self.gs_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


# ── Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    missing = [k for k, v in SOLVERS.items() if not os.path.isfile(v)]
    if missing:
        print(f"⚠️  Missing binaries: {', '.join(missing)}")
        print(f"   Run 'make all_cpu' in {SCRIPT_DIR}")
    app = App()
    app.mainloop()
