"""Render every figure used by the BART slide deck (`slides/slides.md`).

This script is the single source of truth for the deck's PNGs. It does NOT
re-implement the BART algorithm: it reuses the from-scratch NumPy code and the
fits in the slide-source notebook `slides/how_bart_works.py`.

Mechanism
---------
The notebook is a marimo app, so its core functions (`run_bart`, `friedman`,
`bart_pdp`, ...) live inside per-cell closures and are NOT importable. marimo's
`app.run()` also can't be used directly: the `@app.class_definition` `Tree`
class is not injected into the other cells' globals in script mode. Instead we
use the path the project already documents -- `marimo export script` -- which
flattens every cell into one module-level namespace (no per-cell isolation), so
`Tree`, `run_bart`, the fit dicts, etc. are all plain globals. We exec that flat
script once and harvest the globals.

The notebook runs the full sweep of fits on exec, so this takes ~2 minutes.
Run from anywhere:  `pixi run python slides/scripts/make_figures.py`
"""

from __future__ import annotations

import pathlib
import runpy
import subprocess
import sys
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "slides" / "how_bart_works.py"
IMG = ROOT / "slides" / "images"


# --------------------------------------------------------------------------- #
# Harvest the notebook's globals (functions + precomputed fits)
# --------------------------------------------------------------------------- #
def load_notebook_namespace() -> dict:
    """Export the marimo notebook to a flat script and exec it, returning its globals."""
    flat = pathlib.Path(tempfile.gettempdir()) / "nb01_flat_for_figures.py"
    print(f"Exporting {NOTEBOOK.name} -> flat script ...", flush=True)
    subprocess.run(
        ["marimo", "export", "script", str(NOTEBOOK), "-o", str(flat)],
        check=True,
    )
    print("Executing flat script (runs the BART fits, ~2 min) ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ns = runpy.run_path(str(flat))
    print(f"  harvested {len(ns)} globals", flush=True)
    return ns


# --------------------------------------------------------------------------- #
# Slide-friendly matplotlib style (applied AFTER exec, which resets rcParams)
# --------------------------------------------------------------------------- #
def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig, name: str) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    path = IMG / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}", flush=True)


def save_anim(fig, anim, name: str, fps: int) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    path = IMG / name
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}", flush=True)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_gbm_demo(ns):
    d = ns["gbm_demo"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.scatter(
        d["x_train"],
        d["y_train"],
        s=22,
        alpha=0.7,
        color="#4c4c4c",
        label="observations",
    )
    ax.plot(d["x_grid"], d["f_true"], color="#c44e52", lw=2.0, label=r"true $f(x)$")
    ax.plot(
        d["x_grid"],
        d["f_pred"],
        color="#55a868",
        lw=2.4,
        label="gradient-boosting prediction",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "gbm_demo.png")


def fig_bart_hero(ns):
    b = ns["buildup"]
    fit = b["fits"][100]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.scatter(
        b["x_train"],
        b["y_train"],
        s=22,
        alpha=0.7,
        color="#4c4c4c",
        label="observations",
    )
    ax.plot(b["x_grid"], b["f_true"], color="#c44e52", lw=2.0, label=r"true $f(x)$")
    ax.plot(
        b["x_grid"], fit["f_mean"], color="#4c72b0", lw=2.4, label="BART posterior mean"
    )
    ax.fill_between(
        b["x_grid"],
        fit["f_lo"],
        fit["f_hi"],
        color="#4c72b0",
        alpha=0.18,
        label="90% credible band",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "bart_hero.png")


def fig_mcmc_demo(ns):
    draws, accept = ns["demo_rw_metropolis"](
        ns["demo_log_beta"], x0=0.5, step=0.3, n_draws=4000, rng=20260518
    )
    xs = np.linspace(1e-4, 1 - 1e-4, 400)
    truth = 12.0 * xs**2 * (1.0 - xs)  # Beta(3, 2) pdf
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].plot(draws, lw=0.4, color="#4c4c4c")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("x")
    axes[0].set_title(f"trace (accept rate = {accept:.0%})")
    axes[1].hist(
        draws[1000:],
        bins=40,
        density=True,
        alpha=0.75,
        color="#4c72b0",
        edgecolor="white",
        label="MH draws (post burn-in)",
    )
    axes[1].plot(xs, truth, color="#c44e52", lw=2.4, label="Beta(3, 2) target")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("density")
    axes[1].set_title("draws vs target")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "mcmc_demo.png")


def fig_single_tree(ns):
    Tree = ns["Tree"]
    predict = ns["predict"]
    rng = np.random.default_rng(0)
    n = 250
    X = rng.uniform(size=(n, 2))
    f = np.where(
        X[:, 0] < 0.5,
        np.where(X[:, 1] < 0.6, -1.2, 0.4),
        np.where(X[:, 1] < 0.3, 0.9, 1.6),
    )
    y = f + rng.normal(0, 0.2, size=n)

    t = Tree()
    t.split_var = [0, 1, 1, -1, -1, -1, -1]
    t.split_val = [0.5, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0]
    t.left = [1, 3, 5, -1, -1, -1, -1]
    t.right = [2, 4, 6, -1, -1, -1, -1]
    t.parent = [-1, 0, 0, 1, 1, 2, 2]
    t.mu = [0.0, 0.0, 0.0, -1.2, 0.4, 0.9, 1.6]

    grid = np.linspace(0, 1, 200)
    G0, G1 = np.meshgrid(grid, grid)
    Z = predict(t, np.column_stack([G0.ravel(), G1.ravel()])).reshape(G0.shape)

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    im = ax.pcolormesh(G0, G1, Z, cmap="RdBu_r", vmin=-1.8, vmax=1.8, shading="auto")
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="RdBu_r",
        vmin=-1.8,
        vmax=1.8,
        edgecolor="black",
        linewidth=0.4,
        s=24,
    )
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")
    fig.colorbar(im, ax=ax, label=r"leaf $\mu$")
    fig.tight_layout()
    save(fig, "single_tree.png")


def fig_sum_of_trees_panels(ns):
    b = ns["buildup"]
    ms = b["ms"]
    fig, axes = plt.subplots(
        1, len(ms), figsize=(3.0 * len(ms), 3.2), sharex=True, sharey=True
    )
    for ax, m in zip(axes, ms):
        fit = b["fits"][m]
        ax.scatter(b["x_train"], b["y_train"], s=10, alpha=0.55, color="#4c4c4c")
        ax.plot(b["x_grid"], b["f_true"], color="#c44e52", lw=1.2)
        ax.plot(
            b["x_grid"], fit["f_draw"], color="#4c72b0", lw=1.6, drawstyle="steps-mid"
        )
        ax.set_title(f"m = {m}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.suptitle(
        "One posterior draw as m grows: blocky step → smooth staircase", y=1.04
    )
    fig.tight_layout()
    save(fig, "sum_of_trees_panels.png")


def anim_sum_of_trees_buildup(ns):
    """Animated GIF: the fit refining as trees are summed one at a time.

    Reuses the from-scratch sampler. We fit one fresh BART on the 1D toy DGP,
    pick the posterior draw closest to the posterior mean, then animate the
    cumulative sum g_k = sum_{j<=k} tree_j across its m trees. `predict` returns
    SCALED leaf values, so each cumulative curve is inverse-scaled exactly as
    `predict_at` does: f_scaled * y_range + (y_min + 0.5 * y_range).
    """
    b = ns["buildup"]
    run_bart = ns["run_bart"]
    predict = ns["predict"]
    x_train, y_train = b["x_train"], b["y_train"]
    x_grid, f_true = b["x_grid"], b["f_true"]

    print("  fitting toy BART for buildup animation ...", flush=True)
    fit = run_bart(
        x_train[:, None],
        y_train,
        m=100,
        n_iter=400,
        burn_in=200,
        thin=10,
        rng=np.random.default_rng(20260423),
    )

    # Posterior draw closest (L2) to the posterior mean, in scaled training space.
    fd = fit["f_draws_scaled"]  # (kept, n_train), scaled units
    d = int(np.argmin(((fd - fd.mean(axis=0)) ** 2).sum(axis=1)))
    trees = fit["tree_snapshots"][d]
    m = len(trees)

    # Per-tree grid predictions (scaled) -> cumulative inverse-scaled curves.
    y_min, y_range = fit["y_min"], fit["y_range"]
    preds = np.array([predict(t, x_grid[:, None]) for t in trees])  # (m, n_grid)
    cum = np.cumsum(preds, axis=0) * y_range + (y_min + 0.5 * y_range)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=90)
    ax.scatter(x_train, y_train, s=12, alpha=0.55, color="#4c4c4c", label="data")
    ax.plot(x_grid, f_true, color="#c44e52", lw=1.4, label="true f(x)")
    (line,) = ax.plot(
        x_grid,
        cum[0],
        color="#4c72b0",
        lw=1.8,
        drawstyle="steps-mid",
        label="sum of trees",
    )
    pad = 0.15 * (y_train.max() - y_train.min())
    ax.set_ylim(
        min(y_train.min(), f_true.min()) - pad,
        max(y_train.max(), f_true.max()) + pad,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", frameon=False)
    title = ax.set_title(f"Sum of trees: 1 / {m}")
    fig.tight_layout()

    def update(k):
        line.set_ydata(cum[k])
        title.set_text(f"Sum of trees: {k + 1} / {m}")
        return line, title

    hold = 30  # ~2 s dwell on the converged fit before the loop restarts
    frame_seq = list(range(m)) + [m - 1] * hold
    anim = FuncAnimation(fig, update, frames=frame_seq, interval=1000 / 15, blit=False)
    save_anim(fig, anim, "sum_of_trees_buildup.gif", fps=15)


def fig_prior_tree_sizes(ns):
    tsd = ns["terminal_size_dist"]
    betas = [(0.95, 0.5), (0.95, 2.0), (0.95, 4.0)]
    # Shared bins/axes across panels so the distributions are directly comparable:
    # higher beta visibly shifts mass toward fewer terminal nodes.
    bins = np.arange(1, 15 + 2) - 0.5
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True, sharey=True)
    for ax, (a, beta) in zip(axes, betas):
        sizes = tsd(a, beta)
        ax.hist(sizes, bins=bins, density=True, color="#4c72b0", edgecolor="white")
        ax.set_title(rf"$\alpha={a:.2f},\ \beta={beta:.1f}$")
    axes[0].set_ylabel("prior probability")
    fig.supxlabel("number of terminal nodes")
    fig.suptitle(r"Higher $\beta$ punishes depth harder → smaller trees", y=1.04)
    fig.tight_layout()
    save(fig, "prior_tree_sizes.png")


def fig_prior_sample_trees(ns):
    """Six trees drawn from the CGM98 prior at the default (alpha, beta)."""
    Tree = ns["Tree"]
    alpha, beta = 0.95, 2.0

    def sample_tree(rng, max_depth=6):
        t = Tree()
        stack = [(0, 0)]
        while stack:
            node, d = stack.pop()
            p = alpha * (1 + d) ** (-beta) if d < max_depth else 0.0
            if rng.random() < p:
                left = len(t.split_var)
                right = left + 1
                t.split_var[node] = 0
                t.split_val[node] = 0.5
                t.left[node] = left
                t.right[node] = right
                t.split_var += [-1, -1]
                t.split_val += [0.0, 0.0]
                t.left += [-1, -1]
                t.right += [-1, -1]
                t.parent += [node, node]
                t.mu += [0.0, 0.0]
                stack.append((right, d + 1))
                stack.append((left, d + 1))
        return t

    def layout(t):
        pos = {}
        counter = [0]

        def rec(node, depth):
            if t.is_leaf(node):
                x = counter[0]
                counter[0] += 1
                pos[node] = (x, -depth)
                return x
            xl = rec(t.left[node], depth + 1)
            xr = rec(t.right[node], depth + 1)
            x = 0.5 * (xl + xr)
            pos[node] = (x, -depth)
            return x

        rec(0, 0)
        return pos

    rng = np.random.default_rng(20260423)
    trees = [sample_tree(rng) for _ in range(6)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
    for ax, t in zip(axes.ravel(), trees):
        pos = layout(t)
        for i in range(len(t.split_var)):
            if not t.is_leaf(i):
                for child in (t.left[i], t.right[i]):
                    ax.plot(
                        [pos[i][0], pos[child][0]],
                        [pos[i][1], pos[child][1]],
                        color="#999",
                        lw=1.2,
                        zorder=1,
                    )
        for i, (x, y) in pos.items():
            leaf = t.is_leaf(i)
            ax.scatter(
                [x],
                [y],
                s=190,
                zorder=2,
                color="#c44e52" if leaf else "#4c72b0",
                edgecolor="#333",
                linewidth=0.8,
            )
        ax.set_title(f"{len(t.leaves())} leaves", fontsize=12)
        ax.axis("off")
    fig.suptitle(r"Six trees drawn from the prior  ($\alpha=0.95,\ \beta=2.0$)", y=1.0)
    fig.tight_layout()
    save(fig, "prior_sample_trees.png")


# --------------------------------------------------------------------------- #
# Schematic tree-glyph helpers (shared by the two conceptual slides)
# --------------------------------------------------------------------------- #
def _make_tree(ns, depth):
    """A small balanced Tree of the given depth, used purely as a drawn glyph.

    depth=1 -> stump (root + 2 leaves); depth=2 -> 7 nodes. Split fields are
    placeholders: these trees are never evaluated, only laid out and drawn.
    """
    Tree = ns["Tree"]
    t = Tree()
    t.split_var = [-1]
    t.split_val = [0.0]
    t.left = [-1]
    t.right = [-1]
    t.parent = [-1]
    t.mu = [0.0]

    def grow(node, d):
        if d == 0:
            return
        left = len(t.split_var)
        right = left + 1
        t.split_var[node] = 0
        t.split_val[node] = 0.5
        t.left[node] = left
        t.right[node] = right
        t.split_var += [-1, -1]
        t.split_val += [0.0, 0.0]
        t.left += [-1, -1]
        t.right += [-1, -1]
        t.parent += [node, node]
        t.mu += [0.0, 0.0]
        grow(left, d - 1)
        grow(right, d - 1)

    grow(0, depth)
    return t


def _tree_layout(t):
    """Recursive (x, -depth) layout, identical to fig_prior_sample_trees."""
    pos = {}
    counter = [0]

    def rec(node, depth):
        if t.is_leaf(node):
            x = counter[0]
            counter[0] += 1
            pos[node] = (x, -depth)
            return x
        xl = rec(t.left[node], depth + 1)
        xr = rec(t.right[node], depth + 1)
        x = 0.5 * (xl + xr)
        pos[node] = (x, -depth)
        return x

    rec(0, 0)
    return pos


def _draw_glyph(ax, t, ox, oy, scale=1.0, node_s=120, lw=1.1):
    """Draw tree `t` as a glyph centred at (ox, oy) in axis data coords."""
    pos = _tree_layout(t)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    P = {i: (ox + (x - cx) * scale, oy + (y - cy) * scale) for i, (x, y) in pos.items()}
    for i in range(len(t.split_var)):
        if not t.is_leaf(i):
            for child in (t.left[i], t.right[i]):
                ax.plot(
                    [P[i][0], P[child][0]],
                    [P[i][1], P[child][1]],
                    color="#999",
                    lw=lw,
                    zorder=1,
                )
    for i, (x, y) in P.items():
        leaf = t.is_leaf(i)
        ax.scatter(
            [x],
            [y],
            s=node_s,
            zorder=2,
            color="#c44e52" if leaf else "#4c72b0",
            edgecolor="#333",
            linewidth=0.7,
        )


# --------------------------------------------------------------------------- #
# "The BART algorithm -- the idea": Trees x Iterations grid (capstone schematic)
# --------------------------------------------------------------------------- #
def fig_bart_idea_grid(ns):
    """Schematic of the whole sampler, after the ISLP/ESL 'BART algorithm -- the
    idea' figure. Columns are the m trees, rows are MCMC sweeps; each sweep every
    tree is perturbed by an MH move against its partial residual, and averaging the
    per-sweep sums-of-trees gives the final prediction. Pure schematic built from
    the tree-glyph helpers (no fit data).
    """
    col_x = [0.0, 1.5, 3.2]
    col_lab = ["1", "2", "m"]
    # rows 1, 2, 3, then a gap (vertical ellipsis) then B
    row_y = [0.0, -1.15, -2.30, -4.0]
    row_lab = ["1", "2", "3", "B"]
    depths = [
        [0, 0, 0],  # sweep 1: stumps (single nodes)
        [1, 1, 1],  # sweep 2: first splits
        [2, 0, 2],  # sweep 3: tree 2 pruned back to a stump
        [2, 1, 2],  # sweep B: grown
    ]
    GREEN = "#2e7d32"

    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    for ry, drow in zip(row_y, depths):
        for cx, dep in zip(col_x, drow):
            _draw_glyph(
                ax, _make_tree(ns, dep), ox=cx, oy=ry, scale=0.26, node_s=70, lw=0.9
            )

    # top header: "Trees ->" + column numbers (ellipsis before the last)
    ax.text(
        -2.0,
        1.05,
        "Trees",
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#333",
    )
    ax.annotate(
        "",
        xy=(-0.5, 1.05),
        xytext=(-1.2, 1.05),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4),
    )
    for cx, lab in zip(col_x, col_lab):
        ax.text(
            cx,
            1.05,
            lab,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#333",
        )
    ax.text(
        (col_x[1] + col_x[2]) / 2,
        1.05,
        r"$\cdots$",
        ha="center",
        va="center",
        fontsize=15,
        color="#333",
    )

    # left header: "Iterations" + down arrow + row numbers
    ax.text(
        -2.0,
        0.45,
        "Iterations",
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#333",
    )
    ax.annotate(
        "",
        xy=(-1.55, row_y[-1] + 0.2),
        xytext=(-1.55, 0.1),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4),
    )
    for ry, lab in zip(row_y, row_lab):
        ax.text(
            -1.15,
            ry,
            lab,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#333",
        )
    ax.text(
        -1.15,
        (row_y[2] + row_y[3]) / 2,
        r"$\vdots$",
        ha="center",
        va="center",
        fontsize=15,
        color="#333",
    )

    # green annotations on the right
    ax.annotate(
        "perturbations based\non partial residuals",
        xy=(col_x[-1] + 0.55, row_y[1]),
        xytext=(col_x[-1] + 1.5, row_y[1]),
        ha="left",
        va="center",
        fontsize=12,
        color=GREEN,
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8),
    )
    ax.annotate(
        "average across sweeps\n→ final predictions",
        xy=(col_x[-1] + 0.55, row_y[-1]),
        xytext=(col_x[-1] + 1.5, row_y[-1]),
        ha="left",
        va="center",
        fontsize=12,
        color=GREEN,
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8),
    )

    ax.set_xlim(-2.2, 7.8)
    ax.set_ylim(-4.8, 1.5)
    ax.axis("off")
    save(fig, "bart_idea_grid.png")


def fig_conjugate_leaf(ns):
    from scipy.stats import norm

    rng = np.random.default_rng(7)
    n_l, sigma, sigma_mu, true_mu = 14, 0.3, 0.25, 0.5
    residuals = true_mu + rng.normal(0, sigma, size=n_l)
    rbar = residuals.mean()
    post_var = sigma**2 * sigma_mu**2 / (sigma**2 + n_l * sigma_mu**2)
    post_mean = (n_l * sigma_mu**2) / (sigma**2 + n_l * sigma_mu**2) * rbar
    post_sd = np.sqrt(post_var)
    grid = np.linspace(-1.0, 1.5, 600)

    fig, (ax_r, ax_d) = plt.subplots(1, 2, figsize=(11, 3.9))
    ax_r.hist(residuals, bins=8, color="#888", alpha=0.55, edgecolor="white")
    ax_r.axvline(rbar, color="#c44e52", lw=1.6, label=f"mean = {rbar:.2f}")
    ax_r.set_xlabel("residual r")
    ax_r.set_ylabel("count")
    ax_r.set_title(f"Residuals in one leaf  (n = {n_l})")
    ax_r.legend(frameon=False)

    ax_d.plot(
        grid,
        norm.pdf(grid, 0, sigma_mu),
        lw=1.8,
        color="#8c8c8c",
        label=rf"prior $\mathcal{{N}}(0, {sigma_mu:.2f}^2)$",
    )
    ax_d.plot(
        grid,
        norm.pdf(grid, rbar, sigma / np.sqrt(n_l)),
        lw=1.8,
        color="#4c72b0",
        label=rf"likelihood $\propto \mathcal{{N}}({rbar:.2f}, \sigma^2/n)$",
    )
    ax_d.plot(
        grid,
        norm.pdf(grid, post_mean, post_sd),
        lw=2.6,
        color="#c44e52",
        label=rf"posterior $\mathcal{{N}}({post_mean:.2f}, {post_sd:.2f}^2)$",
    )
    for v, c in [(0, "#8c8c8c"), (rbar, "#4c72b0"), (post_mean, "#c44e52")]:
        ax_d.axvline(v, color=c, lw=0.7, ls=":")
    ax_d.set_xlabel(r"$\mu_\ell$")
    ax_d.set_ylabel("density")
    ax_d.set_title("Prior + likelihood = posterior")
    ax_d.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    save(fig, "conjugate_leaf.png")


def fig_coverage(ns):
    fit = ns["friedman_fits"][50]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    panels = [
        (
            a,
            fit["f_true_in"],
            fit["f_mean"],
            fit["f_lo"],
            fit["f_hi"],
            fit["coverage_in"],
            "in-sample",
        ),
        (
            b,
            fit["f_test_true"],
            fit["f_test_mean"],
            fit["f_test_lo"],
            fit["f_test_hi"],
            fit["coverage_oos"],
            "out-of-sample",
        ),
    ]
    for ax, xt, mn, lo, hi, cov, tag in panels:
        order = np.argsort(xt)
        ax.errorbar(
            xt[order],
            mn[order],
            yerr=[(mn - lo)[order], (hi - mn)[order]],
            fmt="o",
            ms=3,
            ecolor="#4c72b0",
            color="#333",
            alpha=0.55,
            elinewidth=0.7,
        )
        lim = (xt.min() - 1, xt.max() + 1)
        ax.plot(lim, lim, "--", color="#c44e52", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(r"true $f(x)$")
        ax.set_ylabel(r"posterior mean $\hat f(x)$ with 90% PI")
        ax.set_title(f"{tag} — 90% coverage: {cov:.0%}")
    fig.tight_layout()
    save(fig, "coverage.png")


def fig_choosing_m(ns):
    fits = ns["toy_calibration_fits"]
    X_toy = ns["X_toy"]
    xg = ns["X_toy_grid"][:, 0]
    y_toy = ns["y_toy"]
    f_true = ns["f_toy_true"]
    ms = [5, 10, 15, 20]

    # Fit panels
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.3), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, ms):
        f = fits[m]
        ax.scatter(X_toy[:, 0], y_toy, s=10, alpha=0.55, color="#4c4c4c", zorder=3)
        ax.plot(xg, f_true, color="#c44e52", lw=1.2, ls="--", zorder=2)
        ax.fill_between(
            xg, f["f_lo_grid"], f["f_hi_grid"], color="#4c72b0", alpha=0.25, zorder=1
        )
        ax.plot(xg, f["f_mean_grid"], color="#4c72b0", lw=1.6, zorder=2)
        ax.set_title(f"m = {m}  (cov {f['coverage']:.0%})")
    for ax in axes[-1, :]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    fig.suptitle("Fit smooths and the band tightens as m grows", y=1.04)
    fig.tight_layout()
    save(fig, "choosing_m_fits.png")

    # Sigma overlay: small, closely-spaced m so the concentration toward the
    # noise floor is visible as a progression (the wider panel range above
    # piles 20/50/200 on top of each other at the floor).
    ms_sigma = [5, 10, 20, 40]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    colors = ["#dd8452", "#55a868", "#4c72b0", "#8172b3"]
    for m, c in zip(ms_sigma, colors):
        ax.hist(
            fits[m]["sigma_draws"],
            bins=30,
            density=True,
            histtype="step",
            lw=2.0,
            color=c,
            label=f"m = {m}",
        )
    ax.axvline(0.15, color="#c44e52", lw=2, ls="--", label=r"true $\sigma = 0.15$")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("posterior density")
    ax.set_title(r"$\sigma$ posterior concentrates toward the noise floor")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "choosing_m_sigma.png")


def fig_restricted_r2(ns):
    fit = ns["friedman_fits"][50]
    ranking = list(fit["ranking"])
    r2 = fit["r2_curve"]
    p = len(r2)
    xs = np.arange(1, p + 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(xs, r2, "o-", color="#4c72b0", lw=1.8)
    ax.axhline(0.95, color="#c44e52", ls="--", lw=1, label=r"$R^2 = 0.95$")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$X_{{{ranking[i]}}}$" for i in range(p)])
    ax.set_xlabel("variables included (cumulative, by inclusion rank)")
    ax.set_ylabel(r"$R^2$ vs. full-model predictions")
    ax.set_title("Restricted-model R² flattens at the 5 relevant variables")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "restricted_r2.png")


def fig_pdp(ns):
    bart_pdp = ns["bart_pdp"]
    X_fried = ns["X_fried"]
    fit_50 = ns["friedman_fits"][50]
    grid = np.linspace(0.0, 1.0, 30)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    specs = [
        (axes[0], 0, r"$x_0$ (sin component)", None),
        (axes[1], 3, r"$x_3$ (linear)", lambda v: 10.0 * v),
    ]
    for ax, j, label, truth in specs:
        d = bart_pdp(fit_50, X_fried, j=j, grid=grid)
        mean = d.mean(axis=0)
        lo = np.quantile(d, 0.05, axis=0)
        hi = np.quantile(d, 0.95, axis=0)
        ax.fill_between(
            grid, lo, hi, color="#4c72b0", alpha=0.25, label="90% posterior band"
        )
        ax.plot(grid, mean, color="#4c72b0", lw=2.4, label="PDP posterior mean")
        if truth is not None:
            tv = np.array([truth(v) for v in grid])
            ax.plot(
                grid,
                tv - tv.mean() + mean.mean(),
                color="#c44e52",
                ls="--",
                lw=1.8,
                label="true shape",
            )
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\mathrm{PD}_j(v)$")
        ax.set_title(f"PDP for {label}")
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "pdp.png")


def fig_ice(ns):
    bart_ice = ns["bart_ice"]
    X_fried = ns["X_fried"]
    fit_50 = ns["friedman_fits"][50]
    grid = np.linspace(0.0, 1.0, 30)
    rng = np.random.default_rng(20260518)
    row_idx = rng.choice(X_fried.shape[0], size=30, replace=False)
    ice_mean = bart_ice(fit_50, X_fried, j=3, grid=grid, row_idx=row_idx).mean(axis=0)
    pdp_mean = ice_mean.mean(axis=0)
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for r in range(ice_mean.shape[0]):
        ax.plot(grid, ice_mean[r], color="#4c72b0", alpha=0.25, lw=0.9)
    ax.plot(grid, pdp_mean, color="#c44e52", lw=2.6, label="PDP (row-average)")
    ax.set_xlabel(r"$x_3$")
    ax.set_ylabel("posterior-mean prediction")
    ax.set_title("ICE curves at 30 random rows — parallel ⇒ additive")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "ice.png")


def fig_sparse_compare(ns):
    sc = ns["sparse_compare"]
    p_sp = sc["p_total"]
    n_rel = sc["n_relevant"]
    inc_uni = sc["uniform_inclusion"]
    inc_sp = sc["sparse_inclusion"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    xs = np.arange(p_sp)
    specs = [
        (
            axes[0],
            inc_uni,
            f"uniform 1/p prior (max noise {inc_uni[n_rel:].max():.3f})",
        ),
        (
            axes[1],
            inc_sp,
            f"Dirichlet a=1 prior (max noise {inc_sp[n_rel:].max():.3f})",
        ),
    ]
    for ax, inc, title in specs:
        cols = ["#c44e52" if i < n_rel else "#bbb" for i in xs]
        ax.bar(xs, inc, color=cols, edgecolor="none")
        ax.set_xlabel("variable index")
        ax.set_title(title, fontsize=12)
    axes[0].set_ylabel("inclusion frequency")
    fig.suptitle("Inclusion frequency: red = relevant, grey = noise", y=1.02)
    fig.tight_layout()
    save(fig, "sparse_compare.png")


def fig_overfitting(ns):
    """BART vs gradient boosting as iterations / rounds grow.

    The classic train/test overfitting picture (cf. the ISLP Heart-data slide),
    reproduced on the deck's own Friedman DGP. Gradient boosting drives its
    TRAINING error toward zero while its TEST error turns back up — the widening
    train/test gap IS overfitting. BART's training and test error both settle
    and stay flat: the posterior average over draws cannot memorise the way a
    longer boosting run does.

    All four curves are MSE against the *observed* (noisy) targets — the
    conventional learning-curve convention, and the only consistent reference on
    which "training error falls toward zero" is meaningful. (Measured against the
    noise-free truth, training and test would coincide and the overfitting gap
    would vanish.) Two regime choices, calibrated empirically, make the contrast
    land: a higher training-noise level (overfitting is invisible on the deck's
    clean noise=1.0 data) and shallow boosting trees (so boosting first reaches a
    competitive minimum rather than overfitting from round one).
    """
    from sklearn.ensemble import GradientBoostingRegressor

    SEED = 20260601
    NOISE = 5.0  # noisier than the deck's 1.0 so overfitting is demonstrable
    N_ROUNDS = 1000  # matches the BART snapshot count below (x-axes aligned)

    run_bart = ns["run_bart"]
    predict_at = ns["predict_at"]
    friedman = ns["friedman"]
    X_tr = ns["X_fried"]
    X_te = ns["X_fried_test"]
    # Re-draw both targets at higher noise (the deck's y_fried is noise=1.0); the
    # same rng stream gives independent train and test noise.
    data_rng = np.random.default_rng(SEED)
    y_tr = friedman(X_tr, noise=NOISE, rng=data_rng)
    y_te = friedman(X_te, noise=NOISE, rng=data_rng)

    # --- BART: cumulative running-mean prediction over the whole chain ------- #
    print("  fitting BART trajectory for overfitting figure ...", flush=True)
    fit = run_bart(
        X_tr,
        y_tr,
        m=100,
        n_iter=2000,
        burn_in=0,
        thin=2,
        rng=np.random.default_rng(SEED),
    )
    draws_tr = predict_at(fit, X_tr)  # (n_snap, n_train), original units
    draws_te = predict_at(fit, X_te)  # (n_snap, n_test)
    n_snap = draws_tr.shape[0]
    counts = np.arange(1, n_snap + 1)[:, None]
    run_tr = np.cumsum(draws_tr, axis=0) / counts  # posterior mean over draws 1..b
    run_te = np.cumsum(draws_te, axis=0) / counts
    bart_train = ((run_tr - y_tr) ** 2).mean(axis=1)
    bart_test = ((run_te - y_te) ** 2).mean(axis=1)
    bart_x = np.arange(1, n_snap + 1)

    # --- Gradient boosting: staged predictions over boosting rounds ---------- #
    print("  fitting gradient boosting for overfitting figure ...", flush=True)
    gbm = GradientBoostingRegressor(
        n_estimators=N_ROUNDS,
        max_depth=2,
        learning_rate=0.05,
        random_state=SEED,
    )
    gbm.fit(X_tr, y_tr)
    gbm_train = np.array([((p - y_tr) ** 2).mean() for p in gbm.staged_predict(X_tr)])
    gbm_test = np.array([((p - y_te) ** 2).mean() for p in gbm.staged_predict(X_te)])
    gbm_x = np.arange(1, N_ROUNDS + 1)

    # --- plot: blue = BART, orange = boosting; solid = test, dashed = train -- #
    BART, GBM = "#4c72b0", "#dd8452"
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(gbm_x, gbm_test, color=GBM, lw=2.6, label="boosting test")
    ax.plot(gbm_x, gbm_train, color=GBM, lw=1.8, ls="--", label="boosting train")
    ax.plot(bart_x, bart_test, color=BART, lw=2.6, label="BART test")
    ax.plot(bart_x, bart_train, color=BART, lw=1.8, ls="--", label="BART train")
    ax.set_xscale("log")
    ax.set_xlim(1, max(n_snap, N_ROUNDS))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("iterations  /  boosting rounds")
    ax.set_ylabel("mean squared error")
    ax.legend(frameon=False, ncol=2, loc="center right", fontsize=10)
    fig.tight_layout()
    save(fig, "overfitting.png")


def main():
    ns = load_notebook_namespace()
    apply_style()
    print("Rendering figures ...", flush=True)
    fig_gbm_demo(ns)
    fig_bart_hero(ns)
    fig_mcmc_demo(ns)
    fig_single_tree(ns)
    fig_sum_of_trees_panels(ns)
    if "--no-animations" not in sys.argv:
        anim_sum_of_trees_buildup(ns)
    fig_prior_tree_sizes(ns)
    fig_prior_sample_trees(ns)
    fig_bart_idea_grid(ns)
    fig_conjugate_leaf(ns)
    fig_coverage(ns)
    fig_choosing_m(ns)
    fig_restricted_r2(ns)
    fig_pdp(ns)
    fig_ice(ns)
    fig_sparse_compare(ns)
    fig_overfitting(ns)
    print(f"\nDone. Figures in {IMG.relative_to(ROOT)}/", flush=True)


if __name__ == "__main__":
    sys.exit(main())
