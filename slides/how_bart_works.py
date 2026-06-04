import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import chi2

    rng = np.random.default_rng(20260423)
    plt.rcParams["figure.dpi"] = 110
    return chi2, np, plt, rng


@app.cell(hide_code=True)
def _():
    # Shared colour scheme for every figure in this notebook. Define once here and
    # reference these names instead of hand-picking hex codes per plot.
    PALETTE = {
        "obs": "#4c4c4c",  # observed data points
        "truth": "#c44e52",  # the true function f(x)
        "posterior": "#4c72b0",  # BART posterior mean / draws / credible band
        "accent": "#dd8452",  # a secondary model (e.g. gradient boosting)
        "muted": "#b0b0b0",  # de-emphasised series
    }
    # Colourblind-friendly categorical cycle for grouped series (m-sweeps, etc.).
    CYCLE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
    return (PALETTE,)


@app.cell(hide_code=True)
def _(np):

    def _make_1d_toy():
        rng_toy = np.random.default_rng(20260423 + 1)
        n_toy = 80
        x = rng_toy.uniform(0, 1, size=n_toy)
        f_true = np.where(x < 0.4, -1.0, 0.0) + 1.5 * np.exp(-((x - 0.75) ** 2) / 0.01)
        y = f_true + rng_toy.normal(0, 0.15, size=n_toy)
        x_grid = np.linspace(0, 1, 200)
        f_grid_true = np.where(x_grid < 0.4, -1.0, 0.0) + 1.5 * np.exp(
            -((x_grid - 0.75) ** 2) / 0.01
        )
        return x[:, None], y, x_grid[:, None], f_grid_true

    X_toy, y_toy, X_toy_grid, f_toy_true = _make_1d_toy()
    return X_toy, X_toy_grid, f_toy_true, y_toy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How BART works

    BART sits between two tree-based models you probably already use.

    **Structurally**, BART is an additive model like **gradient-boosted trees**
    (the method behind libraries such as XGBoost, LightGBM, and CatBoost): the
    prediction is the sum of many small trees. **Behaviourally**, BART trees act
    more like a **random forest**: every tree is a weak learner contributing a
    small piece of the overall fit, and the ensemble reaches a consensus through
    their sum. Two important differences:

    * Gradient boosting fits trees sequentially with gradient descent. BART fits
      them *simultaneously*: every tree is refreshed by a Metropolis-Hastings
      move on every sweep through the data.
    * Random forests force weakness through bagging (row / column subsampling).
      BART forces weakness through a Bayesian **prior** that pulls every tree
      toward shallow shapes and small leaf values.

    The payoff for accepting the Bayesian sampler is a full *posterior
    distribution* over the regression function rather than a single point
    estimate. The model speaks in credible intervals.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Start with a familiar model

    To see what BART adds, start with a model you already know. The plot below
    is plain **gradient boosting** — `sklearn.ensemble.GradientBoostingRegressor`
    with default settings — on a small example: 80 noisy observations (grey
    points) scattered around a true function (red curve) that has a step near
    $x = 0.4$ and a sharp hump near $x = 0.75$. The summed trees track both
    features closely; it is a fast, accurate point predictor. But what it
    returns is a single curve and nothing else — no measure of uncertainty, and
    the fit looks equally committed where the data is dense and where it is
    sparse. Keep this picture in mind: next we fit the *same* data with BART,
    which keeps the sum-of-trees structure but adds a full posterior.
    """)
    return


@app.cell(hide_code=True)
def _(gbm_demo, mo, plt):
    def _gbm_demo_fig():
        if gbm_demo is None:
            return mo.md("*Compute cell hasn't finished yet.*")
        x = gbm_demo["x_train"]
        y = gbm_demo["y_train"]
        xg = gbm_demo["x_grid"]
        f_true = gbm_demo["f_true"]
        f_pred = gbm_demo["f_pred"]

        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.scatter(x, y, s=18, alpha=0.7, color="#333", label="observations")
        ax.plot(xg, f_true, color="#c44e52", lw=1.6, label=r"true $f(x)$")
        ax.plot(
            xg, f_pred, color="#55a868", lw=2.0, label="gradient-boosting prediction"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Gradient Boosting")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        return fig

    _gbm_demo_fig()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the same data, fit with BART. The scatter and the red true function
    $f(x)$ are as before; what is new is the blue **posterior-mean** prediction
    and the shaded **90% credible band** around it. Unlike the gradient-boosting
    fit above, BART does not commit to a single curve — the band widens where
    the data is sparser, which is exactly what a calibrated model should do.
    By the end of this notebook you will have written every line of the BART
    sampler in pure NumPy.
    """)
    return


@app.cell(hide_code=True)
def _(buildup, mo, plt):
    def _section0_headline():
        if buildup is None:
            return mo.md("*Compute cell hasn't finished yet.*")
        x = buildup["x_train"]
        y = buildup["y_train"]
        xg = buildup["x_grid"]
        f_true = buildup["f_true"]
        fit = buildup["fits"][100]

        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.scatter(x, y, s=18, alpha=0.7, color="#333", label="observations")
        ax.plot(xg, f_true, color="#c44e52", lw=1.6, label=r"true $f(x)$")
        ax.plot(xg, fit["f_mean"], color="#4c72b0", lw=2.0, label="BART posterior mean")
        ax.fill_between(
            xg,
            fit["f_lo"],
            fit["f_hi"],
            color="#4c72b0",
            alpha=0.18,
            label="90% credible band",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("BART")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        return fig

    _section0_headline()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian inference, briefly

    Bayesian inference treats a model parameter $\theta$ as having a
    *distribution* rather than a single value. Before seeing data we have a
    **prior** distribution $P(\theta)$; after observing data $D$ we have a
    **posterior** $P(\theta \mid D)$. The bridge is **Bayes' rule**:

    $$
    P(\theta \mid D) \;=\; \frac{P(D \mid \theta)\,P(\theta)}{P(D)}.
    $$

    | Term | Name | Meaning |
    |------|------|---------|
    | $P(\theta)$ | **Prior** | What we believe about $\theta$ before seeing data |
    | $P(D \mid \theta)$ | **Likelihood** | How probable the data is, given a particular $\theta$ |
    | $P(D)$ | **Evidence** (marginal likelihood) | Total probability of the data across all $\theta$ |
    | $P(\theta \mid D)$ | **Posterior** | Our updated belief after seeing the data |

    The denominator $P(D)$ is a constant in $\theta$, so the working form is

    $$
    P(\theta \mid D) \;\propto\; P(D \mid \theta)\,P(\theta).
    $$

    Posterior is proportional to likelihood times prior. With more data the
    likelihood dominates; with less data the prior has more influence.

    **A Gaussian example.** Suppose we want to estimate the true mean $\mu$
    of a noisy sensor. We start with a vague prior:
    $\mu \sim \mathcal{N}(10, 3^2)$. We then collect five readings whose
    sample mean is $\bar y = 12$, with per-reading noise standard deviation
    $2$. The posterior is also Normal:

    $$
    \mu \mid y \;\sim\; \mathcal{N}\!\left(
        \frac{\tau^{-2}\,\mu_0 \,+\, \sigma^{-2}\,n\,\bar y}{\tau^{-2} + \sigma^{-2}\,n},\;\;
        \bigl(\tau^{-2} + \sigma^{-2}\,n\bigr)^{-1}
    \right).
    $$

    The cleanest reading of this formula is **inverse-variance weighting**:
    each source of information contributes a precision (the reciprocal of
    its variance). The prior contributes $1/\tau^2 = 1/9$; each datum
    contributes $1/\sigma^2 = 1/4$, and five of them contribute $5/4 = 1.25$.
    The posterior precision is the sum, $\approx 1.36$, and the posterior
    mean is the precision-weighted average of $\mu_0 = 10$ and $\bar y = 12$.
    Data scientists already use this algebra in weighted least squares and
    Kalman filtering.

    **The prior is regularization.** In standard ML you prevent overfitting
    by adding a penalty term: L2 weight decay in a linear model, a
    `max_depth` cap on a tree, a learning-rate shrinkage on a boosting
    ensemble. A Bayesian **prior** is the same idea expressed as a
    probability distribution. The smaller $\tau$ is in the example above,
    the more aggressively the posterior is pulled toward $\mu_0$ regardless
    of the data. In BART, every tree stays a weak learner because the prior
    on its depth (favours shallow trees) and on its leaf values (favours
    small numbers) is deliberately tight. Not because of bagging, not
    because of learning-rate shrinkage.

    **The marginal likelihood.** The denominator we cancelled,

    $$
    P(D) \;=\; \int P(D \mid \theta)\,P(\theta)\,d\theta,
    $$

    looks like accounting. But when we want to compare two *model
    structures* (here: two candidate tree shapes), the marginal likelihood
    becomes the natural score: it says how well the structure as a whole
    explains the data, with the parameter $\theta$ integrated out. BART
    uses this to compare candidate tree structures *without ever sampling
    leaf values along the way*. This is the trick that makes the
    structural part of the sampler tractable.

    Every leaf in a BART tree is a Gaussian-Gaussian update like the one
    above. The structure of each tree is chosen by scoring candidates via
    the marginal likelihood, in the MCMC sampler we describe next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MCMC, briefly

    Bayesian inference gives us a recipe for the posterior, but in most
    useful models the posterior has no closed form. We sample from it
    instead.

    **Markov chains.** Markov chain Monte Carlo (MCMC) builds a sequence of
    samples $\theta^{(1)}, \theta^{(2)}, \ldots$ in which each draw depends
    only on the previous one. The chain is constructed so that, after enough
    steps, the samples behave like draws from the posterior. The samples are
    correlated (unlike i.i.d. draws), but you can still compute posterior
    means, quantiles, and credible intervals from them.

    **Metropolis-Hastings.** Given the current state $\theta$, propose a new
    state $\theta'$ from some proposal distribution $q(\theta' \mid \theta)$
    and accept it with probability

    $$
    \alpha \;=\; \min\!\left(1,\;\;
        \frac{P(\theta' \mid D)}{P(\theta \mid D)}
        \;\cdot\; \frac{q(\theta \mid \theta')}{q(\theta' \mid \theta)}\right).
    $$

    The leftmost factor, $P(\theta' \mid D) / P(\theta \mid D)$, is the
    **marginal likelihood ratio** from the previous section. When BART
    proposes adding a branch to a tree, it computes the marginal likelihood
    of the new (deeper) tree, compares it to the marginal likelihood of the
    old tree, and accepts the move more often when the new structure scores
    higher. The right-hand factor is a bookkeeping correction for any
    asymmetry in how moves are generated; if the proposal is symmetric (a
    Normal step centred at $\theta$, for example), it equals 1.

    The cell below makes this concrete with a random-walk Metropolis sampler
    on a Beta(3, 2) target. Forty lines of NumPy, one accept / reject
    decision per iteration.
    """)
    return


@app.cell
def _(np):
    def demo_rw_metropolis(log_target, x0, step=0.3, n_draws=4000, rng=None):
        rng = np.random.default_rng(rng)
        x, lp = x0, log_target(x0)
        draws = np.empty(n_draws)
        accepts = 0
        for t in range(n_draws):
            x_new = x + rng.normal(scale=step)
            lp_new = log_target(x_new)
            if np.log(rng.uniform()) < lp_new - lp:
                x, lp = x_new, lp_new
                accepts += 1
            draws[t] = x
        return draws, accepts / n_draws

    def demo_log_beta(x, a=3.0, b=2.0):
        # Unnormalised log-density of Beta(a, b). The normalising constant
        # cancels in the MH acceptance ratio, so we ignore it.
        if x <= 0.0 or x >= 1.0:
            return -np.inf
        return (a - 1.0) * np.log(x) + (b - 1.0) * np.log(1.0 - x)

    return demo_log_beta, demo_rw_metropolis


@app.cell(hide_code=True)
def _(demo_log_beta, demo_rw_metropolis, np, plt):
    _draws, _accept = demo_rw_metropolis(
        demo_log_beta, x0=0.5, step=0.3, n_draws=4000, rng=20260518
    )

    _xs = np.linspace(1e-4, 1 - 1e-4, 400)
    _truth = 12.0 * _xs**2 * (1.0 - _xs)  # Beta(3, 2) pdf

    _fig, _axes = plt.subplots(1, 2, figsize=(9, 2.6))
    _axes[0].plot(_draws, lw=0.4, color="#333")
    _axes[0].set_xlabel("iteration")
    _axes[0].set_ylabel("x")
    _axes[0].set_title(f"trace (accept rate = {_accept:.0%})")
    _axes[1].hist(
        _draws[1000:],
        bins=40,
        density=True,
        alpha=0.75,
        color="#4c72b0",
        edgecolor="white",
        label="MH draws (post burn-in)",
    )
    _axes[1].plot(_xs, _truth, color="#c44e52", lw=2, label="Beta(3, 2) target")
    _axes[1].set_xlabel("x")
    _axes[1].set_ylabel("density")
    _axes[1].set_title("draws vs target")
    _axes[1].legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **From one tree to fifty.** A BART ensemble has many trees, but the MH
    update above describes how to update *one* tree. How does the algorithm
    handle 50 trees at once?

    Mechanically: BART's outer loop is a cycle through the trees. To update
    tree $j$, hold the other trees fixed and subtract their combined
    predictions from the target,

    $$
    r_j \;=\; y \;-\; \sum_{k \ne j} f_k(X),
    $$

    so tree $j$ only has to fit this **partial residual**, the piece of $y$
    that the other trees did not already explain. Run one Metropolis-Hastings
    update against $r_j$, move on to tree $j+1$, and so on. One full sweep
    through the ensemble is one MCMC step. This is the same residual-fitting
    trick gradient boosting uses; BART just applies it symmetrically across
    all trees on every sweep, not sequentially.

    **Convergence diagnostics.** Two numbers tell you whether the chains
    have converged: $\hat R$ measures agreement between independently
    initialised chains (close to $1.0$ is good, above $1.01$ is a red flag),
    and **effective sample size (ESS)** measures how many independent draws
    the correlated MCMC samples are worth (hundreds-of-effective-samples is
    the practical target). The application notebooks 01 through 03 plot both
    for the `pymc-bart` fits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A single regression tree

    A **regression tree** is a piecewise-constant function. Each internal node
    holds a *splitting rule* of the form $x_v \le c$; rows answering "yes" go
    left, "no" go right. Each leaf $\ell$ stores a scalar value $\mu_\ell$.
    Prediction is just routing: walk down from the root following the splits
    until you reach a leaf, then return its $\mu$.

    The implementation below stores the tree as six parallel arrays indexed by
    node id $i = 0, \dots, N-1$:

    | Array | Meaning |
    |-------|---------|
    | `split_var[i]` | $-1$ if node $i$ is a leaf, otherwise the column index to split on |
    | `split_val[i]` | Cut threshold (used only when `split_var[i] \ge 0`) |
    | `left[i]`, `right[i]` | Child node indices, or $-1$ for leaves |
    | `parent[i]` | Parent index, or $-1$ for the root |
    | `mu[i]` | Leaf value $\mu_\ell$ (meaningful only when `split_var[i] == -1`) |

    A leaf is any node whose `split_var` equals $-1$.  This representation is
    tedious to read but cheap to copy, which the sampler does on every proposed
    move.

    The figure below shows what a single shallow tree looks like in 2D: each
    region of the input space is one leaf, painted with that leaf's $\mu$.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plt, predict):

    def _section1_partition():
        rng_demo = np.random.default_rng(0)
        n_demo = 250
        X_demo = rng_demo.uniform(size=(n_demo, 2))
        f_demo = np.where(
            X_demo[:, 0] < 0.5,
            np.where(X_demo[:, 1] < 0.6, -1.2, 0.4),
            np.where(X_demo[:, 1] < 0.3, 0.9, 1.6),
        )
        y_demo = f_demo + rng_demo.normal(0, 0.2, size=n_demo)

        t = Tree()
        t.split_var = [0, 1, 1, -1, -1, -1, -1]
        t.split_val = [0.5, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0]
        t.left = [1, 3, 5, -1, -1, -1, -1]
        t.right = [2, 4, 6, -1, -1, -1, -1]
        t.parent = [-1, 0, 0, 1, 1, 2, 2]
        t.mu = [0.0, 0.0, 0.0, -1.2, 0.4, 0.9, 1.6]

        grid = np.linspace(0, 1, 200)
        G0, G1 = np.meshgrid(grid, grid)
        X_grid = np.column_stack([G0.ravel(), G1.ravel()])
        Z = predict(t, X_grid).reshape(G0.shape)

        fig, ax = plt.subplots(figsize=(5.5, 4.4))
        im = ax.pcolormesh(
            G0, G1, Z, cmap="RdBu_r", vmin=-1.8, vmax=1.8, shading="auto"
        )
        ax.scatter(
            X_demo[:, 0],
            X_demo[:, 1],
            c=y_demo,
            cmap="RdBu_r",
            vmin=-1.8,
            vmax=1.8,
            edgecolor="black",
            linewidth=0.4,
            s=22,
        )
        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$x_1$")
        ax.set_title("Tree partition of 2D input space")
        fig.colorbar(im, ax=ax, label=r"leaf $\mu$")
        fig.tight_layout()

        tree_diagram = mo.mermaid(
            """
            graph TD
                A["x₀ ≤ 0.5"]
                B["x₁ ≤ 0.6"]
                C["x₁ ≤ 0.3"]
                D["μ = -1.2"]
                E["μ = 0.4"]
                F["μ = 0.9"]
                G["μ = 1.6"]
                A --> B
                A --> C
                B --> D
                B --> E
                C --> F
                C --> G
                classDef internal fill:#4c72b0,stroke:#333,stroke-width:1.5px,color:#fff
                classDef leaf fill:#c44e52,stroke:#333,stroke-width:1.5px,color:#fff
                class A,B,C internal
                class D,E,F,G leaf
            """
        )
        return mo.hstack([fig, tree_diagram], justify="space-around", widths="equal")

    _section1_partition()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sum of trees

    BART is **additive**: the model is a sum of $m$ regression trees,

    $$
    f(x) \;=\; \sum_{j=1}^{m} g(x;\, T_j, M_j).
    $$

    Why a sum? A single deep tree is high-variance and brittle — small data
    changes flip the whole topology. Bagging stabilises predictions but the
    Bayesian story is hard to write down. Boosting fits the residual one tree
    at a time but is greedy. BART splits the difference: many *shallow* trees,
    each fit to the residual after subtracting the others, with a prior that
    keeps any individual tree from swallowing the signal alone.

    Use the slider below to watch this buildup. The plot shows a single
    posterior draw — one realization of the $m$-tree ensemble — as the
    piecewise-constant function it actually is. At $m=1$ the fit is one coarse
    step function, visibly blocky. As $m$ grows the ensemble becomes a finer
    staircase; by $m = 100$–$200$ the steps are fine enough to track the smooth
    hump near $x = 0.75$.
    """)
    return


@app.cell(hide_code=True)
def _(X_toy, X_toy_grid, f_toy_true, y_toy):
    # Gradient-boosted-tree fit on the 1D toy DGP for the intro contrast. <50 ms.
    from sklearn.ensemble import GradientBoostingRegressor

    _gbm = GradientBoostingRegressor(random_state=20260423)
    _gbm.fit(X_toy, y_toy)
    gbm_demo = {
        "x_train": X_toy[:, 0],
        "y_train": y_toy,
        "x_grid": X_toy_grid[:, 0],
        "f_true": f_toy_true,
        "f_pred": _gbm.predict(X_toy_grid),
    }
    return (gbm_demo,)


@app.cell(hide_code=True)
def _(mo):
    buildup_m = mo.ui.slider(
        steps=[1, 5, 20, 100, 200],
        value=20,
        label="m (number of trees in ensemble)",
    )
    mo.md(
        rf"""
        **Interactive: choose m.** The draw below comes from a precomputed run; the
        slider just swaps in a stored posterior draw at the selected $m$.

        {buildup_m}
        """
    )
    return (buildup_m,)


@app.cell(hide_code=True)
def _(buildup, buildup_m, mo, plt):
    def _buildup_slider_plot():
        if buildup is None:
            return mo.md("*Compute cell hasn't finished yet.*")
        m = buildup_m.value
        fit = buildup["fits"][m]
        x = buildup["x_train"]
        y = buildup["y_train"]
        xg = buildup["x_grid"]
        f_true = buildup["f_true"]

        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.scatter(x, y, s=18, alpha=0.7, color="#333", label="observations")
        ax.plot(xg, f_true, color="#c44e52", lw=1.4, label=r"true $f(x)$")
        ax.plot(
            xg,
            fit["f_draw"],
            color="#4c72b0",
            lw=1.8,
            drawstyle="steps-mid",
            label=f"one BART draw (m={m})",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"One posterior draw at m={m}")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        return fig

    _buildup_slider_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prior on tree shape

    Now that we have an ensemble of trees, what stops one tree from absorbing
    the whole signal? The **prior**. Following Chipman, George & McCulloch (1998),
    BART places a structural prior that makes deep splits exponentially less
    likely: each internal-or-leaf decision at depth $d$ is a Bernoulli with

    $$
    P(\text{non-terminal at depth }d) \;=\; \alpha (1 + d)^{-\beta}.
    $$

    Higher $\alpha$ means trees grow more eagerly at depth 0; higher $\beta$
    punishes deeper splits harder. The CGM98 default $(\alpha, \beta) = (0.95, 2.0)$
    concentrates prior mass on trees with 2-4 leaves. The Metropolis-Hastings
    sampler we describe shortly will sometimes accept deeper trees if the data
    demand it, but the prior is constantly pulling each tree back toward
    simplicity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    tree_alpha = mo.ui.slider(
        0.1, 0.99, step=0.01, value=0.95, label=r"$\alpha$ (base probability)"
    )
    tree_beta = mo.ui.slider(
        0.0, 10.0, step=0.1, value=2.0, label=r"$\beta$ (depth penalty)"
    )
    mo.md(
        rf"""
        **Prior on tree depth.** The non-terminal probability at depth $d$ is
        $\alpha(1+d)^{{-\beta}}$. Move the sliders to see how the prior concentrates on
        tree sizes.

        {tree_alpha} {tree_beta}
        """
    )
    return tree_alpha, tree_beta


@app.cell(hide_code=True)
def _(np, plt, tree_alpha, tree_beta):
    def terminal_size_dist(alpha, beta, max_depth=8, n_samples=20000, seed=0):
        rng_local = np.random.default_rng(seed)
        sizes = np.empty(n_samples, dtype=int)
        for s in range(n_samples):
            stack = [0]  # list of depths of open (not-yet-resolved) nodes
            leaves = 0
            while stack:
                d = stack.pop()
                p = alpha * (1 + d) ** (-beta) if d < max_depth else 0.0
                if rng_local.random() < p:
                    stack.extend([d + 1, d + 1])
                else:
                    leaves += 1
            sizes[s] = leaves
        return sizes

    _sizes = terminal_size_dist(tree_alpha.value, tree_beta.value)
    _bins = np.arange(1, min(_sizes.max(), 15) + 2) - 0.5
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.hist(_sizes, bins=_bins, density=True, color="#4c72b0", edgecolor="white")
    _ax.set_xlabel("number of terminal nodes")
    _ax.set_ylabel("prior probability")
    _ax.set_title(
        f"Prior over tree size  (α={tree_alpha.value:.2f}, β={tree_beta.value:.1f})"
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, np, tree_alpha, tree_beta):

    def _sample_trees(alpha, beta, n_trees=6, max_depth=6, seed=20260423):
        rng_local = np.random.default_rng(seed)
        trees = []
        for _ in range(n_trees):
            t = Tree()
            stack = [(0, 0)]
            while stack:
                node, d = stack.pop()
                p = alpha * (1 + d) ** (-beta) if d < max_depth else 0.0
                if rng_local.random() < p:
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
            trees.append(t)
        return trees

    def _tree_to_mermaid(tree, title=None):
        lines = ["graph TD"]
        if title is not None:
            lines = ["---", f"title: {title}", "---"] + lines
        internals = []
        leaves = []
        for i in range(len(tree.split_var)):
            if tree.is_leaf(i):
                leaves.append(i)
                lines.append(f'  N{i}["·"]')
            else:
                internals.append(i)
                lines.append(f'  N{i}["x{tree.split_var[i]}"]')
            if tree.parent[i] >= 0:
                lines.append(f"  N{tree.parent[i]} --> N{i}")
        if internals:
            lines.append("  classDef internal fill:#4c72b0,stroke:#333,color:#fff")
            lines.append(f"  class {','.join(f'N{i}' for i in internals)} internal")
        if leaves:
            lines.append("  classDef leaf fill:#c44e52,stroke:#333,color:#fff")
            lines.append(f"  class {','.join(f'N{i}' for i in leaves)} leaf")
        return mo.mermaid("\n".join(lines))

    def _plot_sample_trees():
        trees = _sample_trees(tree_alpha.value, tree_beta.value)
        diagrams = [
            _tree_to_mermaid(t, title=f"{len(t.leaves())} leaves") for t in trees
        ]
        return mo.vstack(
            [
                mo.md(
                    f"**Six trees drawn from the prior at α={tree_alpha.value:.2f}, "
                    f"β={tree_beta.value:.1f}**"
                ),
                mo.hstack(diagrams[:3], justify="space-around", widths="equal"),
                mo.hstack(diagrams[3:], justify="space-around", widths="equal"),
            ]
        )

    _plot_sample_trees()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conjugate leaf updates

    Each tree is two things at once: a *structure* (which splits, in which order)
    and a *labelling* (what $\mu$ value lives at each leaf). BART separates them.
    Conditional on the structure, the leaf values are easy: $\mu_\ell$ has a
    Normal prior, the residuals inside leaf $\ell$ are Normal, and conjugacy
    gives a closed-form Normal posterior.

    Concretely, write $n_\ell$ for the count of training rows landing in
    leaf $\ell$ and $\bar r_\ell$ for their mean residual. The prior is
    $\mu_\ell \sim \mathcal{N}(0,\, \sigma_\mu^2)$ and the leaf-conditional
    likelihood is $\mathcal{N}(\mu_\ell,\, \sigma^2/n_\ell)$. The posterior is

    $$
    \mu_\ell \mid r \;\sim\; \mathcal{N}\!\left(
        \frac{n_\ell \sigma_\mu^2}{\sigma^2 + n_\ell \sigma_\mu^2}\, \bar r_\ell,\;
        \frac{\sigma^2 \sigma_\mu^2}{\sigma^2 + n_\ell \sigma_\mu^2}
    \right).
    $$

    The posterior mean is a *shrinkage estimator*: pull each leaf's sample mean
    toward zero by a factor that depends on how much data the leaf holds. Tiny
    leaves get shrunk hard; big leaves keep close to their sample mean.

    The closed form is also what lets the Metropolis-Hastings sampler in the
    next section integrate $\mu$ out entirely when scoring tree structures.
    That move is the trick that keeps MH from having to do reversible jump
    over leaf dimensions.

    The conjugate update is implemented in `draw_leaf_values` below
    (hidden by default — expand to read). `run_bart` calls it once per
    tree per sweep, after the structure move accepts.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):

    def _section4_conjugate():
        from scipy.stats import norm as _norm

        rng_demo = np.random.default_rng(7)
        n_l = 14
        sigma = 0.3
        sigma_mu = 0.25
        true_mu = 0.5
        residuals = true_mu + rng_demo.normal(0, sigma, size=n_l)

        rbar = residuals.mean()
        post_var = sigma**2 * sigma_mu**2 / (sigma**2 + n_l * sigma_mu**2)
        post_mean = (n_l * sigma_mu**2) / (sigma**2 + n_l * sigma_mu**2) * rbar
        post_sd = np.sqrt(post_var)

        grid = np.linspace(-1.0, 1.5, 600)
        prior_pdf = _norm.pdf(grid, 0, sigma_mu)
        lik_pdf = _norm.pdf(grid, rbar, sigma / np.sqrt(n_l))
        post_pdf = _norm.pdf(grid, post_mean, post_sd)

        fig, (ax_r, ax_d) = plt.subplots(1, 2, figsize=(11, 3.8))

        ax_r.hist(residuals, bins=8, color="#666", alpha=0.5, edgecolor="white")
        ax_r.axvline(rbar, color="#c44e52", lw=1.4, label=f"mean = {rbar:.2f}")
        ax_r.set_xlabel("residual r")
        ax_r.set_ylabel("count")
        ax_r.set_title(f"Residuals in one leaf  (n = {n_l})")
        ax_r.legend(frameon=False)

        ax_d.plot(
            grid,
            prior_pdf,
            lw=1.6,
            color="#8c8c8c",
            label=rf"prior $\mathcal{{N}}(0, {sigma_mu:.2f}^2)$",
        )
        ax_d.plot(
            grid,
            lik_pdf,
            lw=1.6,
            color="#4c72b0",
            label=rf"likelihood $\propto \mathcal{{N}}({rbar:.2f}, \sigma^2/n)$",
        )
        ax_d.plot(
            grid,
            post_pdf,
            lw=2.2,
            color="#c44e52",
            label=rf"posterior $\mathcal{{N}}({post_mean:.2f}, {post_sd:.2f}^2)$",
        )
        ax_d.axvline(0, color="#8c8c8c", lw=0.6, ls=":")
        ax_d.axvline(rbar, color="#4c72b0", lw=0.6, ls=":")
        ax_d.axvline(post_mean, color="#c44e52", lw=0.6, ls=":")
        ax_d.set_xlabel(r"$\mu_\ell$")
        ax_d.set_ylabel("density")
        ax_d.set_title("Prior + likelihood = posterior (Normal-Normal conjugacy)")
        ax_d.legend(frameon=False, fontsize=9)

        fig.tight_layout()
        return fig

    _section4_conjugate()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metropolis-Hastings moves on tree structure

    Trees live in a discrete combinatorial space — you can't take gradients
    over them. The classic CGM98 sampler proposes one of two **structural
    moves** at each step:

    * **Grow.** Pick a leaf at random and turn it into an internal node by
      drawing a uniform split variable and cut from the data in that leaf.
    * **Prune.** Pick a *singly-internal* node (one whose two children are
      both leaves) at random and collapse it back to a leaf.

    Because the leaf values are conjugate-Gaussian (see the previous
    section), we never have to track $\mu$ during the structural move. The
    likelihood factor in the MH acceptance ratio uses the **marginal
    likelihood**, with $\mu$ integrated out leaf-by-leaf:

    $$
    p(r \mid T,\, \sigma^2)
    \;=\; \prod_{\ell}\, \sqrt{\frac{\sigma^2}{\sigma^2 + n_\ell \sigma_\mu^2}}\,
    \exp\!\left(\frac{\sigma_\mu^2\, s_\ell^2}{2\sigma^2(\sigma^2 + n_\ell \sigma_\mu^2)}\right),
    $$

    with $s_\ell = \sum_{i \in \ell} r_i$. The Hastings ratio combines this with
    the **prior shape ratio** (depth-$d$ split probability) and the **move
    ratio** (forward and reverse proposal probabilities). The cartoon below
    sketches one accept/reject decision; the implementation lives in
    `grow_proposal` and `prune_proposal`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):

    def _section5_cartoon():
        diagram = mo.mermaid(
            """
            graph LR
                subgraph T ["current tree T"]
                    direction TB
                    A["x₀ ≤ c"] --> B["μ₁"]
                    A --> C["μ₂"]
                end
                subgraph Tp ["proposed tree T′ (grow μ₂)"]
                    direction TB
                    D["x₀ ≤ c"] --> E["μ₁"]
                    D --> F["x₁ ≤ c′"]
                    F --> G["μ₂ᴸ"]
                    F --> H["μ₂ᴿ"]
                end
                T -.->|propose grow| Tp
                classDef internal fill:#4c72b0,stroke:#333,color:#fff
                classDef leaf fill:#c44e52,stroke:#333,color:#fff
                class A,D,F internal
                class B,C,E,G,H leaf
            """
        )
        caption = mo.md(
            r"""
            $$
            \log A \;=\; \underbrace{\log\frac{p(r\mid T')}{p(r\mid T)}}_{\text{likelihood ratio}}
            \;+\; \underbrace{\log\frac{\pi(T')}{\pi(T)}}_{\text{prior shape ratio}}
            \;+\; \underbrace{\log\frac{q(T\mid T')}{q(T'\mid T)}}_{\text{move ratio}}
            $$

            Accept $T'$ with probability $\min(1,\, e^{\log A})$.
            Prune is the time-reversal: collapse a singly-internal node back into a leaf.
            """
        )
        return mo.vstack([diagram, caption])

    _section5_cartoon()
    return


@app.class_definition
class Tree:
    """Binary regression tree. Starts as a single-leaf root."""

    __slots__ = ("split_var", "split_val", "left", "right", "parent", "mu")

    def __init__(self):
        self.split_var = [-1]
        self.split_val = [0.0]
        self.left = [-1]
        self.right = [-1]
        self.parent = [-1]
        self.mu = [0.0]

    def copy(self):
        # bypass __init__ to clone an existing tree's node arrays in place
        t = Tree.__new__(Tree)
        t.split_var = list(self.split_var)
        t.split_val = list(self.split_val)
        t.left = list(self.left)
        t.right = list(self.right)
        t.parent = list(self.parent)
        t.mu = list(self.mu)
        return t

    def is_leaf(self, i):
        return self.split_var[i] < 0

    def leaves(self):
        return [i for i in range(len(self.split_var)) if self.split_var[i] < 0]

    def internal_nodes(self):
        return [i for i in range(len(self.split_var)) if self.split_var[i] >= 0]

    def depth_of(self, i):
        d = 0
        while self.parent[i] >= 0:
            i = self.parent[i]
            d += 1
        return d

    def singly_internal(self):
        """Internal nodes whose *both* children are leaves (prune candidates)."""
        out = []
        for i in self.internal_nodes():
            if self.is_leaf(self.left[i]) and self.is_leaf(self.right[i]):
                out.append(i)
        return out


@app.cell
def _(np):
    def assign_leaves(tree, X):
        """Return (n,) array giving the leaf index each row of X lands in."""
        n = X.shape[0]
        leaf_of = np.full(n, -1, dtype=np.int64)

        def descend(node, rows):
            if tree.is_leaf(node):
                leaf_of[rows] = node
                return
            col = tree.split_var[node]
            mask = X[rows, col] <= tree.split_val[node]
            descend(tree.left[node], rows[mask])
            descend(tree.right[node], rows[~mask])

        descend(0, np.arange(n))
        return leaf_of

    def predict(tree, X):
        """Return (n,) tree predictions: leaf μ values at each row."""
        leaf_of = assign_leaves(tree, X)
        return np.asarray(tree.mu, dtype=float)[leaf_of]

    return assign_leaves, predict


@app.cell
def _(assign_leaves, np):
    def log_marginal_leaf(n_l, s_l, sigma2, sigma_mu2):
        """Contribution of one leaf to log p(r | T, σ)."""
        denom = sigma2 + n_l * sigma_mu2
        return 0.5 * np.log(sigma2 / denom) + 0.5 * sigma_mu2 * s_l**2 / (
            sigma2 * denom
        )

    def log_marginal_tree(tree, X, r, sigma2, sigma_mu2):
        """Log marginal likelihood of residuals r given tree structure."""
        leaf_of = assign_leaves(tree, X)
        total = 0.0
        for lf in tree.leaves():
            mask = leaf_of == lf
            n_l = int(mask.sum())
            if n_l == 0:
                continue  # empty leaf adds zero
            s_l = r[mask].sum()
            total += log_marginal_leaf(n_l, s_l, sigma2, sigma_mu2)
        return total

    return (log_marginal_tree,)


@app.cell
def _(np):
    def log_prior_tree(tree, alpha, beta):
        """log p(T) under the CGM98 prior, up to constants from split rules."""
        logp = 0.0
        for i in range(len(tree.split_var)):
            d = tree.depth_of(i)
            p_split = alpha * (1.0 + d) ** (-beta)
            if tree.is_leaf(i):
                logp += np.log(1.0 - p_split)
            else:
                logp += np.log(p_split)
        return logp

    return


@app.cell
def _(np):
    def splittable_cuts(X_leaf, col):
        """Unique midpoints of sorted values in column `col` for rows in a leaf."""
        vals = np.unique(X_leaf[:, col])
        if vals.size < 2:
            return np.empty(0)
        return 0.5 * (vals[:-1] + vals[1:])

    def descendants(tree, node, leaves_only=False):
        """Indices of `node` and everything below it."""
        out = [node]
        stack = [node]
        while stack:
            i = stack.pop()
            if not tree.is_leaf(i):
                stack.extend([tree.left[i], tree.right[i]])
                out.extend([tree.left[i], tree.right[i]])
        if leaves_only:
            return [i for i in out if tree.is_leaf(i)]
        return out

    def compact(tree):
        """Rebuild the tree dropping unreachable nodes, renumbering indices."""
        # bypass __init__ — we're rebuilding the node arrays from scratch
        new = Tree.__new__(Tree)
        new.split_var = []
        new.split_val = []
        new.left = []
        new.right = []
        new.parent = []
        new.mu = []
        remap = {}
        order = []
        stack = [0]
        while stack:
            i = stack.pop()
            remap[i] = len(order)
            order.append(i)
            if not tree.is_leaf(i):
                stack.append(tree.right[i])
                stack.append(tree.left[i])
        for old in order:
            new.split_var.append(tree.split_var[old])
            new.split_val.append(tree.split_val[old])
            new.mu.append(tree.mu[old])
            new.left.append(remap[tree.left[old]] if tree.left[old] >= 0 else -1)
            new.right.append(remap[tree.right[old]] if tree.right[old] >= 0 else -1)
            new.parent.append(remap[tree.parent[old]] if tree.parent[old] >= 0 else -1)
        return new

    return compact, splittable_cuts


@app.cell
def _(log_marginal_tree, np, splittable_cuts):
    def grow_proposal(tree, X, leaf_of, r, rng, alpha, beta, sigma2, sigma_mu2):
        """Propose a grow. Returns (new_tree, log_acceptance) or (None, -inf)."""
        leaves = tree.leaves()
        lf = leaves[rng.integers(len(leaves))]
        mask = leaf_of == lf
        if int(mask.sum()) < 2:
            return None, -np.inf
        X_leaf = X[mask]

        p_cols = X.shape[1]
        chosen_col = None
        for c in rng.permutation(p_cols):
            cuts = splittable_cuts(X_leaf, c)
            if cuts.size:
                chosen_col = int(c)
                chosen_cut = float(rng.choice(cuts))
                break
        if chosen_col is None:
            return None, -np.inf

        t_new = tree.copy()
        d = tree.depth_of(lf)
        new_left = len(t_new.split_var)
        new_right = new_left + 1
        t_new.split_var[lf] = chosen_col
        t_new.split_val[lf] = chosen_cut
        t_new.left[lf] = new_left
        t_new.right[lf] = new_right
        t_new.split_var += [-1, -1]
        t_new.split_val += [0.0, 0.0]
        t_new.left += [-1, -1]
        t_new.right += [-1, -1]
        t_new.parent += [lf, lf]
        t_new.mu += [0.0, 0.0]

        ll_new = log_marginal_tree(t_new, X, r, sigma2, sigma_mu2)
        ll_old = log_marginal_tree(tree, X, r, sigma2, sigma_mu2)

        p_split_d = alpha * (1.0 + d) ** (-beta)
        p_split_d1 = alpha * (2.0 + d) ** (-beta)
        log_shape_ratio = (
            np.log(p_split_d) + 2.0 * np.log(1.0 - p_split_d1) - np.log(1.0 - p_split_d)
        )

        b = len(leaves)
        P_grow_fwd = 1.0 if len(tree.internal_nodes()) == 0 else 0.5
        P_prune_bwd = 0.5
        w_new = len(t_new.singly_internal())
        log_move_ratio = np.log(P_prune_bwd / P_grow_fwd) + np.log(b / w_new)

        log_accept = (ll_new - ll_old) + log_shape_ratio + log_move_ratio
        return t_new, log_accept

    return (grow_proposal,)


@app.cell
def _(compact, log_marginal_tree, np):
    def prune_proposal(tree, X, r, rng, alpha, beta, sigma2, sigma_mu2):
        """Propose a prune. Returns (new_tree, log_acceptance) or (None, -inf)."""
        candidates = tree.singly_internal()
        if not candidates:
            return None, -np.inf
        w = len(candidates)
        node = candidates[rng.integers(w)]
        d = tree.depth_of(node)

        t_new = tree.copy()
        child_l, child_r = t_new.left[node], t_new.right[node]
        t_new.split_var[node] = -1
        t_new.split_val[node] = 0.0
        t_new.left[node] = -1
        t_new.right[node] = -1
        t_new.mu[node] = 0.0
        t_new.parent[child_l] = -2
        t_new.parent[child_r] = -2
        t_new = compact(t_new)

        ll_new = log_marginal_tree(t_new, X, r, sigma2, sigma_mu2)
        ll_old = log_marginal_tree(tree, X, r, sigma2, sigma_mu2)

        p_split_d = alpha * (1.0 + d) ** (-beta)
        p_split_d1 = alpha * (2.0 + d) ** (-beta)
        log_shape_ratio = (
            np.log(1.0 - p_split_d) - np.log(p_split_d) - 2.0 * np.log(1.0 - p_split_d1)
        )

        b_new = len(t_new.leaves())
        P_grow_bwd = 1.0 if b_new == 1 else 0.5
        P_prune_fwd = 0.5
        log_move_ratio = np.log(P_grow_bwd / P_prune_fwd) + np.log(w / b_new)

        log_accept = (ll_new - ll_old) + log_shape_ratio + log_move_ratio
        return t_new, log_accept

    return (prune_proposal,)


@app.cell
def _(assign_leaves, np):
    def draw_leaf_values(tree, X, r, rng, sigma2, sigma_mu2):
        """Sample μ_ℓ for every leaf. Returns the updated tree (mutated in place)."""
        leaf_of = assign_leaves(tree, X)
        for lf in tree.leaves():
            mask = leaf_of == lf
            n_l = int(mask.sum())
            if n_l == 0:
                tree.mu[lf] = float(rng.normal(0.0, np.sqrt(sigma_mu2)))
                continue
            s_l = r[mask].sum()
            post_var = sigma2 * sigma_mu2 / (sigma2 + n_l * sigma_mu2)
            post_mean = post_var * s_l / sigma2
            tree.mu[lf] = float(rng.normal(post_mean, np.sqrt(post_var)))
        return tree

    return (draw_leaf_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The σ update

    Once trees and leaf values are drawn, $\sigma^2$ follows a standard
    inverse-$\chi^2$ conditional. Following CGM98 we put a scaled inverse-$\chi^2$
    prior on $\sigma^2$ whose scale $\lambda$ is calibrated by OLS: fit a linear
    model first to get $\hat\sigma$, then choose $\lambda$ so that
    $P(\sigma < \hat\sigma) = q$ (default $q = 0.9$). This makes the prior
    informative about scale but agnostic about *more* than that — BART will
    happily pull $\sigma$ down once the trees explain more of the variance.
    """)
    return


@app.cell
def _(chi2, np):
    def draw_sigma2(residuals, nu, lam, rng):
        """Inverse-χ² conditional for σ²."""
        n = residuals.size
        shape = nu + n
        scale = nu * lam + np.sum(residuals**2)
        chi = rng.chisquare(shape)
        return scale / chi

    def calibrate_sigma_prior(y, X, nu=3.0, q=0.9):
        """Compute λ so that P(σ < σ̂) = q under the prior σ² ~ νλ/χ²_ν.

        σ̂ is the residual std from an OLS fit of y on X (with intercept); if OLS
        is rank-deficient we fall back to sd(y).
        """
        try:
            X_aug = np.column_stack([np.ones(X.shape[0]), X])
            beta_hat, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            resid = y - X_aug @ beta_hat
            sigma_hat = float(np.sqrt(np.mean(resid**2)))
        except np.linalg.LinAlgError:
            sigma_hat = float(y.std())
        # P(σ² < σ̂²) = q  ⇔  νλ/σ̂² = χ²_ν quantile at (1-q)
        lam = sigma_hat**2 * chi2.ppf(1.0 - q, df=nu) / nu
        return sigma_hat, lam

    return calibrate_sigma_prior, draw_sigma2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Putting it together: the BART sampler

    One full sweep through the ensemble looks like this:

    1. **Backfitting.** For each tree $j$ in turn, compute the partial
       residual $R_j = y - \sum_{i \ne j} g(x;\, T_i, M_i)$. Tree $j$ now has
       to fit whatever the other trees did not already explain.
    2. **Structure step.** A Metropolis-Hastings grow / prune update on the
       shape $T_j$ given $R_j$ (the MH section).
    3. **Leaf step.** Sample the leaf values $M_j$ from their conjugate
       posterior given the new structure (the conjugate-leaf-updates
       section).
    4. **Noise step.** Sample $\sigma^2$ from its inverse-$\chi^2$
       conditional given the current residuals (the $\sigma$ update
       section).

    The formal name for this cycle-through-blocks pattern is **Gibbs
    sampling**: each block has its own update rule and one sweep touches
    every parameter exactly once. The implementation is `run_bart`. The
    function body is hidden by default but you can expand the cell to see
    the loop. The figure below is a calibration check on the Friedman DGP,
    where the true $f$ and the true $\sigma = 1$ are both known.

    A few `run_bart` hyperparameters worth naming explicitly (they appear
    in the signature but not in the algorithm text above):

    * **`k`** — leaf-value shrinkage. Each leaf's prior SD is set so that
      $\pm k$ standard deviations of the prior cover roughly the
      observed range of $y$ around $\bar y$. Larger $k$ → tighter
      per-leaf prior → smaller per-tree contribution.
    * **`nu`, `q`** — the inverse-$\chi^2$ prior on $\sigma^2$ is
      calibrated by setting the prior $q$-th quantile equal to a rough
      preliminary estimate of $\sigma$ (e.g. residual SD of an OLS fit).
      `nu` controls how informative that prior is.
    * **`thin`** — post-hoc subsampling of saved draws. Reduces memory
      / storage but does not change the chain itself.
    """)
    return


@app.cell
def _(
    assign_leaves,
    calibrate_sigma_prior,
    draw_leaf_values,
    draw_sigma2,
    grow_proposal,
    np,
    predict,
    prune_proposal,
):
    def run_bart(
        X,
        y,
        m=200,
        n_iter=1000,
        burn_in=500,
        alpha=0.95,
        beta=2.0,
        k=2.0,
        nu=3.0,
        q=0.9,
        sigma2_init=None,
        rng=None,
        verbose=False,
        thin=1,
    ):
        """Fit BART by Bayesian backfitting MCMC.

        Returns a dict containing posterior draws of f on the training data,
        σ, split counts per variable, and a list of tree-ensemble snapshots
        (one per kept iteration) suitable for out-of-sample prediction.
        """
        if rng is None:
            rng = np.random.default_rng()

        y_min, y_max = float(y.min()), float(y.max())
        y_range = y_max - y_min
        y_scaled = (y - y_min) / y_range - 0.5

        sigma_hat, lam = calibrate_sigma_prior(y_scaled, X, nu=nu, q=q)
        sigma2 = sigma_hat**2 if sigma2_init is None else float(sigma2_init)

        sigma_mu = 0.5 / (k * np.sqrt(m))
        sigma_mu2 = sigma_mu**2

        n, p = X.shape
        trees = [Tree() for _ in range(m)]
        tree_preds = np.zeros((m, n))

        n_kept = (n_iter - burn_in + thin - 1) // thin
        f_draws = np.zeros((n_kept, n))
        sigma2_draws = np.zeros(n_kept)
        splits = np.zeros((n_kept, p), dtype=np.int64)
        tree_snapshots = []  # list of list-of-Tree, length n_kept

        accept_stats = {"grow": [0, 0], "prune": [0, 0]}
        kept = 0

        for it in range(n_iter):
            for j in range(m):
                Rj = y_scaled - tree_preds.sum(axis=0) + tree_preds[j]

                if not trees[j].internal_nodes():
                    move = "grow"
                else:
                    move = "grow" if rng.random() < 0.5 else "prune"

                if move == "grow":
                    leaf_of = assign_leaves(trees[j], X)
                    t_new, logA = grow_proposal(
                        trees[j],
                        X,
                        leaf_of,
                        Rj,
                        rng,
                        alpha,
                        beta,
                        sigma2,
                        sigma_mu2,
                    )
                    accept_stats["grow"][1] += 1
                    if t_new is not None and np.log(rng.random()) < logA:
                        trees[j] = t_new
                        accept_stats["grow"][0] += 1
                else:
                    t_new, logA = prune_proposal(
                        trees[j], X, Rj, rng, alpha, beta, sigma2, sigma_mu2
                    )
                    accept_stats["prune"][1] += 1
                    if t_new is not None and np.log(rng.random()) < logA:
                        trees[j] = t_new
                        accept_stats["prune"][0] += 1

                trees[j] = draw_leaf_values(trees[j], X, Rj, rng, sigma2, sigma_mu2)
                tree_preds[j] = predict(trees[j], X)

            f_hat = tree_preds.sum(axis=0)
            resid = y_scaled - f_hat
            sigma2 = draw_sigma2(resid, nu, lam, rng)

            if it >= burn_in and ((it - burn_in) % thin == 0):
                f_draws[kept] = f_hat
                sigma2_draws[kept] = sigma2
                for t in trees:
                    for v in t.split_var:
                        if v >= 0:
                            splits[kept, v] += 1
                tree_snapshots.append([t.copy() for t in trees])
                kept += 1

            if verbose and (it + 1) % max(1, n_iter // 10) == 0:
                print(
                    f"  iter {it + 1}/{n_iter}  σ={np.sqrt(sigma2) * y_range:.3f}  "
                    f"leaves={np.mean([len(t.leaves()) for t in trees]):.1f}"
                )

        f_draws = f_draws[:kept]
        sigma2_draws = sigma2_draws[:kept]
        splits = splits[:kept]

        return {
            "f_draws_scaled": f_draws,
            "sigma2_draws_scaled": sigma2_draws,
            "sigma_draws": np.sqrt(sigma2_draws) * y_range,
            "f_mean": f_draws.mean(axis=0) * y_range + (y_min + 0.5 * y_range),
            "f_lo": np.quantile(f_draws, 0.05, axis=0) * y_range
            + (y_min + 0.5 * y_range),
            "f_hi": np.quantile(f_draws, 0.95, axis=0) * y_range
            + (y_min + 0.5 * y_range),
            "splits": splits,
            "tree_snapshots": tree_snapshots,
            "y_min": y_min,
            "y_range": y_range,
            "accept": accept_stats,
        }

    return (run_bart,)


@app.cell(hide_code=True)
def _(np, predict):
    def predict_at(fit, X_new):
        """Evaluate stored tree ensembles at X_new; return (draws, n_new) array."""
        snapshots = fit["tree_snapshots"]
        y_min, y_range = fit["y_min"], fit["y_range"]
        n_draws = len(snapshots)
        n_new = X_new.shape[0]
        draws = np.zeros((n_draws, n_new))
        for d, trees in enumerate(snapshots):
            f_scaled = np.zeros(n_new)
            for t in trees:
                f_scaled += predict(t, X_new)
            draws[d] = f_scaled * y_range + (y_min + 0.5 * y_range)
        return draws

    return (predict_at,)


@app.cell(hide_code=True)
def _(fit_fried, mo, np, plt):
    def _coverage_plot():
        if fit_fried is None:
            return mo.md("*Compute cell for friedman_m50 hasn't finished yet.*")

        f_true_in = fit_fried["f_true_in"]
        f_mean_in = fit_fried["f_mean"]
        f_lo_in = fit_fried["f_lo"]
        f_hi_in = fit_fried["f_hi"]
        cov_in = fit_fried["coverage_in"]

        f_true_out = fit_fried["f_test_true"] if "f_test_true" in fit_fried else None
        f_mean_out = fit_fried["f_test_mean"]
        f_lo_out = fit_fried["f_test_lo"]
        f_hi_out = fit_fried["f_test_hi"]
        cov_out = fit_fried["coverage_oos"]

        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, xt, m_, lo_, hi_, cov, tag in [
            (a, f_true_in, f_mean_in, f_lo_in, f_hi_in, cov_in, "in-sample"),
            (b, f_true_out, f_mean_out, f_lo_out, f_hi_out, cov_out, "out-of-sample"),
        ]:
            order = np.argsort(xt)
            ax.errorbar(
                xt[order],
                m_[order],
                yerr=[(m_ - lo_)[order], (hi_ - m_)[order]],
                fmt="o",
                ms=3,
                ecolor="#4c72b0",
                color="#333",
                alpha=0.55,
                elinewidth=0.7,
            )
            lim = (xt.min() - 1, xt.max() + 1)
            ax.plot(lim, lim, "--", color="C3", lw=1)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel(r"true $f(x)$")
            ax.set_ylabel(r"posterior predictive mean $\hat f(x)$ with 90% PI")
            ax.set_title(f"{tag} - 90% coverage: {cov:.0%}")

        fig.suptitle(
            "Friedman sanity check: BART is calibrated when the truth is known", y=1.02
        )
        fig.tight_layout()
        return fig

    _coverage_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing $m$

    BART has one knob whose right value is genuinely problem-dependent:
    $m$, the number of trees in the ensemble.  Quiroga *et al.* (2022)
    recommend $m \approx 50$ for exploration and variable-importance work
    and $m \approx 200$ for the final inference.  The intuition:

    * Each tree absorbs a *small* piece of $f(x)$.  More trees → finer
      decomposition → smoother fit, but with diminishing returns.
    * Linero & Yang (2018) show that as $m \to \infty$ the BART prior
      converges to a (nowhere-differentiable) Gaussian Process, which
      explains why "more trees" keeps helping for a while.

    The interactive plot below refits BART at several values of $m$ on the
    1D toy data from earlier.  The left panel shows the posterior mean and
    90% credible band against the true function; the right panel shows the
    posterior for $\sigma$.  Switch $m$ to watch the fit smooth out, the
    band tighten, and the $\sigma$ posterior concentrate around the true
    noise level.
    """)
    return


@app.cell(hide_code=True)
def _(np, rng):
    # Friedman DGP used by the calibration fit and the variable-importance plots.
    def friedman(X, noise=0.0, rng=None):
        y = (
            10 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20 * (X[:, 2] - 0.5) ** 2
            + 10 * X[:, 3]
            + 5 * X[:, 4]
        )
        if noise > 0:
            assert rng is not None
            y = y + rng.normal(0, noise, size=y.shape[0])
        return y

    # Paper §5.2.1 defaults: n=100, p=10.
    n_train, n_feat = 100, 10
    X_fried = rng.uniform(size=(n_train, n_feat))
    y_fried = friedman(X_fried, noise=1.0, rng=rng)
    X_fried_test = rng.uniform(size=(200, n_feat))
    y_fried_test_true = friedman(X_fried_test, noise=0.0)
    return X_fried, X_fried_test, friedman, y_fried, y_fried_test_true


@app.cell(hide_code=True)
def _(
    X_fried,
    X_fried_test,
    friedman,
    np,
    predict,
    predict_at,
    prune_to_subset,
    run_bart,
    y_fried,
    y_fried_test_true,
):
    # Friedman calibration fits at m in {10, 50, 200}. ~10 s total.
    friedman_fits = {}
    for _m in (10, 50, 200):
        _rng = np.random.default_rng(20260423 + _m)
        _fit = run_bart(X_fried, y_fried, m=_m, n_iter=2000, burn_in=200, rng=_rng)
        _f_true_in = friedman(X_fried, noise=0.0)

        # In-sample posterior *prediction* intervals (add observation noise to f draws)
        _noise_in = _rng.normal(
            0,
            np.sqrt(_fit["sigma2_draws_scaled"])[:, None],
            size=_fit["f_draws_scaled"].shape,
        )
        _pred_in_scaled = _fit["f_draws_scaled"] + _noise_in
        _pred_in = _pred_in_scaled * _fit["y_range"] + (
            _fit["y_min"] + 0.5 * _fit["y_range"]
        )

        # Out-of-sample posterior *prediction* intervals
        _test_draws = predict_at(_fit, X_fried_test)
        _noise_test = _rng.normal(
            0, _fit["sigma_draws"][:, None], size=_test_draws.shape
        )
        _pred_test = _test_draws + _noise_test

        _summary = {
            "sigma_draws": _fit["sigma_draws"],
            "f_mean": _pred_in.mean(axis=0),
            "f_lo": np.quantile(_pred_in, 0.05, axis=0),
            "f_hi": np.quantile(_pred_in, 0.95, axis=0),
            "f_test_mean": _pred_test.mean(axis=0),
            "f_test_lo": np.quantile(_pred_test, 0.05, axis=0),
            "f_test_hi": np.quantile(_pred_test, 0.95, axis=0),
            "f_test_true": y_fried_test_true,
            "f_true_in": _f_true_in,
            "splits": _fit["splits"],
        }
        _summary["coverage_in"] = float(
            ((_summary["f_lo"] <= _f_true_in) & (_f_true_in <= _summary["f_hi"])).mean()
        )
        _summary["coverage_oos"] = float(
            (
                (_summary["f_test_lo"] <= y_fried_test_true)
                & (y_fried_test_true <= _summary["f_test_hi"])
            ).mean()
        )
        _summary["mse_oos"] = float(
            np.mean((_summary["f_test_mean"] - y_fried_test_true) ** 2)
        )

        if _m == 50:
            _splits = _fit["splits"]
            _tot = _splits.sum(axis=1, keepdims=True)
            _tot = np.where(_tot == 0, 1, _tot)
            _inclusion = (_splits / _tot).mean(axis=0)
            _ranking = list(np.argsort(-_inclusion))
            _summary["inclusion"] = _inclusion
            _summary["ranking"] = np.asarray(_ranking)

            # Restricted-R^2 curve via prune_to_subset on a thinned set of
            # tree snapshots.
            _snaps = _fit["tree_snapshots"]
            _n_sub = min(120, len(_snaps))
            _idx_sub = np.linspace(0, len(_snaps) - 1, _n_sub, dtype=int)
            _y_min = _fit["y_min"]
            _y_range = _fit["y_range"]

            _full_draws = np.zeros((_n_sub, X_fried.shape[0]))
            for _di, _sidx in enumerate(_idx_sub):
                _fs = np.zeros(X_fried.shape[0])
                for _t in _snaps[_sidx]:
                    _fs += predict(_t, X_fried)
                _full_draws[_di] = _fs * _y_range + (_y_min + 0.5 * _y_range)
            _full_mean = _full_draws.mean(axis=0)

            _r2 = []
            for _k in range(1, len(_ranking) + 1):
                _kept = _ranking[:_k]
                _rd = np.zeros_like(_full_draws)
                for _di, _sidx in enumerate(_idx_sub):
                    _fs = np.zeros(X_fried.shape[0])
                    for _t in _snaps[_sidx]:
                        _fs += predict(prune_to_subset(_t, _kept), X_fried)
                    _rd[_di] = _fs * _y_range + (_y_min + 0.5 * _y_range)
                _rm = _rd.mean(axis=0)
                _r2.append(float(1.0 - np.var(_full_mean - _rm) / np.var(_full_mean)))
            _summary["r2_curve"] = np.asarray(_r2)

            # PDP / ICE reuse these thinned snapshots; stored here so
            # predict_at(friedman_fits[50], X) works as a drop-in evaluator.
            _summary["tree_snapshots"] = [_snaps[i] for i in _idx_sub]
            _summary["y_min"] = _y_min
            _summary["y_range"] = _y_range

        friedman_fits[_m] = _summary

    fit_fried = friedman_fits[50]
    return fit_fried, friedman_fits


@app.cell(hide_code=True)
def _(X_toy, X_toy_grid, np, predict_at, run_bart, y_toy):
    # 1D toy calibration fits at m in {1, 5, 10, 15, 20, 40, 50, 100, 200}. ~1.5 s.
    toy_cal_ms = [1, 5, 10, 15, 20, 40, 50, 100, 200]
    toy_calibration_fits = {}
    for _m in toy_cal_ms:
        _rng = np.random.default_rng(20260423 + 200 + _m)
        _fit = run_bart(X_toy, y_toy, m=_m, n_iter=600, burn_in=200, rng=_rng)
        _draws_grid = predict_at(_fit, X_toy_grid)
        _x_train = X_toy[:, 0]
        _f_true_train = np.where(_x_train < 0.4, -1.0, 0.0) + 1.5 * np.exp(
            -((_x_train - 0.75) ** 2) / 0.01
        )
        _summary = {
            "f_mean_train": _fit["f_mean"],
            "f_lo_train": _fit["f_lo"],
            "f_hi_train": _fit["f_hi"],
            "f_mean_grid": _draws_grid.mean(axis=0),
            "f_lo_grid": np.quantile(_draws_grid, 0.05, axis=0),
            "f_hi_grid": np.quantile(_draws_grid, 0.95, axis=0),
            "sigma_draws": _fit["sigma_draws"],
        }
        _summary["coverage"] = float(
            (
                (_summary["f_lo_train"] <= _f_true_train)
                & (_f_true_train <= _summary["f_hi_train"])
            ).mean()
        )
        _summary["mse"] = float(
            np.mean((_summary["f_mean_train"] - _f_true_train) ** 2)
        )
        toy_calibration_fits[_m] = _summary
    return toy_cal_ms, toy_calibration_fits


@app.cell(hide_code=True)
def _(mo):
    calibration_m = mo.ui.slider(
        steps=[1, 5, 10, 20, 50, 100, 200],
        value=20,
        label="m (number of trees)",
    )
    mo.md(
        rf"""
        **Interactive: choose m.** Pick a value of m from the slider to see the BART
        fit on the 1D toy data and the corresponding σ posterior.

        {calibration_m}
        """
    )
    return (calibration_m,)


@app.cell(hide_code=True)
def _(
    PALETTE,
    X_toy,
    X_toy_grid,
    calibration_m,
    f_toy_true,
    mo,
    plt,
    toy_cal_ms,
    toy_calibration_fits,
    y_toy,
):
    def _calibration_sweep():
        fits = toy_calibration_fits
        if any(fits[m] is None for m in toy_cal_ms):
            return mo.md(
                "*Compute cells for the calibration sweep haven't finished yet.*"
            )

        sel = calibration_m.value
        f = fits[sel]

        fig, (ax_fit, ax_sigma) = plt.subplots(1, 2, figsize=(11, 4.2))

        # Left: fit on 1D toy
        x = X_toy[:, 0]
        xg = X_toy_grid[:, 0]
        ax_fit.scatter(
            x, y_toy, s=18, alpha=0.7, color="#333", label="observations", zorder=3
        )
        ax_fit.plot(
            xg,
            f_toy_true,
            color=PALETTE["truth"],
            lw=1.4,
            ls="--",
            label=r"true $f(x)$",
            zorder=2,
        )
        ax_fit.fill_between(
            xg,
            f["f_lo_grid"],
            f["f_hi_grid"],
            color=PALETTE["posterior"],
            alpha=0.25,
            label="90% credible band",
            zorder=1,
        )
        ax_fit.plot(
            xg,
            f["f_mean_grid"],
            color=PALETTE["posterior"],
            lw=1.6,
            label="posterior mean",
            zorder=2,
        )
        ax_fit.set_xlabel("x")
        ax_fit.set_ylabel("y")
        ax_fit.set_title(f"In-sample fit at m = {sel}")
        ax_fit.legend(frameon=False, loc="upper left")

        # Right: σ posterior
        ax_sigma.hist(
            f["sigma_draws"],
            bins=30,
            color=PALETTE["posterior"],
            edgecolor="white",
            alpha=0.8,
        )
        ax_sigma.axvline(
            0.15, color=PALETTE["truth"], lw=2, ls="--", label=r"true $\sigma = 0.15$"
        )
        ax_sigma.set_xlabel(r"$\sigma$")
        ax_sigma.set_ylabel("posterior draws")
        ax_sigma.set_title(r"Posterior of $\sigma$")
        ax_sigma.legend(frameon=False)

        fig.suptitle(
            f"m = {sel}:  coverage = {f['coverage']:.0%},  MSE = {f['mse']:.3f}",
            y=1.02,
        )
        fig.tight_layout()
        return fig

    _calibration_sweep()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpreting the fit

    Two diagnostic tools come for free from a BART posterior: a **variable
    importance** measure that tells you which inputs actually drive the
    fit, and **partial-dependence / ICE curves** that tell you what shape
    those drivers have.

    ### Variable importance

    Counting how often BART splits on each variable gives a useful
    ranking, but doesn't directly answer "would my predictions get worse
    if I dropped these last $p - k$ variables?".

    Quiroga *et al.* (2022) propose a richer estimator that *uses the same
    posterior trees we already drew*: for each top-$k$ subset (top-1, then
    top-2, …, all $p$), prune posterior trees by collapsing every split on
    an excluded variable into a leaf carrying the mean $\mu$ of its old
    subtree, then predict through the pruned ensemble. The reported
    $R^2_k$ is the coefficient of determination between the full-model
    predictions and the restricted-model predictions:

    $$
    R^2_k \;=\; 1 \;-\; \frac{\operatorname{Var}\!\bigl(\hat f^{\text{full}}(x) \;-\; \hat f^{\text{restricted}}_k(x)\bigr)}{\operatorname{Var}\!\bigl(\hat f^{\text{full}}(x)\bigr)}.
    $$

    We implement the pruning operator below and run it on the Friedman
    fit from earlier — the curve flattens around $k = 5$, recovering the
    true relevant-variable count for the Friedman DGP.
    """)
    return


@app.cell
def _(np):
    def prune_to_subset(tree, kept_vars):
        """Return a copy of `tree` with all splits on excluded variables collapsed.

        Each excluded internal node is turned into a leaf whose μ is the
        unweighted mean of the leaf μ values in the subtree it used to root.
        Descendant slots become unreachable but stay in the parallel arrays —
        ``predict`` only walks the reachable subtree from node 0.
        """
        new_tree = tree.copy()
        kept = set(int(v) for v in kept_vars)
        stack = [0]
        while stack:
            node = stack.pop()
            if new_tree.is_leaf(node):
                continue
            if new_tree.split_var[node] not in kept:
                leaf_mus = []
                sub = [node]
                while sub:
                    i = sub.pop()
                    if new_tree.is_leaf(i):
                        leaf_mus.append(new_tree.mu[i])
                    else:
                        sub.append(new_tree.left[i])
                        sub.append(new_tree.right[i])
                new_tree.split_var[node] = -1
                new_tree.split_val[node] = 0.0
                new_tree.left[node] = -1
                new_tree.right[node] = -1
                new_tree.mu[node] = float(np.mean(leaf_mus))
            else:
                stack.append(new_tree.left[node])
                stack.append(new_tree.right[node])
        return new_tree

    return (prune_to_subset,)


@app.cell(hide_code=True)
def _(fit_fried, mo, np, plt):

    def _restricted_r2_plot():
        if fit_fried is None or "r2_curve" not in fit_fried:
            return mo.md(
                "*The restricted-R² curve isn't in the friedman_m50 fit; "
                "re-run that compute cell.*"
            )

        ranking = list(fit_fried["ranking"])
        r2 = fit_fried["r2_curve"]
        p = len(r2)
        xs = np.arange(1, p + 1)

        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(xs, r2, "o-", color="#4c72b0", lw=1.6)
        ax.axhline(0.95, color="#c44e52", ls="--", lw=1, label=r"$R^2 = 0.95$")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"$X_{{{ranking[i]}}}$" for i in range(p)])
        ax.set_xlabel("variables included (cumulative, by inclusion rank)")
        ax.set_ylabel(r"$R^2$ vs. full-model predictions")
        ax.set_title(
            "Restricted-model R² on Friedman — flattens at the 5 relevant variables"
        )
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        return fig

    _restricted_r2_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Partial dependence and ICE

    A trained BART ensemble is a posterior over functions $f(x)$. Two
    standard interpretability tools reduce that posterior to one feature
    at a time.

    The **partial dependence function** for feature $j$ averages the
    prediction over the empirical distribution of the *other* features:

    $$
    \mathrm{PD}_j(v) \;=\; \frac{1}{n}\sum_{i=1}^{n} f\!\left(x_i^{(j \to v)}\right),
    $$

    where $x_i^{(j \to v)}$ is row $i$ with column $j$ replaced by $v$. This
    is what most people mean when they say "the effect of $x_j$ on $y$":
    every row contributes a counterfactual prediction at the same $v$, and
    the average traces out the marginal shape.

    An **individual conditional expectation (ICE)** curve plots the same
    counterfactual prediction *for one row* across $v$:

    $$
    \mathrm{ICE}_{j, i}(v) \;=\; f\!\left(x_i^{(j \to v)}\right).
    $$

    PDP shows the marginal effect; ICE shows the row-level effects whose
    average *is* the PDP. When ICE curves are parallel, the feature acts
    additively; when they cross, there is interaction.

    Because BART gives a *posterior* over $f$, we get a posterior over each
    PDP and ICE curve for free. The plot below uses the $m = 50$ Friedman
    fit from the "Choosing $m$" section.
    """)
    return


@app.cell
def _(np, predict_at):
    def bart_pdp(fit, X, j, grid):
        """Posterior over the partial-dependence function for feature `j`.

        For each grid value v, copy X, overwrite column j with v, evaluate
        the cached BART ensemble at every row, then average across rows.
        Returns shape (n_draws, len(grid)).
        """
        out = np.empty((len(fit["tree_snapshots"]), len(grid)))
        for g, v in enumerate(grid):
            X_eval = X.copy()
            X_eval[:, j] = v
            draws = predict_at(fit, X_eval)  # (n_draws, n)
            out[:, g] = draws.mean(axis=1)
        return out

    def bart_ice(fit, X, j, grid, row_idx):
        """Posterior over ICE curves at the chosen rows.

        Same as bart_pdp but without the row average; instead the prediction
        is kept per requested row. Returns shape (n_draws, len(row_idx), len(grid)).
        """
        out = np.empty((len(fit["tree_snapshots"]), len(row_idx), len(grid)))
        for g, v in enumerate(grid):
            X_eval = X.copy()
            X_eval[:, j] = v
            draws = predict_at(fit, X_eval)  # (n_draws, n)
            out[:, :, g] = draws[:, row_idx]
        return out

    return bart_ice, bart_pdp


@app.cell(hide_code=True)
def _(X_fried, bart_pdp, friedman_fits, np, plt):
    def _pdp_plot():
        fit_50 = friedman_fits[50]
        grid = np.linspace(0.0, 1.0, 30)

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
        # The Friedman ground truth is
        #   10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4.
        # We plot PDPs for x0 (sin component, expect a hump) and x3 (linear).
        for ax, j, label, truth in [
            (axes[0], 0, r"$x_0$ (sin component)", None),
            (axes[1], 3, r"$x_3$ (linear)", lambda v: 10.0 * v),
        ]:
            pdp_draws = bart_pdp(fit_50, X_fried, j=j, grid=grid)
            pdp_mean = pdp_draws.mean(axis=0)
            pdp_lo = np.quantile(pdp_draws, 0.05, axis=0)
            pdp_hi = np.quantile(pdp_draws, 0.95, axis=0)

            ax.fill_between(
                grid,
                pdp_lo,
                pdp_hi,
                color="#4c72b0",
                alpha=0.25,
                label="90% posterior band",
            )
            ax.plot(grid, pdp_mean, color="#4c72b0", lw=2, label="PDP posterior mean")
            if truth is not None:
                tv = np.array([truth(v) for v in grid])
                # Center the ground-truth curve at the empirical PDP mean so the
                # shape comparison is fair (the additive constant is absorbed by
                # the rest of the ensemble).
                ax.plot(
                    grid,
                    tv - tv.mean() + pdp_mean.mean(),
                    color="#c44e52",
                    ls="--",
                    lw=1.5,
                    label="true shape",
                )
            ax.set_xlabel(label)
            ax.set_ylabel(r"$\mathrm{PD}_j(v)$")
            ax.set_title(f"PDP for {label}")
            ax.legend(frameon=False, fontsize=8)

        fig.suptitle(
            "BART partial-dependence curves with posterior uncertainty", y=1.02
        )
        fig.tight_layout()
        return fig

    _pdp_plot()
    return


@app.cell(hide_code=True)
def _(X_fried, bart_ice, friedman_fits, np, plt):
    def _ice_plot():
        fit_50 = friedman_fits[50]
        grid = np.linspace(0.0, 1.0, 30)
        rng_ice = np.random.default_rng(20260518)
        row_idx = rng_ice.choice(X_fried.shape[0], size=30, replace=False)

        ice_draws = bart_ice(fit_50, X_fried, j=3, grid=grid, row_idx=row_idx)
        ice_mean = ice_draws.mean(axis=0)  # (n_rows, n_grid)

        pdp_mean = ice_mean.mean(axis=0)

        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        for r in range(ice_mean.shape[0]):
            ax.plot(grid, ice_mean[r], color="#4c72b0", alpha=0.25, lw=0.8)
        ax.plot(grid, pdp_mean, color="#c44e52", lw=2, label="PDP (row-average)")
        ax.set_xlabel(r"$x_3$")
        ax.set_ylabel(r"posterior-mean prediction")
        ax.set_title("ICE curves at 30 random training rows (Friedman, m=50)")
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig

    _ice_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The PDP for $x_3$ recovers the linear ramp the Friedman DGP puts in
    (slope 10), and the band tightens around the fit because every training
    row contributes. The PDP for $x_0$ is a milder hump because $x_0$ only
    enters through $\sin(\pi x_0 x_1)$, so averaging over $x_1 \in [0, 1]$
    flattens the effect.

    The ICE curves are nearly parallel for $x_3$, which is what we expect
    from a purely additive feature; crossing curves on a different feature
    would tell us BART has detected an interaction.

    **In the application notebooks.** `pymc-bart` ships `pmb.plot_pdp` and
    `pmb.plot_ice` that produce the same plots from a fitted PyMC model with
    a `pmb.BART` random variable. `01_regression.py` uses `plot_pdp` for the
    F1 fit, and `03_survival.py` uses `plot_ice` because in survival analysis
    the per-individual hazard curve is what stakeholders actually want to see.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A note on Particle Gibbs

    The MH grow / prune sampler we just built mixes well on small trees but
    stalls on large ones: a single grow rarely produces a dramatically better
    tree, and pruning unlucky structure takes many sweeps. Lakshminarayanan et
    al. (2015) replace the structural MH proposal with a **conditional
    sequential Monte Carlo** kernel: propose several whole trees in parallel
    by growing them layer-by-layer with resampling, then draw the next tree
    from the resulting weighted set. One particle is fixed to the current
    tree, which keeps the construction a valid Gibbs step.

    This is the sampler `pymc-bart` uses by default. We do not build it from
    scratch here: a correct PG implementation involves careful particle
    weight bookkeeping and a tuned mix of (`num_particles`, `batch`,
    resampling scheme, leaf-fit step) that distract from the core BART ideas
    in the earlier sections. Notebook 04 exposes the lever directly via
    `pm.sample(step=pmb.PGBART(vars=[eta], num_particles=20, batch=(0.1, 0.15)))`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sparsity in high-dimensional $X$ (Linero's prior)

    The default BART prior draws splitting variables uniformly from
    $\{1, \dots, p\}$. When most of those $p$ covariates are irrelevant
    (common in genomics, drug discovery, or any "throw everything at it"
    workflow) uniform sampling wastes proposal mass on noise.

    Linero (2018) replaces the uniform draw with a Dirichlet–categorical:

    $$
    s \;\sim\; \operatorname{Dir}\!\left(\tfrac{a}{p}, \dots, \tfrac{a}{p}\right),
    \qquad
    v \mid s \;\sim\; \operatorname{Cat}(s).
    $$

    With $a$ small the prior concentrates mass on a few variables; with
    $a$ large it approaches uniform.  We sample $s$ adaptively during
    burn-in by a conjugate Dirichlet update from the current split counts,
    then freeze $s$ for the post-burn-in draws.

    The MH derivation is unchanged: as long as the *proposal* draws
    $v$ from the same $s$, the prior–proposal factor cancels exactly,
    and the existing acceptance ratio is correct.
    """)
    return


@app.cell
def _(log_marginal_tree, np, splittable_cuts):
    def grow_proposal_sparse(
        tree, X, leaf_of, r, rng, alpha, beta, sigma2, sigma_mu2, split_prob
    ):
        """grow_proposal with split-variable drawn from `split_prob` (instead of uniform).

        Both the prior and the proposal use ``split_prob`` for the variable
        choice, so the variable-choice factor still cancels in the MH ratio
        and the acceptance formula is identical to the uniform case.
        """
        leaves = tree.leaves()
        lf = leaves[rng.integers(len(leaves))]
        mask = leaf_of == lf
        if int(mask.sum()) < 2:
            return None, -np.inf
        X_leaf = X[mask]

        chosen_col = int(rng.choice(len(split_prob), p=split_prob))
        cuts = splittable_cuts(X_leaf, chosen_col)
        if cuts.size == 0:
            return None, -np.inf
        chosen_cut = float(rng.choice(cuts))

        t_new = tree.copy()
        d = tree.depth_of(lf)
        new_left = len(t_new.split_var)
        new_right = new_left + 1
        t_new.split_var[lf] = chosen_col
        t_new.split_val[lf] = chosen_cut
        t_new.left[lf] = new_left
        t_new.right[lf] = new_right
        t_new.split_var += [-1, -1]
        t_new.split_val += [0.0, 0.0]
        t_new.left += [-1, -1]
        t_new.right += [-1, -1]
        t_new.parent += [lf, lf]
        t_new.mu += [0.0, 0.0]

        ll_new = log_marginal_tree(t_new, X, r, sigma2, sigma_mu2)
        ll_old = log_marginal_tree(tree, X, r, sigma2, sigma_mu2)

        p_split_d = alpha * (1.0 + d) ** (-beta)
        p_split_d1 = alpha * (2.0 + d) ** (-beta)
        log_shape_ratio = (
            np.log(p_split_d) + 2.0 * np.log(1.0 - p_split_d1) - np.log(1.0 - p_split_d)
        )

        b = len(leaves)
        P_grow_fwd = 1.0 if len(tree.internal_nodes()) == 0 else 0.5
        P_prune_bwd = 0.5
        w_new = len(t_new.singly_internal())
        log_move_ratio = np.log(P_prune_bwd / P_grow_fwd) + np.log(b / w_new)

        log_accept = (ll_new - ll_old) + log_shape_ratio + log_move_ratio
        return t_new, log_accept

    return (grow_proposal_sparse,)


@app.cell
def _(
    assign_leaves,
    calibrate_sigma_prior,
    draw_leaf_values,
    draw_sigma2,
    grow_proposal_sparse,
    np,
    predict,
    prune_proposal,
):
    def run_bart_sparse(
        X,
        y,
        m=50,
        n_iter=1000,
        burn_in=500,
        alpha=0.95,
        beta=2.0,
        k=2.0,
        nu=3.0,
        q=0.9,
        dirichlet_a=1.0,
        sigma2_init=None,
        rng=None,
        thin=1,
    ):
        """MH-BART with adaptive Dirichlet prior on splitting variable.

        During burn-in we re-sample ``split_prob ~ Dir(a/p + counts)`` once per
        sweep, where ``counts`` is the current per-variable split count summed
        across all m trees. After burn-in ``split_prob`` is frozen at its
        current value.
        """
        if rng is None:
            rng = np.random.default_rng()

        y_min, y_max = float(y.min()), float(y.max())
        y_range = y_max - y_min
        y_scaled = (y - y_min) / y_range - 0.5
        sigma_hat, lam = calibrate_sigma_prior(y_scaled, X, nu=nu, q=q)
        sigma2 = sigma_hat**2 if sigma2_init is None else float(sigma2_init)
        sigma_mu = 0.5 / (k * np.sqrt(m))
        sigma_mu2 = sigma_mu**2

        n, p = X.shape
        trees = [Tree() for _ in range(m)]
        tree_preds = np.zeros((m, n))
        split_prob = np.full(p, 1.0 / p)

        n_kept = (n_iter - burn_in + thin - 1) // thin
        f_draws = np.zeros((n_kept, n))
        sigma2_draws = np.zeros(n_kept)
        splits = np.zeros((n_kept, p), dtype=np.int64)
        split_prob_history = np.zeros((n_iter, p))
        kept = 0

        for it in range(n_iter):
            if it < burn_in:
                counts = np.zeros(p, dtype=float)
                for t in trees:
                    for v in t.split_var:
                        if v >= 0:
                            counts[v] += 1
                split_prob = rng.dirichlet(dirichlet_a / p + counts)
            split_prob_history[it] = split_prob

            for j in range(m):
                Rj = y_scaled - tree_preds.sum(axis=0) + tree_preds[j]
                if not trees[j].internal_nodes():
                    move = "grow"
                else:
                    move = "grow" if rng.random() < 0.5 else "prune"
                if move == "grow":
                    leaf_of = assign_leaves(trees[j], X)
                    t_new, logA = grow_proposal_sparse(
                        trees[j],
                        X,
                        leaf_of,
                        Rj,
                        rng,
                        alpha,
                        beta,
                        sigma2,
                        sigma_mu2,
                        split_prob,
                    )
                    if t_new is not None and np.log(rng.random()) < logA:
                        trees[j] = t_new
                else:
                    t_new, logA = prune_proposal(
                        trees[j], X, Rj, rng, alpha, beta, sigma2, sigma_mu2
                    )
                    if t_new is not None and np.log(rng.random()) < logA:
                        trees[j] = t_new
                trees[j] = draw_leaf_values(trees[j], X, Rj, rng, sigma2, sigma_mu2)
                tree_preds[j] = predict(trees[j], X)

            f_hat = tree_preds.sum(axis=0)
            sigma2 = draw_sigma2(y_scaled - f_hat, nu, lam, rng)

            if it >= burn_in and ((it - burn_in) % thin == 0):
                f_draws[kept] = f_hat
                sigma2_draws[kept] = sigma2
                for t in trees:
                    for v in t.split_var:
                        if v >= 0:
                            splits[kept, v] += 1
                kept += 1

        return {
            "sigma_draws": np.sqrt(sigma2_draws[:kept]) * y_range,
            "f_mean": f_draws[:kept].mean(axis=0) * y_range + (y_min + 0.5 * y_range),
            "splits": splits[:kept],
            "split_prob_history": split_prob_history,
            "split_prob_final": split_prob,
        }

    return (run_bart_sparse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Demo: Friedman with $p = 100$, only 5 relevant variables

    We simulate Friedman's function on $X_0..X_4$ and concatenate $95$ columns
    of pure noise that the model has no reason to ever split on. The plot
    below compares **uniform** split-variable sampling (left) against the
    **adaptive Dirichlet** prior described above (right), both fit on
    the same data with `m=20, n_iter=400, burn_in=200`.

    If the sparsity prior is doing its job, the right panel should put almost
    all inclusion mass on the red bars at indices $0..4$ and leave the grey
    "noise" bars near zero. Compare to Quiroga *et al.* (2022) Figure 13.
    """)
    return


@app.cell(hide_code=True)
def _(friedman, np, run_bart, run_bart_sparse):
    # Uniform-prior vs Dirichlet-prior inclusion frequencies on p=100 sparse
    # Friedman. ~3 s.
    _rng_sp = np.random.default_rng(20260423)
    _n_sp = 150
    _p_sp = 100
    _X_sp_rel = _rng_sp.uniform(size=(_n_sp, 5))
    _X_sp_noi = _rng_sp.uniform(size=(_n_sp, _p_sp - 5))
    _X_sp = np.concatenate([_X_sp_rel, _X_sp_noi], axis=1)
    _y_sp = friedman(_X_sp, noise=1.0, rng=_rng_sp)
    _kw = dict(m=20, n_iter=400, burn_in=200, alpha=0.95, beta=2.0)
    _uni_fit = run_bart(_X_sp, _y_sp, rng=np.random.default_rng(20260423), **_kw)
    _sp_fit = run_bart_sparse(
        _X_sp, _y_sp, dirichlet_a=1.0, rng=np.random.default_rng(20260423), **_kw
    )

    def _inc(splits):
        tot = splits.sum(axis=1, keepdims=True)
        tot = np.where(tot == 0, 1, tot)
        return (splits / tot).mean(axis=0)

    sparse_compare = {
        "p_total": _p_sp,
        "n_relevant": 5,
        "uniform_inclusion": _inc(_uni_fit["splits"]),
        "sparse_inclusion": _inc(_sp_fit["splits"]),
    }
    return (sparse_compare,)


@app.cell(hide_code=True)
def _(mo, np, plt, sparse_compare):

    def _sparse_compare_plot():
        if sparse_compare is None:
            return mo.md(
                "*Run the sparse-compare cell above to generate `sparse_compare`.*"
            )
        p_sp = sparse_compare["p_total"]
        n_rel = sparse_compare["n_relevant"]
        inc_uni = sparse_compare["uniform_inclusion"]
        inc_sp = sparse_compare["sparse_inclusion"]

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
        xs = np.arange(p_sp)
        for ax, inc, title in [
            (
                axes[0],
                inc_uni,
                f"uniform 1/p prior  (max irrelevant: {inc_uni[n_rel:].max():.3f})",
            ),
            (
                axes[1],
                inc_sp,
                f"Dirichlet a=1 prior  (max irrelevant: {inc_sp[n_rel:].max():.3f})",
            ),
        ]:
            cols = ["#c44e52" if i < n_rel else "#bbb" for i in xs]
            ax.bar(xs, inc, color=cols, edgecolor="white")
            ax.set_xlabel("variable index")
            ax.set_title(title)
        axes[0].set_ylabel("inclusion frequency")
        fig.suptitle(
            "Inclusion-frequency comparison: uniform vs adaptive Dirichlet prior\n"
            "(red = relevant, grey = noise)",
            y=1.05,
        )
        fig.tight_layout()
        return fig

    _sparse_compare_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading the comparison.**

    The uniform prior (left) does its job badly but predictably: every
    variable gets a baseline $\sim 1/p \approx 0.01$ rate, and the relevant
    red bars at indices $0..4$ rise slightly above the grey floor.
    Inclusion is informative but noisy.

    The Dirichlet prior (right) is sharper but, on this small example,
    locked onto the wrong variables. It found $X_0$ strongly and $X_4$
    weakly (both genuinely relevant) but also reinforced two irrelevant
    noise columns (indices around 21 and 33) at higher rates than the
    remaining three relevant variables. Max-irrelevant inclusion climbs
    from $0.037$ (uniform) to $0.307$ (Dirichlet). The Dirichlet update
    $s \sim \operatorname{Dir}(a/p + \text{counts})$ is a
    reinforcement-style dynamic: with $p = 100$, $a = 1$, only $200$
    burn-in iterations and $m = 20$ trees, whichever variables happen to
    get picked first keep getting rewarded.

    **When to reach for the sparsity prior.** Use it when you suspect
    most of your columns are irrelevant and want BART to ignore them
    (genomics, drug discovery, or any "throw everything at it"
    pipeline). Expect to give it room to work: a longer adaptive phase,
    more trees, sometimes a separate Bayesian update on $a$ itself. The
    production version lives in `pymc-bart` as the `split_prior` argument
    of `pmb.BART`, with those refinements built in. The implementation
    here shows the mechanism, not the tuning that makes it robust.
    """)
    return


@app.cell(hide_code=True)
def _(X_toy, X_toy_grid, f_toy_true, np, predict_at, run_bart, y_toy):
    # 1D buildup posteriors at m in {1, 5, 20, 100, 200} on the example data. ~1 s.
    buildup_ms = [1, 5, 20, 100, 200]
    buildup_fits_raw = {}
    for _m in buildup_ms:
        _rng = np.random.default_rng(20260423 + 100 + _m)
        _f = run_bart(X_toy, y_toy, m=_m, n_iter=600, burn_in=200, rng=_rng)
        _draws = predict_at(_f, X_toy_grid)
        _mean = _draws.mean(axis=0)
        _idx = int(((_draws - _mean) ** 2).sum(axis=1).argmin())
        buildup_fits_raw[_m] = {
            "f_mean": _mean,
            "f_lo": np.quantile(_draws, 0.05, axis=0),
            "f_hi": np.quantile(_draws, 0.95, axis=0),
            "f_draw": _draws[_idx],  # one representative ensemble realization
            "sigma_mean": float(_f["sigma_draws"].mean()),
        }
    buildup = {
        "x_train": X_toy[:, 0],
        "y_train": y_toy,
        "x_grid": X_toy_grid[:, 0],
        "f_true": f_toy_true,
        "ms": buildup_ms,
        "fits": buildup_fits_raw,
    }
    return (buildup,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
