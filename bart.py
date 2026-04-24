# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pymc>=5.28",
#     "pymc-bart>=0.11",
#     "arviz>=0.23",
#     "numpy>=2",
#     "matplotlib>=3.10",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayesian Additive Regression Trees — from scratch

    *PyData London 2026*

    Chipman, George & McCulloch (2010), *BART: Bayesian Additive Regression Trees.*
    Annals of Applied Statistics 4(1): 266–298. [arXiv:0806.3286](https://arxiv.org/abs/0806.3286)

    ## What this notebook does

    We build BART end-to-end: the prior, the marginal likelihood, the
    Metropolis–Hastings moves, the Gibbs backfitting sweep, the classifier, and
    the discrete-time survival wrapper — all in pure NumPy, no PyMC, no sklearn.

    Three worked applications follow: regression on Friedman's test function,
    probit classification, and discrete-time survival. A separate section
    contrasts the paper's MH sampler with the Particle Gibbs sampler used by
    modern BART implementations like `pymc-bart`.

    ## Prerequisites

    Comfortable with Bayesian regression, MCMC at the level of Metropolis–Hastings
    and Gibbs, and decision trees. No prior BART exposure assumed.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import chi2, norm, truncnorm

    rng = np.random.default_rng(20260423)
    plt.rcParams["figure.dpi"] = 110
    return chi2, norm, np, plt, rng, truncnorm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 &nbsp; Why BART?

    We model a continuous response as

    $$
        Y = f(x) + \varepsilon, \qquad \varepsilon \sim \mathcal N(0, \sigma^2),
    $$

    and approximate $f$ by a **sum of regression trees**

    $$
        f(x) \;\approx\; \sum_{j=1}^{m} g(x;\, T_j, M_j),
    $$

    where each $T_j$ is a binary tree structure and $M_j = \{\mu_{j\ell}\}$ the leaf
    values. With $m$ in the hundreds, every tree only has to explain a small
    slice of $f$ — that is what makes them *weak learners*. The Bayesian twist:
    we put a prior on $(T_j, M_j, \sigma)$ and sample the posterior. Uncertainty
    in $f$ then falls out of the MCMC draws — no bootstrap, no delta method.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 &nbsp; The regularisation prior

    Three pieces, each pulling the fit toward "nothing interesting here":

    - **Tree shape.** A node at depth $d$ is non-terminal with probability
      $\alpha(1+d)^{-\beta}$. With the paper's defaults $(\alpha, \beta) = (0.95, 2)$,
      most trees stay at 2–3 leaves.
    - **Leaf values.** $\mu_{j\ell} \sim \mathcal N(0, \sigma_\mu^2)$ with
      $\sigma_\mu = 0.5 / (k\sqrt{m})$ after rescaling $y$ to $[-0.5, 0.5]$.
      So any one tree can only move the prediction a tiny amount.
    - **Noise $\sigma$.** Inverse-$\chi^2$ with $(\nu, q) = (3, 0.9)$ calibrated
      so that the prior is *dominated* by a rough OLS estimate of the residual sd
      — we believe BART will do at least as well as OLS.

    Play with $\alpha$ and $\beta$ below to see the tree-size prior shift.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    tree_alpha = mo.ui.slider(
        0.1, 0.99, step=0.01, value=0.95, label=r"$\alpha$ (base probability)"
    )
    tree_beta = mo.ui.slider(
        0.0, 5.0, step=0.1, value=2.0, label=r"$\beta$ (depth penalty)"
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
def _(mo):
    mo.md(r"""
    ## 3 &nbsp; Representing a tree

    A binary tree is a list of nodes. Each node is either **internal** (has a
    split rule $x_j \le c$) or a **leaf** (has a value $\mu$). We store parent
    and child indices in parallel arrays so that the cheap moves the sampler
    needs — "find this node's sibling", "collapse this internal node into a
    leaf" — are $O(1)$ index lookups rather than pointer chasing.

    Root is always index $0$. `split_var[i] = -1` flags a leaf; otherwise it is
    the column index to split on.
    """)
    return


@app.class_definition
# ─── Binary tree, stored as parallel arrays ──────────────────────────────
#   split_var[i] = -1  iff node i is a leaf, else the column index to split on
#   split_val[i] = cut threshold  (internal)  OR  unused (leaf)
#   left[i], right[i] = children indices       OR  -1, -1 (leaf)
#   parent[i] = parent index, or -1 for the root
#   mu[i]     = leaf value (meaningful only for leaves)


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
        return [
            i for i in range(len(self.split_var)) if self.split_var[i] >= 0
        ]

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Routing data through the tree

    `assign_leaves` walks each row of `X` down the tree, recording which leaf
    it lands in. `predict` then just looks up the leaf's $\mu$. This is the only
    piece of code we call per-observation; keeping it tight matters because it
    runs inside every MCMC sweep.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Quick sanity check

    Hand-build a tree with one split on $x_0$ at $0.5$ and two leaves
    $\mu_L = -1,\ \mu_R = +2$, then route a toy $X$ through it.
    """)
    return


@app.cell
def _(np, predict):
    # Manual 3-node tree: root splits x0 at 0.5, leaves hold μ = -1, +2.
    _demo = Tree()
    _demo.split_var = [0, -1, -1]
    _demo.split_val = [0.5, 0.0, 0.0]
    _demo.left = [1, -1, -1]
    _demo.right = [2, -1, -1]
    _demo.parent = [-1, 0, 0]
    _demo.mu = [0.0, -1.0, 2.0]

    _X_demo = np.array([[0.1], [0.4], [0.6], [0.9]])
    _pred = predict(_demo, _X_demo)
    (_X_demo.ravel(), _pred)  # expected: (0.1, 0.4) -> -1 ; (0.6, 0.9) -> +2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 &nbsp; The marginal likelihood of a tree

    Before proposing a new tree structure we need to score the *current* one.
    Writing $n_\ell$ for the rows in leaf $\ell$ and
    $s_\ell = \sum_{i\in\ell} r_i$ for their residual sum, the leaf prior
    $\mu \sim \mathcal N(0, \sigma_\mu^2)$ combines conjugately with the normal
    data model to give a closed-form marginal:

    $$
        \log p(r \mid T, \sigma)
        \;=\;
        \sum_{\ell}
        \Bigl[
            \tfrac12 \log\!\tfrac{\sigma^2}{\sigma^2 + n_\ell \sigma_\mu^2}
            \;+\;
            \tfrac12 \cdot \tfrac{\sigma_\mu^2\, s_\ell^2}{\sigma^2(\sigma^2 + n_\ell \sigma_\mu^2)}
        \Bigr]
        \;+\; \text{const.}
    $$

    Because $\mu$ is integrated out, Metropolis–Hastings on tree structure is a
    plain random-walk MH — no reversible-jump, no Jacobian.
    """)
    return


@app.cell
def _(assign_leaves, np):
    # ─── Marginal likelihood after integrating out leaf values ───────────────


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The tree shape prior

    CGM98 factorises the prior on a tree as

    $$
        p(T) \;=\; \prod_{\text{internal } i} p_{\text{split}}(d_i)
                    \prod_{\text{leaf } \ell}\, (1 - p_{\text{split}}(d_\ell)),
        \qquad
        p_{\text{split}}(d) = \alpha(1+d)^{-\beta}.
    $$

    The uniform draw of split variable and cut value at each internal node
    adds a constant $1/(p \cdot n_{\text{cuts}})$ factor per internal node;
    because our MH proposals sample the same way, that factor cancels in the
    acceptance ratio and we don't have to track it here.
    """)
    return


@app.cell
def _(np):
    # ─── Tree prior: α(1+d)^{-β} for splitting at depth d ────────────────────


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 &nbsp; Sampling trees with Metropolis–Hastings

    Tree space is *discrete and unbounded*. CGM98 cycles through four local
    moves — **grow** a leaf into an internal node, **prune** a twig back to a
    leaf, **change** a split rule, **swap** two adjacent rules. We implement
    grow + prune only: they already make the chain ergodic, and the other two
    only help mixing on the margin.

    At each sweep we draw $P(\text{grow}) = P(\text{prune}) = 0.5$ if both are
    possible; on a stump only grow is legal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tree-surgery helpers

    Three small utilities the proposals need: enumerate allowable cut points in
    a leaf, walk a subtree, and *compact* a tree after pruning (renumber
    indices so the node arrays are contiguous).
    """)
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
            new.right.append(
                remap[tree.right[old]] if tree.right[old] >= 0 else -1
            )
            new.parent.append(
                remap[tree.parent[old]] if tree.parent[old] >= 0 else -1
            )
        return new

    return compact, splittable_cuts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Grow: deriving the acceptance ratio

    Pick a leaf $\ell$ uniformly at random (probability $1/b$ where $b$ is the
    number of leaves), pick a split variable and cut value the same way the
    prior does, and split. The prior and proposal both contain the same
    $1/(p \cdot n_{\text{cuts}})$ factor for the uniform split rule draw, so
    those cancel. What remains is the **shape ratio**

    $$
        \frac{p_{\text{split}}(d)\,(1-p_{\text{split}}(d+1))^2}
             {1 - p_{\text{split}}(d)}
    $$

    (we turned a leaf at depth $d$ into an internal with two leaves at $d+1$)
    and the **move-probability ratio**

    $$
        \frac{P_{\text{prune} \mid T'}\,/\,w'}{P_{\text{grow} \mid T}\,/\,b},
    $$

    where $w'$ counts *singly-internal* nodes of $T'$ (prune candidates).
    Multiply by the marginal-likelihood ratio and we have $\log A$.
    """)
    return


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
            np.log(p_split_d)
            + 2.0 * np.log(1.0 - p_split_d1)
            - np.log(1.0 - p_split_d)
        )

        b = len(leaves)
        P_grow_fwd = 1.0 if len(tree.internal_nodes()) == 0 else 0.5
        P_prune_bwd = 0.5
        w_new = len(t_new.singly_internal())
        log_move_ratio = np.log(P_prune_bwd / P_grow_fwd) + np.log(b / w_new)

        log_accept = (ll_new - ll_old) + log_shape_ratio + log_move_ratio
        return t_new, log_accept

    return (grow_proposal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prune: the reverse move

    Pick a *singly-internal* node (both children are leaves) uniformly and
    collapse it back to a leaf. The acceptance ratio is exactly the reciprocal
    of the grow ratio — swap $T$ and $T'$, swap $b$ and $w$, flip the sign of
    the shape-ratio log.
    """)
    return


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
            np.log(1.0 - p_split_d)
            - np.log(p_split_d)
            - 2.0 * np.log(1.0 - p_split_d1)
        )

        b_new = len(t_new.leaves())
        P_grow_bwd = 1.0 if b_new == 1 else 0.5
        P_prune_fwd = 0.5
        log_move_ratio = np.log(P_grow_bwd / P_prune_fwd) + np.log(w / b_new)

        log_accept = (ll_new - ll_old) + log_shape_ratio + log_move_ratio
        return t_new, log_accept

    return (prune_proposal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 &nbsp; Gibbs updates for $\mu$ and $\sigma$

    ### Drawing leaf values

    With tree $T$ fixed, each leaf has $n_\ell$ residuals with sum $s_\ell$.
    The normal–normal conjugacy gives

    $$
        \mu_\ell \mid r, T, \sigma
        \;\sim\;
        \mathcal N\!\Bigl(
            \tfrac{s_\ell\,\sigma_\mu^2}{\sigma^2 + n_\ell \sigma_\mu^2},\;
            \tfrac{\sigma^2\,\sigma_\mu^2}{\sigma^2 + n_\ell \sigma_\mu^2}
        \Bigr).
    $$

    Shrinkage toward zero is strong when a leaf is small — the prior takes
    over when the data has little to say.
    """)
    return


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
    ### Updating $\sigma^2$ and the data-calibrated prior

    With the ensemble $f(x) = \sum_j g(x; T_j, M_j)$ in hand, residuals
    $e = y - f$ are exchangeable Gaussians so $\sigma^2$ updates through its
    conjugate inverse-$\chi^2$:

    $$
        \sigma^2 \mid \text{everything}
        \;\sim\;
        \frac{\nu\lambda + \sum_i e_i^2}{\chi^2_{\nu + n}}.
    $$

    CGM98 calibrate $\lambda$ so the prior *quantile* at level $q$ equals a
    rough OLS estimate $\hat\sigma$ — a stand-in for "BART will do at least as
    well as a linear fit." Defaults $(\nu, q) = (3, 0.9)$.
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
    ## 7 &nbsp; The full MH–BART sampler

    Chipman et al.'s *Bayesian backfitting* (Hastie & Tibshirani 2000) drops
    out of the sum-of-trees structure. For each tree $T_j$, define the
    *partial residual*

    $$
        R_j \;=\; y - \sum_{k \ne j} g(x; T_k, M_k).
    $$

    Conditional on the other trees and $\sigma$, updating $(T_j, M_j)$ is a
    one-tree regression on $R_j$ — so each sweep calls grow/prune once per
    tree, then refreshes leaf values, then draws $\sigma$.

    One implementation choice worth pointing out: we keep a running
    `tree_preds[j]` of each tree's contribution on the training rows, so
    computing $R_j$ is $O(n)$ instead of $O(nm)$.
    """)
    return


@app.cell
def _(
    assign_leaves,
    calibrate_sigma_prior,
    draw_leaf_values,
    draw_sigma2,
    grow_proposal,
    mo,
    np,
    predict,
    prune_proposal,
):
    # ─── The BART sampler: Gibbs over trees, σ, and leaves ──────────────────


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

        for it in mo.status.progress_bar(range(n_iter), title="MH-BART sampling"):
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

                trees[j] = draw_leaf_values(
                    trees[j], X, Rj, rng, sigma2, sigma_mu2
                )
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
def _(mo):
    mo.md(r"""
    ### Out-of-sample prediction

    We stored every post-burn-in tree ensemble as a snapshot. To score a new
    `X_new`, just push it through each snapshot and average — giving us the
    same posterior uncertainty at test points as on the training set.
    """)
    return


@app.cell
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


@app.cell
def _(assign_leaves, splittable_cuts):
    # ─── PG-BART helpers ─────────────────────────────────────────────────────


    def _frontier(tree, depth):
        """Leaves of `tree` currently at this depth — the SMC advance frontier."""
        return [lf for lf in tree.leaves() if tree.depth_of(lf) == depth]


    def _split_in_place(tree, leaf, var, cut):
        """Turn `leaf` into an internal node with two fresh leaf children."""
        new_left = len(tree.split_var)
        new_right = new_left + 1
        tree.split_var[leaf] = var
        tree.split_val[leaf] = cut
        tree.left[leaf] = new_left
        tree.right[leaf] = new_right
        tree.split_var += [-1, -1]
        tree.split_val += [0.0, 0.0]
        tree.left += [-1, -1]
        tree.right += [-1, -1]
        tree.parent += [leaf, leaf]
        tree.mu += [0.0, 0.0]


    def grow_particle(tree, X, depth, rng, alpha, beta):
        """Advance one particle by one SMC step.

        For every leaf at the given `depth`, flip the prior split coin; if it
        comes up heads and the leaf has ≥ 2 rows, pick a random (var, cut) from
        the valid options and split in place.
        """
        p = alpha * (1.0 + depth) ** (-beta)
        leaf_of = assign_leaves(tree, X)
        for lf in _frontier(tree, depth):
            if rng.random() >= p:
                continue
            mask = leaf_of == lf
            if int(mask.sum()) < 2:
                continue
            X_leaf = X[mask]
            for c in rng.permutation(X.shape[1]):
                cuts = splittable_cuts(X_leaf, c)
                if cuts.size:
                    _split_in_place(tree, lf, int(c), float(rng.choice(cuts)))
                    break

    return (grow_particle,)


@app.cell
def _(grow_particle, log_marginal_tree, np):
    def particle_gibbs_tree(
        X,
        r,
        sigma2,
        sigma_mu2,
        rng,
        reference_tree=None,
        n_particles=10,
        max_depth=4,
        alpha=0.95,
        beta=2.0,
    ):
        """Conditional-SMC replacement for one tree, given partial residuals r.

        Particle 0 is reserved for the reference tree (held fixed to the current
        tree for tree j — this is what makes the construction a valid PG kernel).
        The remaining particles start as stumps and are grown level-by-level with
        resampling proportional to the marginal-likelihood gain at each level.
        The final draw is over **all** particles (including the reference) — so
        if no new proposal beats the current tree the sampler simply stays put.
        """
        particles = [
            reference_tree.copy() if reference_tree is not None else Tree()
        ]
        for _ in range(n_particles - 1):
            particles.append(Tree())

        log_w = [log_marginal_tree(p, X, r, sigma2, sigma_mu2) for p in particles]

        for d in range(max_depth):
            # Advance the non-reference particles one level.
            for i in range(1, n_particles):
                grow_particle(particles[i], X, d, rng, alpha, beta)

            log_w_new = [
                log_marginal_tree(p, X, r, sigma2, sigma_mu2) for p in particles
            ]
            incr = np.array(log_w_new) - np.array(log_w)
            log_w = log_w_new

            # Systematic resampling of non-reference slots; reference stays at 0.
            w = np.exp(incr - incr.max())
            w = w / w.sum()
            csum = np.cumsum(w)
            u = (rng.random() + np.arange(n_particles - 1)) / (n_particles - 1)
            idx = np.clip(np.searchsorted(csum, u), 0, n_particles - 1)
            particles = [particles[0]] + [particles[int(j)].copy() for j in idx]
            log_w = [
                log_marginal_tree(p, X, r, sigma2, sigma_mu2) for p in particles
            ]

        # Draw the returned tree from ALL particles (including the reference) —
        # excluding the reference would break detailed balance: if every new
        # proposal is worse than the current tree, the chain must be able to
        # stay put.
        final = np.array(log_w)
        final = np.exp(final - final.max())
        final = final / final.sum()
        return particles[rng.choice(n_particles, p=final)]

    return (particle_gibbs_tree,)


@app.cell
def _(draw_leaf_values, np, particle_gibbs_tree, predict):
    _rng_smoke = np.random.default_rng(42)
    _X_step = _rng_smoke.uniform(0, 1, size=(200, 1))
    _y_step = np.where(_X_step[:, 0] < 0.5, -1.0, 1.0) + _rng_smoke.normal(
        0, 0.1, size=200
    )

    _cur = Tree()
    for _ in range(20):
        _cur = particle_gibbs_tree(
            _X_step,
            _y_step,
            sigma2=0.01,
            sigma_mu2=1.0,
            rng=_rng_smoke,
            reference_tree=_cur,
            n_particles=15,
            max_depth=3,
        )
        _cur = draw_leaf_values(_cur, _X_step, _y_step, _rng_smoke, 0.01, 1.0)

    _pred_step = predict(_cur, _X_step)
    (
        float(_pred_step[_X_step[:, 0] < 0.5].mean()),
        float(_pred_step[_X_step[:, 0] >= 0.5].mean()),
        len(_cur.leaves()),
    )
    return


@app.cell
def _(
    calibrate_sigma_prior,
    draw_leaf_values,
    draw_sigma2,
    mo,
    np,
    particle_gibbs_tree,
    predict,
):
    def run_bart_pg(
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
        n_particles=10,
        max_depth=4,
        sigma2_init=None,
        rng=None,
        verbose=False,
        thin=1,
    ):
        """BART fit with particle-Gibbs tree updates in place of MH grow/prune."""
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
        tree_snapshots = []
        kept = 0

        for it in mo.status.progress_bar(range(n_iter), title="PG-BART sampling"):
            for j in range(m):
                Rj = y_scaled - tree_preds.sum(axis=0) + tree_preds[j]
                trees[j] = particle_gibbs_tree(
                    X,
                    Rj,
                    sigma2,
                    sigma_mu2,
                    rng,
                    reference_tree=trees[j],
                    n_particles=n_particles,
                    max_depth=max_depth,
                    alpha=alpha,
                    beta=beta,
                )
                trees[j] = draw_leaf_values(
                    trees[j], X, Rj, rng, sigma2, sigma_mu2
                )
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
        }

    return (run_bart_pg,)


@app.cell
def _(X_train, np, run_bart, run_bart_pg, y_train):
    # Cheap comparison: small m + n_iter so the cell stays interactive.
    _rng_cmp = np.random.default_rng(20260423)
    fit_mh = run_bart(
        X_train,
        y_train,
        m=50,
        n_iter=500,
        burn_in=200,
        alpha=0.95,
        beta=2.0,
        k=2.0,
        nu=3.0,
        q=0.9,
        rng=_rng_cmp,
        verbose=False,
    )
    fit_pg = run_bart_pg(
        X_train,
        y_train,
        m=50,
        n_iter=500,
        burn_in=200,
        n_particles=10,
        max_depth=4,
        alpha=0.95,
        beta=2.0,
        k=2.0,
        nu=3.0,
        q=0.9,
        rng=np.random.default_rng(20260423),
        verbose=False,
    )
    (
        float(fit_mh["sigma_draws"].mean()),
        float(fit_pg["sigma_draws"].mean()),
    )
    return fit_mh, fit_pg


@app.cell(hide_code=True)
def _(fit_mh, fit_pg, np, plt):
    def _running_ess(x, step=10):
        """Naïve running ESS via arviz.ess on growing prefixes."""
        from arviz import ess

        ks = np.arange(step, len(x) + 1, step)
        out = np.empty(len(ks))
        for i, k in enumerate(ks):
            out[i] = float(ess(x[:k]))
        return ks, out


    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 3.8))

    _a.plot(
        fit_mh["sigma_draws"], lw=0.6, alpha=0.9, label="MH-BART", color="#4c72b0"
    )
    _a.plot(
        fit_pg["sigma_draws"], lw=0.6, alpha=0.9, label="PG-BART", color="#c44e52"
    )
    _a.axhline(1.0, lw=1.0, ls="--", color="#333", label=r"true $\sigma = 1$")
    _a.set_xlabel("kept iteration")
    _a.set_ylabel(r"$\sigma$ draw")
    _a.set_title("σ trace")
    _a.legend(frameon=False)

    _ks_mh, _ess_mh = _running_ess(fit_mh["sigma_draws"])
    _ks_pg, _ess_pg = _running_ess(fit_pg["sigma_draws"])
    _b.plot(
        _ks_mh,
        _ess_mh,
        color="#4c72b0",
        label=f"MH-BART (final ESS={_ess_mh[-1]:.1f})",
    )
    _b.plot(
        _ks_pg,
        _ess_pg,
        color="#c44e52",
        label=f"PG-BART (final ESS={_ess_pg[-1]:.1f})",
    )
    _b.set_xlabel("kept iteration")
    _b.set_ylabel(r"ESS of $\sigma$ draws up to iter")
    _b.set_title("running effective sample size")
    _b.legend(frameon=False)

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(np, rng):
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
    X_train = rng.uniform(size=(n_train, n_feat))
    y_train = friedman(X_train, noise=1.0, rng=rng)
    X_test = rng.uniform(size=(200, n_feat))
    y_test_true = friedman(X_test, noise=0.0)
    y_train.shape, X_train.shape
    return X_test, X_train, friedman, y_test_true, y_train


@app.cell
def _(X_train, np, run_bart, y_train):
    # Fit BART with paper defaults: m=200 trees, (ν,q,k) = (3, 0.9, 2).
    # 5000 kept draws after 1000 burn-in (matches Fig. 3 of the paper).
    fit = run_bart(
        X_train,
        y_train,
        m=200,
        n_iter=3000,
        burn_in=1000,
        thin=1,
        alpha=0.95,
        beta=2.0,
        k=2.0,
        nu=3.0,
        q=0.9,
        rng=np.random.default_rng(20260423),
        verbose=True,
    )
    return (fit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In-sample fit, $\sigma$ mixing, and the $\sigma$ posterior

    Three diagnostics from one run:

    - Posterior mean of $f$ vs. the known truth with a 90% credible band.
      Because this is the Friedman signal we *know* the truth — so this is the
      cleanest picture of BART's recovery.
    - $\sigma$ trace — should hover around the truth of 1 after burn-in.
    - $\sigma$ posterior density — quantifies how well MCMC pins down the
      noise scale.
    """)
    return


@app.cell(hide_code=True)
def _(X_train, fit, friedman, np, plt):
    # In-sample fit: posterior mean of f(x) with 90% CI against the known
    # underlying signal.  Unlike out-of-sample, here we *know* f_true, so this
    # is the cleanest picture of how well BART has recovered the regression
    # function at the observed x's.
    _f_true_train = friedman(X_train, noise=0.0)
    _f_mean = fit["f_mean"]
    _f_lo = fit["f_lo"]
    _f_hi = fit["f_hi"]

    _in_cov = ((_f_lo <= _f_true_train) & (_f_true_train <= _f_hi)).mean()

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(13, 4))

    # (a) In-sample posterior vs truth
    _order = np.argsort(_f_true_train)
    _ax1.errorbar(
        _f_true_train[_order],
        _f_mean[_order],
        yerr=[(_f_mean - _f_lo)[_order], (_f_hi - _f_mean)[_order]],
        fmt="o",
        ms=3,
        ecolor="#4c72b0",
        color="#333",
        alpha=0.6,
        elinewidth=0.8,
    )
    _lim = (_f_true_train.min() - 1, _f_true_train.max() + 1)
    _ax1.plot(_lim, _lim, "--", color="C3", lw=1)
    _ax1.set_xlim(_lim)
    _ax1.set_ylim(_lim)
    _ax1.set_xlabel(r"true $f(x)$")
    _ax1.set_ylabel(r"posterior mean $\hat f(x)$ with 90% CI")
    _ax1.set_title(f"In-sample fit — 90% coverage: {_in_cov:.0%}")

    # (b) σ trace
    _sigma = fit["sigma_draws"]
    _ax2.plot(_sigma, ",", color="#333", alpha=0.4)
    _ax2.axhline(1.0, color="C3", lw=1.5, label=r"true $\sigma = 1$")
    _ax2.set_xlabel("MCMC iteration (post burn-in)")
    _ax2.set_ylabel(r"$\sigma$ draw")
    _ax2.set_title(r"$\sigma$ trace — checks mixing")
    _ax2.legend()

    # (c) σ posterior density
    _ax3.hist(
        _sigma,
        bins=40,
        density=True,
        color="#4c72b0",
        edgecolor="white",
        alpha=0.85,
    )
    _ax3.axvline(1.0, color="C3", lw=1.5, label=r"true $\sigma = 1$")
    _ax3.set_xlabel(r"$\sigma$")
    _ax3.set_ylabel("posterior density")
    _ax3.set_title(
        rf"$\sigma$ posterior: {_sigma.mean():.2f} [{np.quantile(_sigma, 0.05):.2f}, "
        rf"{np.quantile(_sigma, 0.95):.2f}]"
    )
    _ax3.legend()

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Out-of-sample coverage

    Apply each stored ensemble to `X_test` and average. We know the noiseless
    $f(x_{\text{test}})$ for this simulation, so we can check calibration: the
    nominal 90% bands should cover roughly 90% of truths.
    """)
    return


@app.cell
def _(X_test, fit, np, predict_at):
    f_test_draws = predict_at(fit, X_test)
    f_test_mean = f_test_draws.mean(axis=0)
    f_test_lo = np.quantile(f_test_draws, 0.05, axis=0)
    f_test_hi = np.quantile(f_test_draws, 0.95, axis=0)
    return f_test_hi, f_test_lo, f_test_mean


@app.cell(hide_code=True)
def _(f_test_hi, f_test_lo, f_test_mean, np, plt, y_test_true):
    _covered = (f_test_lo <= y_test_true) & (y_test_true <= f_test_hi)

    _fig, _ax = plt.subplots(figsize=(6, 5))
    _order = np.argsort(y_test_true)
    _ax.errorbar(
        y_test_true[_order],
        f_test_mean[_order],
        yerr=[
            (f_test_mean - f_test_lo)[_order],
            (f_test_hi - f_test_mean)[_order],
        ],
        fmt="o",
        ms=3,
        ecolor="#4c72b0",
        color="#333",
        alpha=0.6,
        elinewidth=0.8,
    )
    _lim = (
        min(y_test_true.min(), f_test_mean.min()) - 1,
        max(y_test_true.max(), f_test_mean.max()) + 1,
    )
    _ax.plot(_lim, _lim, "--", color="C3", lw=1)
    _ax.set_xlim(_lim)
    _ax.set_ylim(_lim)
    _ax.set_xlabel(r"true $f(x)$ (out-of-sample)")
    _ax.set_ylabel(r"posterior mean $\hat f(x)$ with 90% interval")
    _ax.set_title(f"Coverage of nominal 90% interval: {_covered.mean():.0%}")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Which variables are BART using?

    CGM98 Eq. 20 defines the *inclusion frequency* $v_i$ as the average (over
    MCMC draws) fraction of tree split rules that use variable $i$. A simple,
    robust signal: variables with little effect rarely get split on.
    """)
    return


@app.cell
def _(X_train, fit, np):
    # Average fraction of split rules using each variable, per CGM Eq. 20:
    # v_i = (1/K) Σ_k z_{ik}, where z_{ik} is the fraction of splits using var i.
    _total_per_iter = fit["splits"].sum(axis=1, keepdims=True)
    _total_per_iter = np.where(_total_per_iter == 0, 1, _total_per_iter)
    inclusion = (fit["splits"] / _total_per_iter).mean(axis=0)
    inclusion_labels = [f"x{i + 1}" for i in range(X_train.shape[1])]
    return inclusion, inclusion_labels


@app.cell(hide_code=True)
def _(inclusion, inclusion_labels, np, plt):
    _order = np.argsort(-inclusion)
    _colors = [
        "#4c72b0" if int(inclusion_labels[i][1:]) <= 5 else "#bbbbbb"
        for i in _order
    ]
    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    _ax.bar(
        [f"${inclusion_labels[i]}$" for i in _order],
        inclusion[_order],
        color=_colors,
        edgecolor="white",
    )
    _ax.set_ylabel("relative inclusion frequency")
    _ax.set_title("Variable importance — blue = truly relevant")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Partial dependence plots

    Eq. 19 of the paper: sweep one coordinate across its quantile grid while
    fixing all others at observed values, and average predictions. The five
    signal variables show clearly bent/nonlinear dependences; the other five
    are flat with a posterior band tight around zero.
    """)
    return


@app.cell(hide_code=True)
def _(X_train, fit, np, plt, predict_at):
    # Partial dependence (Eq. 19): f_s(x_s) = (1/n) Σ_i f(x_s, x_{i,c})
    # For each variable, sweep x_s across its quantile grid, fix all other
    # columns to training values, evaluate every posterior draw, aggregate.
    def partial_dependence(fit, X, var_idx, grid_size=20):
        xs = np.quantile(X[:, var_idx], np.linspace(0.05, 0.95, grid_size))
        n = X.shape[0]
        pdp_draws = np.zeros((len(fit["tree_snapshots"]), grid_size))
        for k, x_s in enumerate(xs):
            X_rep = X.copy()
            X_rep[:, var_idx] = x_s
            # average across observations for each draw
            draws = predict_at(fit, X_rep)  # (n_draws, n)
            pdp_draws[:, k] = draws.mean(axis=1)
        return xs, pdp_draws


    _fig, _axes = plt.subplots(2, 5, figsize=(12, 5), sharey=True)
    for v in range(X_train.shape[1]):
        ax = _axes.flat[v]
        xs, pdp_draws = partial_dependence(fit, X_train, v, grid_size=15)
        m_ = pdp_draws.mean(axis=0)
        lo, hi = np.quantile(pdp_draws, [0.05, 0.95], axis=0)
        color = "#4c72b0" if v < 5 else "#888"
        ax.fill_between(xs, lo, hi, alpha=0.25, color=color)
        ax.plot(xs, m_, color=color)
        ax.set_title(f"$x_{{{v + 1}}}$", fontsize=9)
        ax.set_xticks([])
    _fig.suptitle(
        "Partial dependence — only $x_1, \\ldots, x_5$ carry signal",
        y=1.02,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(np, truncnorm):
    def sample_latent_z(y_bin, G, rng):
        """Draw Z_i truncated at 0 with mean G(x_i), variance 1."""
        z = np.empty_like(G)
        # Positive cases: truncate left at 0
        pos = y_bin == 1
        a_pos = (0.0 - G[pos]) / 1.0
        z[pos] = truncnorm.rvs(
            a_pos, np.inf, loc=G[pos], scale=1.0, random_state=rng
        )
        # Negative cases: truncate right at 0
        neg = ~pos
        b_neg = (0.0 - G[neg]) / 1.0
        z[neg] = truncnorm.rvs(
            -np.inf, b_neg, loc=G[neg], scale=1.0, random_state=rng
        )
        return z

    return (sample_latent_z,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What changes in the sampler

    With $\sigma$ fixed and no $y$-rescaling, the backfitting loop is nearly
    identical to `run_bart`. We substitute $Z$ for the response and refresh
    $Z$ at the top of every sweep from the truncated-normal conditional.
    """)
    return


@app.cell
def _(
    assign_leaves,
    draw_leaf_values,
    grow_proposal,
    mo,
    norm,
    np,
    predict,
    prune_proposal,
    sample_latent_z,
):
    def run_bart_probit(
        X,
        y_bin,
        m=50,
        n_iter=1000,
        burn_in=500,
        alpha=0.95,
        beta=2.0,
        k=2.0,
        rng=None,
        verbose=False,
        thin=1,
    ):
        """BART probit classifier. Returns draws of G(x) and implied p(x) = Φ(G)."""
        if rng is None:
            rng = np.random.default_rng()

        # σ fixed at 1; leaf prior is 3/(k√m)
        sigma2 = 1.0
        sigma_mu = 3.0 / (k * np.sqrt(m))
        sigma_mu2 = sigma_mu**2

        n, p = X.shape
        trees = [Tree() for _ in range(m)]
        tree_preds = np.zeros((m, n))

        n_kept = (n_iter - burn_in + thin - 1) // thin
        G_draws = np.zeros((n_kept, n))
        splits = np.zeros((n_kept, p), dtype=np.int64)
        tree_snapshots = []
        kept = 0

        # Initialize Z from prior centered at 0
        z = sample_latent_z(y_bin, np.zeros(n), rng)

        for it in mo.status.progress_bar(
            range(n_iter), title="BART probit sampling"
        ):
            G_curr = tree_preds.sum(axis=0)

            # 1) draw latent Z | G, y
            z = sample_latent_z(y_bin, G_curr, rng)

            # 2) backfitting sweep over trees, fitting to Z (no y rescaling —
            #    Z is already on the probit scale).
            for j in range(m):
                Rj = z - tree_preds.sum(axis=0) + tree_preds[j]

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
                    if t_new is not None and np.log(rng.random()) < logA:
                        trees[j] = t_new
                else:
                    t_new, logA = prune_proposal(
                        trees[j], X, Rj, rng, alpha, beta, sigma2, sigma_mu2
                    )
                    if t_new is not None and np.log(rng.random()) < logA:
                        trees[j] = t_new

                trees[j] = draw_leaf_values(
                    trees[j], X, Rj, rng, sigma2, sigma_mu2
                )
                tree_preds[j] = predict(trees[j], X)

            if it >= burn_in and ((it - burn_in) % thin == 0):
                G_draws[kept] = tree_preds.sum(axis=0)
                for t in trees:
                    for v in t.split_var:
                        if v >= 0:
                            splits[kept, v] += 1
                tree_snapshots.append([t.copy() for t in trees])
                kept += 1

            if verbose and (it + 1) % max(1, n_iter // 10) == 0:
                print(
                    f"  iter {it + 1}/{n_iter}  "
                    f"leaves={np.mean([len(t.leaves()) for t in trees]):.1f}  "
                    f"p̂ mean={norm.cdf(tree_preds.sum(axis=0)).mean():.3f}"
                )

        G_draws = G_draws[:kept]
        p_draws = norm.cdf(G_draws)
        return {
            "G_draws": G_draws,
            "p_draws": p_draws,
            "p_mean": p_draws.mean(axis=0),
            "p_lo": np.quantile(p_draws, 0.05, axis=0),
            "p_hi": np.quantile(p_draws, 0.95, axis=0),
            "splits": splits[:kept],
            "tree_snapshots": tree_snapshots,
        }


    def predict_probit_at(fit, X_new):
        """Return (n_draws, n_new) array of p(x) draws at X_new."""
        snaps = fit["tree_snapshots"]
        n_new = X_new.shape[0]
        draws = np.zeros((len(snaps), n_new))
        for d, trees in enumerate(snaps):
            g = np.zeros(n_new)
            for t in trees:
                g += predict(t, X_new)
            draws[d] = norm.cdf(g)
        return draws

    return predict_probit_at, run_bart_probit


@app.cell
def _(np, rng):
    def simulate_binary(n, p, rng):
        X = rng.uniform(-1, 1, size=(n, p))
        logit = 2.5 * X[:, 0] * X[:, 1] + 3 * (X[:, 2] > 0.2) - 0.5
        prob_true = 1 / (1 + np.exp(-logit))
        y = rng.binomial(1, prob_true)
        return X, y, prob_true


    X_cls, y_cls, prob_true = simulate_binary(500, 20, rng)
    X_cls_test, y_cls_test, prob_test_true = simulate_binary(300, 20, rng)
    f"train active rate: {y_cls.mean():.2%}"
    return X_cls, X_cls_test, prob_test_true, y_cls, y_cls_test


@app.cell
def _(X_cls, np, run_bart_probit, y_cls):
    # Probit BART: m=50 trees (paper uses m=50 for the drug-discovery example),
    # (ν, q) have no role under probit, σ is fixed at 1.
    fit_cls = run_bart_probit(
        X_cls,
        y_cls,
        m=50,
        n_iter=1500,
        burn_in=500,
        alpha=0.95,
        beta=2.0,
        k=2.0,
        rng=np.random.default_rng(20260423),
        verbose=True,
    )
    return (fit_cls,)


@app.cell
def _(X_cls_test, fit_cls, np, predict_probit_at):
    p_test_draws = predict_probit_at(fit_cls, X_cls_test)
    p_test_mean = p_test_draws.mean(axis=0)
    p_test_lo = np.quantile(p_test_draws, 0.05, axis=0)
    p_test_hi = np.quantile(p_test_draws, 0.95, axis=0)
    return p_test_hi, p_test_lo, p_test_mean


@app.cell(hide_code=True)
def _(np, p_test_hi, p_test_lo, p_test_mean, plt, prob_test_true, y_cls_test):
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(10, 4))
    _order = np.argsort(prob_test_true)
    _a.errorbar(
        prob_test_true[_order],
        p_test_mean[_order],
        yerr=[
            (p_test_mean - p_test_lo)[_order],
            (p_test_hi - p_test_mean)[_order],
        ],
        fmt=".",
        ms=3,
        alpha=0.5,
        elinewidth=0.7,
        color="#4c72b0",
    )
    _a.plot([0, 1], [0, 1], "--", color="C3", lw=1)
    _a.set_xlabel(r"true $\Pr(Y=1\mid x)$")
    _a.set_ylabel("posterior mean ± 90% interval")
    _a.set_title("Calibration on the test set")

    # Top-20 predicted probabilities — paper's Fig. 9 idea
    _top_idx = np.argsort(-p_test_mean)[:20]
    _positions = np.arange(len(_top_idx))
    _colors = ["C2" if y_cls_test[i] == 1 else "C3" for i in _top_idx]
    _b.errorbar(
        _positions,
        p_test_mean[_top_idx],
        yerr=[
            (p_test_mean - p_test_lo)[_top_idx],
            (p_test_hi - p_test_mean)[_top_idx],
        ],
        fmt="none",
        ecolor="#888",
        elinewidth=0.8,
    )
    _b.scatter(_positions, p_test_mean[_top_idx], c=_colors, s=35, zorder=3)
    _b.set_xticks(_positions)
    _b.set_xticklabels([])
    _b.set_xlabel("top-20 ranked compounds")
    _b.set_ylabel(r"$\Pr(Y=1\mid x)$ with 90% CI")
    _b.set_title(
        f"Top-20 hit rate: {y_cls_test[_top_idx].sum()}/20 active "
        f"(base {y_cls_test.mean():.0%})"
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(np, rng):
    def simulate_survival(n, rng, max_time=12):
        x1 = rng.uniform(-1, 1, size=n)
        x2 = rng.uniform(-1, 1, size=n)
        rows_y, rows_t, rows_x1, rows_x2 = [], [], [], []
        for i in range(n):
            for t in range(1, max_time + 1):
                logit_h = -3.0 + 0.7 * x1[i] + 0.4 * x2[i] * np.log(t)
                p = 1 / (1 + np.exp(-logit_h))
                y = rng.binomial(1, p)
                rows_y.append(y)
                rows_t.append(t)
                rows_x1.append(x1[i])
                rows_x2.append(x2[i])
                if y == 1:
                    break
        return (
            np.array(rows_y, dtype=int),
            np.column_stack([rows_t, rows_x1, rows_x2]).astype(float),
        )


    surv_y, surv_X = simulate_survival(150, rng, max_time=12)
    f"{surv_X.shape[0]} person-time rows, {surv_y.sum()} events"
    return surv_X, surv_y


@app.cell
def _(np, run_bart_probit, surv_X, surv_y):
    # Discrete-time hazards: for each subject-time row, p = Pr(event at t | x, at-risk).
    # Using fewer trees since this is a smaller dataset with a simple signal.
    fit_surv = run_bart_probit(
        surv_X,
        surv_y,
        m=50,
        n_iter=1000,
        burn_in=400,
        alpha=0.95,
        beta=2.0,
        k=2.0,
        rng=np.random.default_rng(20260423),
        verbose=False,
    )
    return (fit_surv,)


@app.cell
def _(fit_surv, np, predict_probit_at):
    times = np.arange(1, 13)
    profile_low = np.column_stack([times, np.full(12, -0.8), np.full(12, 0.0)])
    profile_high = np.column_stack([times, np.full(12, 0.8), np.full(12, 0.0)])

    p_low_draws = predict_probit_at(fit_surv, profile_low)  # (n_draws, 12)
    p_high_draws = predict_probit_at(fit_surv, profile_high)

    # Survival: S(t|x) = ∏_{s≤t} (1 − p_s)
    S_low = np.cumprod(1 - p_low_draws, axis=1)
    S_high = np.cumprod(1 - p_high_draws, axis=1)
    return S_high, S_low, times


@app.cell(hide_code=True)
def _(S_high, S_low, np, plt, times):
    _m_low = S_low.mean(axis=0)
    _lo_low, _hi_low = np.quantile(S_low, [0.05, 0.95], axis=0)
    _m_high = S_high.mean(axis=0)
    _lo_high, _hi_high = np.quantile(S_high, [0.05, 0.95], axis=0)

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.step(times, _m_low, where="post", color="#4c72b0", label=r"$x_1 = -0.8$")
    _ax.fill_between(
        times, _lo_low, _hi_low, step="post", color="#4c72b0", alpha=0.25
    )
    _ax.step(times, _m_high, where="post", color="#c44e52", label=r"$x_1 = +0.8$")
    _ax.fill_between(
        times, _lo_high, _hi_high, step="post", color="#c44e52", alpha=0.25
    )
    _ax.set_ylim(0, 1.02)
    _ax.set_xlabel("time $t$")
    _ax.set_ylabel(r"$S(t\mid x)$")
    _ax.set_title("Predicted survival, posterior mean and 90% band")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 &nbsp; PG-BART: a better sampler

    MH grow/prune is charmingly simple, but it moves **one leaf at a time**.
    If the posterior has two well-separated modes — say, "split on $x_2$ at
    the root" vs. "split on $x_5$ at the root" — the only way to get from one
    to the other is to prune the tree back to a stump and then grow the other
    direction, and the intermediate stump has low marginal likelihood, so the
    chain rarely accepts it. In practice the chain can get stuck.

    Lakshminarayanan, Roy & Teh (2015) replace this local MH move with a
    **non-local proposal**: at each tree update, run a small sequential
    Monte Carlo sampler that grows a whole fresh tree from scratch, with
    resampling focussing compute on the promising branches, and then swap
    the tree in. This is what `pymc-bart` actually does.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Particle Gibbs: the idea

    Keep $N$ candidate trees ("particles") in flight. Evolve them level-by-level:
    at depth $d$, every particle independently decides — for each of its leaves
    currently at depth $d$ — whether to split (with prior probability
    $p_{\text{split}}(d) = \alpha(1+d)^{-\beta}$) and, if so, draws a uniform
    (variable, cut). After each level we reweight each particle by the gain in
    $\log p(r \mid T, \sigma)$ and **resample** — promising structures
    replicate, unpromising ones die.

    To make this a valid Markov-chain update we use **conditional SMC**: one
    particle is fixed to be the *current* tree for tree $j$ (the "reference
    particle"). The remaining particles propose fresh alternatives; we keep
    one at random proportional to its final weight.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Smoke test

    Recovering a 1-D step function with a single tree. If the sampler works,
    left-side predictions cluster near $-1$ and right-side near $+1$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The full PG-BART sampler

    `run_bart_pg` is a mechanical substitution: replace the MH grow/prune
    update inside `run_bart`'s backfitting loop with `particle_gibbs_tree`,
    passing the *current* tree as the reference particle. Leaf draws,
    $\sigma^2$ updates, and snapshot bookkeeping are unchanged.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Side-by-side: MH-BART vs PG-BART on Friedman

    We compare on a deliberately constrained budget — small $m$, short chain —
    so mixing differences are visible. Both samplers target the *same*
    posterior, but on a short chain the more-autocorrelated sampler may report
    a posterior mean that simply reflects wherever the chain got stuck rather
    than the true centre.

    What to look for in the plots below:

    1. **MH-BART's $\sigma$ trace** has long runs of near-identical draws
       (rejected proposals); **PG-BART's** moves every step.
    2. **PG-BART's ESS** grows roughly linearly with iteration count;
       MH-BART's grows much more slowly and plateaus early. A higher ESS per
       unit of compute is the whole pitch of PG-BART, and the effect is
       visible even on this small budget.

    The two $\sigma$ posterior means may not agree here — MH's ESS is low
    enough that its chain hasn't explored, so its mean is a sample artefact,
    not a reliable estimate. Run MH longer (or at a larger $m$) and it
    converges to the same target as PG.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Trace and running-ESS diagnostics

    Two panels:

    - $\sigma$ **trace** — overlay of the two chains. MH-BART typically shows
      visible staircasing (consecutive draws stuck at the same value because
      every grow/prune was rejected) whereas PG-BART moves on every step.
    - **Running ESS** — effective sample size of $\sigma$ computed on the
      growing prefix of the chain. Since PG-BART has lower autocorrelation,
      its running ESS grows roughly linearly with iteration count; MH-BART's
      growth is much slower.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### When to prefer which

    - **Use MH-BART** when you want the sampler you can read top-to-bottom,
      or the trees are small so local moves are enough (CGM's defaults with
      $m=200$, $(\alpha, \beta) = (0.95, 2)$ keep most trees at 2–3 leaves).
    - **Use PG-BART** when the truth needs deeper trees ($\beta \le 1$, or a
      small $m$), or when long runs are showing poor mixing. It's also
      easier to parallelise (each particle's SMC run is independent).

    `pymc-bart` implements the PG-BART version with extra tricks for
    performance — the sampler in this notebook is the minimal pedagogical
    form of the same idea.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 &nbsp; Regression on Friedman's test function

    CGM98 Section 5.2.1. The underlying signal is

    $$
        f(x) = 10\sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10 x_4 + 5 x_5,
    $$

    defined on $[0,1]^{10}$; $x_6, \ldots, x_{10}$ are pure noise we hand BART
    to see if it resists them. Paper defaults: $n=100$ train, $n=200$ test,
    $m=200$ trees, $(\nu, q, k) = (3, 0.9, 2)$. Training response gets
    Gaussian noise with $\sigma = 1$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 &nbsp; BART for classification

    For binary $Y$, CGM98 follow Albert & Chib (1993): introduce a latent
    $Z_i$ with

    $$
        Z_i \mid x, G, Y_i = 1 \;\sim\; \mathcal{TN}(G(x_i), 1; \;[0,\infty)),
        \qquad
        Z_i \mid x, G, Y_i = 0 \;\sim\; \mathcal{TN}(G(x_i), 1; \;(-\infty,0]),
    $$

    where $G(x) = \sum_j g(x; T_j, M_j)$ lives on the probit scale and
    $p(x) = \Phi(G(x))$. Given $Z$, the sum-of-trees becomes a Gaussian
    regression with known $\sigma = 1$ — so the *exact same* backfitting
    sweep works.

    Two changes to the prior:

    - $\sigma$ is fixed at $1$, not sampled.
    - $\sigma_\mu = 3 / (k\sqrt{m})$ — the extra factor of 6 (vs. the
      regression $0.5/(k\sqrt{m})$) is because $G$ lives on the probit scale
      and $\pm 3$ covers most of $\Phi$'s probability mass (paper Eq. 24).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Probit BART on a simulated classification task

    $n = 500$ train, 20 predictors with only $x_1, x_2, x_3$ active. With
    $m=50$ trees (paper's drug-discovery setting) we expect calibration on
    the test set to hug the 45° line and the top-20 scored cases to be
    enriched for the active class.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11 &nbsp; Discrete-time survival

    A survival problem with time measured in integer units and competing
    hazards at each interval becomes a classification problem on the
    *person–time* data set: for every (subject, $t$) row where the subject
    was still at risk, model $\Pr(\text{event at } t \mid x)$ with BART probit.

    The running $\hat\Pr$ at each $t$ are the discrete hazards, and
    $S(t \mid x) = \prod_{s \le t}\!(1 - \hat h_s(x))$ turns them back into a
    survival curve — with uncertainty carried through every product because
    each $\hat h_s$ is a posterior draw.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12 &nbsp; Takeaways

    - BART is a sum of hundreds of tiny trees with priors that keep them
      *tiny*. The posterior concentrates not because any single tree is
      confident but because the ensemble votes.
    - Bayesian backfitting factorises the problem: each tree is a one-tree
      regression on partial residuals. The sampler is just MH on tree
      structure + Gibbs on leaves + Gibbs on $\sigma$.
    - Because $\mu$ marginalises out of the MH step, we never need
      reversible-jump.
    - The same machinery handles classification (Albert–Chib latent $Z$) and
      discrete-time survival (person–time data).

    **Further reading.**
    Chipman et al. 2010 (original paper);
    Pratola, Chipman, et al. 2013 (parallel BART);
    Lakshminarayanan, Roy & Teh 2015 (Particle Gibbs for BART — subject of
    the next notebook section);
    `pymc-bart` — production-grade implementation used by this tutorial's
    author in practice.
    """)
    return


if __name__ == "__main__":
    app.run()
