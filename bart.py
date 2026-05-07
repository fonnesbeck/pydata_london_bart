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
        particles = [reference_tree.copy() if reference_tree is not None else Tree()]
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
            log_w = [log_marginal_tree(p, X, r, sigma2, sigma_mu2) for p in particles]

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
        batch_frac=1.0,
        sigma2_init=None,
        rng=None,
        verbose=False,
        thin=1,
    ):
        """BART fit with particle-Gibbs tree updates in place of MH grow/prune.

        ``batch_frac`` controls what fraction of the m trees is refreshed per
        sweep. ``pymc-bart`` defaults to 0.1 (only 10% of trees touched per step)
        because each PG tree update is much more expensive than an MH grow/prune
        and most trees are already near-stationary. We default to 1.0 so the
        outer loop matches MH-BART tree-for-tree; pass ``batch_frac=0.1`` to
        recover the pymc-bart behaviour.
        """
        if rng is None:
            rng = np.random.default_rng()
        batch_size = max(1, int(np.ceil(batch_frac * m)))

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
            if batch_size >= m:
                batch = range(m)
            else:
                batch = rng.choice(m, size=batch_size, replace=False)
            for j in batch:
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
        }

    return (run_bart_pg,)


@app.cell
def _(X_fried, np, run_bart, run_bart_pg, y_fried):
    # Cheap comparison on Friedman (known σ=1) so the two samplers
    # can be benchmarked against a shared reference line.
    _rng_cmp = np.random.default_rng(20260423)
    fit_mh = run_bart(
        X_fried,
        y_fried,
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
        X_fried,
        y_fried,
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

    _a.plot(fit_mh["sigma_draws"], lw=0.6, alpha=0.9, label="MH-BART", color="#4c72b0")
    _a.plot(fit_pg["sigma_draws"], lw=0.6, alpha=0.9, label="PG-BART", color="#c44e52")
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
    X_fried = rng.uniform(size=(n_train, n_feat))
    y_fried = friedman(X_fried, noise=1.0, rng=rng)
    X_fried_test = rng.uniform(size=(200, n_feat))
    y_fried_test_true = friedman(X_fried_test, noise=0.0)
    y_fried.shape, X_fried.shape
    return X_fried, X_fried_test, friedman, y_fried, y_fried_test_true


@app.cell
def _(X_fried, np, run_bart, y_fried):
    # Preamble: fit BART on the Friedman DGP where f(x) and σ are both
    # known. Small m and few iterations — this is a calibration sanity check,
    # not the main demonstration. The "real data" fit lives below.
    fit_fried = run_bart(
        X_fried,
        y_fried,
        m=50,
        n_iter=500,
        burn_in=200,
        thin=1,
        alpha=0.95,
        beta=2.0,
        k=2.0,
        nu=3.0,
        q=0.9,
        rng=np.random.default_rng(20260423),
        verbose=False,
    )
    return (fit_fried,)


@app.cell(hide_code=True)
def _(
    X_fried,
    X_fried_test,
    fit_fried,
    friedman,
    np,
    plt,
    predict_at,
    y_fried_test_true,
):
    # 90% credible-interval coverage on the Friedman DGP, in- and out-of-sample,
    # plotted against the known true f(x). This is the calibration picture that
    # lets us trust BART's uncertainty quantification before turning it on data
    # whose answer we don't know.
    _f_true_in = friedman(X_fried, noise=0.0)
    _f_mean_in = fit_fried["f_mean"]
    _f_lo_in = fit_fried["f_lo"]
    _f_hi_in = fit_fried["f_hi"]
    _cov_in = ((_f_lo_in <= _f_true_in) & (_f_true_in <= _f_hi_in)).mean()

    _f_draws_out = predict_at(fit_fried, X_fried_test)
    _f_mean_out = _f_draws_out.mean(axis=0)
    _f_lo_out = np.quantile(_f_draws_out, 0.05, axis=0)
    _f_hi_out = np.quantile(_f_draws_out, 0.95, axis=0)
    _cov_out = (
        (_f_lo_out <= y_fried_test_true) & (y_fried_test_true <= _f_hi_out)
    ).mean()

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for _ax, _xt, _m, _lo, _hi, _cov, _tag in [
        (_a, _f_true_in, _f_mean_in, _f_lo_in, _f_hi_in, _cov_in, "in-sample"),
        (
            _b,
            y_fried_test_true,
            _f_mean_out,
            _f_lo_out,
            _f_hi_out,
            _cov_out,
            "out-of-sample",
        ),
    ]:
        _order = np.argsort(_xt)
        _ax.errorbar(
            _xt[_order],
            _m[_order],
            yerr=[(_m - _lo)[_order], (_hi - _m)[_order]],
            fmt="o",
            ms=3,
            ecolor="#4c72b0",
            color="#333",
            alpha=0.55,
            elinewidth=0.7,
        )
        _lim = (_xt.min() - 1, _xt.max() + 1)
        _ax.plot(_lim, _lim, "--", color="C3", lw=1)
        _ax.set_xlim(_lim)
        _ax.set_ylim(_lim)
        _ax.set_xlabel(r"true $f(x)$")
        _ax.set_ylabel(r"posterior mean $\hat f(x)$ with 90% CI")
        _ax.set_title(f"{_tag} — 90% coverage: {_cov:.0%}")

    _fig.suptitle(
        "Friedman sanity check: BART is calibrated when the truth is known", y=1.02
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### How many trees? — choosing $m$

        BART has one knob whose right value is genuinely problem-dependent:
        $m$, the number of trees in the ensemble.  Quiroga *et al.* (2022)
        recommend $m \approx 50$ for exploration and variable-importance work
        and $m \approx 200$ for the final inference.  The intuition:

        * Each tree absorbs a *small* piece of $f(x)$.  More trees → finer
          decomposition → smoother fit, but with diminishing returns.
        * Linero & Yang (2018) show that as $m \to \infty$ the BART prior
          converges to a (nowhere-differentiable) Gaussian Process, which
          explains why "more trees" keeps helping for a while.
        * `pymc-bart` lets users compare $m$ values via PSIS-LOO-CV
          (`az.compare`); we'll skip that machinery here and just look at the
          σ posterior, in-sample coverage, and out-of-sample MSE on the
          Friedman DGP — three quantities that move together as $m$ grows.

        The cell below is **disabled by default** (it refits BART three times).
        Toggle the cell on to run the comparison.
        """
    )
    return


@app.cell(disabled=True, hide_code=True)
def _(
    X_fried,
    X_fried_test,
    friedman,
    np,
    plt,
    predict_at,
    run_bart,
    y_fried,
    y_fried_test_true,
):
    _ms = [10, 50, 200]
    _colors = ["#dd8452", "#4c72b0", "#c44e52"]
    _fits = {}
    for _m in _ms:
        _fits[_m] = run_bart(
            X_fried,
            y_fried,
            m=_m,
            n_iter=500,
            burn_in=200,
            thin=1,
            alpha=0.95,
            beta=2.0,
            k=2.0,
            nu=3.0,
            q=0.9,
            rng=np.random.default_rng(20260423),
        )

    _f_true_in = friedman(X_fried, noise=0.0)
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.8))

    for _m, _c in zip(_ms, _colors):
        _sd = _fits[_m]["sigma_draws"]
        _axes[0].hist(
            _sd,
            bins=30,
            density=True,
            alpha=0.5,
            color=_c,
            label=f"m={_m} (mean σ={_sd.mean():.2f})",
        )
    _axes[0].axvline(1.0, color="#333", ls="--", lw=1)
    _axes[0].set_xlabel(r"$\sigma$")
    _axes[0].set_ylabel("posterior density")
    _axes[0].set_title("σ posterior shrinks as m grows")
    _axes[0].legend(frameon=False, fontsize=8)

    _cov_in = []
    for _m in _ms:
        _lo = _fits[_m]["f_lo"]
        _hi = _fits[_m]["f_hi"]
        _cov_in.append(((_lo <= _f_true_in) & (_f_true_in <= _hi)).mean())
    _axes[1].bar([str(m) for m in _ms], _cov_in, color=_colors, edgecolor="white")
    _axes[1].axhline(0.9, color="#333", ls="--", lw=1, label="nominal 90%")
    _axes[1].set_ylim(0, 1.05)
    _axes[1].set_xlabel("m (number of trees)")
    _axes[1].set_ylabel("in-sample 90% coverage of f")
    _axes[1].set_title("Coverage stays calibrated")
    _axes[1].legend(frameon=False)

    _mse_oos = []
    for _m in _ms:
        _draws = predict_at(_fits[_m], X_fried_test)
        _f_mean = _draws.mean(axis=0)
        _mse_oos.append(float(np.mean((_f_mean - y_fried_test_true) ** 2)))
    _axes[2].bar([str(m) for m in _ms], _mse_oos, color=_colors, edgecolor="white")
    _axes[2].set_xlabel("m (number of trees)")
    _axes[2].set_ylabel("out-of-sample MSE")
    _axes[2].set_title("Test-set fit improves with m")

    _fig.suptitle("Choosing m on the Friedman DGP (n=100, σ=1)", y=1.02)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(Path, os, pl):
    # 2024 British Grand Prix lap data (FastF1 / F1 live timing API).
    # One row per clean racing lap for the 19 classified drivers. Silverstone
    # 2024 had a wet phase, so INTERMEDIATE tyres appear alongside the dry
    # SOFT/MEDIUM/HARD compounds — rich nonlinearity for BART to learn.
    # Source: https://github.com/theOehrly/Fast-F1 (cached under .cache/fastf1).
    # Pull script: scripts/pull_f1_laps.py
    def load_f1_laps():
        override = os.environ.get("F1_LAPS_CSV")
        candidates = []
        if override:
            candidates.append(Path(override))
        candidates.extend(
            [
                Path.cwd() / "data" / "f1_laps.csv",
                Path.home() / "repos" / "pydata_london_bart" / "data" / "f1_laps.csv",
            ]
        )
        for p in candidates:
            if p.exists():
                return pl.read_csv(p)
        raise FileNotFoundError(
            "f1_laps.csv not found. Set F1_LAPS_CSV env var or place the file "
            "at ./data/f1_laps.csv. Regenerate with scripts/pull_f1_laps.py "
            "(requires fastf1 + an internet connection on first run)."
        )

    f1_df = load_f1_laps()
    f1_df.shape
    return (f1_df,)


@app.cell
def _(f1_df, np):
    # Feature set: three in-race progress variables (tyre_life, lap_number, stint),
    # four weather variables (air_temp, track_temp, humidity, wind_speed), and
    # three compound indicators (MEDIUM is the baseline — it was the most-run
    # dry compound at Silverstone 2024). Target: lap time in seconds.
    _num_cols = [
        "tyre_life",
        "lap_number",
        "stint",
        "air_temp",
        "track_temp",
        "humidity",
        "wind_speed",
    ]
    # MEDIUM is the baseline; the three indicator columns capture the rest.
    _compounds_in_data = ["SOFT", "HARD", "INTERMEDIATE"]
    f1_feature_names = list(_num_cols) + [f"compound_{c}" for c in _compounds_in_data]

    _X_num = f1_df.select(_num_cols).to_numpy().astype(float)
    _X_cmp = np.column_stack(
        [(f1_df["compound"].to_numpy() == c).astype(float) for c in _compounds_in_data]
    )
    _X = np.concatenate([_X_num, _X_cmp], axis=1)
    _y = f1_df["lap_time_s"].to_numpy().astype(float)

    _n_train = 500
    _n_test = 200
    _rng_f1 = np.random.default_rng(20260423)
    _perm = _rng_f1.permutation(_X.shape[0])
    X_train = _X[_perm[:_n_train]]
    y_train = _y[_perm[:_n_train]]
    X_test = _X[_perm[_n_train : _n_train + _n_test]]
    y_test = _y[_perm[_n_train : _n_train + _n_test]]

    (
        f"n_train={X_train.shape[0]}, n_test={X_test.shape[0]}, "
        f"p={X_train.shape[1]}, "
        f"train lap time mean={y_train.mean():.2f}s (sd={y_train.std():.2f}s)"
    )
    return X_test, X_train, f1_feature_names, y_test, y_train


@app.cell
def _(X_train, np, run_bart, y_train):
    # Main §7 fit: paper defaults (m=200 trees, (ν,q,k) = (3, 0.9, 2))
    # on the wine-quality training set. 2000 kept draws after 1000 burn-in.
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
def _(X_train, fit, np, plt, predict_at, y_train):
    # In-sample fit on F1 lap times. Unlike the Friedman preamble we no longer
    # have a true f(x) — we compare the posterior predictive (f + σ·N(0,1))
    # against observed lap times in seconds, and look at the σ posterior on
    # its own (no reference line). σ is the residual driver/noise scale after
    # BART absorbs compound + weather + in-race progress signals.
    _f_mean = fit["f_mean"]
    _sigma_in = fit["sigma_draws"]

    # Posterior predictive draws for in-sample coverage: add σ to the f draws.
    _f_draws_in = predict_at(fit, X_train)
    _rng_pp = np.random.default_rng(20260423)
    _pp_draws_in = (
        _f_draws_in + _rng_pp.standard_normal(_f_draws_in.shape) * _sigma_in[:, None]
    )
    _pp_lo_in = np.quantile(_pp_draws_in, 0.05, axis=0)
    _pp_hi_in = np.quantile(_pp_draws_in, 0.95, axis=0)
    _pp_cov_in = ((_pp_lo_in <= y_train) & (y_train <= _pp_hi_in)).mean()

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(13, 4))

    # (a) Observed y vs posterior mean + 90% posterior predictive interval.
    _order = np.argsort(y_train)
    _ax1.errorbar(
        y_train[_order],
        _f_mean[_order],
        yerr=[(_f_mean - _pp_lo_in)[_order], (_pp_hi_in - _f_mean)[_order]],
        fmt="o",
        ms=3,
        ecolor="#4c72b0",
        color="#333",
        alpha=0.55,
        elinewidth=0.7,
    )
    _lim = (y_train.min() - 1, y_train.max() + 1)
    _ax1.plot(_lim, _lim, "--", color="C3", lw=1)
    _ax1.set_xlim(_lim)
    _ax1.set_ylim(_lim)
    _ax1.set_xlabel("observed lap time (s)")
    _ax1.set_ylabel(r"posterior mean $\hat f(x)$ with 90% PI")
    _ax1.set_title(f"In-sample fit — predictive coverage: {_pp_cov_in:.0%}")

    # (b) σ trace
    _ax2.plot(_sigma_in, ",", color="#333", alpha=0.4)
    _ax2.set_xlabel("MCMC iteration (post burn-in)")
    _ax2.set_ylabel(r"$\sigma$ draw (s)")
    _ax2.set_title(r"$\sigma$ trace — checks mixing")

    # (c) σ posterior density
    _ax3.hist(
        _sigma_in,
        bins=40,
        density=True,
        color="#4c72b0",
        edgecolor="white",
        alpha=0.85,
    )
    _ax3.set_xlabel(r"$\sigma$ (s)")
    _ax3.set_ylabel("posterior density")
    _ax3.set_title(
        rf"$\sigma$ posterior: {_sigma_in.mean():.2f}s "
        rf"[{np.quantile(_sigma_in, 0.05):.2f}, {np.quantile(_sigma_in, 0.95):.2f}]"
    )

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(X_test, fit, np, predict_at):
    f_test_draws = predict_at(fit, X_test)
    f_test_mean = f_test_draws.mean(axis=0)

    # Posterior predictive intervals = f draws + σ · N(0, 1) per posterior draw.
    _sigma_test = fit["sigma_draws"]
    _rng_oos = np.random.default_rng(20260423)
    pp_test_draws = (
        f_test_draws
        + _rng_oos.standard_normal(f_test_draws.shape) * _sigma_test[:, None]
    )
    pp_test_lo = np.quantile(pp_test_draws, 0.05, axis=0)
    pp_test_hi = np.quantile(pp_test_draws, 0.95, axis=0)
    return f_test_mean, pp_test_hi, pp_test_lo


@app.cell(hide_code=True)
def _(f_test_mean, np, plt, pp_test_hi, pp_test_lo, y_test):
    _covered = (pp_test_lo <= y_test) & (y_test <= pp_test_hi)

    _fig, _ax = plt.subplots(figsize=(6, 5))
    _order = np.argsort(y_test)
    _ax.errorbar(
        y_test[_order],
        f_test_mean[_order],
        yerr=[
            (f_test_mean - pp_test_lo)[_order],
            (pp_test_hi - f_test_mean)[_order],
        ],
        fmt="o",
        ms=4,
        ecolor="#4c72b0",
        color="#333",
        alpha=0.6,
        elinewidth=0.8,
    )
    _lim = (
        min(y_test.min(), f_test_mean.min()) - 1,
        max(y_test.max(), f_test_mean.max()) + 1,
    )
    _ax.plot(_lim, _lim, "--", color="C3", lw=1)
    _ax.set_xlim(_lim)
    _ax.set_ylim(_lim)
    _ax.set_xlabel("observed lap time (s) — held-out laps")
    _ax.set_ylabel(r"posterior mean $\hat f(x)$ with 90% posterior-predictive interval")
    _ax.set_title(f"Out-of-sample predictive coverage: {_covered.mean():.0%}")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(f1_feature_names, fit, np):
    # Average fraction of split rules using each variable, per CGM Eq. 20:
    # v_i = (1/K) Σ_k z_{ik}, where z_{ik} is the fraction of splits using var i.
    _total_per_iter = fit["splits"].sum(axis=1, keepdims=True)
    _total_per_iter = np.where(_total_per_iter == 0, 1, _total_per_iter)
    inclusion = (fit["splits"] / _total_per_iter).mean(axis=0)
    inclusion_labels = list(f1_feature_names)
    return inclusion, inclusion_labels


@app.cell(hide_code=True)
def _(inclusion, inclusion_labels, np, plt):
    _order = np.argsort(-inclusion)
    _fig, _ax = plt.subplots(figsize=(8, 3.8))
    _ax.bar(
        [inclusion_labels[i] for i in _order],
        inclusion[_order],
        color="#4c72b0",
        edgecolor="white",
    )
    _ax.set_ylabel("relative inclusion frequency")
    _ax.set_title("Variable importance — 2024 British GP lap times")
    _ax.tick_params(axis="x", labelrotation=45)
    for _lbl in _ax.get_xticklabels():
        _lbl.set_ha("right")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(X_train, f1_feature_names, fit, np, plt, predict_at):
    # Partial dependence (Eq. 19): f_s(x_s) = (1/n) Σ_i f(x_s, x_{i,c})
    # For each variable, sweep x_s across its quantile grid, fix all other
    # columns to training values, evaluate every posterior draw, aggregate.
    def partial_dependence(fit, X, var_idx, grid_size=20):
        xs = np.quantile(X[:, var_idx], np.linspace(0.05, 0.95, grid_size))
        pdp_draws = np.zeros((len(fit["tree_snapshots"]), grid_size))
        for k, x_s in enumerate(xs):
            X_rep = X.copy()
            X_rep[:, var_idx] = x_s
            draws = predict_at(fit, X_rep)  # (n_draws, n)
            pdp_draws[:, k] = draws.mean(axis=1)
        return xs, pdp_draws

    _n_feat = X_train.shape[1]
    _ncols = 4
    _nrows = (_n_feat + _ncols - 1) // _ncols
    _fig, _axes = plt.subplots(
        _nrows, _ncols, figsize=(3 * _ncols, 2.3 * _nrows), sharey=True
    )
    for v in range(_n_feat):
        ax = _axes.flat[v]
        xs, pdp_draws = partial_dependence(fit, X_train, v, grid_size=15)
        m_ = pdp_draws.mean(axis=0)
        lo, hi = np.quantile(pdp_draws, [0.05, 0.95], axis=0)
        ax.fill_between(xs, lo, hi, alpha=0.25, color="#4c72b0")
        ax.plot(xs, m_, color="#4c72b0")
        ax.set_title(f1_feature_names[v], fontsize=9)
        ax.set_xticks([])
    for v in range(_n_feat, _axes.size):
        _axes.flat[v].axis("off")
    _fig.suptitle("Partial dependence of lap time (s) on each feature", y=1.01)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### How many variables actually matter?

        The inclusion-frequency bar chart above ranks variables by how often
        BART chose to split on each one — useful, but not directly answering
        "would my predictions get worse if I dropped these last $p - k$
        variables?".

        Quiroga *et al.* (2022) propose a richer estimator that *uses the same
        posterior trees we already drew*: for each top-$k$ subset (top-1, then
        top-2, …, all $p$), prune posterior trees by collapsing every split on
        an excluded variable into a leaf carrying the mean $\mu$ of its old
        subtree, then predict through the pruned ensemble. The reported
        $R^2_k$ is the coefficient of determination between the full-model
        predictions and the restricted-model predictions:

        $$
        R^2_k \;=\; 1 - \frac{\operatorname{Var}\!\bigl(\hat f^{\text{full}}(x)
        - \hat f^{\text{restricted}}_k(x)\bigr)}
        {\operatorname{Var}\!\bigl(\hat f^{\text{full}}(x)\bigr)}.
        $$

        Read the curve from left to right: the point where it flattens tells
        you how many variables are doing real work.
        """
    )
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
def _(
    X_train,
    f1_feature_names,
    fit,
    inclusion,
    np,
    plt,
    predict,
    predict_at,
    prune_to_subset,
):
    # Top-k restricted-model R² (Quiroga §4.1). Use a subsample of posterior
    # draws to keep this evaluation under ~30s — the curve is stable.
    _ranking = list(np.argsort(-inclusion))
    _p = len(_ranking)

    _full_pred = predict_at(fit, X_train).mean(axis=0)

    _snaps = fit["tree_snapshots"]
    _n_sub = min(150, len(_snaps))
    _idx_sub = np.linspace(0, len(_snaps) - 1, _n_sub, dtype=int)

    _y_min, _y_range = fit["y_min"], fit["y_range"]
    _r2_curve = []
    for _k in range(1, _p + 1):
        _kept = _ranking[:_k]
        _restricted_draws = np.zeros((_n_sub, X_train.shape[0]))
        for _di, _snap_idx in enumerate(_idx_sub):
            _trees = fit["tree_snapshots"][_snap_idx]
            _f_scaled = np.zeros(X_train.shape[0])
            for _t in _trees:
                _t_pruned = prune_to_subset(_t, _kept)
                _f_scaled += predict(_t_pruned, X_train)
            _restricted_draws[_di] = _f_scaled * _y_range + (_y_min + 0.5 * _y_range)
        _restricted_mean = _restricted_draws.mean(axis=0)
        _r2 = 1.0 - np.var(_full_pred - _restricted_mean) / np.var(_full_pred)
        _r2_curve.append(float(_r2))

    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _xs = np.arange(1, _p + 1)
    _ax.plot(_xs, _r2_curve, "o-", color="#4c72b0", lw=1.6)
    _ax.axhline(0.95, color="#c44e52", ls="--", lw=1, label=r"$R^2 = 0.95$")
    _ax.set_xticks(_xs)
    _ax.set_xticklabels(
        [f1_feature_names[_ranking[i]] for i in range(_p)],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    _ax.set_xlabel("variables included (cumulative, by inclusion rank)")
    _ax.set_ylabel(r"$R^2$ vs. full-model predictions")
    _ax.set_title("Restricted-model R² — when does adding variables stop helping?")
    _ax.set_ylim(0, 1.02)
    _ax.legend(frameon=False, loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Sparsity in high-dimensional $X$ — Linero's prior

        The default BART prior draws splitting variables uniformly from
        $\{1, \dots, p\}$.  When most of those $p$ covariates are irrelevant
        — common in genomics, drug discovery, or any "throw everything at
        it" workflow — uniform sampling wastes proposal mass on noise.

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
        """
    )
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
    mo,
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

        for it in mo.status.progress_bar(range(n_iter), title="sparse-BART sampling"):
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
    mo.md(
        r"""
        #### Demo: Friedman with $p = 100$, only 5 relevant variables

        We simulate Friedman's function plus 95 columns of pure noise.  The
        plot below shows inclusion frequencies for two MH-BART runs on the
        same data: one with the uniform prior (left) and one with the
        adaptive Dirichlet prior (right).  The Dirichlet version should
        concentrate sharply on $X_0..X_4$, mirroring Quiroga *et al.* Figure 13.

        Cell is **disabled by default** (it fits two BART models on $p = 100$).
        """
    )
    return


@app.cell(disabled=True, hide_code=True)
def _(friedman, np, plt, run_bart, run_bart_sparse):
    _rng_sp = np.random.default_rng(20260423)
    _n_sp = 150
    _p_sp = 100
    _X_sp_relevant = _rng_sp.uniform(size=(_n_sp, 5))
    _X_sp_noise = _rng_sp.uniform(size=(_n_sp, _p_sp - 5))
    _X_sp = np.concatenate([_X_sp_relevant, _X_sp_noise], axis=1)
    _y_sp = friedman(_X_sp, noise=1.0, rng=_rng_sp)

    _kw = dict(
        m=20,
        n_iter=400,
        burn_in=200,
        alpha=0.95,
        beta=2.0,
        k=2.0,
        nu=3.0,
        q=0.9,
        rng=np.random.default_rng(20260423),
    )
    _fit_uni = run_bart(_X_sp, _y_sp, **_kw)
    _fit_sp = run_bart_sparse(_X_sp, _y_sp, dirichlet_a=1.0, **_kw)

    def _inclusion(splits):
        tot = splits.sum(axis=1, keepdims=True)
        tot = np.where(tot == 0, 1, tot)
        return (splits / tot).mean(axis=0)

    _inc_uni = _inclusion(_fit_uni["splits"])
    _inc_sp = _inclusion(_fit_sp["splits"])

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    _xs = np.arange(_p_sp)
    for _ax, _inc, _title in [
        (
            _axes[0],
            _inc_uni,
            f"uniform 1/p prior  (max irrelevant: {_inc_uni[5:].max():.3f})",
        ),
        (
            _axes[1],
            _inc_sp,
            f"Dirichlet a=1 prior  (max irrelevant: {_inc_sp[5:].max():.3f})",
        ),
    ]:
        _colors = ["#c44e52" if i < 5 else "#bbb" for i in _xs]
        _ax.bar(_xs, _inc, color=_colors, edgecolor="white")
        _ax.set_xlabel("variable index")
        _ax.set_title(_title)
    _axes[0].set_ylabel("inclusion frequency")
    _fig.suptitle(
        "Sparsity-inducing prior concentrates inclusion on the 5 relevant variables",
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
        z[pos] = truncnorm.rvs(a_pos, np.inf, loc=G[pos], scale=1.0, random_state=rng)
        # Negative cases: truncate right at 0
        neg = ~pos
        b_neg = (0.0 - G[neg]) / 1.0
        z[neg] = truncnorm.rvs(-np.inf, b_neg, loc=G[neg], scale=1.0, random_state=rng)
        return z

    return (sample_latent_z,)


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

        for it in mo.status.progress_bar(range(n_iter), title="BART probit sampling"):
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

                trees[j] = draw_leaf_values(trees[j], X, Rj, rng, sigma2, sigma_mu2)
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
    _ax.fill_between(times, _lo_low, _hi_low, step="post", color="#4c72b0", alpha=0.25)
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
    import pymc as pm
    import pymc_bart as pmb
    import arviz as az

    return az, pm, pmb


@app.cell
def _(X_fried, pm, pmb, y_fried):
    with pm.Model():
        # m=200 matches the bespoke Friedman preamble above so this is a like-for-like
        # comparison on the same synthetic data with known σ = 1.
        μ_bart = pmb.BART("μ", X=X_fried, Y=y_fried, m=200)
        σ_reg = pm.HalfNormal("σ", 1.0)
        pm.Normal("y", mu=μ_bart, sigma=σ_reg, observed=y_fried)
        idata_fried = pm.sample(
            draws=500,
            tune=500,
            chains=2,
            cores=1,
            random_seed=20260423,
            progressbar=False,
        )
    return (idata_fried,)


@app.cell(hide_code=True)
def _(az, fit_fried, idata_fried, np, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 3.5))
    _sigma_bespoke = fit_fried["sigma_draws"]
    _sigma_pymc = np.asarray(az.extract(idata_fried, var_names="σ").values).ravel()
    _bins = np.linspace(
        min(_sigma_bespoke.min(), _sigma_pymc.min()) * 0.9,
        max(_sigma_bespoke.max(), _sigma_pymc.max()) * 1.05,
        40,
    )
    _ax.hist(
        _sigma_bespoke,
        bins=_bins,
        alpha=0.55,
        density=True,
        label=f"bespoke preamble (n={_sigma_bespoke.size})",
        color="#4c72b0",
    )
    _ax.hist(
        _sigma_pymc,
        bins=_bins,
        alpha=0.55,
        density=True,
        label=f"pymc-bart (n={_sigma_pymc.size})",
        color="#c44e52",
    )
    _ax.axvline(1.0, color="#333", ls="--", lw=1, label=r"true $\sigma = 1$")
    _ax.set_xlabel(r"$\sigma$")
    _ax.set_ylabel("posterior density")
    _ax.set_title("σ posterior on Friedman — bespoke preamble vs. pymc-bart")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(np):
    import os
    from pathlib import Path
    import polars as pl

    def load_gss():
        override = os.environ.get("GSS_CSV")
        candidates = []
        if override:
            candidates.append(Path(override))
        candidates.extend(
            [
                Path.cwd() / "data" / "gss_2022.csv",
                Path.home() / "repos" / "Koenigsberg_Bayes" / "data" / "gss_2022.csv",
            ]
        )
        for p in candidates:
            if p.exists():
                return pl.read_csv(p)
        raise FileNotFoundError(
            "gss_2022.csv not found. Set GSS_CSV env var, place the file at "
            "./data/gss_2022.csv, or clone the Koenigsberg_Bayes repo."
        )

    _gss_raw = load_gss()
    # Response + predictors
    _cont = ["age"]
    _ordinal = ["stress", "feelnerv", "worry", "anxiety", "finrela"]
    _categ = ["sex", "degree", "race", "relig"]
    _cols = ["satjob"] + _cont + _ordinal + _categ

    _df = _gss_raw.select(_cols).drop_nulls()
    y_ord = _df["satjob"].to_numpy().astype(int) - 1  # shift to {0,1,2,3}

    # One-hot encode the low-card categoricals; keep continuous + ordinal as-is.
    _X_parts = [_df[_cont + _ordinal].to_numpy().astype(float)]
    for c in _categ:
        _dummies = _df[c].to_dummies(drop_first=True).to_numpy().astype(float)
        _X_parts.append(_dummies)
    X_ord = np.concatenate(_X_parts, axis=1)

    f"n={len(y_ord)}, p={X_ord.shape[1]}, classes={np.bincount(y_ord).tolist()}"
    return Path, X_ord, os, pl, y_ord


@app.cell
def _(X_ord, np, pm, pmb, y_ord):
    with pm.Model() as model_sat:
        η = pmb.BART("η", X=X_ord, Y=y_ord.astype(float), m=50)
        γ_free = pm.Normal(
            "γ_free",
            mu=np.array([1.0, 2.0]),
            sigma=1.0,
            size=2,
            transform=pm.distributions.transforms.ordered,
            initval=np.array([1.0, 2.0]),
        )
        cutpoints = pm.Deterministic("cutpoints", pm.math.concatenate([[0.0], γ_free]))
        pm.OrderedProbit(
            "y", eta=η, cutpoints=cutpoints, observed=y_ord, compute_p=False
        )
        idata_sat = pm.sample(
            draws=500,
            tune=500,
            chains=2,
            cores=1,
            random_seed=20260423,
            progressbar=False,
        )
    return (idata_sat,)


@app.cell
def _(az, idata_sat):
    _summary = az.summary(
        idata_sat,
        var_names=["γ_free"],
        round_to=3,
    )
    _summary
    return


@app.cell(hide_code=True)
def _(X_ord, idata_sat, pmb):
    # Variable importance: ranks predictor columns by inclusion frequency.
    _ax = pmb.plot_variable_importance(
        idata_sat,
        bartrv=idata_sat.posterior["η"],
        X=X_ord,
    )
    _ax
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
