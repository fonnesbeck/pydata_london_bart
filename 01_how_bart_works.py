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

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import chi2, norm, truncnorm

    rng = np.random.default_rng(20260423)
    plt.rcParams["figure.dpi"] = 110
    return chi2, np, plt, rng


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
        n_iter=2000,
        burn_in=100,
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
    mo.md(r"""
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
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How many variables actually matter?

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
    R^2_k \;=\; 1 - \frac{\operatorname{Var}\!\bigl(\hat f^{\text{full}}(x)
    - \hat f^{\text{restricted}}_k(x)\bigr)}
    {\operatorname{Var}\!\bigl(\hat f^{\text{full}}(x)\bigr)}.
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
def _(X_fried, fit_fried, np, plt, predict, predict_at, prune_to_subset):
    # Restricted-model R² on the Friedman fit. Variables are ranked by
    # inclusion frequency, then for each top-k subset we collapse splits on
    # excluded variables and recompute predictions. The curve flattens at
    # k=5 — the true number of relevant variables in Friedman's DGP.
    _splits = fit_fried["splits"]
    _tot = _splits.sum(axis=1, keepdims=True)
    _tot = np.where(_tot == 0, 1, _tot)
    _inclusion = (_splits / _tot).mean(axis=0)
    _ranking = list(np.argsort(-_inclusion))
    _p = len(_ranking)

    _full_pred = predict_at(fit_fried, X_fried).mean(axis=0)

    _snaps = fit_fried["tree_snapshots"]
    _n_sub = min(150, len(_snaps))
    _idx_sub = np.linspace(0, len(_snaps) - 1, _n_sub, dtype=int)

    _y_min = fit_fried["y_min"]
    _y_range = fit_fried["y_range"]
    _r2_curve = []
    for _k in range(1, _p + 1):
        _kept = _ranking[:_k]
        _restricted_draws = np.zeros((_n_sub, X_fried.shape[0]))
        for _di, _snap_idx in enumerate(_idx_sub):
            _trees = _snaps[_snap_idx]
            _f_scaled = np.zeros(X_fried.shape[0])
            for _t in _trees:
                _t_pruned = prune_to_subset(_t, _kept)
                _f_scaled += predict(_t_pruned, X_fried)
            _restricted_draws[_di] = _f_scaled * _y_range + (_y_min + 0.5 * _y_range)
        _restricted_mean = _restricted_draws.mean(axis=0)
        _r2 = 1.0 - np.var(_full_pred - _restricted_mean) / np.var(_full_pred)
        _r2_curve.append(float(_r2))

    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _xs = np.arange(1, _p + 1)
    _ax.plot(_xs, _r2_curve, "o-", color="#4c72b0", lw=1.6)
    _ax.axhline(0.95, color="#c44e52", ls="--", lw=1, label=r"$R^2 = 0.95$")
    _ax.set_xticks(_xs)
    _ax.set_xticklabels([f"$X_{{{_ranking[i]}}}$" for i in range(_p)])
    _ax.set_xlabel("variables included (cumulative, by inclusion rank)")
    _ax.set_ylabel(r"$R^2$ vs. full-model predictions")
    _ax.set_title(
        "Restricted-model R² on Friedman — flattens at the 5 relevant variables"
    )
    _ax.set_ylim(0, 1.02)
    _ax.legend(frameon=False, loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    mo.md(r"""
    #### Demo: Friedman with $p = 100$, only 5 relevant variables

    We simulate Friedman's function plus 95 columns of pure noise.  The
    plot below shows inclusion frequencies for two MH-BART runs on the
    same data: one with the uniform prior (left) and one with the
    adaptive Dirichlet prior (right).  The Dirichlet version should
    concentrate sharply on $X_0..X_4$, mirroring Quiroga *et al.* Figure 13.

    Cell is **disabled by default** (it fits two BART models on $p = 100$).
    """)
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
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
