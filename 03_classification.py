import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # BART for classification

        Binary classification with BART means modelling
        $\Pr(Y = 1 \mid x) = \Phi\bigl(g(x)\bigr)$, where $g$ is a sum of
        trees and $\Phi$ is the standard-normal CDF (the probit link). The
        BART prior shrinks $g$ toward zero, which corresponds to a
        baseline probability of $1/2$ — useful regularisation when the
        signal is weak.

        Two demos:

        1. **Synthetic binary data** with known true probabilities — lets
           us check calibration directly.
        2. **GSS 2022 ordinal outcome** — life satisfaction modelled
           with an ordered probit, demonstrating that BART composes
           naturally with PyMC's other likelihoods.
        """
    )
    return


@app.cell
def _():
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)

    import os
    from pathlib import Path

    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import pymc as pm
    import pymc_bart as pmb

    RANDOM_SEED = 20260608
    rng = np.random.default_rng(RANDOM_SEED)
    return Path, RANDOM_SEED, az, np, os, pl, plt, pm, pmb, rng


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
def _(RANDOM_SEED, X_cls, pm, pmb, y_cls):
    # Probit BART: BART output enters Bernoulli through invprobit (the
    # standard-normal CDF, which maps a real-valued BART score onto a
    # probability in [0, 1]). Probit pairs naturally with BART's Gaussian
    # leaf prior; logit would also work but introduces an extra scale
    # parameter. We wrap X in pm.Data so we can swap it for out-of-sample
    # predictions — see the "PyMC patterns for prediction" aside in 02.
    with pm.Model() as model_cls:
        X_data = pm.Data("X_data", X_cls)
        eta = pmb.BART("eta", X=X_data, Y=y_cls.astype(float), m=100)
        p = pm.Deterministic("p", pm.math.invprobit(eta))
        pm.Bernoulli("y", p=p, observed=y_cls, shape=p.shape)
        idata_cls = pm.sample(random_seed=RANDOM_SEED)
    return eta, idata_cls, model_cls


@app.cell
def _(RANDOM_SEED, X_cls_test, idata_cls, model_cls, pm):
    with model_cls:
        pm.set_data({"X_data": X_cls_test})
        pp_cls = pm.sample_posterior_predictive(
            idata_cls,
            var_names=["p"],
            sample_vars=["eta"],
            predictions=True,
            random_seed=RANDOM_SEED,
        )
    return (pp_cls,)


@app.cell(hide_code=True)
def _(np, plt, pp_cls, prob_test_true, y_cls_test):
    _p_draws = pp_cls.predictions["p"].stack(sample=("chain", "draw")).values
    _p_mean = _p_draws.mean(axis=1)
    _p_lo, _p_hi = np.quantile(_p_draws, [0.05, 0.95], axis=1)

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 4.0))
    _order = np.argsort(prob_test_true)
    _a.errorbar(
        prob_test_true[_order],
        _p_mean[_order],
        yerr=[(_p_mean - _p_lo)[_order], (_p_hi - _p_mean)[_order]],
        fmt=".",
        ms=3,
        alpha=0.5,
        elinewidth=0.7,
        color="#4c72b0",
    )
    _a.plot([0, 1], [0, 1], "--", color="C3", lw=1)
    _a.set_xlabel(r"true $\Pr(Y=1 \mid x)$")
    _a.set_ylabel("posterior mean (90% CI)")
    _a.set_title("Calibration on the test set")

    _top = np.argsort(-_p_mean)[:20]
    _pos = np.arange(len(_top))
    _colors = ["C2" if y_cls_test[i] == 1 else "C3" for i in _top]
    _b.errorbar(
        _pos,
        _p_mean[_top],
        yerr=[(_p_mean - _p_lo)[_top], (_p_hi - _p_mean)[_top]],
        fmt="none",
        ecolor="#888",
        elinewidth=0.8,
    )
    _b.scatter(_pos, _p_mean[_top], c=_colors, s=35, zorder=3)
    _b.set_xticks(_pos)
    _b.set_xticklabels([])
    _b.set_xlabel("top-20 ranked test cases")
    _b.set_ylabel(r"$\Pr(Y=1 \mid x)$ with 90% CI")
    _b.set_title(
        f"Top-20 hit rate: {y_cls_test[_top].sum()}/20  "
        f"(base rate {y_cls_test.mean():.0%})"
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(X_cls, eta, idata_cls, model_cls, pm, pmb):
    # Variable importance: only X_0, X_1, X_2 generated the labels in
    # simulate_binary. The plot should rank those three at the top. We
    # reset X_data first because the OOS-prediction cell may have mutated
    # it.
    with model_cls:
        pm.set_data({"X_data": X_cls})
        _vi_cls = pmb.compute_variable_importance(idata_cls, eta, X_cls)
    pmb.plot_variable_importance(_vi_cls)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Backward variable importance and submodel agreement

        `compute_variable_importance` has two ranking methods:

        - `method="VI"` (the default, used above) ranks features by what
          they *add* when included. This is a forward selection view built
          from partial dependence variance.
        - `method="backward"` ranks features by what is *lost* when each is
          removed. This is a backward elimination view. The two rankings
          usually agree on the top features but disagree in the long tail.

        `pmb.plot_scatter_submodels` then visualises the agreement. A
        **submodel** here is the full BART fit restricted to a prefix of
        the importance ranking (the top-1 feature, then top-2, then
        top-3, etc.). The grid below shows one panel per submodel; each
        panel plots the submodel's predictions (x-axis) against the
        full-model predictions (y-axis). Panels that hug the diagonal
        mean the submodel's features are sufficient — adding the
        remaining features wouldn't change predictions much. That's how
        you justify stopping at $k$ features rather than carrying all
        $p$.
        """
    )
    return


@app.cell(hide_code=True)
def _(X_cls, eta, idata_cls, model_cls, pm, pmb):
    with model_cls:
        pm.set_data({"X_data": X_cls})
        _vi_backward = pmb.compute_variable_importance(
            idata_cls, eta, X_cls, method="backward"
        )
    pmb.plot_scatter_submodels(_vi_backward)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Sparse splitting via `split_prior`

        BART's default prior picks a splitting variable uniformly at random
        from the $p$ available features. When most features are noise
        (3-of-20 here), uniform sampling wastes tree capacity exploring
        irrelevant directions. `split_prior` lets you bias that choice:
        pass a length-$p$ array of positive weights, and each split draws
        a variable with probability proportional to its weight. Higher
        weight $\Rightarrow$ higher selection probability; the default is
        uniform.

        Below we upweight the truly relevant features
        $\{X_0, X_1, X_2\}$ by a factor of 10, i.e. `split_prior =
        np.array([10., 10., 10.] + [1.]*17)`. This is the Linero (2018)
        sparse splitting prior, hand-tuned here for *demonstrating the
        interface*. In practice you obtain the weights from domain
        knowledge or from a first-pass inclusion analysis (and Linero's
        full method places a Dirichlet prior on the weights and learns
        them; see notebook 1's `run_bart_sparse` implementation).

        We compare inclusion frequencies (how often each variable is
        chosen as a splitting variable, averaged over posterior trees)
        between the uniform-prior fit above and the sparse-prior fit,
        via `pmb.plot_variable_inclusion`. Inclusion frequency is the
        raw counting statistic — distinct from the restricted-$R^2$
        "variable importance" plot earlier, which weighs each feature
        by how much it actually moves predictions.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, X_cls, np, pm, pmb, y_cls):
    # Same model spec as model_cls, but with split_prior upweighting
    # X_0..X_2 10x.
    _split_prior = np.array([10.0, 10.0, 10.0] + [1.0] * 17)
    with pm.Model() as model_cls_sp:
        X_data_sp = pm.Data("X_data", X_cls)
        eta_sp = pmb.BART(
            "eta", X=X_data_sp, Y=y_cls.astype(float), m=100, split_prior=_split_prior
        )
        p_sp = pm.Deterministic("p", pm.math.invprobit(eta_sp))
        pm.Bernoulli("y", p=p_sp, observed=y_cls, shape=p_sp.shape)
        idata_cls_sp = pm.sample(random_seed=RANDOM_SEED)
    return eta_sp, idata_cls_sp, model_cls_sp


@app.cell(hide_code=True)
def _(X_cls, idata_cls, idata_cls_sp, plt, pmb):
    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(13, 4.0), sharey=True)
    pmb.plot_variable_inclusion(idata_cls, X_cls, ax=_ax0)
    _ax0.set_title("Uniform split prior (baseline)")
    pmb.plot_variable_inclusion(idata_cls_sp, X_cls, ax=_ax1)
    _ax1.set_title(r"Sparse $split\_prior = [10,10,10,1,\ldots,1]$")
    _fig.suptitle(
        "Inclusion frequencies: sparse prior concentrates splits on $X_0..X_2$",
        y=1.02,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(az, idata_cls):
    # BART RVs need different convergence diagnostics than scalars:
    # plot_convergence_dist shows ECDFs of ESS and R-hat across every node
    # of the BART variable.
    az.plot_convergence_dist(idata_cls, var_names=["eta"])
    return


@app.cell
def _(idata_cls):
    _stats = idata_cls.sample_stats
    (
        f"divergences: {int(_stats['diverging'].sum())}"
        if "diverging" in _stats
        else "PGBART-only model (no HMC step) — divergence diagnostic not applicable"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Ordinal outcome: GSS 2022 life satisfaction

        BART composes with any PyMC likelihood. Here we model life
        satisfaction (`lifenow`, originally 1–10) as an ordered probit
        whose latent score is BART. Categories 1–3 are nearly empty
        (≈20 obs combined), so we collapse 1–4 into a single "low"
        bucket, giving **$K = 7$** categories: `<=4, 5, 6, 7, 8, 9, 10`.

        Predictors: age, financial-relative-position scale, education,
        and five self-reported mental-health/wellbeing scales (anxiety,
        work-meaningfulness, stress, feeling-nervous, worry); plus sex
        (binary), fulltime employment indicator (binary), race (multi-
        level), and religion (grouped into 5 buckets — None, Protestant,
        Catholic, Christian-other, Non-Christian + residual). **Twelve
        features total.**

        The cutpoints are estimated jointly with the trees: the first
        cutpoint is fixed at zero for identifiability; the remaining
        $K - 2 = 5$ are constrained ordered via
        `pm.distributions.transforms.ordered`.
        """
    )
    return


@app.cell
def _(Path, np, os, pl):
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

    def _int_code(col):
        # Map any 1-D array of category labels to contiguous integer
        # codes 0..K-1. SubsetSplitRule operates on these directly.
        _, codes = np.unique(col, return_inverse=True)
        return codes.astype(float)

    def _relig_group(raw):
        # Collapse the GSS `relig` codes into 5 buckets so the SubsetSplitRule
        # has interpretable levels: None=0 (reference), Protestant=1,
        # Catholic=2, Christian-other={10,11,13}=3, Non-Christian={3,5,6,7,
        # 8,9,12}=4. Anything else falls into a residual bucket=5.
        out = np.full(raw.shape, 5, dtype=int)
        out[raw == 4] = 0
        out[raw == 1] = 1
        out[raw == 2] = 2
        out[np.isin(raw, [10, 11, 13])] = 3
        out[np.isin(raw, [3, 5, 6, 7, 8, 9, 12])] = 4
        return out.astype(float)

    _kept_cols = [
        "lifenow",
        "age",
        "finrela",
        "degree",
        "anxiety",
        "wrkmeangfl",
        "stress",
        "feelnerv",
        "worry",
        "sex",
        "wrkstat",
        "race",
        "relig",
    ]
    _gss = (
        load_gss()
        .select(_kept_cols)
        .filter(pl.col("lifenow").is_between(1, 10))
        .drop_nulls()
    )

    # Collapse lifenow 1-4 into a single low bucket -> 7 categories
    # indexed 0..6 (categories 1-3 hold ~20 obs combined and would
    # destabilise the ordered probit).
    y_ord = np.clip(_gss["lifenow"].to_numpy().astype(int), 4, None) - 4
    K_ord = 7

    feature_names_ord = [
        "age",
        "finrela",
        "degree",
        "anxiety",
        "wrkmeangfl",
        "stress",
        "feelnerv",
        "worry",
        "sex",
        "fulltime",
        "race",
        "relig",
    ]

    # 8 continuous/ordinal columns (age + 7 ordinal scales treated as
    # equal-interval), then 2 binary indicators (sex=female, fulltime
    # = wrkstat==1), then 2 multi-level unordered categoricals
    # (race integer-coded; relig grouped via _relig_group).
    X_ord = np.column_stack(
        [
            _gss["age"].to_numpy().astype(float),
            _gss["finrela"].to_numpy().astype(float),
            _gss["degree"].to_numpy().astype(float),
            _gss["anxiety"].to_numpy().astype(float),
            _gss["wrkmeangfl"].to_numpy().astype(float),
            _gss["stress"].to_numpy().astype(float),
            _gss["feelnerv"].to_numpy().astype(float),
            _gss["worry"].to_numpy().astype(float),
            (_gss["sex"].to_numpy() == 2).astype(float),
            (_gss["wrkstat"].to_numpy() == 1).astype(float),
            _int_code(_gss["race"].to_numpy()),
            _relig_group(_gss["relig"].to_numpy()),
        ]
    )

    f"n={len(y_ord)}, p={X_ord.shape[1]}, K={K_ord}, classes={np.bincount(y_ord).tolist()}"
    return K_ord, X_ord, feature_names_ord, y_ord


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### `SubsetSplitRule` for multi-level categoricals

        BART's split rules are per-column. Three are bundled with
        `pymc-bart`:

        - `ContinuousSplitRule` (default): splits on $x_j \le c$ for some
          cut value $c$. Correct for numeric or ordinal predictors.
        - `OneHotSplitRule`: splits on $x_j = k$ for a single level $k$.
          Correct for binary indicators (e.g. `sex`).
        - `SubsetSplitRule`: splits on $x_j \in S$ for an arbitrary
          subset $S \subset \{0, 1, \ldots, K-1\}$ of the $K$ levels.
          Correct for unordered categoricals with $K > 2$ levels
          (e.g. `degree`, `race`, `relig`).

        Without `SubsetSplitRule`, a $K$-level unordered categorical
        either has to be one-hot expanded (which inflates $p$ and forces
        BART to chain $K-1$ binary splits across multiple depth levels to
        recover a single subset split), or treated as continuous (which
        imposes a spurious ordering). `SubsetSplitRule` does the right
        thing in a single node.

        Here `split_rules` is a length-$p$ list with one rule per column,
        applied to the design matrix from the previous cell whose layout
        is `[age, finrela, degree, anxiety, wrkmeangfl, stress, feelnerv,
        worry, sex, fulltime, race, relig]` — 8 `ContinuousSplitRule`,
        2 `OneHotSplitRule` (binary indicators), 2 `SubsetSplitRule`
        (multi-level unordered).

        The ordered-probit cutpoints below need an identifying constraint.
        We fix $\gamma_0 = 0$ and place a Normal prior on $(\gamma_1,
        \gamma_2)$ transformed by `pm.distributions.transforms.ordered`,
        which reparameterises the pair onto the constrained simplex
        $\gamma_1 < \gamma_2$. `initval` gives the chain a starting point
        already inside the constraint (a non-ordered start would error
        before the first NUTS step). `compute_p=False` tells PyMC not to
        materialise the per-category probabilities at each draw — we only
        need the cutpoints and the likelihood, and the probabilities are
        cheap to recompute post-hoc.
        """
    )
    return


@app.cell
def _(K_ord, RANDOM_SEED, X_ord, np, pm, pmb, y_ord):
    # 8 continuous/ordinal columns, then 2 binary OneHot (sex, fulltime),
    # then 2 multi-level Subset (race, relig).
    _split_rules = (
        [pmb.ContinuousSplitRule] * 8
        + [pmb.OneHotSplitRule] * 2
        + [pmb.SubsetSplitRule] * 2
    )
    with pm.Model() as model_ord:
        eta_ord = pmb.BART(
            "eta",
            X=X_ord,
            Y=y_ord.astype(float),
            m=100,
            split_rules=_split_rules,
        )
        gamma_free = pm.Normal(
            "gamma_free",
            mu=np.arange(1, K_ord - 1, dtype=float),
            sigma=1.0,
            size=K_ord - 2,
            transform=pm.distributions.transforms.ordered,
            initval=np.arange(1, K_ord - 1, dtype=float),
        )
        cutpoints = pm.Deterministic(
            "cutpoints", pm.math.concatenate([[0.0], gamma_free])
        )
        pm.OrderedProbit(
            "y", eta=eta_ord, cutpoints=cutpoints, observed=y_ord, compute_p=False
        )
        # target_accept=0.95: ordered-probit + BART geometry has funnel-
        # like ridges; the higher target keeps divergences down.
        idata_ord = pm.sample(random_seed=RANDOM_SEED, target_accept=0.95)
    return eta_ord, idata_ord, model_ord


@app.cell
def _(az, idata_ord):
    az.summary(idata_ord, var_names=["gamma_free", "cutpoints"], round_to=3)
    return


@app.cell(hide_code=True)
def _(X_ord, eta_ord, feature_names_ord, idata_ord, model_ord, pmb):
    with model_ord:
        _vi_ord = pmb.compute_variable_importance(idata_ord, eta_ord, X_ord)
    pmb.plot_variable_importance(_vi_ord, labels=feature_names_ord)
    return


@app.cell(hide_code=True)
def _(az, idata_ord):
    az.plot_convergence_dist(idata_ord, var_names=["eta"])
    return


@app.cell
def _(az, idata_ord):
    _n_div = int(idata_ord.sample_stats["diverging"].sum())
    _summary = az.summary(idata_ord, var_names=["gamma_free"], round_to=3)
    f"divergences: {_n_div}   |   gamma_free max R-hat = {_summary['r_hat'].max():.3f}, min ESS = {_summary['ess_bulk'].min():.0f}"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Posterior predictive check

        Sample `y` draws from the fitted ordered probit and compare the
        per-category counts to the observed histogram. If the model
        captures the marginal distribution the observed counts (red dots)
        sit inside the 90% predictive band (blue error bars).
        """
    )
    return


@app.cell
def _(RANDOM_SEED, idata_ord, model_ord, pm):
    with model_ord:
        ppc_ord = pm.sample_posterior_predictive(
            idata_ord,
            var_names=["y"],
            sample_vars=["eta"],
            random_seed=RANDOM_SEED,
        )
    return (ppc_ord,)


@app.cell(hide_code=True)
def _(K_ord, np, plt, ppc_ord, y_ord):
    _y_rep = ppc_ord.posterior_predictive["y"].values.reshape(-1, len(y_ord))
    _obs = np.bincount(y_ord, minlength=K_ord)
    _rep = np.stack([np.bincount(row, minlength=K_ord) for row in _y_rep], axis=0)
    _mean = _rep.mean(axis=0)
    _lo, _hi = np.quantile(_rep, [0.05, 0.95], axis=0)

    _fig, _ax = plt.subplots(figsize=(7.5, 4.0))
    _pos = np.arange(K_ord)
    _ax.errorbar(
        _pos,
        _mean,
        yerr=[_mean - _lo, _hi - _mean],
        fmt="none",
        ecolor="#4c72b0",
        elinewidth=1.5,
        capsize=4,
        label="posterior predictive (90% band)",
    )
    _ax.scatter(_pos, _obs, color="C3", zorder=3, label="observed")
    _ax.set_xticks(_pos)
    _ax.set_xticklabels(["<=4", "5", "6", "7", "8", "9", "10"])
    _ax.set_xlabel("lifenow category (after collapse)")
    _ax.set_ylabel("count")
    _ax.set_title("PPC: category counts")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Partial dependence

        PDP for the five continuous/ordinal predictors (age and four
        wellbeing scales). The y-axis is the latent $\eta$ (probit
        scale), not a probability — interpret signs and relative
        magnitudes, not absolute levels. `pmb.plot_pdp` reads feature
        names from a duck-typed DataFrame's `.columns`, so we pass a
        polars `DataFrame` view of `X_ord`.
        """
    )
    return


@app.cell(hide_code=True)
def _(X_ord, eta_ord, feature_names_ord, pl, pmb):
    _X_df = pl.DataFrame(X_ord, schema=feature_names_ord)
    _axes = pmb.plot_pdp(
        eta_ord,
        X=_X_df,
        Y=None,
        xs_interval="quantiles",
        var_idx=[0, 1, 2, 3, 4],
        var_discrete=[1, 2, 3, 4],
        figsize=(12, 6.5),
        sharey=True,
        color="C0",
    )
    _axes[0].figure.tight_layout()
    _axes[0].figure
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
