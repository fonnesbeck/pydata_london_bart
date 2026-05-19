import marimo

__generated_with = "0.21.1"
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
        2. **GSS 2022 ordinal outcome** — job satisfaction modelled with
           an ordered probit, demonstrating that BART composes naturally
           with PyMC's other likelihoods.
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
    # Probit BART: BART output enters Bernoulli through invprobit. We wrap
    # X in pm.Data so we can swap it for out-of-sample predictions.
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

        `pmb.plot_scatter_submodels` then visualises the agreement: each
        panel plots the full-model predictions against a pruned submodel's
        predictions. The submodels that hug the diagonal are the ones
        whose feature subsets are sufficient. That is how you justify
        stopping at $k$ features rather than carrying all $p$.
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
        chosen as a splitting variable) between the uniform-prior fit
        above and the sparse-prior fit, via `pmb.plot_variable_inclusion`.
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
        ## Ordinal outcome: GSS 2022 job satisfaction

        BART composes with any PyMC likelihood. Here we model job
        satisfaction (`satjob`, four ordered levels) as an ordered probit
        whose latent score is BART. Predictors: age, five self-reported
        anxiety/stress/finance scales, and four unordered categoricals
        (sex, degree, race, religion) kept as integer-coded columns so
        the BART splitter can apply category-aware split rules to them.

        The cutpoints are estimated jointly with the trees: the first
        cutpoint is fixed at zero for identifiability; the remaining two
        are constrained ordered.
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

    def _integer_code(col):
        # Map any 1-D array of category labels to contiguous integer
        # codes 0..K-1. SubsetSplitRule operates on these integer codes
        # directly; no need to one-hot expand.
        _, codes = np.unique(col, return_inverse=True)
        return codes.astype(float)

    _gss_raw = load_gss()
    _cont = ["age"]
    _ordinal = ["stress", "feelnerv", "worry", "anxiety", "finrela"]
    _categ = ["sex", "degree", "race", "relig"]
    _cols = ["satjob"] + _cont + _ordinal + _categ

    _df = _gss_raw.select(_cols).drop_nulls()
    y_ord = _df["satjob"].to_numpy().astype(int) - 1

    # Keep age + ordinal scales as continuous; integer-code the unordered
    # categoricals (sex, degree, race, relig) so SubsetSplitRule can act
    # on them directly. n_cont = 6 (age + 5 ordinal columns treated as
    # continuous).
    _X_cont_ordinal = _df[_cont + _ordinal].to_numpy().astype(float)
    _X_cat = np.column_stack([_integer_code(_df[c].to_numpy()) for c in _categ])
    X_ord = np.concatenate([_X_cont_ordinal, _X_cat], axis=1)
    n_cont = _X_cont_ordinal.shape[1]

    f"n={len(y_ord)}, p={X_ord.shape[1]}, n_continuous={n_cont}, classes={np.bincount(y_ord).tolist()}"
    return X_ord, n_cont, y_ord


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
        is `[age, stress, feelnerv, worry, anxiety, finrela, sex,
        degree, race, relig]`.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, X_ord, n_cont, np, pm, pmb, y_ord):
    # 6 continuous (age + 5 ordinal scales), then OneHot for binary sex,
    # then SubsetSplitRule for the three multi-level unordered
    # categoricals (degree, race, relig).
    _split_rules = (
        [pmb.ContinuousSplitRule] * n_cont
        + [pmb.OneHotSplitRule]
        + [pmb.SubsetSplitRule] * 3
    )
    with pm.Model() as model_sat:
        eta_sat = pmb.BART(
            "eta",
            X=X_ord,
            Y=y_ord.astype(float),
            m=100,
            split_rules=_split_rules,
        )
        gamma_free = pm.Normal(
            "gamma_free",
            mu=np.array([1.0, 2.0]),
            sigma=1.0,
            size=2,
            transform=pm.distributions.transforms.ordered,
            initval=np.array([1.0, 2.0]),
        )
        cutpoints = pm.Deterministic(
            "cutpoints", pm.math.concatenate([[0.0], gamma_free])
        )
        pm.OrderedProbit(
            "y", eta=eta_sat, cutpoints=cutpoints, observed=y_ord, compute_p=False
        )
        idata_sat = pm.sample(random_seed=RANDOM_SEED)
    return eta_sat, idata_sat, model_sat


@app.cell
def _(az, idata_sat):
    az.summary(idata_sat, var_names=["gamma_free", "cutpoints"], round_to=3)
    return


@app.cell(hide_code=True)
def _(X_ord, eta_sat, idata_sat, model_sat, pmb):
    with model_sat:
        _vi_sat = pmb.compute_variable_importance(idata_sat, eta_sat, X_ord)
    pmb.plot_variable_importance(_vi_sat)
    return


@app.cell(hide_code=True)
def _(az, idata_sat):
    az.plot_convergence_dist(idata_sat, var_names=["eta"])
    return


@app.cell
def _(az, idata_sat):
    _n_div = int(idata_sat.sample_stats["diverging"].sum())
    _summary = az.summary(idata_sat, var_names=["gamma_free"], round_to=3)
    f"divergences: {_n_div}   |   gamma_free max R-hat = {_summary['r_hat'].max():.3f}, min ESS = {_summary['ess_bulk'].min():.0f}"
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
