import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # BART for classification: predicting life satisfaction

    How satisfied are people with their lives, and what predicts it? We use
    the **2022 General Social Survey** — a long-running, nationally
    representative US survey — in which respondents rate their life on a
    0–10 ladder (`lifenow`). We model that rating with BART at **two
    resolutions**:

    1. **A binary question first** — is a person *highly* satisfied (9–10)
       or not? This is **probit BART**:
       $\Pr(Y = 1 \mid x) = \Phi\bigl(g(x)\bigr)$, where $g$ is a sum of
       trees and $\Phi$ is the standard-normal CDF. The BART prior shrinks
       $g$ toward zero — i.e. toward a baseline probability of $\tfrac12$ —
       useful regularisation when the signal is weak.
    2. **The full ordinal scale next** — an **ordered probit** that recovers
       the gradation the yes/no question throws away.

    Same data, same predictors, same train/test split throughout. We start
    simple, see what a binary model can and cannot say, then let the outcome
    get richer.
    """)
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
    return Path, RANDOM_SEED, az, np, os, pl, plt, pm, pmb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The data

    We keep respondents who answered the life-satisfaction ladder and a
    battery of wellbeing items. That battery is asked of only part of the
    sample, so after dropping incomplete records we have **n ≈ 1,600**
    complete cases — modest, which is realistic for survey work and keeps
    the models honest about uncertainty.

    **Outcome.** `lifenow` runs 1–10. For the binary model we define
    **"highly satisfied" = `lifenow ≥ 9`** (about 41% of respondents) — a
    balanced, interpretable threshold. For the ordinal model we keep the
    full scale, collapsing the sparse bottom (1–4, ~20 people combined) into
    a single "≤4" category, giving **$K = 7$** ordered levels.

    **Predictors (12).** Age; relative financial standing (`finrela`);
    education (`degree`); five self-reported wellbeing scales (anxiety,
    work-meaningfulness, stress, feeling-nervous, worry); sex; full-time
    employment; race; and religion (grouped into 5 buckets). Encodings
    matter for the trees — see the split-rules note below.

    We hold out a random **20%** of respondents as a test set, shared by
    both models, so any calibration claim is about data the model never saw.
    """)
    return


@app.cell(hide_code=True)
def _(Path, RANDOM_SEED, np, os, pl, pmb):
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
        # Map any 1-D array of category labels to contiguous integer codes
        # 0..K-1. SubsetSplitRule operates on these directly.
        _, codes = np.unique(col, return_inverse=True)
        return codes.astype(float)

    def _relig_group(raw):
        # Collapse the GSS `relig` codes into 5 buckets so SubsetSplitRule has
        # interpretable levels: None=0 (reference), Protestant=1, Catholic=2,
        # Christian-other={10,11,13}=3, Non-Christian={3,5,6,7,8,9,12}=4.
        # Anything else falls into a residual bucket=5.
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

    _lifenow = _gss["lifenow"].to_numpy().astype(int)

    # Binary target: "highly satisfied" = top of the ladder (9 or 10). ~41%.
    y_bin = (_lifenow >= 9).astype(int)

    # Ordinal target: collapse lifenow 1-4 into a single low bucket -> 7
    # categories indexed 0..6 (categories 1-3 hold ~20 obs combined and would
    # destabilise the ordered probit).
    y_ord = np.clip(_lifenow, 4, None) - 4
    K_ord = 7

    feature_names = [
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
    # = wrkstat==1), then 2 multi-level unordered categoricals (race
    # integer-coded; relig grouped via _relig_group).
    X = np.column_stack(
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

    # Per-column split rules, one per design-matrix column: 8 continuous,
    # 2 one-hot (binary sex/fulltime), 2 subset (multi-level race/relig).
    # Shared by the probit and ordered-probit models.
    split_rules = (
        [pmb.ContinuousSplitRule] * 8
        + [pmb.OneHotSplitRule] * 2
        + [pmb.SubsetSplitRule] * 2
    )

    # Shared 80/20 train/test split (seeded) used by both models.
    _rng_split = np.random.default_rng(RANDOM_SEED)
    _perm = _rng_split.permutation(X.shape[0])
    _n_train = int(0.8 * X.shape[0])
    train_idx = _perm[:_n_train]
    test_idx = _perm[_n_train:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_bin_train, y_bin_test = y_bin[train_idx], y_bin[test_idx]
    y_ord_train, y_ord_test = y_ord[train_idx], y_ord[test_idx]

    (
        f"n={X.shape[0]} (train {len(train_idx)} / test {len(test_idx)}), p={X.shape[1]} | "
        f"highly-satisfied rate={y_bin.mean():.1%} | "
        f"ordinal K={K_ord}, classes={np.bincount(y_ord).tolist()}"
    )
    return (
        K_ord,
        X_test,
        X_train,
        feature_names,
        split_rules,
        y_bin_test,
        y_bin_train,
        y_ord_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A binary question: who is highly satisfied?

    The simplest version of the question is yes/no. BART produces a
    real-valued score $\eta = g(x)$ for each person; the probit link
    $\Phi(\eta)$ squashes it into a probability, and a Bernoulli likelihood
    ties it to the observed 0/1 label. Probit pairs naturally with BART's
    Gaussian leaf prior; logit would work too but adds a scale parameter.

    We wrap the design matrix in `pm.Data` so we can swap in the held-out
    respondents for prediction without rebuilding the model — see the "PyMC
    patterns for prediction" aside in notebook 02 for `set_data` /
    `sample_posterior_predictive`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Split rules: telling BART each column's type

    BART's split rules are per-column. Three ship with `pymc-bart`:

    - **`ContinuousSplitRule`** (default): splits on $x_j \le c$. Correct
      for numeric or ordinal predictors (age, the wellbeing scales).
    - **`OneHotSplitRule`**: splits on $x_j = k$ for a single level. Correct
      for binary indicators (`sex`, `fulltime`).
    - **`SubsetSplitRule`**: splits on $x_j \in S$ for an arbitrary subset
      $S \subset \{0, \ldots, K-1\}$ in a single node. Correct for unordered
      categoricals with $K > 2$ levels (`race`, `relig`).

    Without `SubsetSplitRule`, a $K$-level unordered categorical must either
    be one-hot expanded (inflating $p$ and forcing BART to chain $K-1$ binary
    splits across depth levels) or treated as continuous (imposing a
    spurious ordering). The subset rule does the right thing in one split. We
    pass `split_rules` as a length-$p$ list in design-matrix order:
    `[age, finrela, degree, anxiety, wrkmeangfl, stress, feelnerv, worry,
    sex, fulltime, race, relig]`.
    """)
    return


@app.cell
def _(RANDOM_SEED, X_train, pm, pmb, split_rules, y_bin_train):
    # Probit BART: the BART score enters a Bernoulli through invprobit (the
    # standard-normal CDF, mapping the real-valued score onto [0, 1]).
    with pm.Model() as model_cls:
        X_data = pm.Data("X_data", X_train)
        eta = pmb.BART(
            "eta",
            X=X_data,
            Y=y_bin_train.astype(float),
            m=100,
            split_rules=split_rules,
        )
        p = pm.Deterministic("p", pm.math.invprobit(eta))
        pm.Bernoulli("y", p=p, observed=y_bin_train, shape=p.shape)
        idata_cls = pm.sample(random_seed=RANDOM_SEED)
    return eta, idata_cls, model_cls


@app.cell
def _(RANDOM_SEED, X_test, idata_cls, model_cls, pm):
    with model_cls:
        pm.set_data({"X_data": X_test})
        pp_cls = pm.sample_posterior_predictive(
            idata_cls,
            var_names=["p"],
            sample_vars=["eta"],
            predictions=True,
            random_seed=RANDOM_SEED,
        )
    return (pp_cls,)


@app.cell(hide_code=True)
def _(np, plt, pp_cls, y_bin_test):
    _p_draws = pp_cls.predictions["p"].stack(sample=("chain", "draw")).values
    _p_mean = _p_draws.mean(axis=1)
    _p_lo, _p_hi = np.quantile(_p_draws, [0.05, 0.95], axis=1)

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 4.0))

    # Reliability diagram: bin held-out predictions into deciles and compare
    # mean predicted probability to the empirical positive rate in each bin.
    _bins = np.linspace(0, 1, 11)
    _which = np.clip(np.digitize(_p_mean, _bins) - 1, 0, 9)
    _xs, _ys, _ns = [], [], []
    for _k in range(10):
        _mask = _which == _k
        if _mask.sum() > 0:
            _xs.append(_p_mean[_mask].mean())
            _ys.append(y_bin_test[_mask].mean())
            _ns.append(int(_mask.sum()))
    _a.plot([0, 1], [0, 1], "--", color="C3", lw=1)
    _a.scatter(
        _xs, _ys, s=[max(25, n * 4) for n in _ns], color="#4c72b0", alpha=0.8, zorder=3
    )
    _a.set_xlim(0, 1)
    _a.set_ylim(0, 1)
    _a.set_xlabel(r"mean predicted $\Pr(\text{highly satisfied})$")
    _a.set_ylabel("empirical frequency")
    _a.set_title("Held-out reliability (decile bins)")

    _top = np.argsort(-_p_mean)[:20]
    _pos = np.arange(len(_top))
    _colors = ["C2" if y_bin_test[i] == 1 else "C3" for i in _top]
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
    _b.set_ylabel(r"$\Pr(\text{highly satisfied})$ with 90% CI")
    _b.set_title(
        f"Top-20 hit rate: {int(y_bin_test[_top].sum())}/20  "
        f"(base rate {y_bin_test.mean():.0%})"
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading these panels

    **Left — reliability.** We bin the held-out respondents by their
    predicted probability of being highly satisfied (deciles) and plot, per
    bin, the mean prediction against the actual fraction who were highly
    satisfied. Points on the diagonal mean the probabilities are honest:
    when the model says "60%", about 60% really are. Above the line is
    under-confidence, below is over-confidence; marker size is the bin count.

    **Right — ranking.** The 20 test respondents the model is most confident
    about, coloured green if they really were highly satisfied. A useful
    model concentrates the hits at the top, even if its absolute
    probabilities are not perfectly calibrated.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Which predictors matter?

    `compute_variable_importance` ranks features by inclusion frequency (how
    often each appears as a splitting variable across the posterior trees)
    and measures how a restricted model's $R^2$ recovers the full-model fit.
    The curve plateaus at the smallest sufficient subset — a defensible
    feature selection. On real predictors (no planted ground truth) this is
    a genuine finding, not a recovery check.
    """)
    return


@app.cell(hide_code=True)
def _(X_train, eta, feature_names, idata_cls, model_cls, pm, pmb):
    # Reset X_data first: the OOS-prediction cell swapped in the test matrix.
    with model_cls:
        pm.set_data({"X_data": X_train})
        _vi_cls = pmb.compute_variable_importance(idata_cls, eta, X_train)
    pmb.plot_variable_importance(_vi_cls, labels=feature_names)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Backward variable importance and submodel agreement

    `compute_variable_importance` has two ranking methods:

    - `method="VI"` (the default, used above) ranks features by what they
      *add* when included — a forward-selection view from partial-dependence
      variance.
    - `method="backward"` ranks features by what is *lost* when each is
      removed — a backward-elimination view. The two usually agree on the top
      features and disagree in the long tail.

    `pmb.plot_scatter_submodels` visualises the agreement. A **submodel** is
    the full BART fit restricted to a prefix of the importance ranking
    (top-1, then top-2, …). Each panel plots the submodel's predictions
    (x-axis) against the full-model predictions (y-axis); panels that hug the
    diagonal mean those features are sufficient — adding the rest would
    barely change predictions. That is how you justify stopping at $k$
    features rather than carrying all $p$.
    """)
    return


@app.cell(hide_code=True)
def _(X_train, eta, feature_names, idata_cls, model_cls, np, pm, pmb):
    class _LabeledMatrix:
        def __init__(self, values, columns):
            self._values = values
            self.columns = np.asarray(columns)
            self.shape = values.shape

        def to_numpy(self):
            return self._values


    with model_cls:
        pm.set_data({"X_data": X_train})
        vi_backward = pmb.compute_variable_importance(
            idata_cls,
            eta,
            _LabeledMatrix(X_train, feature_names),
            method="backward",
        )
    return (vi_backward,)


@app.cell(hide_code=True)
def _(pmb, vi_backward):
    pmb.plot_scatter_submodels(vi_backward, grid=(3, 4), figsize=(12, 10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `split_prior`: encoding a domain prior

    BART picks a splitting variable uniformly at random by default.
    `split_prior` lets you bias that choice: pass a length-$p$ array of
    positive weights, and each split draws a variable with probability
    proportional to its weight. This is Linero's (2018) sparse-splitting
    idea — used here not to recover planted signal, but to **encode a prior
    belief**: that self-reported wellbeing (anxiety, work-meaning, stress,
    nervousness, worry) should drive life satisfaction more than the
    demographics. We upweight those five columns 10×:
    `split_prior = np.ones(12)` with `[3:8] = 10`.

    The comparison below shows **inclusion frequencies** — how often each
    variable is chosen to split, averaged over posterior trees — under the
    uniform prior versus the domain prior, via `pmb.plot_variable_inclusion`.
    This is the raw counting statistic, distinct from the restricted-$R^2$
    importance plot above. In practice you set these weights from domain
    knowledge or a first-pass inclusion analysis; Linero's full method places
    a Dirichlet prior on the weights and learns them (see notebook 1's
    `run_bart_sparse`).
    """)
    return


@app.cell
def _(RANDOM_SEED, X_train, np, pm, pmb, split_rules, y_bin_train):
    # Same spec as model_cls, but with split_prior upweighting the five
    # wellbeing-scale columns (indices 3-7) 10x.
    _split_prior = np.ones(12)
    _split_prior[3:8] = 10.0
    with pm.Model() as model_cls_sp:
        X_data_sp = pm.Data("X_data", X_train)
        eta_sp = pmb.BART(
            "eta",
            X=X_data_sp,
            Y=y_bin_train.astype(float),
            m=100,
            split_prior=_split_prior,
            split_rules=split_rules,
        )
        p_sp = pm.Deterministic("p", pm.math.invprobit(eta_sp))
        pm.Bernoulli("y", p=p_sp, observed=y_bin_train, shape=p_sp.shape)
        idata_cls_sp = pm.sample(random_seed=RANDOM_SEED)
    return (idata_cls_sp,)


@app.cell(hide_code=True)
def _(X_train, feature_names, idata_cls, idata_cls_sp, plt, pmb):
    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(13, 4.0), sharey=True)
    pmb.plot_variable_inclusion(idata_cls, X_train, labels=feature_names, ax=_ax0)
    _ax0.set_title("Uniform split prior")
    pmb.plot_variable_inclusion(idata_cls_sp, X_train, labels=feature_names, ax=_ax1)
    _ax1.set_title(r"Domain prior: wellbeing scales upweighted 10$\times$")
    _fig.suptitle(
        "Inclusion frequencies: the domain prior concentrates splits on the wellbeing scales",
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
    mo.md(r"""
    ## From yes/no to the full scale

    Collapsing satisfaction to "highly satisfied or not" threw away real
    information: it treats someone who rates their life a 5 the same as an 8,
    and a 9 the same as a 10. If we care about *where* people fall, not just
    whether they clear a bar, we should model the whole ladder. That is an
    **ordered probit**: the same BART score $\eta$, now sliced by a set of
    estimated cutpoints into ordered categories.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The full scale: ordered-probit life satisfaction

    We model `lifenow` as **$K = 7$** ordered categories
    (`<=4, 5, 6, 7, 8, 9, 10`). The latent score is the same BART function;
    an ordered set of cutpoints turns it into category probabilities. The
    first cutpoint is fixed at $0$ for identifiability and the remaining
    $K - 2 = 5$ are constrained increasing via
    `pm.distributions.transforms.ordered` (with an `initval` already inside
    the constraint, since a non-ordered start errors before the first NUTS
    step). `compute_p=False` skips materialising per-category probabilities
    at every draw — we only need the cutpoints and likelihood, and the
    probabilities are cheap to recompute post-hoc.

    Split rules carry over from the binary model: 8 continuous, 2 one-hot
    (`sex`, `fulltime`), 2 subset (`race`, `relig`). This fit uses
    `target_accept=0.95` — the ordered-probit + BART geometry has funnel-like
    ridges, and the higher target keeps divergences down. It is the slowest
    fit in the tutorial.
    """)
    return


@app.cell
def _(K_ord, RANDOM_SEED, X_train, np, pm, pmb, split_rules, y_ord_train):
    with pm.Model() as model_ord:
        eta_ord = pmb.BART(
            "eta",
            X=X_train,
            Y=y_ord_train.astype(float),
            m=100,
            split_rules=split_rules,
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
            "y", eta=eta_ord, cutpoints=cutpoints, observed=y_ord_train, compute_p=False
        )
        idata_ord = pm.sample(chains=2, random_seed=RANDOM_SEED, target_accept=0.95)
    return eta_ord, idata_ord, model_ord


@app.cell
def _(az, idata_ord):
    az.summary(idata_ord, var_names=["gamma_free", "cutpoints"], round_to=3)
    return


@app.cell(hide_code=True)
def _(X_train, eta_ord, feature_names, idata_ord, model_ord, pmb):
    with model_ord:
        _vi_ord = pmb.compute_variable_importance(idata_ord, eta_ord, X_train)
    pmb.plot_variable_importance(_vi_ord, labels=feature_names)
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
    mo.md(r"""
    ### Posterior predictive check

    Sample `y` draws from the fitted ordered probit and compare the
    per-category counts to the observed histogram (on the training set). If
    the model captures the marginal distribution the observed counts (red
    dots) sit inside the 90% predictive band (blue error bars).
    """)
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
def _(K_ord, np, plt, ppc_ord, y_ord_train):
    _y_rep = ppc_ord.posterior_predictive["y"].values.reshape(-1, len(y_ord_train))
    _obs = np.bincount(y_ord_train, minlength=K_ord)
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
    mo.md(r"""
    ### Partial dependence

    PDP for the five continuous/ordinal predictors (age and four wellbeing
    scales). The y-axis is the latent $\eta$ (probit scale), not a
    probability — interpret signs and relative magnitudes, not absolute
    levels. `pmb.plot_pdp` reads feature names from a duck-typed DataFrame's
    `.columns`, so we pass a polars `DataFrame` view of the design matrix.
    """)
    return


@app.cell(hide_code=True)
def _(X_train, eta_ord, feature_names, pl, pmb):
    _X_df = pl.DataFrame(X_train, schema=feature_names)
    _pdp_features = [0, 1, 2, 3, 4]
    _pdp_labels = [feature_names[i] for i in _pdp_features]
    _axes = pmb.plot_pdp(
        eta_ord,
        X=_X_df,
        Y=None,
        xs_interval="quantiles",
        xs_values=[0.05, 0.25, 0.5, 0.75, 0.95],
        var_idx=_pdp_features,
        var_discrete=[1, 2, 3, 4],
        func=lambda x: x - x.mean(),
        samples=80,
        grid=(3, 2),
        figsize=(11, 8.5),
        sharey=False,
        color="C0",
        color_mean="#1f77b4",
        alpha=0.14,
    )
    _fig = _axes[0].figure
    for _ax, _label in zip(_axes, _pdp_labels):
        _ax.set_title(_label, fontsize=12, fontweight="semibold", pad=8)
        _ax.set_xlabel(_label, fontsize=10)
        _ax.tick_params(axis="both", labelsize=9)
        _ax.grid(axis="y", alpha=0.18)
    for _ax in _axes[len(_pdp_labels):]:
        _ax.set_visible(False)
    _fig.suptitle("Partial dependence on latent life-satisfaction score", fontsize=14, y=0.995)
    _fig.supxlabel("Predictor value", fontsize=11)
    _fig.supylabel("Centered partial dependence (probit scale)", fontsize=11)
    _fig.tight_layout(rect=(0.02, 0.03, 1, 0.96))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What predicts life satisfaction?

    Across both resolutions the same story emerges from the
    variable-importance and partial-dependence views: the self-reported
    wellbeing scales — stress, worry, anxiety, how meaningful work feels —
    move predicted satisfaction far more than demographics like age or
    education. The binary model answers a sharp question ("who clears the
    bar?") with a calibration and ranking we can check on held-out data; the
    ordered probit recovers the gradation in between, at the cost of a much
    heavier fit. Same BART score underneath — only the likelihood changed.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
