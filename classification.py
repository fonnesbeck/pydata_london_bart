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
    1–10 ladder (`lifenow`). We model that rating with BART at **two
    resolutions**:

    1. **A binary question first** — is a person *highly* satisfied (9–10)
       or not? This is **probit BART**:
       $\Pr(Y = 1 \mid x) = \Phi\bigl(g(x)\bigr)$, where $g$ is a sum of
       trees and $\Phi$ is the standard-normal CDF. The BART prior shrinks
       $g$ toward zero — i.e. toward a baseline probability of $\tfrac12$ —
       useful regularisation when the signal is weak.
    2. **The full ordinal scale next** — an **ordered probit** that recovers
       the gradation the yes/no question throws away.

    We start with the binary question, check held-out probabilities, compare
    BART with a logistic-regression baseline, and then return to the full
    ordinal scale.
    """)
    return


@app.cell(hide_code=True)
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    RANDOM_SEED = 20260608
    return (
        LogisticRegression,
        Path,
        RANDOM_SEED,
        az,
        brier_score_loss,
        log_loss,
        np,
        os,
        pl,
        plt,
        pm,
        pmb,
        roc_auc_score,
    )


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

    **Encodings.** We encode unordered categoricals as contiguous integer
    levels so `SubsetSplitRule` can act on them directly. `relig` is grouped
    into five interpretable buckets (None, Protestant, Catholic,
    Christian-other, Non-Christian, plus a residual bucket), while `race` is
    integer-coded from the observed labels. The design matrix combines 8
    continuous/ordinal columns, 2 binary indicators, and 2 unordered
    multi-level categoricals.

    We hold out a random **20%** of respondents as a test set for the binary
    BART model and the logistic baseline, so calibration and ranking claims
    are about data those models never saw. The ordered-probit extension later
    trains on the same training rows and is evaluated in-sample.
    """)
    return


@app.cell(hide_code=True)
def _(Path, os, pl):
    def _load_gss():
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
    gss_df = (
        _load_gss()
        .select(_kept_cols)
        .filter(pl.col("lifenow").is_between(1, 10))
        .drop_nulls()
    )
    gss_df.shape
    return (gss_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The data-subsample control defaults to 25% to keep all seven ordinal
    classes represented in training.
    """)
    return


@app.cell(hide_code=True)
def _(gss_df, mo):
    _n_train = int(0.8 * gss_df.height)
    _subsample_options = {
        f"Full (~{_n_train} respondents)": 1.0,
        f"50% (~{int(0.5 * _n_train)} respondents)": 0.5,
        f"25% (~{int(0.25 * _n_train)} respondents)": 0.25,
        f"10% (~{int(0.1 * _n_train)} respondents)": 0.1,
    }
    subsample = mo.ui.radio(
        options=_subsample_options,
        value=f"25% (~{int(0.25 * _n_train)} respondents)",
        label="**Data subsample** &mdash; smaller = faster fits for iteration",
    )
    subsample
    return (subsample,)


@app.cell(hide_code=True)
def _(RANDOM_SEED, gss_df, np, pmb, subsample):
    def _int_code(col):
        _, codes = np.unique(col, return_inverse=True)
        return codes.astype(float)

    def _relig_group(raw):
        out = np.full(raw.shape, 5, dtype=int)
        out[raw == 4] = 0
        out[raw == 1] = 1
        out[raw == 2] = 2
        out[np.isin(raw, [10, 11, 13])] = 3
        out[np.isin(raw, [3, 5, 6, 7, 8, 9, 12])] = 4
        return out.astype(float)

    _lifenow = gss_df["lifenow"].to_numpy().astype(int)
    y_bin = (_lifenow >= 9).astype(int)
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

    X = np.column_stack(
        [
            gss_df["age"].to_numpy().astype(float),
            gss_df["finrela"].to_numpy().astype(float),
            gss_df["degree"].to_numpy().astype(float),
            gss_df["anxiety"].to_numpy().astype(float),
            gss_df["wrkmeangfl"].to_numpy().astype(float),
            gss_df["stress"].to_numpy().astype(float),
            gss_df["feelnerv"].to_numpy().astype(float),
            gss_df["worry"].to_numpy().astype(float),
            (gss_df["sex"].to_numpy() == 2).astype(float),
            (gss_df["wrkstat"].to_numpy() == 1).astype(float),
            _int_code(gss_df["race"].to_numpy()),
            _relig_group(gss_df["relig"].to_numpy()),
        ]
    )
    # Match each design-matrix column to the split rule BART should use:
    # - ContinuousSplitRule: ordered numeric columns; split at thresholds x <= c.
    # - OneHotSplitRule: binary indicators; split on off/on.
    # - SubsetSplitRule: unordered categorical codes; split subsets of levels.
    split_rules = (
        [pmb.ContinuousSplitRule] * 8
        + [pmb.OneHotSplitRule] * 2
        + [pmb.SubsetSplitRule] * 2
    )

    _rng_split = np.random.default_rng(RANDOM_SEED)
    _perm = _rng_split.permutation(X.shape[0])
    _n_train = int(0.8 * X.shape[0])
    test_idx = _perm[_n_train:]
    _n_sub = int(subsample.value * _n_train)
    train_idx = _perm[:_n_sub]
    X_train, X_test = X[train_idx], X[test_idx]
    y_bin_train, y_bin_test = y_bin[train_idx], y_bin[test_idx]
    y_ord_train = y_ord[train_idx]

    (
        f"n={X.shape[0]} (train {len(train_idx)} [{subsample.value:.0%} of {_n_train}] / test {len(test_idx)}), p={X.shape[1]} | "
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
    ## Who is highly satisfied?

    The regression notebook put uncertainty bands around a continuous mean.
    Here the posterior draws are probabilities:
    $\Pr(\text{highly satisfied} \mid x)$, not just class labels.

    BART produces a real-valued score $\eta = g(x)$ for each person; the
    probit link $\Phi(\eta)$ squashes it into a probability, and a Bernoulli
    likelihood ties it to the observed 0/1 label. If you have seen logistic
    regression, probit is the same idea with the standard-normal CDF instead
    of the logistic sigmoid. Probit pairs naturally with BART's Gaussian leaf
    prior; logit would work too but adds a scale parameter.

    We wrap the design matrix in `pm.Data` so we can swap in the held-out
    respondents for prediction without rebuilding the model.

    ### Split rules for the GSS columns

    The GSS design matrix mixes ordered numbers, binary flags, and unordered
    labels. BART needs a split rule that matches each column's semantics:

    - **`ContinuousSplitRule`** is for ordered numeric columns. A tree tries
      threshold splits such as `age <= 45` versus `age > 45`, so larger and
      smaller values must be meaningful.
    - **`OneHotSplitRule`** is for 0/1 indicators. The split is simply "off"
      versus "on"; this is what we use for `sex` and `fulltime`.
    - **`SubsetSplitRule`** is for unordered multi-level categoricals. A tree
      can send any subset of levels left and the remaining levels right —
      e.g. one set of religion groups versus the others — without pretending
      the integer codes have a numeric order.

    In this notebook that means 8 continuous/ordinal columns (`age`,
    `finrela`, `degree`, and the wellbeing scales), 2 one-hot indicators
    (`sex`, `fulltime`), and 2 subset-split categoricals (`race`, `relig`).
    We pass `split_rules` in design-matrix order:
    `[age, finrela, degree, anxiety, wrkmeangfl, stress, feelnerv, worry,
    sex, fulltime, race, relig]`.
    """)
    return


@app.cell
def _(RANDOM_SEED, X_train, pm, pmb, split_rules, y_bin_train):
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
        pm.compute_log_likelihood(idata_cls)
    return eta, idata_cls, model_cls


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Convergence diagnostics

    BART random variables need a different diagnostic view from scalar
    parameters: `az.plot_convergence_dist` summarizes ESS and $\hat R$ across
    every node of the latent BART score rather than plotting a wall of traces.
    """)
    return


@app.cell
def _(az, idata_cls):
    az.plot_convergence_dist(idata_cls, var_names=["eta"])
    return


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
def _(
    LogisticRegression,
    RANDOM_SEED,
    X_test,
    X_train,
    brier_score_loss,
    feature_names,
    log_loss,
    np,
    pl,
    pp_cls,
    roc_auc_score,
    y_bin_test,
    y_bin_train,
):
    _p_draws = pp_cls.predictions["p"].stack(sample=("chain", "draw")).values
    p_bart_test = _p_draws.mean(axis=1)

    _logistic = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    _logistic.fit(X_train, y_bin_train)
    p_logistic_test = _logistic.predict_proba(X_test)[:, 1]

    _eps = 1e-15
    _p_bart_clip = np.clip(p_bart_test, _eps, 1.0 - _eps)
    _p_logistic_clip = np.clip(p_logistic_test, _eps, 1.0 - _eps)

    assert X_train.shape[1] == X_test.shape[1] == len(feature_names)
    assert len(p_bart_test) == len(p_logistic_test) == len(y_bin_test)
    assert np.isfinite(X_train).all() and np.isfinite(X_test).all()
    assert np.isfinite(p_bart_test).all() and np.isfinite(p_logistic_test).all()

    _bart_brier = brier_score_loss(y_bin_test, p_bart_test)
    _logistic_brier = brier_score_loss(y_bin_test, p_logistic_test)
    _bart_log_loss = log_loss(y_bin_test, _p_bart_clip)
    _logistic_log_loss = log_loss(y_bin_test, _p_logistic_clip)
    _bart_auc = roc_auc_score(y_bin_test, p_bart_test)
    _logistic_auc = roc_auc_score(y_bin_test, p_logistic_test)

    _rng_boot = np.random.default_rng(RANDOM_SEED)
    _boot_idx = _rng_boot.integers(0, len(y_bin_test), size=(1000, len(y_bin_test)))
    _brier_diff = np.array(
        [
            brier_score_loss(y_bin_test[_idx], p_bart_test[_idx])
            - brier_score_loss(y_bin_test[_idx], p_logistic_test[_idx])
            for _idx in _boot_idx
        ]
    )
    _diff_lo, _diff_hi = np.quantile(_brier_diff, [0.05, 0.95])
    _diff_note = (
        "inconclusive on this split"
        if _diff_lo <= 0.0 <= _diff_hi or abs(_bart_brier - _logistic_brier) < 0.01
        else "BART lower Brier"
        if _bart_brier < _logistic_brier
        else "logistic lower Brier"
    )

    cls_comparison = pl.DataFrame(
        [
            {
                "section": "prevalence",
                "metric": "positive rate",
                "BART": f"train={y_bin_train.mean():.1%}",
                "logistic": f"test={y_bin_test.mean():.1%}",
                "note": "base-rate context",
            },
            {
                "section": "performance",
                "metric": "Brier score",
                "BART": f"{_bart_brier:.3f}",
                "logistic": f"{_logistic_brier:.3f}",
                "note": f"BART - logistic = {_bart_brier - _logistic_brier:+.3f}; 90% bootstrap CI [{_diff_lo:+.3f}, {_diff_hi:+.3f}] ({_diff_note})",
            },
            {
                "section": "performance",
                "metric": "log loss",
                "BART": f"{_bart_log_loss:.3f}",
                "logistic": f"{_logistic_log_loss:.3f}",
                "note": f"probabilities clipped to [{_eps:g}, 1-{_eps:g}]",
            },
            {
                "section": "performance",
                "metric": "AUC",
                "BART": f"{_bart_auc:.3f}",
                "logistic": f"{_logistic_auc:.3f}",
                "note": "ranking only",
            },
        ]
    )
    cls_comparison
    return p_bart_test, p_logistic_test


@app.cell(hide_code=True)
def _(np, p_bart_test, p_logistic_test, plt, pp_cls, y_bin_test):
    _p_draws = pp_cls.predictions["p"].stack(sample=("chain", "draw")).values
    _p_mean = p_bart_test
    _p_lo, _p_hi = np.quantile(_p_draws, [0.05, 0.95], axis=1)

    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(11, 4.0))

    _bins = np.linspace(0, 1, 11)

    def _reliability_points(_p):
        _which = np.clip(np.digitize(_p, _bins) - 1, 0, 9)
        _xs, _ys, _ns = [], [], []
        for _k in range(10):
            _mask = _which == _k
            if _mask.sum() > 0:
                _xs.append(_p[_mask].mean())
                _ys.append(y_bin_test[_mask].mean())
                _ns.append(int(_mask.sum()))
        return _xs, _ys, _ns

    _bart_x, _bart_y, _bart_n = _reliability_points(_p_mean)
    _log_x, _log_y, _log_n = _reliability_points(p_logistic_test)
    _a.plot([0, 1], [0, 1], "--", color="C3", lw=1)
    _a.scatter(
        _bart_x,
        _bart_y,
        s=[max(25, n * 4) for n in _bart_n],
        color="#4c72b0",
        alpha=0.8,
        zorder=3,
        label="BART",
    )
    _a.scatter(
        _log_x,
        _log_y,
        s=[max(25, n * 4) for n in _log_n],
        color="#dd8452",
        marker="s",
        alpha=0.75,
        zorder=3,
        label="logistic",
    )
    _a.set_xlim(0, 1)
    _a.set_ylim(0, 1)
    _a.set_xlabel(r"mean predicted $\Pr(\text{highly satisfied})$")
    _a.set_ylabel("empirical frequency")
    _a.set_title("Held-out reliability (decile bins; empty bins omitted)")
    _a.legend(frameon=False)

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

    **Left — reliability.** We bin the held-out respondents by predicted
    probability of being highly satisfied (deciles) and plot, per bin, the
    mean prediction against the actual fraction who were highly satisfied.
    Circles are BART, squares are logistic regression, and marker size is the
    bin count. Logistic regression is the linear/additive baseline; BART can
    improve on it when nonlinearities or interactions matter, but a single
    split should be read as illustrative, especially when metric gaps are
    small or the bootstrap interval includes zero. Points on the diagonal mean
    the probabilities are honest: when the model says "60%", about 60% really
    are. Above the line is under-confidence, below is over-confidence.

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

    `compute_variable_importance` first ranks features by **posterior inclusion
    frequency**: how often each feature appears as a splitting variable across the
    sampled trees. The plot then follows that ranking cumulatively: the first row is
    the top-inclusion feature alone, the next row adds the second feature, and so on.

    So the order matters for reading the path, but it is **not** a causal ranking and
    not the same as a single-feature effect size. In this fit, `race` enters first
    because it was the most frequently used splitting variable in the posterior. On
    its own it recovers very little of the full fit ($R^2 \approx 0.02$ here), so the
    substantive signal is in the later jumps from `finrela` and the wellbeing scales,
    not in `race` alone.
    """)
    return


@app.cell(hide_code=True)
def _(X_train, eta, feature_names, idata_cls, model_cls, np, plt, pm, pmb):
    with model_cls:
        pm.set_data({"X_data": X_train})
        _vi_cls = pmb.compute_variable_importance(idata_cls, eta, X_train)

    _ordered_features = np.asarray(feature_names)[_vi_cls["indices"]]
    _cumulative_labels = [
        name if i == 0 else f"+ {name}" for i, name in enumerate(_ordered_features)
    ]
    _y = np.arange(len(_cumulative_labels))
    _r2_mean = _vi_cls["r2_mean"]
    _r2_hdi = _vi_cls["r2_hdi"]
    _xerr = np.vstack(
        [
            np.clip(_r2_mean - _r2_hdi[:, 0], 0, None),
            np.clip(_r2_hdi[:, 1] - _r2_mean, 0, None),
        ]
    )


    def _pearson_r2(a, b):
        return float(np.corrcoef(a, b)[0, 1] ** 2)


    _preds_all = _vi_cls["preds_all"]
    _ref_r2 = np.array(
        [_pearson_r2(_preds_all[i], _preds_all[i + 1]) for i in range(len(_preds_all) - 1)]
    )
    _ref_lo, _ref_hi = np.quantile(_ref_r2, [0.05, 0.95])
    _ref_mean = _ref_r2.mean()

    _fig, _ax = plt.subplots(figsize=(8.5, 6.0))
    _ax.axvspan(_ref_lo, _ref_hi, color="0.5", alpha=0.12, lw=0)
    _ax.axvline(_ref_mean, color="0.45", ls="--", lw=1.5, label="full-model agreement")
    _ax.errorbar(
        _r2_mean,
        _y,
        xerr=_xerr,
        fmt="o",
        ms=6,
        color="#2f2f2f",
        mfc="white",
        mec="#2f2f2f",
        ecolor="#2f2f2f",
        elinewidth=1.4,
        capsize=3,
    )
    _ax.set_yticks(_y, _cumulative_labels)
    _ax.invert_yaxis()
    _ax.set_xlim(0, 1)
    _ax.set_xlabel(r"Agreement with full BART fit ($R^2$)")
    _ax.set_ylabel("Cumulative predictors (inclusion-frequency order)")
    _ax.set_title("Restricted submodels recover the full BART fit", pad=12)
    _ax.grid(axis="x", alpha=0.18)
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.legend(frameon=False, loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first row is a useful warning about variable-importance plots. `race` appears
    first because it is the most frequently selected splitting variable, not because
    it explains much by itself. Its restricted fit has almost no agreement with the
    full BART surface, so the model is using it as an early partition rather than as
    the main source of predictive signal.

    The larger jumps come after `finrela` and the wellbeing measures enter the
    cumulative submodel. Read the plot as: **which additions make the restricted
    ensemble behave like the full ensemble?** On that scale, the financial and
    self-reported wellbeing variables carry the interpretable signal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The remaining sections compute backward variable importance, compare a
    domain-informed split prior, and fit the ordered-probit model on the
    full satisfaction scale.
    """)
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


@app.cell
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


@app.cell
def _(plt, pmb, vi_backward):
    _fig = pmb.plot_scatter_submodels(vi_backward, grid=(3, 4), figsize=(12, 10))
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A domain prior with `split_prior`

    BART picks a splitting variable uniformly at random by default.
    `split_prior` lets you bias that choice: pass a length-$p$ array of
    positive weights, and each split draws a variable with probability
    proportional to its weight. This is Linero's (2018) sparse-splitting idea,
    used here to **encode a prior belief**: self-reported wellbeing (anxiety,
    work-meaning, stress, nervousness, worry) should drive life satisfaction
    more than the demographics. We upweight those five columns 10×:
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
    _split_prior = np.ones(12)
    _split_prior[3:8] = 10.0
    with pm.Model():
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
        pm.compute_log_likelihood(idata_cls_sp)
    return (idata_cls_sp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The uniform and domain-prior BART fits share the same binary outcome,
    likelihood, training rows, and tree count, so PSIS-LOO isolates the effect
    of the split prior. Higher `elpd_loo` is better; differences smaller than a
    few standard errors are not practically meaningful.
    """)
    return


@app.cell
def _(az, idata_cls, idata_cls_sp):
    cls_loo_comparison = az.compare(
        {
            "uniform split prior": idata_cls,
            "wellbeing split prior": idata_cls_sp,
        },
        ic="loo",
    )
    cls_loo_comparison
    return


@app.cell
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
def _(mo):
    mo.md(r"""
    ## From yes/no to the full scale

    Collapsing satisfaction to a binary variable threw away a lot of
    information: it treats someone who rates their life a 5 the same as an 8,
    and a 9 the same as a 10. To model an ordinal range as an outcome, we employ a
    **ordered probit** model: the same BART score $\eta$, now sliced by a set of
    estimated cutpoints into ordered categories.

    We model `lifenow` as **$K = 7$** ordered categories
    (`<=4, 5, 6, 7, 8, 9, 10`). Imagine an unobserved satisfaction score
    $\eta$: higher values mean higher satisfaction, but the survey only
    records which interval the score fell into. Ordered cutpoints are the
    thresholds between those intervals.

    One cutpoint is arbitrarily fixed at $0$ (a **baseline**) for identifiability: if we shifted every
    $\eta$ and every cutpoint by the same amount, the category probabilities
    would not change. The remaining $K - 2 = 5$ cutpoints are constrained
    increasing via `pm.distributions.transforms.ordered` (with an `initval`
    already inside the constraint, since a non-ordered start errors before the
    first NUTS step). `compute_p=False` skips materialising per-category
    probabilities at every draw; we only need the cutpoints and likelihood, and
    the probabilities are cheap to recompute post-hoc.

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


@app.cell
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
    per-category counts to the observed histogram on the **training set**.
    This is a marginal PPC for the fitted likelihood, not held-out
    calibration. If the model captures the marginal distribution the observed
    counts (red dots) sit inside the 90% predictive band (blue error bars).
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


@app.cell
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
    for _ax in _axes[len(_pdp_labels) :]:
        _ax.set_visible(False)
    _fig.suptitle(
        "Partial dependence on latent life-satisfaction score", fontsize=14, y=0.995
    )
    _fig.supxlabel("Predictor value", fontsize=11)
    _fig.supylabel("Centered partial dependence (probit scale)", fontsize=11)
    _fig.tight_layout(rect=(0.02, 0.03, 1, 0.96))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What predicts life satisfaction?

    Across the binary model and the ordinal model, the same story emerges from
    the variable-importance and partial-dependence views:
    the self-reported wellbeing scales, which include stress, worry, anxiety, how
    meaningful work feels, move predicted satisfaction far more than
    demographics like age or education.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
