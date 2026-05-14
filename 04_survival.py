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
    mo.md(
        r"""
        # BART for survival analysis

        Time-to-event data is inherently uncertain, and BART's flexibility
        is particularly valuable when the hazard function has unknown
        shape. We model **discrete-time hazards**: at each integer time $t$
        the event occurs with probability
        $$h(t \mid x) = \Phi\bigl(g(t, x)\bigr),$$
        where $g$ is a sum of trees and $\Phi$ is the probit link. Each
        subject contributes one row per time period they're at risk; the
        outcome is "did the event occur in this period?" — exactly the
        binary classification setup from notebook 3, applied to expanded
        person-time data.

        Survival follows from the hazards:
        $$S(t \mid x) = \prod_{s \le t} \bigl(1 - h(s \mid x)\bigr).$$
        """
    )
    return


@app.cell
def _():
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)

    import matplotlib.pyplot as plt
    import numpy as np
    import pymc as pm
    import pymc_bart as pmb

    rng = np.random.default_rng(20260423)
    return np, plt, pm, pmb, rng


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
def _(pm, pmb, surv_X, surv_y):
    # Discrete-time hazards modelled as repeated probit BART. The first
    # column of X is time t; BART learns h(t, x) jointly so the hazard
    # shape doesn't have to be specified up front.
    with pm.Model() as model_surv:
        X_data = pm.Data("X_data", surv_X)
        eta = pmb.BART("eta", X=X_data, Y=surv_y.astype(float), m=50)
        p = pm.Deterministic("p", pm.math.invprobit(eta))
        pm.Bernoulli("event", p=p, observed=surv_y, shape=p.shape)
        idata_surv = pm.sample(
            draws=300,
            tune=300,
            chains=2,
            cores=1,
            random_seed=20260423,
            progressbar=False,
        )
    return idata_surv, model_surv


@app.cell
def _(idata_surv, model_surv, np, pm):
    # Predict hazards at two contrasting risk profiles, then accumulate
    # into survival curves.
    times = np.arange(1, 13)
    profile_low = np.column_stack([times, np.full(12, -0.8), np.full(12, 0.0)])
    profile_high = np.column_stack([times, np.full(12, 0.8), np.full(12, 0.0)])
    X_profiles = np.concatenate([profile_low, profile_high], axis=0)

    with model_surv:
        pm.set_data({"X_data": X_profiles})
        pp_surv = pm.sample_posterior_predictive(
            idata_surv,
            var_names=["p"],
            predictions=True,
            random_seed=20260423,
            progressbar=False,
        )

    _p_draws = pp_surv.predictions["p"].stack(sample=("chain", "draw")).values
    p_low_draws = _p_draws[:12, :].T
    p_high_draws = _p_draws[12:, :].T
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
    _ax.set_ylabel(r"$S(t \mid x)$")
    _ax.set_title("Predicted survival, posterior mean and 90% band")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
