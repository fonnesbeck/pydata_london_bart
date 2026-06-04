"""Micro-test the size=2 + separate_trees heteroscedastic pattern.

Two questions to answer before committing to the plan:

1. Does ``pmb.BART(shape=(2, n))`` accept multi-output construction at all?
2. With Y on the natural y scale (mean far from zero), does the
   scalar-Y.mean() init blow up when row 1 is used as log_sigma
   (exp(Y.mean()) -> huge sigma)?

The teaching demo wants Fit A vs Fit B vs Fit C to be visibly different
but all stable. If Fit A NaNs out at init on a realistic dataset, the
plan needs adjusting (switch heteroscedastic section to a synthetic
dataset where Y.mean() is near zero, leaving F1 for the homoscedastic
diagnosis).
"""

from __future__ import annotations

import multiprocessing as mp

mp.set_start_method("fork", force=True)

import numpy as np
import pymc as pm
import pymc_bart as pmb

RANDOM_SEED = 20260608


def synthetic_hetero(n=400, p=6, seed=RANDOM_SEED):
    """mu uses x[0], x[1]; log_sigma uses x[2]; x[3..p-1] are nuisance.

    With p > 1 and disjoint driving variables, separate_trees=False is
    forced to pick splits that compromise between the two outputs,
    whereas separate_trees=True can let each ensemble specialise.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    mu_true = 2 * np.sin(np.pi * x[:, 0]) + x[:, 1]
    log_sigma_true = -1.0 + x[:, 2]
    sigma_true = np.exp(log_sigma_true)
    y = mu_true + sigma_true * rng.standard_normal(n)
    return x, y, mu_true, sigma_true


def try_fit(label, separate_trees, x, y):
    log_dev = np.log(np.abs(y - y.mean()) + 0.5)
    Y_stacked = np.column_stack([y, log_dev])
    print(
        f"\n[{label}] Y_stacked shape={Y_stacked.shape}, "
        f"Y.mean()={Y_stacked.mean():.3f} (would init log_sigma at exp(mean))"
    )
    with pm.Model() as model:
        X_data = pm.Data("X_data", x)
        try:
            ms = pmb.BART(
                "mu_sigma",
                X=X_data,
                Y=Y_stacked,
                m=50,
                separate_trees=separate_trees,
                shape=(2, len(y)),
            )
        except Exception as e:
            print(f"  -> BART construction FAILED: {type(e).__name__}: {e}")
            return None
        sigma = pm.Deterministic("sigma", pm.math.exp(ms[1]))
        pm.Normal("y", mu=ms[0], sigma=sigma, observed=y)
        try:
            idata = pm.sample(
                draws=200,
                tune=200,
                chains=2,
                random_seed=RANDOM_SEED,
                progressbar=False,
            )
        except Exception as e:
            print(f"  -> sampling FAILED: {type(e).__name__}: {e}")
            return None
    mu_hat = idata.posterior["mu_sigma"].mean(("chain", "draw")).values[0]
    sigma_hat = idata.posterior["sigma"].mean(("chain", "draw")).values
    print(f"  -> mu_hat range: [{mu_hat.min():.2f}, {mu_hat.max():.2f}]")
    print(f"  -> sigma_hat range: [{sigma_hat.min():.3f}, {sigma_hat.max():.3f}]")
    # variable inclusion: how often each ensemble used each feature
    vi = idata.posterior.get("variable_inclusion", None)
    if vi is not None:
        vi_mean = vi.mean(("chain", "draw")).values
        print(f"  -> variable_inclusion mean: {vi_mean}")
    return idata


def main() -> None:
    x, y, mu_true, sigma_true = synthetic_hetero()
    print(
        f"synthetic: n={len(y)}, "
        f"mu_true range [{mu_true.min():.2f}, {mu_true.max():.2f}], "
        f"sigma_true range [{sigma_true.min():.3f}, {sigma_true.max():.3f}]"
    )

    try_fit("separate_trees=False", False, x, y)
    try_fit("separate_trees=True", True, x, y)


if __name__ == "__main__":
    main()
