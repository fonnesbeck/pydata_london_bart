"""Verify the centering fix works for size=2 + separate_trees on F1 data."""

from __future__ import annotations

import multiprocessing as mp

mp.set_start_method("fork", force=True)

import numpy as np
import polars as pl
import pymc as pm
import pymc_bart as pmb

RANDOM_SEED = 20260608


def main() -> None:
    df = pl.read_csv("data/f1_laps.csv")
    num_cols = [
        "tyre_life",
        "lap_number",
        "stint",
        "air_temp",
        "track_temp",
        "humidity",
        "wind_speed",
    ]
    compounds = sorted(df["compound"].unique().to_list())
    code = {c: i for i, c in enumerate(compounds)}
    X_num = df.select(num_cols).to_numpy().astype(float)
    X_cmp = np.array([code[c] for c in df["compound"].to_numpy()], dtype=float)[:, None]
    X = np.concatenate([X_num, X_cmp], axis=1)
    y = df["lap_time_s"].to_numpy().astype(float)
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(X.shape[0])
    X_train = X[perm[:500]]
    y_train = y[perm[:500]]

    # Centering: subtract means so Y for both columns has mean ~0,
    # avoiding the exp(Y.mean()) blowup on log_sigma init.
    y_mean = float(y_train.mean())
    y_centered = y_train - y_mean
    log_dev = np.log(np.abs(y_centered) + 0.5)
    log_dev_mean = float(log_dev.mean())
    log_dev_centered = log_dev - log_dev_mean
    Y_stacked = np.column_stack([y_centered, log_dev_centered])
    print(
        f"y_train mean={y_mean:.2f}, std={y_train.std():.2f}"
        f"\ny_centered mean={y_centered.mean():.3f}, range "
        f"[{y_centered.min():.2f}, {y_centered.max():.2f}]"
        f"\nlog_dev_centered mean={log_dev_centered.mean():.3f}, range "
        f"[{log_dev_centered.min():.2f}, {log_dev_centered.max():.2f}]"
        f"\nY_stacked.mean()={Y_stacked.mean():.3f} (the scalar init value)"
    )

    rules = [pmb.ContinuousSplitRule] * 7 + [pmb.OneHotSplitRule]
    with pm.Model() as model:
        X_data = pm.Data("X_data", X_train)
        ms = pmb.BART(
            "mu_log_sigma",
            X=X_data,
            Y=Y_stacked,
            m=50,
            separate_trees=True,
            shape=(2, len(y_train)),
            split_rules=rules,
        )
        mu = pm.Deterministic("mu", ms[0] + y_mean)
        log_sigma = pm.Deterministic("log_sigma", ms[1] + log_dev_mean)
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_train)
        idata = pm.sample(
            draws=200,
            tune=200,
            chains=2,
            random_seed=RANDOM_SEED,
            progressbar=False,
        )
    mu_hat = idata.posterior["mu"].mean(("chain", "draw")).values
    sig_hat = idata.posterior["sigma"].mean(("chain", "draw")).values
    print(
        f"\nmu_hat range:    [{mu_hat.min():.2f}, {mu_hat.max():.2f}]  (y_train range [{y_train.min():.2f}, {y_train.max():.2f}])"
        f"\nsigma_hat range: [{sig_hat.min():.3f}, {sig_hat.max():.3f}]  (y_train std={y_train.std():.2f})"
    )


if __name__ == "__main__":
    main()
