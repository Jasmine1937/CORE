"""Select features across development windows and apply SMOTE-ENN to training data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from clean import (cli_config, fit_preprocessor, load_state, numeric_frame,
                   save_csv, save_state, transform)
from windows import make_window_specs, validate_windows


def xgb_model(cfg):
    parameters = {"n_estimators": 100, "max_depth": 6, "learning_rate": .3,
                  "subsample": 1., "colsample_bytree": 1.}
    parameters.update(cfg.get("features", {}).get("xgb", {}))
    parameters.update({"objective": "binary:logistic", "eval_metric": "auc",
                       "random_state": int(cfg.get("seed", 42)), "verbosity": 0})
    parameters.setdefault("n_jobs", 1)
    return XGBClassifier(**parameters)


def resample_training(x, y, cfg):
    settings = cfg.get("preprocessing", {})
    neighbors = int(settings.get("smote_neighbors", 5))
    enn_neighbors = int(settings.get("enn_neighbors", 3))
    if neighbors < 1 or enn_neighbors < 1:
        raise ValueError("SMOTE and ENN neighbors must be positive.")
    _, counts = np.unique(y, return_counts=True)
    if len(counts) != 2 or counts.min() <= neighbors:
        raise ValueError("Each training class needs more rows than smote_neighbors.")
    seed = int(cfg.get("seed", 42))
    sampler = SMOTEENN(
        random_state=seed,
        smote=SMOTE(random_state=seed, k_neighbors=neighbors, sampling_strategy="auto"),
        enn=EditedNearestNeighbours(n_neighbors=enn_neighbors, sampling_strategy="all"))
    values, labels = sampler.fit_resample(x, y)
    if len(np.unique(labels)) != 2:
        raise ValueError("SMOTE-ENN removed a training class.")
    return np.asarray(values, dtype=float), np.asarray(labels, dtype=int)


def select_features(cfg):
    previous = load_state(cfg)
    if "windows" not in previous:
        raise ValueError("Run windows.py before features.py.")
    frame = previous["data"]
    specs = make_window_specs(cfg)
    validate_windows(frame, specs, cfg)
    # Reset downstream model state before fitting features.
    state = {"data": frame, "candidate_features": previous["candidate_features"], "windows": specs}
    save_state(cfg, state)
    raw_windows = []
    for spec in specs:
        train = frame[frame["year"].between(spec["train_start"], spec["train_end"])].copy()
        valid = frame[frame["year"].isin(spec["validation_years"])].copy()
        raw_windows.append((spec, train, valid))

    candidates = list(state["candidate_features"])
    eligible = [name for name in candidates
                if all(not numeric_frame(train, [name])[name].isna().all()
                       for _, train, _ in raw_windows)]
    if not eligible:
        raise ValueError("No feature is observed in every training window.")
    prepared = []
    rank_vectors = []
    seed = int(cfg.get("seed", 42))
    for spec, train, valid in raw_windows:
        preprocessor = fit_preprocessor(train, eligible, cfg)
        x_train = transform(train, preprocessor)
        x_validation = transform(valid, preprocessor)
        y_train = train["target"].to_numpy(dtype=int)
        y_validation = valid["target"].to_numpy(dtype=int)
        xgb = xgb_model(cfg)
        xgb.fit(x_train, y_train)
        l1 = LogisticRegression(penalty="l1", solver="liblinear",
                                C=float(cfg.get("features", {}).get("l1_c", 1)),
                                max_iter=1000, random_state=seed)
        l1.fit(StandardScaler().fit_transform(x_train), y_train)
        xgb_rank = pd.Series(xgb.feature_importances_).rank(ascending=False, method="average").to_numpy()
        l1_rank = pd.Series(np.abs(l1.coef_[0])).rank(ascending=False, method="average").to_numpy()
        rank_vectors.append((xgb_rank + l1_rank) / 2)
        prepared.append({"spec": spec, "train": train, "valid": valid,
                         "x_train": x_train, "x_validation": x_validation,
                         "y_train": y_train, "y_validation": y_validation})

    mean_rank = np.mean(rank_vectors, axis=0)
    order = sorted(range(len(eligible)), key=lambda i: (mean_rank[i], i))
    ranked = [eligible[i] for i in order]
    curve = []
    for size in range(1, len(ranked) + 1):
        columns = order[:size]
        aucs = []
        for window in prepared:
            model = xgb_model(cfg)
            model.fit(window["x_train"][:, columns], window["y_train"])
            scores = model.predict_proba(window["x_validation"][:, columns])[:, 1]
            aucs.append(float(roc_auc_score(window["y_validation"], scores)))
        curve.append({"feature_count": size, "mean_auc": float(np.mean(aucs)),
                      "se_auc": float(np.std(aucs, ddof=1) / np.sqrt(len(aucs)))})
        print(f"Features {size}/{len(ranked)}: validation AUC {curve[-1]['mean_auc']:.4f}", flush=True)
    best = max(curve, key=lambda row: row["mean_auc"])
    cutoff = best["mean_auc"] - best["se_auc"]
    chosen = min(row["feature_count"] for row in curve if row["mean_auc"] >= cutoff - 1e-15)
    selected = ranked[:chosen]

    windows = []
    for window in prepared:
        preprocessor = fit_preprocessor(window["train"], selected, cfg)
        x_train, y_train = resample_training(
            transform(window["train"], preprocessor), window["y_train"], cfg)
        windows.append({**window["spec"], "x_train": x_train, "y_train": y_train,
                        "x_validation": transform(window["valid"], preprocessor),
                        "y_validation": window["y_validation"],
                        "preprocessor": preprocessor, "validation_frame": window["valid"],
                        "train_rows_before_resampling": len(window["train"])})
    state["features"] = selected
    state["windows"] = windows
    state["feature_curve"] = pd.DataFrame(curve)
    state["feature_cutoff"] = float(cutoff)
    ranking = pd.DataFrame({"feature": ranked, "rank": range(1, len(ranked) + 1),
                            "mean_rank": [float(mean_rank[i]) for i in order],
                            "selected": [name in selected for name in ranked]})
    excluded = [name for name in candidates if name not in eligible]
    if excluded:
        ranking = pd.concat([ranking, pd.DataFrame(
            {"feature": excluded, "rank": np.nan, "mean_rank": np.nan, "selected": False})],
            ignore_index=True)
    save_csv(cfg, "features.csv", ranking)
    save_csv(cfg, "windows.csv", pd.DataFrame(specs))
    save_state(cfg, state)
    return state


def main(argv=None):
    cfg = cli_config("Select features and prepare training windows.", argv)
    state = select_features(cfg)
    print(f"Selected {len(state['features'])} features; prepared {len(state['windows'])} windows.")


if __name__ == "__main__":
    main()
