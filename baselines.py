"""Fit, select and evaluate frozen baseline models."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from clean import cli_config, load_state, save_state, transform
from prepare import evaluate_metrics, youden_threshold


def build_estimator(name, parameters, options, seed, n_features=None):
    """Construct a baseline estimator from configured parameters and options."""
    kwargs = {**options, **parameters}
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=seed, **kwargs)
    if name == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis(**kwargs)
    if name == "dt":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=seed, **kwargs)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, **kwargs)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, **kwargs)
    if name == "ngb":
        from ngboost import NGBClassifier
        from ngboost.distns import Bernoulli
        from ngboost.scores import LogScore
        # NGBoost >= 0.5.11 handles singular Fisher matrices and keeps at least one column.
        return NGBClassifier(Dist=Bernoulli, Score=LogScore,
                             random_state=seed, **kwargs)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        if "hidden_layer_sizes" in kwargs:
            kwargs["hidden_layer_sizes"] = tuple(kwargs["hidden_layer_sizes"])
        return MLPClassifier(random_state=seed, **kwargs)
    if name == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier
        return TabNetClassifier(seed=seed, **kwargs)
    raise ValueError(f"Unknown trainable baseline: {name}")


def score_model(fitted, frame, matrix=None):
    if fitted["model"] == "zscore":
        z = pd.to_numeric(frame["zscore"], errors="raise").to_numpy(float)
        return -np.where(np.isfinite(z), z, fitted["z_median"])
    x = transform(frame, fitted["preprocessor"]) if matrix is None else matrix
    if fitted["model"] == "tabnet":
        x = np.asarray(x, dtype=np.float32)
    estimator = fitted["estimator"]
    # NGBoost supplies Bernoulli probabilities in class order [0, 1].
    classes = np.asarray([0, 1] if fitted["model"] == "ngb" else estimator.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError("A fitted baseline must have positive class 1.")
    return np.asarray(estimator.predict_proba(x), dtype=float)[:, positive[0]]


def metric_row(fitted, frame, year, phase, matrix=None):
    scores = score_model(fitted, frame, matrix)
    metrics = evaluate_metrics(frame.target.to_numpy(int), scores, fitted["threshold"])
    return {
        "method": fitted["model"], "model": fitted["model"],
        "source_window": fitted["source_window"],
        "source_validation_year": fitted["source_validation_year"],
        "predictor_id": f'{fitted["model"]}_w{fitted["source_window"]}',
        "year": int(year), "phase": phase, "threshold": fitted["threshold"],
        "seed": fitted["seed"], "n_selected": 1,
        "auc": metrics["auc"], "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "n": len(frame), "n_positive": int(frame.target.sum()),
        "parameters": json.dumps(fitted["parameters"], sort_keys=True),
    }


def run_baselines(state, cfg):
    panel, windows = state["data"], state["windows"]
    options = cfg["baselines"]
    names = [name.lower() for name in options["models"]]
    allowed = {"zscore", "lr", "lda", "dt", "rf", "xgb", "ngb", "mlp", "tabnet"}
    if not names or len(set(names)) != len(names) or set(names) - allowed:
        raise ValueError("Choose unique paper baseline names.")
    if not windows:
        raise ValueError("Prepare source windows first.")
    final_years = list(cfg["windows"]["final_years"])
    development_years = sorted({int(y) for w in windows for y in w["validation_years"]})
    if set(development_years) & set(final_years):
        raise ValueError("Development and final years must be disjoint.")
    seed = int(cfg["seed"]) % (2 ** 32)
    matrices = {}
    forward_rows, final_rows, selected_models = [], [], []
    for name in names:
        source_models = []
        for window in windows:
            wid = int(window["index"])
            print(f'Baseline {name}: source window {wid}, validation {window["validation_years"]}', flush=True)
            validation = window["validation_frame"]
            xv, yv = window["x_validation"], np.asarray(window["y_validation"], dtype=int)
            if len(validation) != len(yv) or not np.array_equal(validation.target.to_numpy(int), yv):
                raise ValueError("Validation rows and matrices must keep identical order.")
            fitted = {
                "model": name, "source_window": wid,
                "source_validation_year": int(window["validation_year"]),
                "preprocessor": window["preprocessor"], "seed": seed,
                "parameters": {}, "threshold": 0., "estimator": None,
            }
            if name == "zscore":
                training = panel[panel.year.between(window["train_start"], window["train_end"])]
                z = pd.to_numeric(training["zscore"], errors="raise").to_numpy(float)
                z = z[np.isfinite(z)]
                if not len(z):
                    raise ValueError(f"Z-score is missing throughout source training window {wid}.")
                fitted["z_median"] = float(np.median(z))
                scores = score_model(fitted, validation)
            else:
                fixed = options["settings"][name]
                best_auc = -np.inf
                for parameters in ParameterGrid(options["search_spaces"][name]):
                    candidate = build_estimator(name, parameters, fixed["estimator"], seed, n_features=window["x_train"].shape[1])
                    if name == "tabnet":
                        candidate.fit(
                            np.asarray(window["x_train"], dtype=np.float32),
                            np.asarray(window["y_train"], dtype=int),
                            eval_set=[(np.asarray(xv, dtype=np.float32), yv)],
                            **fixed["fit"],
                        )
                    else:
                        # Fit the prepared SMOTE-ENN training matrix.
                        candidate.fit(window["x_train"], window["y_train"], **fixed["fit"])
                    trial = {**fitted, "estimator": candidate}
                    prediction = score_model(trial, validation, xv)
                    auc = evaluate_metrics(yv, prediction, 0.5)["auc"]
                    if not np.isfinite(auc):
                        raise ValueError(f"Undefined source-validation AUC for {name}, window {wid}.")
                    if auc > best_auc:  # Equal validation AUC retains the first grid setting.
                        best_auc, scores = float(auc), prediction
                        fitted["estimator"], fitted["parameters"] = candidate, dict(parameters)
                if fitted["estimator"] is None:
                    raise ValueError(f"Empty parameter grid for {name}.")
            fitted["threshold"] = float(youden_threshold(yv, scores))
            source_aucs = []
            for year in development_years:
                if year < min(window["validation_years"]):
                    continue
                frame = panel[panel.year == year]
                if frame.empty:
                    raise ValueError(f"Missing development cohort {year}.")
                key = (wid, year)
                if name != "zscore" and key not in matrices:
                    matrices[key] = transform(frame, window["preprocessor"])
                phase = "source_validation" if year in window["validation_years"] else "rolling_forward"
                row = metric_row(fitted, frame, year, phase, matrices.get(key))
                source_aucs.append(row["auc"])
                forward_rows.append(row)
            if not source_aucs or not np.isfinite(source_aucs).all():
                raise ValueError(f"Undefined development mean for {name}, window {wid}.")
            fitted["development_mean_auc"] = float(np.mean(source_aucs))
            source_models.append(fitted)
        # Freeze the model with the highest mean AUC across its development cohorts.
        chosen = max(source_models, key=lambda model: model["development_mean_auc"])
        selected_models.append(chosen)
        for row in forward_rows:
            if row["model"] == name:
                row["selected_for_final"] = row["source_window"] == chosen["source_window"]
        for year in final_years:
            frame = panel[panel.year == year]
            if frame.empty:
                raise ValueError(f"Missing final cohort {year}.")
            row = metric_row(chosen, frame, year, "final")
            row["development_mean_auc"] = chosen["development_mean_auc"]
            final_rows.append(row)
    state["baselines"] = selected_models
    return pd.DataFrame(forward_rows), pd.DataFrame(final_rows)


def main(argv=None):
    cfg = cli_config("Fit and evaluate frozen paper baselines.", argv)
    state = load_state(cfg)
    forward, final = run_baselines(state, cfg)
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    forward.to_csv(output / "baselines_forward.csv", index=False)
    final.to_csv(output / "baselines_final.csv", index=False)
    save_state(cfg, state)
    print(f"Saved {len(forward)} development rows and {len(final)} final rows.")


if __name__ == "__main__":
    main()
