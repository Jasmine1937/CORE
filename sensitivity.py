"""Evaluate single-seed parameter sensitivity on frozen G0 archives."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import qmc, spearmanr
from clean import cli_config, load_state
from core import core_s, core_g
from test import evaluate_selection

COMPLEXITY_KEYS = ("T", "D", "O", "U", "Nx", "G", "H")
GREEDY_KEYS = ("auc", "stability", "complexity", "coverage")
PROPERTIES = (
    "C_global", "T", "D", "O", "U", "Nx", "G", "H",
    "structural_stability", "temporal_sd_z", "temporal_drop_z",
    "temporal_stability", "stability",
)
METRICS = ("auc", "macro_f1", "balanced_accuracy")


def make_design(cfg):
    """Generate a 15-dimensional maximin Latin-hypercube design with the default configuration."""
    options = cfg["sensitivity"]
    count, attempts = int(options["count"]), int(options["design_attempts"])
    if count < 2 or attempts < 1:
        raise ValueError("Sensitivity requires at least two settings and one design attempt.")
    seed = int(options["design_seed"])
    best_distance, best = -np.inf, None
    for attempt in range(attempts):
        points = qmc.LatinHypercube(d=15, scramble=True, seed=seed + attempt).random(count - 1)
        distance = float(pdist(points).min()) if len(points) > 1 else np.inf
        if distance > best_distance:
            best_distance, best = distance, points
    base = copy.deepcopy(cfg["core"])
    base["random_seed"] = int(cfg["seed"])
    base["final_test_years"] = list(cfg["windows"]["final_years"])
    base["selection_budget"] = 3
    low, high = map(float, options["weight_factor_range"])
    if not 0 < low <= high:
        raise ValueError("Weight factors must be positive and ordered.")

    def linear(name, unit):
        bounds = options[name]
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError(f"Invalid sensitivity range: {name}")
        return float(bounds[0] + unit * (bounds[1] - bounds[0]))

    design = [("C000", base)]
    for index, point in enumerate(best, 1):
        parameters = copy.deepcopy(base)
        factors = np.exp(np.log(low) + point * np.log(high / low))
        parameters["complexity_weights"] = {
            key: float(base["complexity_weights"][key] * factors[j])
            for j, key in enumerate(COMPLEXITY_KEYS)
        }
        parameters["greedy_weights"] = {
            key: float(base["greedy_weights"][key] * factors[11 + j])
            for j, key in enumerate(GREEDY_KEYS)
        }
        parameters["stability_lambda"] = linear("lambda_range", point[7])
        parameters["shrinkage_nu"] = linear("nu_range", point[8])
        parameters["std_weight"] = linear("std_weight_range", point[9])
        parameters["worst_drop_weight"] = 1. - parameters["std_weight"]
        parameters["temporal_weight"] = linear("temporal_weight_range", point[10])
        parameters["structural_weight"] = 1. - parameters["temporal_weight"]
        design.append((f"C{index:03d}", parameters))
    return design


def parameter_columns(parameters):
    row = {}
    for prefix, group in (("a", "complexity_weights"), ("b", "greedy_weights")):
        total = float(sum(parameters[group].values()))
        if total <= 0:
            raise ValueError("Relative weights require a positive total.")
        for key, value in parameters[group].items():
            row[f"{prefix}_{key}"] = float(value)
            row[f"{prefix}_share_{key}"] = float(value) / total
    for name in ("stability_lambda", "shrinkage_nu", "std_weight",
                 "worst_drop_weight", "temporal_weight", "structural_weight",
                 "selection_budget"):
        row[name] = parameters[name]
    return row


def summarize_selection(annual, selected, cfg):
    """Compute each formula's annual sample SD before averaging formulas."""
    rows = []
    expected = set(map(int, cfg["windows"]["final_years"]))
    for (_, _), group in annual.groupby(["source_window", "expression"], sort=False):
        if group.year.duplicated().any() or set(group.year) != expected:
            raise ValueError("Each selected formula needs every final year exactly once.")
        item = {}
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            item[f"{metric}_mean"] = float(values.mean()) if np.isfinite(values).all() else np.nan
            item[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 and np.isfinite(values).all() else np.nan
        rows.append(item)
    if not rows:
        raise ValueError("No final predictions for a selected configuration.")
    per_formula = pd.DataFrame(rows)
    summary = {
        column: float(values.mean()) if np.isfinite(values.to_numpy()).all() else np.nan
        for column, values in per_formula.items()
    }
    for name in PROPERTIES:
        if name not in selected:
            raise ValueError(f"CORE did not provide the realized property {name}.")
        values = selected[name].to_numpy(float)
        summary[name] = float(values.mean()) if np.isfinite(values).all() else np.nan
    return summary


def run_sensitivity(state, cfg):
    candidates = state["prepared"].loc[state["prepared"].variant == "G0"].copy()
    if candidates.empty:
        raise ValueError("Sensitivity requires prepared G0 archives.")
    if set(cfg["windows"]["final_years"]) & {
        int(y) for window in state["windows"] for y in window["validation_years"]
    }:
        raise ValueError("Final years cannot enter CORE development screening.")
    selections, configurations, menus, all_selected = {}, {}, {}, []
    # Select formulas for every prespecified configuration before final evaluation.
    design = make_design(cfg)
    for position, (config_id, parameters) in enumerate(design, 1):
        if position == 1 or position % 25 == 0 or position == len(design):
            print(f"CORE sensitivity: selecting {position}/{len(design)}", flush=True)
        menu, _ = core_s(candidates, parameters)
        chosen, _ = core_g(menu, parameters)
        if chosen.empty:
            raise ValueError(f"No expression selected for {config_id}.")
        chosen = chosen.copy()
        chosen["method"] = "sensitivity"
        selections[config_id], configurations[config_id] = chosen, parameters
        menus[config_id] = len(menu)
        all_selected.append(chosen)
    unique = pd.concat(all_selected, ignore_index=True).drop_duplicates(["source_window", "expression"])
    print(f"Evaluating {len(unique)} distinct source/formula pairs on final years.", flush=True)
    evaluated = evaluate_selection(state, unique, cfg, method="sensitivity")
    cache = {
        (int(source), str(expression)): group.copy()
        for (source, expression), group in evaluated.groupby(["source_window", "expression"], sort=False)
    }
    default = set(selections["C000"]["canonical_expression"])
    records = []
    for config_id, selected in selections.items():
        annual = pd.concat(
            [cache[(int(row.source_window), str(row.expression))] for row in selected.itertuples()],
            ignore_index=True,
        )
        current = set(selected["canonical_expression"])
        union = default | current
        record = {
            "config_id": config_id, "seed": int(cfg["seed"]), "variant": "G0",
            "design_seed": int(cfg["sensitivity"]["design_seed"]),
            "n_selected": len(selected), "menu_size": menus[config_id],
            "jaccard": len(default & current) / len(union) if union else 1.,
            "expressions": json.dumps(
                [{"source_window": int(row.source_window), "expression": str(row.expression)}
                 for row in selected.itertuples()], separators=(",", ":")
            ),
            **parameter_columns(configurations[config_id]),
            **summarize_selection(annual, selected, cfg),
        }
        records.append(record)
    return pd.DataFrame(records)


def association_statistics(data, cfg):
    """Fit univariate HC3 regressions and Spearman correlations across configurations from one GP seed."""
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    shares = [column for column in data if column.startswith(("a_share_", "b_share_"))]
    performance = [
        "C_global", *COMPLEXITY_KEYS, "structural_stability", "temporal_stability",
        "stability", *shares, "stability_lambda", "shrinkage_nu",
        "std_weight", "temporal_weight",
    ]
    dispersion = [
        "structural_stability", "temporal_sd_z", "temporal_drop_z",
        "temporal_stability", "stability", "b_share_stability",
    ]
    if data.config_id.duplicated().any() or data.seed.nunique() != 1:
        raise ValueError("Use one observed summary per configuration from one GP seed.")
    alpha = float(cfg["sensitivity"]["alpha"])
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    rows = []
    for panel, predictors, suffix in (
        ("A_mean_performance", performance, "mean"),
        ("B_temporal_dispersion", dispersion, "sd"),
    ):
        for predictor in predictors:
            for metric in METRICS:
                outcome = f"{metric}_{suffix}"
                if predictor not in data or outcome not in data:
                    raise ValueError(f"Missing sensitivity column: {predictor} or {outcome}")
                x, y = data[predictor].to_numpy(float), data[outcome].to_numpy(float)
                step = 1. if predictor in (*COMPLEXITY_KEYS, "shrinkage_nu") else (
                    .5 if predictor == "stability_lambda" else .1
                )
                row = {
                    "panel": panel, "predictor": predictor, "outcome": outcome,
                    "seed": int(cfg["seed"]), "n_configurations": len(data),
                    "effect_step": step,
                    "scope": "single_seed_configuration_association",
                    "intercept": np.nan, "slope": np.nan, "hc3_se": np.nan,
                    "hc3_p": np.nan, "slope_ci_low": np.nan, "slope_ci_high": np.nan,
                    "effect": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan,
                    "r_squared": np.nan,
                }
                if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
                    rows.append({**row, "status": "insufficient_defined_configurations"})
                    continue
                if np.ptp(x) == 0 or np.ptp(y) == 0:
                    rows.append({**row, "status": "constant_property_or_outcome"})
                    continue
                fitted = sm.OLS(y, sm.add_constant(x, has_constant="add")).fit(cov_type="HC3")
                interval = np.asarray(fitted.conf_int(alpha=alpha))[1]
                correlation = spearmanr(x, y)
                rows.append({
                    **row, "status": "estimated", "intercept": float(fitted.params[0]),
                    "slope": float(fitted.params[1]), "hc3_se": float(fitted.bse[1]),
                    "hc3_p": float(fitted.pvalues[1]), "slope_ci_low": float(interval[0]),
                    "slope_ci_high": float(interval[1]), "effect": float(fitted.params[1] * step),
                    "spearman_rho": float(correlation.statistic),
                    "spearman_p": float(correlation.pvalue), "r_squared": float(fitted.rsquared),
                })
    result = pd.DataFrame(rows)
    for statistic in ("hc3", "spearman"):
        result[f"{statistic}_q"] = np.nan
        for _, indices in result.groupby("panel").groups.items():
            observed = [i for i in indices if np.isfinite(result.loc[i, f"{statistic}_p"])]
            if observed:
                result.loc[observed, f"{statistic}_q"] = multipletests(
                    result.loc[observed, f"{statistic}_p"], alpha=alpha, method="fdr_bh"
                )[1]
    return result


def plot_sensitivity(data, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    pairs = (
        ("C_global", "auc_mean", "Complexity", "Final mean AUC"),
        ("stability", "auc_mean", "Stability", "Final mean AUC"),
        ("stability", "auc_sd", "Stability", "Annual AUC SD"),
    )
    default = data.loc[data.config_id == "C000"].iloc[0]
    for axis, (x, y, xlabel, ylabel) in zip(axes, pairs):
        usable = np.isfinite(data[x]) & np.isfinite(data[y])
        axis.scatter(data.loc[usable, x], data.loc[usable, y], s=13, alpha=.55)
        if np.isfinite(default[x]) and np.isfinite(default[y]):
            axis.scatter([default[x]], [default[y]], marker="*", s=100, color="darkred", label="Default")
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=.2)
    axes[0].legend(frameon=False)
    figure.suptitle("Single-seed frozen G0 sensitivity")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main(argv=None):
    cfg = cli_config("Evaluate prespecified frozen-G0 sensitivity.", argv)
    state = load_state(cfg)
    data = run_sensitivity(state, cfg)
    statistics = association_statistics(data, cfg)
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "sensitivity.csv", index=False)
    statistics.to_csv(output / "stats.csv", index=False)
    plot_sensitivity(data, output / "sensitivity.png")
    print(f"Saved {len(data)} settings and {len(statistics)} single-seed associations.")


if __name__ == "__main__":
    main()
