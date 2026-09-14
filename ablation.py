"""Compare one GP-only predictor with one CORE predictor for each GP variant (Q1)."""
from __future__ import annotations

import copy
from pathlib import Path
import numpy as np
import pandas as pd
from clean import cli_config, load_state, save_state
from core import core_s, core_g, characterize_candidates
from test import evaluate_selection


def select_gp(champions):
    """Select a final-generation champion by mean development AUC."""
    if champions.empty:
        raise ValueError("No prepared GP-only champions are available.")
    means = []
    for row in champions.itertuples():
        values = np.asarray([row.validation_auc, *row.forward_auc], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("GP source selection requires finite development AUCs.")
        means.append(float(values.mean()))
    work = champions.copy()
    work["development_mean_auc"] = means
    # np.argmax resolves ties by source-window order.
    return work.iloc[[int(np.argmax(means))]].copy()


def run_ablation(state, cfg):
    prepared, champions = state["prepared"], state["prepared_gp"]
    parameters = copy.deepcopy(cfg["core"])
    parameters["random_seed"] = int(cfg["seed"])
    parameters["final_test_years"] = list(cfg["windows"]["final_years"])
    parameters["selection_budget"] = 1
    selections, rows = [], []
    for variant, group in prepared.groupby("variant", sort=True):
        menu, _ = core_s(group, parameters)
        selected, _ = core_g(menu, parameters)
        gp = select_gp(champions[champions.variant == variant])
        gp = characterize_candidates(gp, menu, parameters, errors="record")
        for label, chosen in (("GP-only", gp), ("GP+CORE", selected)):
            chosen = chosen.copy()
            chosen["method"] = label
            selections.append(chosen)
            evaluated = evaluate_selection(state, chosen, cfg, method=label)
            evaluated["comparison"] = "Q1"
            evaluated["n_selected"] = len(chosen)
            evaluated["selection_budget"] = 1
            rows.append(evaluated)
    if not rows:
        raise ValueError("No GP variants are available for the Q1 comparison.")
    state["ablation_selected"] = pd.concat(selections, ignore_index=True)
    return pd.concat(rows, ignore_index=True)


def main(argv=None):
    cfg = cli_config("Compare one GP expression with one CORE expression.", argv)
    state = load_state(cfg)
    results = run_ablation(state, cfg)
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "ablation.csv", index=False)
    save_state(cfg, state)
    print(f"Saved {len(results)} Q1 final-year records.")


if __name__ == "__main__":
    main()
