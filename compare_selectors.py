"""Compare CORE and four selectors using a common candidate archive and output budget."""
from __future__ import annotations

import copy
from pathlib import Path
import pandas as pd
from clean import cli_config, load_state, save_state
from core import core_s, core_g, select_candidates, characterize_candidates
from test import evaluate_selection
from ablation import select_gp


def run_selectors(state, cfg):
    parameters = copy.deepcopy(cfg["core"])
    parameters["random_seed"] = int(cfg["seed"])
    parameters["final_test_years"] = list(cfg["windows"]["final_years"])
    parameters["selection_budget"] = 3
    labels = {
        "auc_best": "AUC-best", "one_se_simplest": "1SE-simplest",
        "pareto_knee": "Pareto-knee", "one_se_random": "1SE-Random",
    }
    rows, selections = [], []
    for variant, group in state["prepared"].groupby("variant", sort=True):
        menu, _ = core_s(group, parameters)
        full, _ = core_g(menu, parameters)
        choices = [("CORE", full)]
        for method, label in labels.items():
            chosen, _ = select_candidates(group, parameters, method=method, reference_menu=menu)
            chosen = characterize_candidates(chosen, menu, parameters, errors="record")
            choices.append((label, chosen))
        gp = select_gp(state["prepared_gp"].loc[state["prepared_gp"].variant == variant])
        gp = characterize_candidates(gp, menu, parameters, errors="record")
        choices.append(("GP-only", gp))
        for label, chosen in choices:
            chosen = chosen.copy()
            chosen["method"] = label
            selections.append(chosen)
            evaluated = evaluate_selection(state, chosen, cfg, method=label)
            complete = len(chosen) == (1 if label == "GP-only" else 3)
            evaluated["comparison"] = "GP_reference" if label == "GP-only" else ("Q3" if complete else "Q3_incomplete")
            if not complete:
                print(f"{variant} {label}: budget shortfall; selected {len(chosen)} distinct eligible formulas.")
            evaluated["n_selected"] = len(chosen)
            evaluated["selection_budget"] = 1 if label == "GP-only" else 3
            rows.append(evaluated)
    if not rows:
        raise ValueError("No prepared archives are available.")
    state["selectors_selected"] = pd.concat(selections, ignore_index=True)
    return pd.concat(rows, ignore_index=True)


def main(argv=None):
    cfg = cli_config("Compare frozen archive selectors.", argv)
    state = load_state(cfg)
    results = run_selectors(state, cfg)
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "selectors.csv", index=False)
    save_state(cfg, state)
    print(f"Saved {len(results)} selector final-year records.")


if __name__ == "__main__":
    main()
