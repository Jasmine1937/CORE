"""Create chronological training and validation windows."""
from __future__ import annotations

import pandas as pd

from clean import cli_config, load_state, save_csv, save_state


def _positive_integer(value, name):
    number = int(value)
    if isinstance(value, bool) or number != float(value) or number < 1:
        raise ValueError(name + " must be a positive integer.")
    return number


def make_window_specs(cfg):
    settings = cfg["windows"]
    start = int(settings["start_year"])
    if start != float(settings["start_year"]):
        raise ValueError("start_year must be an integer.")
    length = _positive_integer(settings["train_years"], "train_years")
    count = _positive_integer(settings["count"], "count")
    validation = _positive_integer(settings.get("validation_years", 1), "validation_years")
    step = _positive_integer(settings.get("step_years", 1), "step_years")
    if count < 2:
        raise ValueError("Feature selection requires at least two development windows.")
    final = [int(year) for year in settings["final_years"]]
    if not final or len(final) != len(set(final)):
        raise ValueError("final_years must be a nonempty list of unique years.")
    if any(y != float(raw) for y, raw in zip(final, settings["final_years"])):
        raise ValueError("Final years must be integers.")
    specs = []
    for index in range(count):
        first = start + index * step
        last = first + length - 1
        years = list(range(last + 1, last + 1 + validation))
        if years[-1] >= min(final):
            raise ValueError("All development windows must end before the final period.")
        specs.append({"index": index + 1, "train_start": first, "train_end": last,
                      "validation_year": years[-1], "validation_years": years})
    return specs


def validate_windows(frame, specs, cfg):
    available = set(frame["year"].unique())
    final = set(map(int, cfg["windows"]["final_years"]))
    if final - available:
        raise ValueError("Missing final years: " + ", ".join(map(str, sorted(final - available))))
    for spec in specs:
        required = set(range(spec["train_start"], spec["train_end"] + 1)) | set(spec["validation_years"])
        if required - available:
            raise ValueError(f"Window {spec['index']} is missing years: {sorted(required - available)}")
        train = frame[frame["year"].between(spec["train_start"], spec["train_end"])]
        valid = frame[frame["year"].isin(spec["validation_years"])]
        if train["target"].nunique() != 2 or valid["target"].nunique() != 2:
            raise ValueError(f"Window {spec['index']} needs both classes in training and validation.")


def build_windows(cfg):
    old = load_state(cfg)
    specs = make_window_specs(cfg)
    validate_windows(old["data"], specs, cfg)
    state = {"data": old["data"], "candidate_features": old["candidate_features"], "windows": specs}
    save_csv(cfg, "windows.csv", pd.DataFrame(specs))
    save_state(cfg, state)
    return state


def main(argv=None):
    cfg = cli_config("Define chronological development windows.", argv)
    state = build_windows(cfg)
    print(f"Saved {len(state['windows'])} training/validation windows.")


if __name__ == "__main__":
    main()
