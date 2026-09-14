"""Load a labeled panel and manage configuration, preprocessing and saved state."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_config(path="config.json"):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8-sig") as stream:
        cfg = json.load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a JSON object.")
    cfg.setdefault("seed", 42)
    cfg.setdefault("preprocessing", {})
    cfg.setdefault("features", {})
    cfg.setdefault("data", {})
    for key, value in {"id": "entity_id", "year": "year", "target": "target",
                       "zscore": "zscore", "features": None}.items():
        cfg["data"].setdefault(key, value)
    if not cfg["data"].get("file"):
        raise ValueError("data.file must specify a labeled panel CSV.")
    for owner, key in ((cfg["data"], "file"), (cfg, "output")):
        value = Path(owner.get(key, "output")).expanduser()
        owner[key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    return cfg


def cli_config(description, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--out", help="Override the output directory.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.out:
        cfg["output"] = str(Path(args.out).expanduser().resolve())
    return cfg


def load_state(cfg):
    path = Path(cfg["output"]) / "state.pkl"
    if not path.is_file():
        raise FileNotFoundError("Run clean.py first; state.pkl does not exist.")
    with path.open("rb") as stream:
        state = pickle.load(stream)
    if not isinstance(state, dict):
        raise ValueError("state.pkl must contain a dictionary.")
    return state


def save_state(cfg, state):
    folder = Path(cfg["output"])
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "state.pkl").open("wb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)


def save_csv(cfg, name, frame):
    folder = Path(cfg["output"])
    folder.mkdir(parents=True, exist_ok=True)
    destination = (folder / name).resolve()
    if destination == Path(cfg["data"]["file"]).resolve():
        raise ValueError("Input and output CSV paths must differ.")
    frame.to_csv(destination, index=False)


def numeric_frame(frame, features):
    names = list(features)
    if not names or len(names) != len(set(names)):
        raise ValueError("Provide a nonempty, unique list of feature columns.")
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError("Missing feature columns: " + ", ".join(sorted(missing)))
    return frame[names].apply(pd.to_numeric, errors="coerce").astype(float).replace(
        [np.inf, -np.inf], np.nan)


def fit_preprocessor(frame, features, cfg):
    """Fit clipping limits, medians and scaling parameters on training data."""
    if frame.empty:
        raise ValueError("Training data are empty.")
    final = cfg["windows"]["final_years"]
    if not final:
        raise ValueError("Set windows.final_years.")
    if "year" in frame and (frame["year"] >= min(map(int, final))).any():
        raise ValueError("Final-period observations cannot fit preprocessing.")
    settings = cfg.get("preprocessing", {})
    lower = float(settings.get("lower_quantile", .01))
    upper = float(settings.get("upper_quantile", .99))
    if not 0 <= lower < upper <= 1:
        raise ValueError("Preprocessing quantiles must satisfy 0 <= lower < upper <= 1.")
    values = numeric_frame(frame, features)
    unavailable = values.columns[values.isna().all()].tolist()
    if unavailable:
        raise ValueError("Entirely missing training columns: " + ", ".join(unavailable))
    lo, hi = values.quantile(lower), values.quantile(upper)
    clipped = values.clip(lower=lo, upper=hi, axis=1)
    medians = clipped.median()
    filled = clipped.fillna(medians)
    minima = filled.min()
    spans = (filled.max() - minima).replace(0., 1.)
    return {"features": list(features), "lower": lo.tolist(), "upper": hi.tolist(),
            "medians": medians.tolist(), "minima": minima.tolist(), "spans": spans.tolist()}


def transform(frame, preprocessor):
    """Transform predictors using fitted training parameters."""
    values = numeric_frame(frame, preprocessor["features"]).to_numpy()
    values = np.clip(values, preprocessor["lower"], preprocessor["upper"])
    values = np.where(np.isnan(values), np.asarray(preprocessor["medians"]), values)
    values = (values - np.asarray(preprocessor["minima"])) / np.asarray(preprocessor["spans"])
    if not np.isfinite(values).all():
        raise ValueError("Transformed values must be finite.")
    return values


def clean_data(cfg):
    settings = cfg["data"]
    source = Path(settings["file"])
    encoding = settings.get("encoding", "utf-8-sig")
    with source.open(encoding=encoding, newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError("The input CSV needs unique column names.")
    frame = pd.read_csv(source, dtype="string", encoding=encoding)
    roles = {key: settings[key] for key in ("id", "year", "target", "zscore")}
    if len(set(roles.values())) != 4:
        raise ValueError("Data id, year, target and zscore must refer to different columns.")
    missing = set(roles.values()) - set(frame.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    candidates = settings.get("features")
    if candidates is None:
        candidates = [name for name in frame.columns if name not in roles.values()]
    if not isinstance(candidates, list) or not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("data.features must be null or a nonempty unique column list.")
    if set(candidates) & (set(roles.values()) | {"entity_id", "year", "target", "zscore"}):
        raise ValueError("Identifiers, year, target and the baseline zscore are not candidate features.")
    absent = set(candidates) - set(frame.columns)
    if absent:
        raise ValueError("Missing candidate columns: " + ", ".join(sorted(absent)))
    rename = {roles["id"]: "entity_id", roles["year"]: "year",
              roles["target"]: "target", roles["zscore"]: "zscore"}
    frame = frame[list(roles.values()) + candidates].rename(columns=rename)
    frame["entity_id"] = frame["entity_id"].str.strip()
    if frame["entity_id"].isna().any() or frame["entity_id"].eq("").any():
        raise ValueError("Every row needs an entity identifier.")
    for name in ("year", "target"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if frame[name].isna().any() or not np.isfinite(frame[name].to_numpy(dtype=float)).all():
            raise ValueError(name + " must be observed and finite.")
    years = frame["year"].to_numpy(dtype=float)
    if not np.equal(years, np.floor(years)).all():
        raise ValueError("Years must be integers.")
    if not frame["target"].isin([0, 1]).all():
        raise ValueError("Targets must be observed 0 or 1 labels.")
    frame["year"] = frame["year"].astype(int)
    frame["target"] = frame["target"].astype(int)
    frame[candidates + ["zscore"]] = numeric_frame(frame, candidates + ["zscore"])
    frame = frame.drop_duplicates().copy()
    if frame.duplicated(["entity_id", "year"]).any():
        raise ValueError("Conflicting records for the same entity and year.")
    frame = frame.sort_values(["year", "entity_id"], kind="stable").reset_index(drop=True)
    state = {"data": frame, "candidate_features": candidates}
    save_csv(cfg, "data.csv", frame)
    save_state(cfg, state)
    return state


def main(argv=None):
    cfg = cli_config("Read and standardize a labeled panel.", argv)
    state = clean_data(cfg)
    print(f"Saved {len(state['data'])} rows and {len(state['candidate_features'])} candidate columns.")


if __name__ == "__main__":
    main()
