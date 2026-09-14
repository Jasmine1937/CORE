# CORE

CORE combines genetic programming, cross-window formula selection, and temporal evaluation for financial distress prediction. The workflow covers data preparation, GP and CORE, model comparisons, and parameter sensitivity. Settings are stored in [config.json](config.json).

## Installation

Use Python 3.11 and JDK 17. Make `java` and `javac` available on `PATH`.

From this directory, run the following commands in PowerShell:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
javac -version
java -version
```

On Linux or macOS, the environment's Python executable is `.venv/bin/python`.

## Input and configuration

Prepare a UTF-8 CSV with one row per company and financial year. Data sources and target construction are described in [data.md](data.md).

| Field | Default column | Content |
|---|---|---|
| Company identifier | `entity_id` | String identifier, including leading zeros |
| Financial year | `year` | Integer year |
| Two-year target | `target` | Observed binary label: 1 for distress, 0 otherwise |
| Z-score | `zscore` | Precomputed score, with higher values indicating better financial health |
| Candidate predictors | Remaining columns | Numeric financial variables available at the predictor date |

Set `data.file` and the column mappings in `config.json`. Use `data.features` to specify the candidate column names. With `data.features` set to `null`, the remaining columns form the candidate set. Feature selection determines the retained subset from validation performance.

Use plain numeric values, such as `1234.5`, with a consistent unit for each column. Represent missing financial values with empty cells. The input panel must contain observed targets and consistent company identifiers.

The `windows` section controls the first training year, training length, number of windows, validation length, step size, and final-test years. Defaults are five training years, one validation year, a one-year step, nine source windows, and a 2017-2021 final horizon. Feature selection requires at least two development windows. The final horizon follows the development period.

## 1. Data preparation

```powershell
.\.venv\Scripts\python.exe clean.py
.\.venv\Scripts\python.exe windows.py
.\.venv\Scripts\python.exe features.py
```

`clean.py` standardizes columns, checks identifiers and binary labels, converts financial values, and resolves identical duplicate rows. Conflicting company-year records produce an error. `windows.py` creates the chronological training and validation partitions.

`features.py` combines XGB and L1-regularized logistic-regression rankings across development windows. It evaluates ranked feature prefixes and retains the smallest subset within one empirical standard error of the best mean validation AUC.

For each source window, preprocessing parameters are estimated from its training observations. The configured defaults apply 1st/99th-percentile clipping, median imputation, and min-max scaling. Training-derived transformations are stored with the selected feature order. Each training matrix then passes through SMOTEENN. Validation, forward-evaluation, and final-test cohorts retain their observed rows and class distributions.

## 2. GP and CORE

```powershell
.\.venv\Scripts\python.exe gp.py
.\.venv\Scripts\python.exe prepare.py
.\.venv\Scripts\python.exe core.py
.\.venv\Scripts\python.exe test.py
.\.venv\Scripts\python.exe summary.py
```

`gp.py` compiles and runs G0, G1, and G2. Defaults are a population of 500, 200 generations, and an archive capacity of 200 candidates per run. G0 optimizes training AUC, G1 uses penalized scalar fitness, and G2 uses AUC and tree size as Pareto objectives. Source window `w` uses GP seed `seed + w`, with window indices starting at 1.

`prepare.py` computes validation metrics, expression complexity, and later-development performance. Complexity accounts for polynomial expansion and cancellation. `core.py` applies the five CORE-S screening steps and the CORE-G cross-window selection procedure. The default final budget is three distinct expressions per GP variant.

For forward evaluation, each CORE-S representative retains its source-window expression, feature mapping, preprocessing, and calibrated threshold while predicting subsequent development validation cohorts.

For final evaluation, each CORE-G expression receives a Youden threshold calibrated on its source validation cohort and subsequent development validation cohorts. The expression and its source preprocessing remain fixed. The resulting predictor is evaluated separately for each final year. Classification uses `score > threshold`; ties in the Youden objective select the largest threshold. Reported metrics include AUC, macro-F1, and balanced accuracy. Each expression is evaluated individually.

`summary.py` reports candidate counts, selected-formula counts, and GP execution times by source window.

## 3. Model comparisons

```powershell
.\.venv\Scripts\python.exe baselines.py
.\.venv\Scripts\python.exe ablation.py
.\.venv\Scripts\python.exe compare_selectors.py
```

`baselines.py` evaluates Z-score, LR, LDA, DT, RF, XGB, NGB, MLP, and TabNet. Trainable baselines use the prepared training matrices and the parameter grids in `config.json`. Validation AUC determines the parameters for each source-window model. Forward evaluation retains the fitted model and its source threshold. The source model with the highest mean AUC across its source and subsequent development validation years proceeds to final evaluation. The Z-score baseline uses `-zscore` as its risk score and a source-training median for missing Z-scores.

`ablation.py` compares GP and GP+CORE with one expression per method and GP variant (Q1). `compare_selectors.py` compares AUC-best, 1SE-simplest, Pareto-knee, 1SE-Random, and CORE using the candidate archives and a three-expression budget (Q3). A one-expression GP reference is identified separately as `GP_reference`. The tables contain annual results for individual expressions. `n_selected` gives the actual output count; `Q3_incomplete` identifies an eligible set smaller than the configured budget.

## 4. Parameter sensitivity

```powershell
.\.venv\Scripts\python.exe sensitivity.py
```

The default scan evaluates the CORE configuration plus 299 maximin Latin-hypercube combinations against the G0 candidate archive. Its 15 continuous dimensions comprise seven complexity weights, the recurrence exponent, the shrinkage strength, two stability mixture weights, and four global selection weights.

Each setting undergoes CORE selection, development-based threshold calibration, and final evaluation. Statistical summaries use univariate regressions with HC3 standard errors, Spearman correlations, and Benjamini-Hochberg adjusted p-values. The estimates describe associations among parameter settings for the configured G0 archive. A three-panel figure summarizes the scan.

## Outputs

The `output` setting selects the results directory. Each script accepts `--config path/to/config.json` and an optional `--out path/to/results` override. Relative paths in the configuration are resolved from the configuration file's directory.

| File | Contents |
|---|---|
| `data.csv`, `windows.csv`, `features.csv` | Standardized panel, window definitions, and feature rankings |
| `candidates.csv`, `formulas.csv` | GP candidate archives and selected formulas |
| `gp.csv` | Window-level counts and GP execution times |
| `forward.csv`, `final.csv` | CORE forward and final-year metrics |
| `baselines_forward.csv`, `baselines_final.csv` | Baseline development and final-year metrics |
| `ablation.csv`, `selectors.csv` | GP and selector comparisons |
| `sensitivity.csv`, `stats.csv`, `sensitivity.png` | Parameter scan, statistical summaries, and figure |

The results directory also contains `state.pkl`, which connects the workflow stages by storing prepared matrices, feature order, preprocessing parameters, candidates, and models. Its `work` subdirectory contains Java build files. After changing an upstream setting, rerun the affected preparation stage and the subsequent stages in order.

## References

The GP framework draws on the Java implementation accompanying Poli, Langdon, and McPhee, *A Field Guide to Genetic Programming* (2008). G1 and G2 implement the penalized and Pareto search variants within this framework. Third-party Python packages provide their respective license and attribution information in their distributions.
