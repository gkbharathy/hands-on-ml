# Example Run Tutorial: Chapter 2 End-to-End Machine Learning Project

## Note on how this record was produced

This document reconstructs the actual session in which Chapter 2 of this
repository was run. The commands below were executed through a Claude Code
assistant session (an agentic coding tool), acting on this machine at the
user's direction, not typed manually at an interactive terminal. This
matters for one practical reason: the commands do **not** appear in
`~/.bash_history` (that file was checked and contains only unrelated,
pre-existing history from other work). The source of truth for this
tutorial is therefore the session's own tool-call record, cross-checked
against `git log`, `git status`, `git diff`, the executed notebook file,
and file timestamps on disk. Every command, output, and file path below was
taken from one of those sources; nothing has been invented. Anything that
could not be confirmed this way is listed under "Open questions" rather
than guessed.

---

## 1. Overview

This repository (`gkbharathy/hands-on-ml`) is a fork of
[`ageron/handson-ml3`](https://github.com/ageron/handson-ml3), the notebook
collection accompanying the book *Hands-On Machine Learning with
Scikit-Learn, Keras and TensorFlow* (3rd edition).

The notebook run was **`02_end_to_end_machine_learning_project.ipynb`**
("Chapter 2 - End-to-end Machine Learning project", per the notebook's own
first markdown cell). It works through a full, guided ML workflow on the
California housing census dataset:

- Load and explore ~20,640 rows of California district-level census data
  (columns include `longitude`, `latitude`, `housing_median_age`,
  `total_rooms`, `total_bedrooms`, `population`, `households`,
  `median_income`, `median_house_value`, `ocean_proximity`), confirmed by
  `housing.info()` output showing `RangeIndex: 20640 entries, 0 to 20639`.
- Split the data into training and test sets (`train_test_split`, and later
  a stratified split via `StratifiedShuffleSplit`).
- Build preprocessing pipelines (imputation, scaling, custom transformers,
  a `ColumnTransformer`).
- Train and compare several regression models (linear regression, decision
  tree, random forest) via cross-validation.
- Tune the best-performing pipeline with `GridSearchCV`.
- Evaluate the final model on the held-out test set.
- Save the trained model with `joblib`.
- An "extra material" appendix demonstrates writing custom
  scikit-learn-compatible estimator/transformer classes (`FeatureFromRegressor`,
  `StandardScalerClone`) and validates them with
  `sklearn.utils.estimator_checks.check_estimator`.

**Task type:** supervised regression - predicting `median_house_value` for
a district from its other census features.

---

## 2. Prerequisites

Confirmed by inspecting the machine directly during the session:

| Item | Value | Source |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS (Noble Numbat), running under WSL2 (kernel `6.6.87.2-microsoft-standard-WSL2`) | `/etc/os-release`, session environment info |
| Python version | 3.11.7 | `python3 --version`, and `.venv/bin/python --version` |
| Environment manager | `venv` (Python's built-in virtual environment tool), created fresh for this project | this session |
| CPU | 20 logical processors available (`nproc`) | this session |
| RAM | 31 GiB total, ~28-29 GiB free at the time | `free -h` |
| GPU | Not used. No GPU-dependent package (e.g. TensorFlow) was installed or required for this notebook. | see Section 3 |

The repository's own `INSTALL.md` recommends Anaconda/`conda` with
`environment.yml` (creating an environment named `homl3`), and both
`requirements.txt` and `environment.yml` pin **scikit-learn 1.3** (`~=1.3.2`
/ `=1.3`). This tutorial did **not** follow that documented conda route; a
lighter-weight plain `venv` was used instead, with only the subset of
packages this specific notebook actually imports (see below). This is a
deliberate deviation from `INSTALL.md`, not an error.

---

## 3. Environment setup

Exact commands, in the order they were run:

```bash
# 1. Clone the fork
mkdir -p /home/bharathy/examples
git clone https://github.com/gkbharathy/hands-on-ml.git /home/bharathy/examples/hands-on-ml
```
Output (captured):
```
Cloning into '/home/bharathy/examples/hands-on-ml'...
```

```bash
# 2. Move into the repo and create a virtual environment
cd /home/bharathy/examples/hands-on-ml
python3 -m venv .venv
```

```bash
# 3. Activate it and install the packages this notebook actually needs
source .venv/bin/activate
pip install --upgrade pip -q
pip install -q jupyterlab "numpy~=1.26.2" "pandas~=2.1.3" "matplotlib~=3.8.1" \
  "scipy~=1.11.3" "scikit-learn~=1.3.2" joblib packaging
```
These installs used the `-q` (quiet) flag and returned no captured stdout.
The choice of package set (rather than the full `requirements.txt`, which
also includes TensorFlow, XGBoost, `transformers`, TensorBoard, etc.) was
based on parsing the notebook's own `import` statements, which showed only
`numpy`, `pandas`, `matplotlib`, `scipy`, `sklearn`, `joblib`, and
`packaging` were used in Chapter 2 - no TensorFlow-related import appears
in this notebook.

Versions confirmed installed (via a Python one-liner printing
`__version__` for each package):
```
numpy 1.26.4
pandas 2.1.4
matplotlib 3.8.4
scipy 1.11.4
sklearn 1.3.2
jupyterlab 4.6.3
```

```bash
# 4. (Optional, for interactive use) Launch Jupyter Lab
nohup jupyter lab --no-browser --ip=127.0.0.1 --port=8888 > .jupyter.log 2>&1 &
```
This produced a running server on `http://127.0.0.1:8888/lab?token=<redacted>`.
The token has been redacted here deliberately: Jupyter generates a new
random token each time the server starts, it is a live access credential
for that local server process, and it should never be committed to a
repository. This Jupyter Lab server was started but the notebook was in
the end executed non-interactively (see Section 5), not run by hand inside
this Lab session.

---

## 4. Plan

The objective agreed at the start of the session was: take the forked
repository, pick a worked example, and run it, confirming it executes
successfully end-to-end. The sequence of steps that followed was:

1. Locate the user's fork of `handson-ml3` via the GitHub CLI (`gh repo
   list` / `gh repo view`) and confirm its parent/upstream repository.
2. Clone the fork locally.
3. Choose Chapter 2 (`02_end_to_end_machine_learning_project.ipynb`) as the
   example to run.
4. Build a minimal Python environment covering only what that notebook
   imports.
5. Execute the notebook non-interactively from the command line
   (`jupyter nbconvert --execute`), first without tolerating errors, to see
   whether it ran clean.
6. When it did not, diagnose the failure, re-run tolerating errors to
   capture full output, and identify exactly which cells failed and why.
7. Decide, with the user, whether to patch the notebook's code or upgrade
   the pinned dependency - the user chose to upgrade scikit-learn.
8. Verify the fix in isolation on the two affected checks before
   re-running the whole notebook.
9. Re-run the full notebook again and confirm zero errors across all
   cells.

---

## 5. Steps and commands

Note on scope: the source notebook contains 174 code cells. Reproducing
every individual cell and its output verbatim here would make this
document unusably long and would not add information beyond what is
already in the notebook file itself. What follows is the real,
command-level sequence of *executions* (each a full run of the notebook),
plus the specific milestone outputs that were inspected to confirm success
or diagnose failure. The complete cell-by-cell record is preserved in
`02_end_to_end_machine_learning_project.executed.ipynb` in the repository
root (see Section 7).

### Run 1 - first execution attempt (strict, stop on first error)

```bash
cd /home/bharathy/examples/hands-on-ml
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 \
  --output 02_end_to_end_machine_learning_project.executed.ipynb \
  02_end_to_end_machine_learning_project.ipynb
```

Result: **failed**, 91% of the way through (159th of 174 code cells), with
no output notebook written (nbconvert does not save partial output on an
unhandled error by default). The failing cell and the tail of the real
traceback:

```
from sklearn.utils.estimator_checks import check_estimator

test_results = check_estimator(FeatureFromRegressor(KNeighborsRegressor()))
```
```
ValueError: When creating aligned memmap-backed arrays, input must be a
single array or a sequence of arrays
```
raised from inside scikit-learn's own
`sklearn.utils.estimator_checks.check_estimator` -> `check_estimators_pickle`
-> `create_memmap_backed_data` test utility.

### Run 2 - re-run tolerating errors, to capture full output

```bash
jupyter nbconvert --to notebook --execute --allow-errors \
  --ExecutePreprocessor.timeout=1800 \
  --output 02_end_to_end_machine_learning_project.executed.ipynb \
  02_end_to_end_machine_learning_project.ipynb
```

This time it completed and wrote the output notebook
(`2,523,975` bytes at that point). A scripted scan of the saved notebook's
cell outputs (looking for `output_type == "error"`) found **6 failing
cells**, all in the "extra material" appendix, none in the core project:

| Code-cell index (of 174) | First line of cell | Error |
|---|---|---|
| 159 | `from sklearn.utils.estimator_checks import check_estimator` | `ValueError: When creating aligned memmap-backed arrays...` |
| 167 | `import numpy as np` (defines `StandardScalerClone`) | `ImportError: cannot import name 'validate_data' from 'sklearn.utils.validation'` |
| 168 | `from sklearn.utils.estimator_checks import check_estimator` | `ValueError: When creating aligned memmap-backed arrays...` |
| 171 | `scaler = StandardScalerClone()` | `AttributeError: 'StandardScalerClone' object has no attribute 'inverse_transform'` |
| 172 | `assert np.all(scaler.get_feature_names_out() ...)` | `AttributeError: 'StandardScalerClone' object has no attribute 'get_feature_names_out'` |
| 173 | `df = pd.DataFrame(...)` | `AttributeError: 'StandardScalerClone' object has no attribute 'feature_names_in_'` |

Diagnosis: cell 167 imports `validate_data` from
`sklearn.utils.validation`, a function not present in scikit-learn 1.3.2
(confirmed: `python -c "from sklearn.utils.validation import validate_data"`
raised `ImportError` under 1.3.2). That import failure meant the
`StandardScalerClone` class body never finished executing, so it lacked
the methods (`inverse_transform`, `get_feature_names_out`) that later
cells (171-173) call - explaining the cascade of `AttributeError`s. The
`check_estimator` failures (159, 168) were a separate, unrelated issue
inside scikit-learn 1.3.2's own test utilities.

Confirmed both `requirements.txt` and `environment.yml` pin scikit-learn to
1.3.x:
```
requirements.txt:scikit-learn~=1.3.2
environment.yml:  - scikit-learn=1.3  # machine learning library
```
`pip index versions scikit-learn` showed 1.9.1 as the latest available
release at the time, with 1.3.2 as the installed one.

### Fix and verification (in isolation, before re-running the whole notebook)

```bash
pip install -q -U scikit-learn
```
Result: scikit-learn upgraded to **1.9.1**. Verified:
```
sklearn version: 1.9.1
validate_data import OK
```

Both affected `check_estimator` calls were then tested standalone (the
`StandardScalerClone` and `FeatureFromRegressor` class definitions copied
exactly from the notebook cells), before touching the full notebook again:
```
check_estimator(StandardScalerClone()) PASSED
check_estimator(FeatureFromRegressor(...)) PASSED
```
(each printed one harmless `SkipTestWarning` about `SCIPY_ARRAY_API` not
being set, which is not an error.)

### Run 3 - final full re-run, with scikit-learn upgraded

```bash
jupyter nbconvert --to notebook --execute --allow-errors \
  --ExecutePreprocessor.timeout=1800 \
  --output 02_end_to_end_machine_learning_project.executed.ipynb \
  02_end_to_end_machine_learning_project.ipynb
```
Output notebook written: `5,706,579` bytes. A repeat of the same
error-scanning script over this file found:
```
total code cells: 174
error cells: 0
```
All 174 code cells executed without error.

---

## 6. Script changes and modifications

**No changes were made to the notebook or to any other tracked file in
this repository.** This was checked directly:

```bash
git status
```
```
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.execute.log
	.execute2.log
	.execute2.pid
	.execute3.log
	.execute3.pid
	.jupyter.log
	.venv/
	02_end_to_end_machine_learning_project.executed.ipynb

nothing added to commit but untracked files present (use "git add" to track)
```

```bash
git diff --stat
```
produced no output at all - i.e. zero lines changed in any tracked file.
`requirements.txt` and `environment.yml` remain exactly as cloned; the
scikit-learn version pin in those files was **not** edited.

The only actual "modification" was to the local Python environment: the
`scikit-learn` package inside `.venv` was upgraded from the version
`requirements.txt`/`environment.yml` specify (`~=1.3.2`) to `1.9.1`, purely
as an installed package in a `.venv` directory that is itself untracked by
git. If this deviation should instead be reflected in the repository's own
pinned requirements (so the fix is documented for future readers), that
would be a deliberate follow-up edit to `requirements.txt`/`environment.yml`
- it has not been made here.

---

## 7. Outputs and results

All figures, model files, and error logs below are actual real files
produced on disk during this session, or actual outputs read directly from
the executed notebook's saved cell outputs.

### Model comparison (cross-validated RMSE, `cv=10`, on training data)

| Model | Cross-val RMSE - mean | std | min | max |
|---|---|---|---|---|
| Linear regression | 69,847.92 | 4,078.41 | 65,659.76 | 80,685.25 |
| Decision tree | 67,153.32 | 1,963.58 | 63,925.25 | 70,664.64 |
| Random forest | 47,002.93 | 1,048.45 | 45,667.06 | 49,354.71 |

(Source: `pd.Series(lin_rmses).describe()`, `pd.Series(tree_rmses).describe()`,
`pd.Series(forest_rmses).describe()` cell outputs.)

### Grid search on the random-forest pipeline

```
grid_search.best_params_
{'preprocessing__geo__n_clusters': 15, 'random_forest__max_features': 6}
```

### Final model evaluation on the held-out test set

```
final_rmse = 41549.20158097943
```
(from `root_mean_squared_error(y_test, final_predictions)`, `X_test`/`y_test`
being the 20% stratified test split held out at the start of the
notebook.)

### A separate SVR exploration (extra material, not the final chosen model)

```
rnd_search.best_params_  # RandomizedSearchCV over an SVR pipeline
{'svr__C': 10000.0, 'svr__kernel': 'linear'}
```

### Saved artefacts on disk (all git-ignored - see `.gitignore`: `/datasets`,
`/images`, `my_*`)

| Path | Size | Notes |
|---|---|---|
| `datasets/housing.tgz` | 449,115 bytes | downloaded from `https://github.com/ageron/data/raw/main/housing.tgz` |
| `datasets/housing/housing.csv` | 1,423,529 bytes | extracted from the tarball |
| `my_california_housing_model.pkl` | 144,913,395 bytes (~138 MB) | the final fitted pipeline, saved via `joblib.dump(final_model, "my_california_housing_model.pkl")` |
| `images/end_to_end_project/*.png` (10 files, e.g. `california_housing_prices_plot.png`, `scatter_matrix_plot.png`, `district_cluster_plot.png`, `attribute_histogram_plots.png`, etc.) | various | generated by the notebook's own `save_fig()` calls during execution |
| `02_end_to_end_machine_learning_project.executed.ipynb` | 5,706,579 bytes | the fully executed notebook with all real cell outputs, at repository root (not git-ignored, currently untracked) |

`images/end_to_end_project/california.png` was **not** generated by this
run - it already existed in the repository at clone time (a static base-map
image used as a plot background) and its timestamp predates the notebook
execution.

---

## 8. Troubleshooting

**Problem 1: `check_estimator(FeatureFromRegressor(...))` raised
`ValueError: When creating aligned memmap-backed arrays, input must be a
single array or a sequence of arrays`.**
- Cause: a bug/limitation inside scikit-learn 1.3.2's own
  `create_memmap_backed_data` test helper, triggered by
  `check_estimators_pickle`'s `readonly_memmap=True` case.
- Resolution: confirmed fixed by upgrading scikit-learn to 1.9.1; the same
  call passed cleanly afterwards (with one harmless `SkipTestWarning`).

**Problem 2: `ImportError: cannot import name 'validate_data' from
'sklearn.utils.validation'`, breaking the `StandardScalerClone` class and
cascading into three further `AttributeError`s in later cells.**
- Cause: the notebook's code (as currently written in this fork) calls
  `validate_data`, which does not exist in scikit-learn 1.3.2, the version
  pinned by this repository's own `requirements.txt` and `environment.yml`.
  This is a version-skew issue in the source repository's state at this
  commit, not something introduced during this run.
- Two options were identified: (a) upgrade scikit-learn so the notebook's
  existing code works as written, or (b) leave scikit-learn pinned at
  1.3.2 and rewrite the affected cells to use the older API. The user
  chose option (a).
- Resolution: `pip install -U scikit-learn` (-> 1.9.1). Verified the import
  succeeded and both affected `check_estimator` calls passed in isolation
  before re-running the full notebook, which then completed with 0 errors
  across all 174 cells.

**Note on shell history:** the commands used to do all of the above were
not found in `~/.bash_history`. That file exists and was checked, but
contains only unrelated prior history. This is flagged here so a future
reader does not assume the history file is authoritative for this project.

---

## 9. Reproduction checklist

- [ ] Clone the fork: `git clone https://github.com/gkbharathy/hands-on-ml.git`
- [ ] `cd hands-on-ml`
- [ ] Create a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
- [ ] `pip install --upgrade pip`
- [ ] Install the packages Chapter 2 needs, with scikit-learn from the
      start at a version that has `validate_data` (>=1.6; this run used
      1.9.1), not the `~=1.3.2` pinned in `requirements.txt`:
      `pip install jupyterlab numpy pandas matplotlib scipy scikit-learn joblib packaging`
- [ ] Confirm versions with a quick `import <pkg>; print(<pkg>.__version__)`
      check for each package.
- [ ] Execute the notebook headlessly to verify it runs end-to-end:
      `jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=1800 --output 02_end_to_end_machine_learning_project.executed.ipynb 02_end_to_end_machine_learning_project.ipynb`
      (or open it in Jupyter Lab/Notebook and run all cells interactively).
- [ ] Confirm no error cells remain in the output notebook.
- [ ] Confirm `final_rmse` prints approximately `41549.2` on the test set
      (exact value may vary slightly with library versions, since only the
      random seeds fixed in the notebook, e.g. `np.random.seed(42)` and
      `random_state=42` parameters, are controlled).
- [ ] Check `my_california_housing_model.pkl` and the
      `images/end_to_end_project/` figures were written.

---

## 10. References and attribution

- Source repository (upstream): [ageron/handson-ml3](https://github.com/ageron/handson-ml3),
  accompanying *Hands-On Machine Learning with Scikit-Learn, Keras and
  TensorFlow*, 3rd edition, by Aurelien Geron (O'Reilly Media). Licensed
  under the Apache License, Version 2.0 (see the repository's `LICENSE`
  file).
- This fork: [gkbharathy/hands-on-ml](https://github.com/gkbharathy/hands-on-ml)
  (`origin` remote for this local clone).
- Dataset: California housing census data, fetched from
  `https://github.com/ageron/data/raw/main/housing.tgz` (as referenced in
  the notebook's own `load_housing_data()` function).
- This tutorial documents a run performed in a local clone of the fork; it
  is not affiliated with, and was not submitted to, the upstream
  `ageron/handson-ml3` repository.

---

## Open questions

- **Exact `pip install` resolved sub-dependency versions** (e.g. exact
  `threadpoolctl`, `contourpy`, `joblib` build-chain versions) were not
  individually recorded beyond the packages explicitly checked in Section
  3 and the `pip freeze` snapshot below; not fabricated here.
- **Why the notebook's committed code assumes a newer scikit-learn than
  the repository's own pinned `requirements.txt`/`environment.yml`** could
  not be determined from this session alone (e.g. whether upstream updated
  the notebook after the last pin bump, or the pin was never updated to
  match). This would need checking the upstream `ageron/handson-ml3`
  commit history directly, which was not done here.
- **Whether this scikit-learn-1.3.2-specific `check_estimator` /
  `create_memmap_backed_data` behaviour is a recognised, tracked bug
  upstream in scikit-learn** was not confirmed against scikit-learn's own
  issue tracker in this session.

For completeness, the full installed-package snapshot at the end of the
session (`pip freeze` for the packages explicitly installed):
```
joblib==1.6.0
jupyterlab==4.6.3
matplotlib==3.8.4
numpy==1.26.4
packaging==26.3
pandas==2.1.4
scikit-learn==1.9.1
scipy==1.11.4
```
