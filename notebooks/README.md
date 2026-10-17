# EE2211 Solver notebooks

Nine Jupyter notebooks mirror the nine website tabs. They run with **Python 3.11**
and use the existing calculator modules in the repository. Keep this folder inside
the repository; the notebooks locate the modules automatically when Jupyter starts
from the repository root or this folder. No Streamlit server is needed.

## Your Conda environment

If you already have an environment, activate it and install the packages:

```powershell
conda activate YOUR_ENV_NAME
conda install -c conda-forge python=3.11 "numpy>=1.26,<2.4" "pandas>=2.2,<3" "scikit-learn>=1.4,<2" "sympy>=1.12,<2" "jupyterlab>=4,<5" ipykernel "nbformat>=5.10,<6" "nbclient>=0.10,<1"
```

`YOUR_ENV_NAME` is your environment's actual name. NumPy, pandas, scikit-learn, and
SymPy provide calculations and tables. JupyterLab opens the notebooks; ipykernel
runs their Python cells. nbformat and nbclient support the verification command
below. The conservative NumPy and pandas ranges also support the calculator APIs
used here, without adopting future major changes.

Alternatively, create an environment from the supplied file, from the repository
root:

```powershell
conda env create -f notebooks/environment.yml
conda activate ee2211-notebooks
```

If you prefer creating it with a single command:

```powershell
conda create -n ee2211-notebooks -c conda-forge python=3.11 "numpy>=1.26,<2.4" "pandas>=2.2,<3" "scikit-learn>=1.4,<2" "sympy>=1.12,<2" "jupyterlab>=4,<5" ipykernel "nbformat>=5.10,<6" "nbclient>=0.10,<1"
conda activate ee2211-notebooks
```

There is no need to install the root project's package for the notebook edition.
This folder's environment file supplies its dependencies independently.

## Open and use

With your environment active, run this from the repository root:

```powershell
python -m jupyterlab notebooks
```

Open a notebook and select your environment's Python kernel. If you launch
JupyterLab from a different environment or use VS Code, you can register your
active environment explicitly:

```powershell
python -m ipykernel install --user --name ee2211-notebooks --display-name "EE2211 (Python 3.11)"
```

The registration command names the active interpreter; run it only after activating
your intended Python 3.11 environment. Select **EE2211 (Python 3.11)** in the notebook.

Run the setup cell once. After updating these files, reload the notebook from disk
and use **Restart Kernel and Run All Cells** to load the revised helpers and inputs.
Each numbered section has editable input cells and
calculation cells. Change its inputs and rerun that section. Use **Restart Kernel
and Run All Cells** for a fresh run of every example. You can run notebooks in any
order; the Cache notebook works with an empty cache too. Each notebook's setup
prints the actual Python version and cache location.

If VS Code underlines `CommonUtils`, `OneHotLinearClassification`, or
`RegressionUtils`, these are local project modules in the folder above `notebooks/`.
They do not need a Conda installation. Import-search settings are included for
opening either the repository root or `notebooks/` as the VS Code workspace.
Use **Developer: Reload Window** after updating these files. The helper also
locates its parent repository directly when Python imports it, independently of
the current working directory. Keep the complete repository together.

| Website tab | Notebook | Included workflows |
| --- | --- | --- |
| Array | [01_array.ipynb](01_array.ipynb) | Shape, rank, determinant, transpose, inverse/left/right inverse, pseudoinverse; dot/matrix/elementwise products, add/subtract/trace; min-max scaling, z-scores, imputation; quartiles and statistics |
| Solve | [02_solve.ipynb](02_solve.ipynb) | Xw = y and wX = y; exact/minimum-norm/least-squares solutions; predictions, residuals, MSE and root MSE |
| Regression | [03_regression.ipynb](03_regression.ipynb) | Linear, polynomial, ridge, ridge-polynomial; numeric/sign/one-hot targets; test predictions and errors; normal equations, diagnostics, polynomial features and model sizes |
| Pearson | [04_pearson.ipynb](04_pearson.ipynb) | Pairwise correlations; means, spread, covariance and input orientation |
| Gradient | [05_gradient.ipynb](05_gradient.ipynb) | Symbolic gradient descent and history; scalar/vector derivatives; classifier scores and boundaries; minimum-point checks; MATLAB-style matrix differentiation |
| Decision Tree | [06_decision_tree.ipynb](06_decision_tree.ipynb) | Gini, entropy, misclassification; parent/child weighting and gains; supplied regression thresholds and best split search |
| K-means | [07_kmeans.ipynb](07_kmeans.ipynb) | Initial centroids, assignments, costs, convergence and iteration history; optional distance matrices |
| Probability | [08_probability.ipynb](08_probability.ipynb) | Solve unknown event probabilities from known values; discrete PMFs/CDFs and moments, Bayes, joint/conditional events, densities, without-replacement sampling, grouped/pooled rates |
| Cache | [09_cache.ipynb](09_cache.ipynb) | Shared saved matrices, full-precision copyable lists, manual saves, retention, removal and clearing |

Matrices accept website-style strings (`"1 2; 3 4"`), newline-separated rows,
fractions, nested lists, and NumPy arrays. Optional inputs accept `None`, `"None"`,
or blank text. Required text class labels keep their names; to supply a literal
test class named `None`, use a nested label list such as `[["None"]]`. Missing
entries use `?` or `NA` in the data-preparation and probability workflows.

## Reusing results

Calculations save named results in `notebooks/.cache/matrices.sqlite3`, shared
between kernels and retained after restarts. This is independent of the website's
saved state. SQLite comes with Python. The generated cache is ignored by Git.

After any notebook's setup cell, you can use:

```python
display(cache.table())
X = cache.load("prepared_data")
print(cache.text("prepared_data"))
```

For an input cell that accepts matrix text, you can also assign the loaded array
directly, such as `X = cache.load("prepared_data")` in Regression.

## Shorter, annotated inputs

Every input field now has an inline comment showing its accepted format or choices.
Regression uses short model and target names and a single ridge-penalty switch:

```python
X = "1 3 -2;-4 0 -1;3 1 8;2 1 6;8 4 6"  # Sample rows, feature columns.
Y = "1;1;2;3;3"  # Class labels, one per sample.
X_test = "1 -2 4"  # Prediction row with the same three features.
Y_test = None  # Optional true target; "None" and "" also skip test errors.
model = "linear"  # linear | poly.
target = "onehot"  # auto | numeric | sign | onehot.
bias = False  # Include/exclude the constant term for either model type.
ridge = 0  # 0 = OLS; positive number = ridge penalty.
degree = 2  # Polynomial only; positive integer, at most 5000 expanded columns.
form = "auto"  # auto | primal | dual; only used with positive ridge.
```

This example predicts class **2**. One-hot class columns follow first appearance
in training, here `[1, 2, 3]`. Omitting a true test label still computes prediction
scores and the predicted class, while leaving test errors empty. Real unknown test
classes and invalid supplied matrices continue to raise input errors.

The original long option names still work in `fit_regression_model`. The compact
`fit_model` helper handles matrix text directly. For polynomial fitting, `bias=False`
now excludes the constant term, matching the setting's meaning.

## Dynamic probability solver

The first calculation in [08_probability.ipynb](08_probability.ipynb) accepts
known probability assignments and solves quantities marked `???`:

```text
a = 0.4
a*b = 0.1
a+b = 0.7
a|b = 0.25
b = ???
```

This returns **P(B) = 2/5 = 0.4**. Use `a*b` for intersection (A and B),
`a+b` for union (A or B), `a|b` for conditional probability, and `~a` or `a'`
for a complement. `P(A)` / `P(A*B)` / `P(A|B)` wrappers work too. Both uppercase
and lowercase names are accepted. Parentheses combine events, such as `~(a+b)`.
Use one assignment per line, or semicolons, with decimal, fraction, or percentage
values. More than one unknown can be requested.

The solver accepts **A, B, and C**. Use `a*b*c` for a triple intersection,
`a+b+c` for a three-event union, and `a|(b+c)` or `a|(b*c)` to condition on a
combined event. For example:

```text
a*c = 0.12
a|c = 0.3
c = ???
```

This returns **P(C) = 2/5 = 0.4**. Independence is not assumed, and knowing
pairwise probabilities does not automatically determine a triple intersection.
If the known values do not determine a unique answer, it reports the exact feasible
range. `[` / `]` include an endpoint; `(` / `)` exclude it. Conflicting or
impossible probabilities produce an input error. Conditional answers flag any
possibility that the conditioning event has probability zero. Fractions are
useful for exact repeating values such as `1/3`.

Only uniquely determined, defined answers are saved in the shared cache, under
names such as `P(B)`. Enable `show_related` to see other probabilities. The same
solver is available on the website under **Probability → Solve from known
probabilities**. It uses the existing packages; no extra installation is needed.

Names return the newest matching result; IDs select an exact history entry. The
default limit is 50 entries across all notebooks. Rerunning a calculation adds
entries and can evict older ones. The Cache notebook can adjust the limit to
1–1000, remove individual entries, or clear the history. Its modification switches
are disabled by default so a normal Run All keeps your results.

## Verify the notebook edition

From the repository root, with your environment active:

```powershell
python -m unittest discover -s notebooks -p test_notebooks.py
python -m unittest test_probability_solver
python notebooks/validate_notebooks.py
```

The test commands check known numeric answers, alternate model options, cache
persistence, dynamic probability queries, and Python 3.11 syntax. The validation
command checks notebook structure and runs every cell in a fresh Jupyter kernel
using the active Python interpreter.
Verification uses a temporary cache and leaves your saved results and notebooks
unchanged. These commands run locally.

Verified with Python 3.11.17, NumPy 2.3.5, pandas 2.3.3, scikit-learn 1.9.1,
and SymPy 1.14.0. Checks cover all nine notebooks, the reported one-hot input,
optional fields, alternate calculator workflows, polynomial bias, and cached values.
