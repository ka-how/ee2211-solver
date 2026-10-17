# EE2211 Solver

[![EE2211 Solver cover](images/cover.png)](https://youtu.be/58JrtpwjFVY)
(click the image to watch the [demo video](https://youtu.be/58JrtpwjFVY), or try webui app [on streamlit](https://nus-ee2211-solver-hy.streamlit.app/))

EE2211 Solver is a small helper app I made for working through EE2211 (Introduction to Machine Learning) from NUS (National University of Singapore) machine learning calculation questions.

The goal is like:
let the app handle repeated matrix / regression / gradient / tree / clustering
calculation steps, so you can spend more exam time thinking about conceptual questions.

The main way to use it is the Streamlit WebUI. There is also a terminal menu if
you prefer command line tools.

A [Jupyter notebook edition](notebooks/README.md) provides one notebook for each
of the nine website tabs, with editable examples and a shared matrix cache.
It supports Python 3.11 and includes Conda setup instructions and an
[environment file](notebooks/environment.yml).

The Probability notebook and website tab also include a dynamic solver: enter
known values such as `a=.4`, `a*b=.1`, `a+b=.7`, and request `b=???` to obtain
`P(B)=.4`. Events A, B, and C are supported, including `a*b*c` and `a|(b+c)`.
Here `*` means intersection, `+` union, and `|` conditional probability.
The solver checks consistency and reports bounds when an answer is not unique.

## What It Can Do

- Parse matrices and quickly check shape, rank, transpose, inverse, left inverse,
  and right inverse.
- Solve linear systems such as `Xw = y` and `wX = y`.
- Fit linear, polynomial, ridge, and ridge-polynomial regression models.
- Run one-hot linear / polynomial classification workflows.
- Calculate Pearson correlation for one or more columns.
- Step through symbolic gradient descent with iteration history.
- Compute decision-tree impurity with Gini, entropy, and classification error.
- Check one-level regression-tree splits and find the best threshold.
- Run K-means clustering from points and initial centroids.
- Cache matrices and results in the WebUI so you can copy and reuse them.
- Scale features, standardize columns, and impute missing values from observed data.
- Calculate matrix products, vector dot products, sums, differences, and traces.
- Work with discrete probabilities, Bayes updates, joint events, and finite-support densities.
- Evaluate scalar/vector functions and inspect gradients, Jacobians, and Hessians at a point.
- Solve from supplied normal equation summaries and preview polynomial features without fitting.
- Check a candidate minimum point, distinguishing maxima, saddles, and inconclusive cases.
- Differentiate real matrix expressions using MATLAB-style syntax, with optional numeric evaluation.
- Find quartiles, compute probabilities for draws without replacement, and compare grouped success rates.

The additional calculators are grouped under task selectors in the existing
Array, Gradient, and Regression tabs. Probability is one additional tab;
reference settings and detailed checks are expandable.

| New workflow | Location |
| --- | --- |
| Min-max scaling, z-scores, mean imputation | Array > Prepare data |
| Matrix and vector operations | Array > Calculate |
| Unknown probabilities, moments, Bayes, events, density intervals | Probability |
| Function values, derivatives, supplied classifier scores | Gradient > Function and derivatives at a point |
| OLS/ridge from `X.T @ X` and `X.T @ Y` | Regression > Use normal equation summaries |
| Monomial counts, feature maps, primal/dual shapes | Regression > Explore polynomial features |
| Residuals, total SSE, normal equation and gradient checks | Regression > Fit from samples > Normal equations and error checks |
| Candidate minima, maxima, and saddles | Gradient > Check a minimum point |
| MATLAB matrix derivatives and Jacobians | Gradient > Differentiate a matrix expression |
| Quartiles and descriptive statistics | Array > Summarize data |
| Finite-bag draws and grouped success rates | Probability > Sampling without replacement / Compare grouped rates |
| Boundary of a supplied binary score | Gradient > Function and derivatives at a point > Sign classification > Decision boundary |

No tabs were added for Paper B. Each tab shows only its selected task, with
derivative checks, numeric evaluation, and pooling details in collapsed panels.

For example, **Gradient > Check a minimum point** accepts `(x-3)^2+2` and
candidate `3`, reporting a strict global minimum with value `2`. For `x^3` at
`0`, it reports neither minimum nor maximum. Multivariable functions are
supported; singular Hessians may need further analysis and are reported as
inconclusive. This is an unconstrained smooth interior-point check. Coordinates
can be exact fractions or expressions such as `pi/2`; approximate stationary
coordinates can have a small nonzero gradient.

For **Gradient > Differentiate a matrix expression**, enter:

```matlab
% Matrix expression
x'*A*x
% Matrices and constants
A = [1 2;0 3]
syms x [2 1] real
% Optional numeric evaluation
x = [1;1]
```

Choose `x` as the differentiation variable. The evaluated gradient is `[4;8]`;
the derivative accounts for both `A` and its transpose. Symbolic matrices can
be declared with `syms A [2 2] real`. Supported expressions include
`norm(X*w-y)^2 + lambda*(w'*w)`, `trace(W'*W)`, and
`sum((X*W-Y).^2,'all')`. Scalar gradients keep the variable's matrix shape;
Jacobians and Hessians use column-major entry order. The tool supports a
documented subset of real MATLAB arithmetic and functions, rather than full
MATLAB scripts. The syntax reference is inside its collapsed help panel.

## Beginner Setup

For worked examples, input formats, settings, and outputs for every calculator,
see the [EE2211 Solver User Guide](docs/EE2211_Solver_User_Guide_Updated.pdf).
It covers all nine WebUI tabs, the fifteen numbered terminal tools, and the
extra calculator shortcuts.

You do not need to be good at Python to run this.

The easiest option is to use the hosted WebUI:
[nus-ee2211-solver-hy.streamlit.app](https://nus-ee2211-solver-hy.streamlit.app/)

If you want to run it on your own computer offline, the simplest local setup is:

1. install Python;
2. clone this repo;
3. install the dependencies;
4. start Streamlit.

Virtual environments and `uv` are both optional. They are useful if you already
use them, but you can skip them at first.

### 0. Install Python

Install Python 3.11 or newer from
[python.org](https://www.python.org/downloads/).

After installing, check that Python works:

```bash
python --version
```

If that command does not work on macOS or Linux, try:

```bash
python3 --version
```

On Windows, try:

```powershell
py --version
```

### 1. Clone This Repo

```bash
git clone https://github.com/hykzr/ee2211-solver.git
cd ee2211-solver
```

### 2. Install The App Dependencies

Run this from inside the `ee2211-solver` folder:

```bash
python -m pip install -e .
```

If you used `python3` or `py` when checking your Python version, use the same
command here:

```bash
python3 -m pip install -e .
```

```powershell
py -m pip install -e .
```

This may take a little while the first time. It installs the Python packages the solver needs.

If Python says the environment is externally managed, use the optional virtual environment setup below, then run the install command again.

### 3. Start The WebUI

Run:

```bash
python -m streamlit run webui.py
```

Again, if you used `python3` or `py` above, use that instead:

```bash
python3 -m streamlit run webui.py
```

```powershell
py -m streamlit run webui.py
```

Streamlit will print a local link, usually something like:

```text
2026-06-26 23:28:29.624 Uvicorn server started on 0.0.0.0:8501

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://...:8501
  
  For better performance, install the Watchdog module:
  ...
```

Open that link after `Local URL:` in your browser (for the example above, the link to copy is `http://localhost:8501`). It may take some seconds for the web page to load

### Optional: Use A Virtual Environment

A virtual environment keeps this app's Python packages separate from your other
Python projects. You do not need one just to try the solver, but it is a tidy
habit if you install a lot of Python tools.

On macOS or Linux, run:

```bash
python -m venv .venv
source .venv/bin/activate
```

If `python` does not work, use `python3` instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

After activation, your terminal prompt may show `(.venv)`.

Then go back to step 2 and install the dependencies.

### Optional: Use `uv` Instead

If you already have `uv`, you can let it manage setup for you:

```bash
uv sync
uv run streamlit run webui.py
```

If you do not have `uv` and want to try it, see the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## How To Type Matrices

Most input boxes accept simple rows like this:

```text
1 2
3 4
```

You can also use semicolons:

```text
1 2; 3 4
```

Or Python-style arrays:

```text
[[1, 2], [3, 4]]
```

Fractions also work: `1/3 2/3; -1/2 3/4`. In Prepare data and probability
tables, `?` or `NA` represents a missing entry. Ordinary numeric matrices
continue to require finite values.

For long one-dimensional data, a useful trick is to type it sideways first:

```text
0 1 2 3 4 5
```

Then click the `Transpose` button to turn it into a column.

## Cache Tip

The WebUI saves recent inputs and results in the `Cache` tab. This is useful
when you want to copy a model matrix, a weight vector, predictions, centroids,
or any intermediate result and paste it into another tab.

The app stores temporary WebUI state and cached matrices under `temp/`. That
folder is ignored by Git.

Copied arrays and transposed inputs retain full stored floating-point precision,
even though displayed results are rounded for readability.

## Decision Tree Inputs

Classification impurity defaults to a parent row followed by its child rows.
Keep **First row is the parent node** checked for:

```text
8 5 5
6 4 0
2 1 5
```

The parent row must equal the sum of the children in each class. Only the two
children are weighted: their weights are `10/18` and `8/18`, giving weighted
Gini impurity `0.5027777778`. The result shows each child's contribution,
the parent impurity, and the impurity gain.

If you enter children only, uncheck the parent option. For regression trees,
the course book's threshold convention is **left: x < threshold** and
**right: x >= threshold**, including when a sample equals the threshold.

## Optional: Terminal Version

If you prefer a text menu instead of the browser app, run this from the repo
folder after installing the dependencies:

```bash
python main.py
```

If you used `python3` or `py` during setup, use that instead.
With `uv`, run `uv run python main.py` instead.

You can choose menu items by number, or use shortcuts like `linear`, `ridge`,
`pearson`, `gradient`, `impurity`, and `split`.

The new workflows also have direct shortcuts: `matrix`, `prepare`,
`probability`, `functions`, `normal`, `features`, `minimum`, `matdiff`, and `stats`.
Within `probability`, choose `sampling` or `rates` for the Paper B calculators.

## Run Tests

If you want to check that the solver logic is still working:

```bash
python run_tests.py
```

If you used `python3` or `py` during setup, use that instead.
With `uv`, run `uv run python run_tests.py` instead.

To exercise the WebUI forms with isolated saved state, run:

```bash
python -m unittest test_webui -v
```

The new numerical tools and Paper A/B examples are checked with:

```bash
python -m unittest test_general_tools test_paper_b_tools -v
```

The printable guide can be regenerated with `python docs/build_user_guide.py`.
Its builder uses `reportlab`, which is needed only when rebuilding the guide.

## Exam Rules And Academic Integrity

Please check the current EE2211 exam rules before using this in any exam.
For my year, pre-written offline code / offline materials were allowed for the
open-book Examplify exams, as long as they followed the stated restrictions.
That does **not** automatically mean the same rules apply every year. The
teaching team can change the policy, and your exam instructions are the source
of truth.

Before relying on this solver in an exam, confirm things like:

- whether pre-coded scripts are allowed;
- whether using code from others (e.g. from a public repository) is allowed.

The solver itself, although coded partially by llm, does not need internet access, nor does it use any local llm based tools.

Use the repo responsibly. It is a study and calculation helper, not permission
to ignore course or exam rules.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

That means you can use, modify, and share it, but it is provided as-is with no
warranty. The authors are not responsible for misuse, wrong answers, rule
violations, or any consequences from using it in a setting where it is not
allowed.
