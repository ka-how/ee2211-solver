"""Build the printable manual with reportlab (a documentation-only dependency)."""

from pathlib import Path
import os
import tempfile
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)


OUTPUT = Path(__file__).with_name("EE2211_Solver_User_Guide_Updated.pdf")
styles = getSampleStyleSheet()
for name in ("Title", "Heading1", "Heading2", "Normal"):
    styles[name].textColor = colors.black
styles["Title"].fontName = "Helvetica-Bold"
styles["Title"].fontSize = 25
styles["Title"].leading = 30
styles["Title"].alignment = TA_LEFT
styles["Title"].spaceAfter = 15
styles["Heading1"].fontSize = 20
styles["Heading1"].leading = 25
styles["Heading1"].spaceAfter = 13
styles["Heading2"].fontSize = 12.5
styles["Heading2"].leading = 16
styles["Heading2"].spaceBefore = 9
styles["Heading2"].spaceAfter = 5
styles["Normal"].fontSize = 11
styles["Normal"].leading = 14
styles["Normal"].spaceAfter = 6
styles.add(ParagraphStyle("CodeText", fontName="Courier", fontSize=9.5, leading=12, spaceAfter=6))
styles.add(ParagraphStyle("CellText", fontName="Helvetica", fontSize=9.5, leading=13))
styles.add(ParagraphStyle("HeaderText", parent=styles["CellText"], fontName="Helvetica-Bold", textColor=colors.white))
styles.add(ParagraphStyle("SmallText", parent=styles["Normal"], fontSize=9.4, leading=13))
story = []


def p(text, style="Normal"):
    story.append(Paragraph(text, styles[style]))


def heading(text):
    p(text, "Heading2")


def code(text):
    p(escape(text).replace("\n", "<br/>"), "CodeText")


def table(rows, widths, padding=5):
    cells = [
        [Paragraph(escape(str(cell)), styles["HeaderText" if i == 0 else "CellText"]) for cell in row]
        for i, row in enumerate(rows)
    ]
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#203E56")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F7")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9D9D9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
    ]))
    story.extend([t, Spacer(1, 10)])


def page(title):
    story.append(PageBreak())
    p(title, "Heading1")


p("EE2211 Solver User Guide", "Title")
p("How to enter data, choose a calculator, and interpret its results", "Heading2")
p("Use this guide for matrix, regression, classification, probability, and clustering calculations. It covers every browser tab, all 15 numbered terminal tools, and the extra calculator shortcuts, with inputs you can paste and results you can check.")
p("Updated 14 October 2026", "SmallText")
heading("Start the program")
p("Install Python 3.11 or newer. Open a terminal in the solver folder, install the app, and start the browser interface:")
code("python -m pip install -e .\npython -m streamlit run webui.py")
p("Open the Local URL printed by Streamlit, usually http://localhost:8501. Keep the terminal open while using the app. For the terminal menu, run <font face='Courier'>python main.py</font>. If you use uv, run <font face='Courier'>uv sync</font> and <font face='Courier'>uv run streamlit run webui.py</font>.")
heading("Find the right tool")
table([
    ["Browser tab", "What it does", "Guide page"],
    ["Array", "Matrices, data preparation, and quartiles", "2, 11, 17"],
    ["Solve", "Solve Xw = y or wX = y", "2"],
    ["Regression", "Models, summaries, and polynomial features", "3-5, 13-14"],
    ["Pearson", "Correlation between columns", "6"],
    ["Gradient", "Descent, derivatives, and minimum checks", "6, 13, 15-16"],
    ["Decision Tree", "Classification impurity or regression splits", "7 and 8"],
    ["K-means", "Cluster points using initial centroids", "9"],
    ["Probability", "Distributions, events, draws, and grouped rates", "12 and 17"],
    ["Cache", "Copy and manage recent matrices", "9"],
], [90, 298, 80])
p("The terminal menu map is on page 10. Additional calculator modes are on pages 11 to 17; the Paper B tool map is on page 18. Choose a task, enter its data, then press its main action button.")

page("Matrix inputs and linear systems")
heading("Enter rows and columns")
p("In browser boxes, use one sample per row and one feature per column. Separate values with spaces or commas; separate rows with new lines or semicolons. All rows must have equal widths. Numeric inputs must be finite; empty arrays, NaN, infinity, and three-dimensional arrays are rejected.")
code("1 2; 3 4\n\n[[1, 2], [3, 4]]")
p("Both examples mean a 2 by 2 matrix. For a column of values, use <font face='Courier'>0;1;2;3</font>. Typing <font face='Courier'>0 1 2 3</font> creates a row; Transpose changes it into a column. A Python list such as <font face='Courier'>[0,1,2,3]</font> is interpreted as a column. Use a nested list <font face='Courier'>[[0,1,2,3]]</font> for an explicit row.")
p("Each browser matrix box has a Preview and a Transpose button. Transpose changes the input; press the tool's main action button afterwards. The terminal accepts one line at a time, so use semicolons or a one-line Python array.")
heading("Array inspection")
p("Open Array, enter a matrix, and press Inspect. The result shows its dimensions, rank, determinant if square, and whether ordinary, left, and right inverses exist. A left inverse requires independent columns; a right inverse requires independent rows.")
code("Array: 1 2; 3 4")
p("Expected rank: 2. Determinant: -2. Inverse: <font face='Courier'>[[-2,1],[1.5,-0.5]]</font>. For a rank-deficient matrix, the app explains which inverses do not exist.")
heading("Solve Xw equals y")
p("Open Solve and select Xw = y. X and y must have the same number of rows. Each column of y is a separate right-hand side. Enter the example below and press Solve.")
code("X: 1 1; 1 2; 1 3\ny: 2; 3; 4")
p("Expected w: <font face='Courier'>[[1],[1]]</font>, giving y exactly. The app distinguishes a unique exact solution, many exact solutions, and a least-squares fit when no exact solution exists. For many solutions, it returns the one with minimum Euclidean norm.")
heading("Solve wX equals y")
p("Choose wX = y when the unknown multiplies X on the left. X and y must have the same number of columns. Enter X as <font face='Courier'>1 2 3; 0 1 0</font> and y as <font face='Courier'>2 7 6</font>. The answer is w = <font face='Courier'>[[2,3]]</font>.")
p("The residual is fitted y minus entered y. MSE is the average squared residual per target. The displayed standard error is the square root of MSE, also called RMSE; it is not an estimate of uncertainty in the coefficients.")

page("Linear and polynomial regression")
p("Regression uses X for training features and Y for matching targets. Use one sample per row and keep feature columns in the same order in X_test. Multiple numeric target columns can be fitted together.")
heading("Linear regression")
p("In Regression, select Model = Linear, Y interpretation = Numeric regression, and Regularization = None. Keep Add bias for linear model checked if the model needs an intercept. Enter the following data and press Fit:")
code("X: 0;1;2;3\nY: 1;3;5;7")
p("The fitted model is y = 1 + 2x. Weights are <font face='Courier'>[[1],[2]]</font>, with the intercept first. Training MSE is approximately zero. With bias enabled, enter raw features rather than a manual column of ones.")
p("Enable Predict X_test to make predictions. Enter X_test = <font face='Courier'>4;5</font>; predictions are <font face='Courier'>9;11</font>. Optionally enter Y_test = <font face='Courier'>9;12</font> to obtain test SSE = 1 and test MSE = 0.5. Leave Y_test empty if you only need predictions.")
heading("Polynomial regression")
p("Choose Model = Polynomial, set Polynomial order = 2, keep Numeric regression and Regularization = None, and enter:")
code("X: 0;1;2\nY: 1;4;9\nX_test: 3")
p("Press Fit with Predict X_test enabled. The model is y = 1 + 2x + x squared, with weights <font face='Courier'>[[1],[2],[1]]</font>. The prediction at x = 3 is 16.")
p("Polynomial expansion always includes the constant term. Do not add a bias column yourself. For two input features and order 2, columns of the model matrix P are 1, x1, x2, x1 squared, x1 times x2, and x2 squared. The app displays P so you can match each weight to its feature.")
heading("Read the outputs")
p("Model Matrix is the data actually used for fitting. Weights contains the fitted coefficient vector or matrix. Training Prediction gives the fitted targets. Square error is SSE, the sum of squared errors; MSE divides each target column's SSE by the number of samples. Test errors use only the entered test samples.")
p("Dependent feature columns are supported through a minimum-norm least-squares fit. A zero training error alone does not establish that the model will predict new data well. Choose an appropriate polynomial degree for the problem.")
heading("Terminal equivalents")
p("Use menu 4 for linear regression or menu 5 for polynomial regression. Follow the prompts for X, Y, bias or order, then optionally X_test and Y_test. A blank optional input finishes the calculation.")

page("Ridge regression")
p("Ridge penalizes large coefficients using Lambda. It works with either a linear or polynomial model, including square model matrices. Lambda must be finite and nonnegative. The implementation penalizes every model coefficient, including the constant term.")
heading("Ridge linear regression")
p("In Regression, select Linear, Numeric regression, and Regularization = Ridge. Keep Add bias checked, set Lambda = 1, and select Ridge form = auto. Enter:")
code("X: 0;1\nY: 1;3\nX_test: 2")
p("Enable Predict X_test and press Fit. The two weights are <font face='Courier'>[[1],[1]]</font>. Predictions on the training samples are 1 and 2, so training SSE = 1 and MSE = 0.5. The test prediction at x = 2 is 3.")
p("These weights differ from the unregularized solution <font face='Courier'>[[1],[2]]</font> because Lambda applies even when the model matrix is square.")
heading("Ridge polynomial regression")
p("Choose Polynomial, order 2, Numeric regression, and Ridge. Set Lambda = 1 and auto, then enter:")
code("X: 0;1\nY: 1;3\nX_test: 2")
p("Press Fit with prediction enabled. Polynomial columns are 1, x, and x squared. The weights are approximately <font face='Courier'>[[0.8571429],[0.7142857],[0.7142857]]</font>; the prediction at x = 2 is approximately 5.1428571.")
heading("Choose a form")
table([
    ["Ridge form", "Use"],
    ["auto", "Primal when samples >= model columns; dual when samples < columns"],
    ["primal form", "Solve the regularized system over model columns"],
    ["dual form", "Solve the regularized system over sample rows"],
], [105, 363])
p("Primal and dual forms produce the same ridge solution up to numerical rounding. Lambda = 0 gives the ordinary minimum-norm least-squares solution, including when columns are dependent. Increasing Lambda usually shrinks weights and can increase training error.")
heading("Terminal equivalents")
p("Menu 6 runs ridge linear regression and always adds a bias term. Menu 7 runs ridge polynomial regression, which includes a constant automatically. Enter X, Y, Lambda, and form; menu 7 also asks for polynomial order. Then enter optional test features and targets.")

page("Classification with regression models")
p("The Regression tab also fits classification scores. Select Linear or Polynomial and choose a Y interpretation. These tools use least-squares or ridge fitting; their raw scores can fall outside the range from zero to one.")
heading("One hot linear classification")
p("Choose Linear, One-hot classification, Regularization = None, and Add bias checked. Enable Predict X_test and enter:")
code("X: 0;1;2;3\nY: low;low;high;high\nX_test: 2.5\nY_test: high")
p("Press Fit. Classes are stored in order of first appearance: low, then high. The test scores are approximately <font face='Courier'>[[0.1,0.9]]</font>, so the predicted label is high. The largest score chooses the class. A tie chooses the first class column.")
p("You may use a single column of text or numeric labels. Each unique label becomes a separate class. Alternatively enter one-hot rows such as <font face='Courier'>1 0;1 0;0 1;0 1</font>. Each row must contain exactly one 1, with all other entries 0.")
heading("One hot polynomial classification")
p("Select Polynomial and order 2, then use the same training and test data. The polynomial model includes a constant automatically. It predicts high at 2.5. Menu 8 runs one-hot linear classification; menu 9 runs one-hot polynomial classification. The terminal prints scores and classified labels when labels were supplied.")
heading("Sign classification")
p("For binary labels -1 and +1, choose Sign classification in the browser. With Linear, Add bias unchecked, no regularization, X = <font face='Courier'>-1;1</font>, Y = <font face='Courier'>-1;1</font>, and X_test = <font face='Courier'>-2;0;2</font>, predictions and signs are <font face='Courier'>-2;0;2</font> and <font face='Courier'>-1;0;1</font>. A score of exactly zero is shown as class 0.")
heading("Choose Auto deliberately")
table([
    ["Y data", "Auto interpretation"],
    ["Text labels", "One-hot classification"],
    ["One numeric column containing only -1 and +1", "Sign classification"],
    ["Valid one-hot rows with multiple columns", "One-hot classification"],
    ["Other numeric targets", "Numeric regression"],
], [255, 213])
p("Select Numeric regression explicitly if your multi-column numeric targets happen to look one-hot. Select One-hot classification explicitly for numeric category labels such as 0, 1, and 2.")
p("For label-based training, Y_test must use known training labels. For one-hot training, use one-hot test rows in the same column order, or a single column of zero-based class indices. Test MSE compares score vectors with one-hot targets; it is not classification accuracy.")

page("Pearson correlation and gradient descent")
heading("Pearson correlation")
p("Open Pearson. Enter X and Y with observations in rows and variables in columns, then press Calculate. The output includes means, population standard deviations and variances, covariance, and Pearson correlation for each X-column and Y-column pair.")
code("X: 1;2;3;4\nY: 2;4;6;8")
p("Expected correlation: 1. X has mean 2.5 and variance 1.25; Y has mean 5 and variance 5. Covariance is 2.5. A value near -1 means a negative linear relationship; near +1 means a positive linear relationship; near zero means little linear association.")
p("Both inputs need matching observation counts and at least two observations. Constant columns have undefined correlation and are rejected. Very small but nonconstant values are accepted. Matching single-row inputs are automatically transposed; certain matching-column layouts are also transposed. Prefer observation rows to make the intended orientation clear. Menu 10 runs the same tool.")
heading("Gradient descent")
p("Open Gradient. Enter a cost function with Python or SymPy notation: use <font face='Courier'>**</font> for powers, <font face='Courier'>*</font> for multiplication, and radians for trigonometric functions. The displayed Variable order is alphabetical; enter one initial value per variable in that order, separated by commas.")
code("Cost function: x**2 + x*y**2\nInitial values: 3, 2\nLearning rate: 0.2\nIterations: 1\nTolerance: leave empty")
p("Press Run. At the initial point, the gradient is <font face='Courier'>[10,12]</font>. One simultaneous update gives x = 1 and y = -0.4. The gradient at that new point is <font face='Courier'>[2.16,-0.8]</font>, and the cost is 1.16.")
p("The history includes iteration 0, followed by each update. Derivatives and costs in a row are evaluated at the parameters in that row. Final Parameters contains the last parameter vector.")
p("Learning rate must be positive and finite. Leave Tolerance empty to run the requested number of updates. With tolerance set, the program stops when the largest absolute parameter change is at most that nonnegative tolerance. Equal consecutive costs alone do not stop an oscillating sequence.")
p("If a function or gradient leaves its real, finite domain, the program reports an error. Check the initial point and reduce the learning rate if needed. Menu 12 asks for the same function, initial values, learning rate, iterations, and optional tolerance.")

page("Classification decision tree impurity")
p("Open Decision Tree and choose Classification impurity. Each row contains class counts in the same class order. By default, First row is the parent node is checked: enter the parent first, followed by its disjoint children. The parent is used for comparison and gain; only the children contribute to the weighted result.")
heading("Calculate the example with nodes A B and C")
code("8 5 5\n6 4 0\n2 1 5")
p("A is the parent, B is the first child, and C is the second child. Keep the parent option checked, select gini or all, and press Calculate. The parent row must equal the column-wise sum of its children; here A = B + C.")
table([
    ["Node", "Samples", "Weight", "Gini", "Contribution"],
    ["B child", "10", "10/18", "0.48", "0.2666667"],
    ["C child", "8", "8/18", "0.53125", "0.2361111"],
], [90, 64, 84, 104, 126])
p("Weighted Gini is 0.5027778. Gini gain is parent impurity minus weighted child impurity: 0.1453704. The parent samples are already represented in B and C, so including A in the weighted average would count them twice.")
heading("Choose an impurity measure")
table([
    ["Measure", "Parent", "Weighted children", "Gain"],
    ["gini", "0.6481481", "0.5027778", "0.1453704"],
    ["entropy", "1.5466316", "1.1166592", "0.4299724"],
    ["misclassification", "0.5555556", "0.3888889", "0.1666667"],
], [119, 107, 133, 109])
p("For a node, divide each class count by its sample total to get class probabilities. Gini is one minus the sum of squared probabilities. Entropy uses base-2 logarithms, with zero-probability terms contributing zero. Misclassification impurity is one minus the largest class probability. Select all to see all three measures.")
heading("Enter children only")
p("Uncheck First row is the parent node and enter <font face='Courier'>6 4 0;2 1 5</font> to get the same weighted result. The app reconstructs the parent by summing children. Each child weight is its size divided by the sum of child sizes. A zero-sample branch has weight and contribution zero; at least one branch must contain samples.")
p("Use disjoint nodes from a single split or a set of disjoint leaves. Do not mix several ancestor levels into the same calculation. Menu 13 asks whether the first row is the parent. The default answer is yes.")
p("Course reference: ee2211book.pdf, Chapter 9, pages 62 to 66; weighting and gain are illustrated in equations 9.6 and 9.7 on page 64.", "SmallText")

page("Regression decision tree splits")
p("Open Decision Tree and choose Regression MSE / split. Enter one feature x and one target y for every sample. The calculator evaluates a single split; it does not grow a complete tree. Feature and target vectors must have matching lengths and finite values.")
heading("Check a threshold")
code("x values: 1;2;3\ny values: 1;5;9\nDecision threshold: 2")
p("Press Calculate. The split convention matches the course book: the left child contains x values strictly less than the threshold; the right contains x values greater than or equal to it. Thus x = 2 belongs to the right child.")
table([
    ["Quantity", "Expected result"],
    ["Root MSE", "10.6666667"],
    ["Left mean and MSE", "1 and 0"],
    ["Right mean and MSE", "7 and 4"],
    ["Weighted split MSE", "(1/3) times 0 + (2/3) times 4 = 2.6666667"],
], [175, 293])
p("A node predicts the mean of its training targets. Its MSE is the average squared distance from that mean. Overall split MSE weights each child's MSE by its number of samples divided by the parent sample count.")
heading("Find the best one level split")
p("Enable Find best one-level split. You can leave Decision threshold empty to search automatically. The calculator keeps x and y paired while sorting x, evaluates thresholds between distinct x values, and chooses the lowest weighted MSE. Repeated feature values stay together. If candidates tie, the smallest candidate threshold is shown.")
p("For the small example above, thresholds 1.5 and 2.5 have equal MSE, so the selected best threshold is 1.5. A manually entered threshold must put at least one sample on each side. Automatic search needs at least two distinct x values.")
heading("Reproduce the book house price example")
code("x values: 0.5;0.6;1;2;3;3.2;3.8\ny values: 0.19;0.23;0.28;0.42;0.53;0.75;0.80")
p("Enable the search and press Calculate. The best threshold is 2.5. Weighted split MSE is approximately 0.0102095, which rounds to the book's 0.0102. To compare several features, rerun with each feature's x values while keeping targets aligned with their samples.")
p("Menu 14 accepts the same vectors and optional threshold, then asks whether to search for the best split. Cache entries named threshold_summary contain threshold, root MSE, left MSE, right MSE, and split MSE in that order. best_split contains threshold followed by split MSE.")
p("Course reference: ee2211book.pdf, Chapter 9, pages 67 and 68; equations 9.17 to 9.19 define node MSE and sample-weighted split MSE.", "SmallText")

page("K means clustering and cached results")
heading("K means clustering")
p("Open K-means. Enter Points X with one point per row and Initial centroids with one centroid per row. The number of centroid rows determines k. Both matrices must have the same number of feature columns.")
code("Points X: 1 1;1.5 2;5 7;6 8\nInitial centroids: 1 1;5 7\nMax iterations: leave empty")
p("Press Run. The point assignments are clusters 1, 1, 2, and 2. Final centroids are <font face='Courier'>[[1.25,1.5],[5.5,7.5]]</font>. The initial cost is 3.25 and the final cost is 1.625.")
p("Points are assigned to the nearest centroid by Euclidean distance. Each centroid is then replaced by the mean of its assigned points. Cost is the sum of squared distances from points to their assigned centroids. Cluster numbers start at 1 in the interface; a distance tie chooses the first centroid.")
p("An empty cluster keeps its previous centroid. Leave Max iterations empty to run until convergence, or enter a positive integer to cap updates. With Max iterations = 1, the example computes the updated centroids but may stop before confirming convergence. The initial assignment is iteration 0. Menu 15 runs the same clustering workflow.")
heading("Copy and reuse results")
p("Open Cache to see recent inputs, weights, predictions, tree calculations, and centroids. Entries show their source, dimensions, and time. New entries appear first. Use Copy to paste an array into another tool; use Remove for one item or Clear cache for all cached matrices.")
p("Max allowed cache controls how many matrices are retained; the default is 50. Reducing the limit removes older entries. Displayed numbers are rounded for readability, while copied arrays and transposed inputs retain their stored floating-point precision.")
p("The browser saves widget inputs and cache data under <font face='Courier'>temp/webui_state.json</font>, so local sessions can restore them. Cached matrices are working data, not a fitted model object. To predict with the interface, enter the original training data and optional X_test in Regression, or copy a coefficient matrix into Solve when appropriate.")
heading("Terminal cache")
p("Menu 11 shows the last numeric input and last result. At a numeric matrix prompt, a single <font face='Courier'>-</font> reuses the last input and <font face='Courier'>_</font> reuses the last result. A minus sign within a number still denotes a negative value. The terminal cache lasts only for the current run.")
p("If Copy is unavailable in your browser, use the displayed Python array text in the cache entry. Keep the complete array when transferring a result so rounding does not alter later calculations.")

page("Terminal tools and troubleshooting")
heading("Terminal menu map")
table([
    ["Menu", "Tool", "Shortcut", "Page"],
    ["1", "Array inspection", "inspect", "2"],
    ["2", "Solve Xw = y", "solve", "2"],
    ["3", "Solve wX = y", "left solve", "2"],
    ["4", "Linear regression", "linear", "3"],
    ["5", "Polynomial regression", "poly", "3"],
    ["6", "Ridge regression", "ridge", "4"],
    ["7", "Ridge polynomial regression", "ridgepoly", "4"],
    ["8", "One-hot linear classification", "onehot", "5"],
    ["9", "One-hot polynomial classification", "onehotpoly", "5"],
    ["10", "Pearson correlation", "pearson", "6"],
    ["11", "Show cached input and result", "cache", "9"],
    ["12", "Gradient descent", "gradient", "6"],
    ["13", "Classification tree impurity", "impurity", "7"],
    ["14", "Regression tree MSE and split", "split", "8"],
    ["15", "K-means clustering", "kmeans", "9"],
], [44, 224, 132, 68], padding=4)
p("Press Enter to accept bracketed defaults or skip optional inputs. Enter 0, exit, or q at the main menu to exit. Ctrl+C during a calculator returns to the menu.")
heading("Resolve common input problems")
p("<b>Sample count mismatch:</b> X and Y need the same number of sample rows. Use semicolons for a column or press Transpose. <b>Feature mismatch:</b> X_test must have the same feature columns as raw X. <b>Parent class count mismatch:</b> ensure each parent class count equals the sum of that class in the children, or select children-only input.")
p("<b>Unknown test class:</b> use a training label, or a valid zero-based column index when training targets were one-hot. <b>Undefined Pearson correlation:</b> remove or change constant columns. <b>Empty tree split:</b> choose a threshold that separates distinct feature values. <b>Nonfinite gradient:</b> check the function domain and learning rate.")
heading("Verify the installation and calculations")
code("python run_tests.py\npython -m unittest test_webui -v")
p("The first command checks numerical calculations. The second exercises the browser forms with isolated test state. See README.md for alternative Python commands and virtual environment setup.")


page("Matrix operations and data preparation")
heading("Matrix and vector calculations")
p("In Array, select Array task = Calculate. Choose Dot product, Matrix product, Add, Subtract, Elementwise product, or Trace. Enter A and B, then press Calculate. Trace needs only a square A. Addition, subtraction, and elementwise products require equal shapes; a matrix product requires A's column count to equal B's row count.")
code("Dot product\nA: 2;-1;3\nB: 0;4;1")
p("The result is -1. Dot product accepts row or column vectors with equal lengths. For A = <font face='Courier'>1 2;3 4</font>, Trace returns 5. Fractions such as <font face='Courier'>1/3</font> are accepted in numeric inputs, including nested Python-style arrays.")
heading("Scale with known or fitted parameters")
p("In Array, select Prepare data, then choose Min-max scaling or Z-score standardization. Enter Data with one feature per column. By default, statistics are fitted on its observed values. To use statistics from a question, check Use supplied feature parameters. Enter one parameter per column, or one number to apply to every column.")
code("Min-max scaling\nData: 18\nFeature minimums: 10\nFeature maximums: 30")
p("Press Transform to obtain 0.4. For z-scores, Data = 17, mean = 11, and standard deviation = 3 gives 2. Fitted z-scores use population standard deviation. Standardization changes location and scale; it does not make a distribution Gaussian.")
p("Reference and output settings lets you fit statistics on separate training rows, choose a different min-max output interval, and reverse a transform. Reversal needs supplied parameters or original reference data. New values are not clipped to the reference range. Constant columns are rejected because their scaling denominator is zero.")
heading("Impute missing entries")
p("Choose Mean imputation and enter <font face='Courier'>2;4;12;?</font>. Press Transform to obtain <font face='Courier'>2;4;12;6</font>. The column mean uses observed entries only. Use ?, NA, or None for missing entries, or set a Missing numeric marker such as -1 in the settings expander.")
p("You may supply separate reference rows to fit the imputation means. A reference column with no observed values cannot supply a mean. Impute missing entries before applying a scaling transform. Parameters used shows the statistics and missing counts; prepared_data is copied through Cache.")
p("Terminal shortcuts: matrix and prepare. The terminal preparation shortcut supports fitted or supplied parameters and an optional missing numeric marker. Reference datasets, inverse transforms, and custom output intervals are available in the browser.", "SmallText")

page("Probability calculators")
p("Probability has one task selector. Only the selected calculator's inputs are shown. Invalid probability tables are rejected rather than silently normalized. Fractions work in text inputs, and ? can represent one unknown probability or prior.")
heading("Discrete distributions")
p("Choose Discrete distribution and enter outcome-probability pairs, one pair per row. Choose a comparison and boundary; between includes both endpoints. Calculate returns the event probability, mean, variance, and standard deviation. Cumulative probabilities is an optional panel.")
code("Outcome and probability:\n-2 1/4\n0 ?\n3 1/2\nEvent: >=   Boundary: 0")
p("The inferred probability is 1/4, and the event probability is 3/4. One unknown is recovered from the requirement that the probabilities sum to 1. Duplicate outcomes are combined when forming cumulative probabilities.")
heading("Bayes updates")
p("Choose Bayes update. Enter P(group) and P(event given group) in each row. Groups must be mutually exclusive and cover all possibilities. Group labels in the results refer to your input row numbers. Choose whether the event occurred, then Calculate.")
code("Prior and event likelihood:\n0.30 0.10\n0.70 0.04")
p("For Event occurred, the evidence probability is 0.058 and posterior probabilities are approximately 0.5172414 and 0.4827586. The result separates prior, likelihood, joint probability, and posterior. Event did not occur uses each likelihood's complement. Conditioning on evidence of probability zero is undefined and rejected.")
heading("Two events")
p("Choose Two events, enter P(A) and P(B), and supply either their intersection or union probability. Calculate returns both intersection and union, P(A given B), P(B given A), and independence and exclusivity checks. Joint table shows all four combinations of A and B. A conditional with a zero-probability conditioning event is shown as undefined.")
p("For P(A) = 0.5, P(B) = 0.4, and intersection = 0.2, the events are independent and overlap. Independence and mutual exclusivity are separate properties. Numerical comparisons use a small floating-point tolerance.")
heading("Continuous density areas")
p("Choose Continuous density, enter a scalar f(x), finite support endpoints, and query interval endpoints. Integrate verifies that the density is continuous and nonnegative on its support and integrates to 1. It then integrates over the query's overlap with the support.")
p("For f(x) = 2*x on [0,1], querying [0.5,1] gives 0.75. Densities may exceed 1; probabilities are areas, and single-point queries give zero. Unverified expressions are rejected. Terminal shortcut: probability.")

page("Function derivatives and regression summaries")
heading("Evaluate a scalar or vector function")
p("In Gradient, select Function and derivatives at a point. Enter one scalar expression per output, separated by semicolons or new lines. Coordinate order is inferred alphabetically. Enter comma-separated point coordinates and press Evaluate. No gradient descent update is performed.")
code("Function outputs: x1**2 + 3*x1*x2 + 2*x2**2\nPoint coordinates: 1,-1")
p("The gradient column is [-1,-1]. A scalar output has a gradient and Hessian; vector outputs have a Jacobian. The optional symbolic panel shows formulas and identifies strictly linear, affine, or nonlinear polynomial outputs.")
p("To preserve unused input coordinates, expand Variable order and specify all variable names. For <font face='Courier'>x1**2+x2;x1*x2</font> with variables <font face='Courier'>x1,x2,x3</font> and point <font face='Courier'>2,-1,7</font>, the Jacobian is <font face='Courier'>[[4,1,0],[-1,2,0]]</font>: two output rows and three input columns.")
heading("Classify a supplied score")
p("Select Interpret result as Sign classification for one scalar score, or Class scores for several outputs. Sign uses -1, 0, or +1. Class scores chooses the largest output and reports its zero-based index. No fitting is needed.")
code("Function outputs: 1-2*x1+x1*x2\nPoint coordinates: 2,1\nInterpret result as: Sign classification")
p("The score and predicted label are -1. For constant scores, leave Point coordinates empty. A class-score tie chooses the first output. Terminal shortcut: functions.")
heading("Solve from supplied normal equation summaries")
p("In Regression, select Use normal equation summaries. Enter X.T @ X and X.T @ Y directly. Choose None for ordinary least squares, or Ridge with Lambda. The summaries must have compatible dimensions, and the Gram matrix must be symmetric and positive semidefinite.")
code("X.T @ X: 4 0;0 9\nX.T @ Y: 8;18\nRegularization: Ridge   Lambda: 1")
p("Solve returns weights [1.6,1.8]. Optional prediction rows must already include the bias and any polynomial columns in the same coefficient order. A prediction row <font face='Courier'>1 1</font> gives 3.4. The equation check shows the regularized matrix and residual. These summaries alone do not determine training SSE. Terminal shortcut: normal.")
heading("Check a fitted model")
p("After Fit from samples, expand Normal equations and error checks. It shows X.T @ X, X.T @ Y, residuals E, X.T @ E, the objective gradient, total SSE, and SSE per sample across outputs. For ordinary least squares, X.T @ E is approximately zero. Ridge's gradient also includes its penalty term. Matrix dimensions distinguishes primal and dual systems.")

page("Polynomial features and Paper A workflows")
heading("Explore a feature map without targets")
p("In Regression, select Explore polynomial features. Choose the number of raw inputs, polynomial order, and whether to include a constant, then press Preview. It reports the coefficient count and the exact column order. A full polynomial of total degree at most d in n inputs has (n+d choose d) coefficients including the constant.")
code("Input features: 3\nPolynomial order: 2\nInclude constant: checked")
p("There are 10 monomials: 1, x1, x2, x3, x1 squared, x1*x2, x1*x3, x2 squared, x2*x3, and x3 squared. Without the constant, there are 9. The app previews at most 5000 columns to avoid an impractical expansion.")
p("Expand Evaluate points and check model sizes to enter points to transform and sample/output counts. Matrix dimensions shows design, weight, primal, and dual shapes. With 39 inputs, order 1, a constant, and 9 samples, the model has 40 coefficients: its primal system is 40 by 40 and its dual system is 9 by 9. Terminal shortcut: features.")
heading("Choose a tool for Paper A")
table([
    ["Paper A questions", "App workflow", "Page"],
    ["4 to 6", "Array > Prepare data", "11"],
    ["7 to 10", "Probability > choose the relevant task", "12"],
    ["11", "Array > Calculate > Dot product", "11"],
    ["12 and 13", "Solve or Array > Inspect", "2"],
    ["14 to 16 and 24", "Gradient > Function and derivatives at a point", "13"],
    ["17 to 20", "Regression > Fit from samples; expand error checks", "3 and 13"],
    ["21", "Regression > Use normal equation summaries", "13"],
    ["22, 23 and 25", "Regression > Explore polynomial features", "14"],
], [108, 288, 72])
p("The same workflows accept other values, dimensions, and functions. Conceptual questions about learning definitions or measurement scales still require interpreting the question; the calculator modes support the numerical parts.")
heading("Verify the new tools")
code("python -m unittest test_general_tools -v\npython -m unittest test_webui -v")
p("The numerical tests cover Paper A examples and generalized cases; browser tests exercise mode changes, validation, and compact default forms. The terminal accepts matrix, prepare, probability, functions, normal, and features directly at its main menu.")
p("Practice reference: ee2211-four-midterm-practice-papers.pdf, Paper A, pages 1 to 5. Use the mode selectors to expose additional controls only when you need them.", "SmallText")

page("Check whether a point is a minimum")
heading("Supply the function and candidate point")
p("Open Gradient and select Check a minimum point. Enter one scalar function and its candidate coordinates. Use * for multiplication and ^ or ** for powers. Variable order is inferred alphabetically; the optional Variable order panel lets you specify it explicitly, including unused coordinates.")
code("Function f: (x-3)^2+2\nCandidate point: 3")
p("Press Check point. The result is Strict global minimum, and the function value is 2. The candidate x = 3 is the arg minimum; the minimum value is f(3) = 2. This tool checks the point you supply; it does not search for all minimizers.")
p("For several variables, enter coordinates separated by commas. Exact fractions and numeric expressions such as 1/3, sqrt(2), and pi/2 are accepted. Use an exact point when possible: a rounded approximation to a stationary point can have a small nonzero gradient.")
heading("Read the classification")
table([
    ["Result", "What establishes it"],
    ["Not a minimum", "Nonzero gradient at a smooth interior point, or a verified negative-curvature direction."],
    ["Strict local minimum", "Zero gradient and positive definite Hessian; or a positive first nonzero derivative of even order in one variable."],
    ["Strict global minimum / global minimum", "A stationary point of a convex quadratic or affine function. Singular curvature gives a non-strict result."],
    ["Strict local maximum", "Zero gradient and negative definite Hessian, or a negative first nonzero derivative of even order."],
    ["Saddle / neither minimum nor maximum", "Indefinite Hessian, or a first nonzero derivative of odd order in one variable."],
    ["Inconclusive", "The derivative test cannot prove a classification; further analysis is needed."],
], [153, 315], padding=3)
heading("A zero gradient can be misleading")
code("x^3 at 0: neither minimum nor maximum\nx^4 at 0: strict local minimum\nx^2-y^2 at 0,0: saddle point\nx^2-y^4 at 0,0: inconclusive Hessian test")
p("Expand Derivative checks for symbolic and evaluated gradients, Hessians, and Hessian eigenvalues. One-variable higher derivative checks stop at order 12. The general multivariable singular-Hessian case is left inconclusive; a positive semidefinite Hessian at one point alone does not prove a minimum.")
p("This tool tests unconstrained smooth interior points. Nondifferentiable functions and domain-boundary candidates need other tests and may be reported as inconclusive. It reports a global result only with a convex quadratic certificate. Terminal shortcut: minimum.", "SmallText")

page("Differentiate MATLAB matrix expressions")
heading("Declare the matrices, then differentiate")
p("Open Gradient and select Differentiate a matrix expression. Enter a MATLAB-style expression, the matrix and constant definitions, and one differentiation variable name. Use one definition per line or semicolons between definitions. syms declares independent real symbolic entries.")
code("Matrix expression: x'*A*x\nMatrices and constants:\nA = [1 2;0 3]\nsyms x [2 1] real\nDifferentiate with respect to: x")
p("Press Differentiate. The symbolic gradient is [2*x1+2*x2; 2*x1+6*x2]. For a general real matrix A, the quadratic gradient is (A + A transpose)*x; A need not be symmetric. The result therefore does not replace it with 2*A*x unless symmetry is actually established.")
heading("Evaluate the derivative")
p("Expand Evaluate at values (optional), enter the following Numeric assignments, and differentiate again:")
code("x = [1;1]")
p("The evaluated gradient is [4;8]. Expanded expression, value, and Hessian shows f(x) = 6 and Hessian [2 2;2 6]. Numeric results can be reused in Cache. Keep the differentiation variable symbolic in its definition; put numeric candidate values in the evaluation panel.")
heading("Keep coefficients symbolic")
code("syms A [2 2] real\nsyms x [2 1] real\nsyms lambda real")
p("The derivative retains A's entries. Optional evaluation can assign A = [1 2;0 3] and x = [1;1]. Every evaluated matrix must match its declared shape. Scalars and explicit symbolic arrays, such as x = [u;v], also work if the differentiation entries are distinct symbols.")
heading("Regression losses and matrix outputs")
code("norm(X*w-y)^2 + lambda*(w'*w)\ntrace((X*W-Y)'*(X*W-Y)) + lambda*trace(W'*W)\nsum((X*W-Y).^2,'all')")
p("Define X, y or Y, lambda, and symbolic w or W before differentiating. Matrix residuals use the Frobenius norm or a trace/sum over all squared entries. norm(A) is accepted for vectors; use norm(A,'fro') for a matrix.")
p("A scalar output gives a gradient with the same shape as the differentiation variable and a Hessian. A vector or matrix output gives a Jacobian. Jacobians flatten output and variable entries column by column; Hessian coordinates follow the displayed entry order. A = [1 2;3 4] with expression A*x has Jacobian A. The complete syntax reference is on page 18 and inside the app. Terminal shortcut: matdiff.", "SmallText")

page("Quartiles, draws, and grouped rates")
heading("Quartiles and descriptive statistics")
p("Open Array and choose Summarize data. Enter observations one per row; multiple feature columns are summarized independently. Press Summarize for minimum, Q1, median, Q3, maximum, and interquartile range (IQR).")
code("Observations: 1;2;3;4;8;9;10;12")
p("Q1 = 2.5, median = 6, Q3 = 9.5, and IQR = 7. The default convention takes the medians of the lower and upper halves. For an odd sample size, exclude the middle observation from both halves; for one observation, both quartiles equal that value.")
p("Quartile convention offers linear interpolation as an alternative, which can give different answers. Mean, spread, and outlier fences shows the mean, population standard deviation, and the fences Q1 - 1.5*IQR and Q3 + 1.5*IQR. Terminal shortcut: stats.")
heading("Draw without replacement")
p("Open Probability and choose Sampling without replacement. Marked items are the category you count; X is their count in a uniformly drawn sample. Choose a comparison or an inclusive between interval, enter the bag counts and event boundary, and press Calculate.")
code("Total items: 6\nMarked items: 2 (blue)\nNumber of draws: 2\nEvent: >=   Count boundary: 1")
p("The probability of at least one blue is 3/5 = 0.6. The result includes an exact fraction. Count distribution and formula shows the hypergeometric probability for each feasible number of marked items. Draws and marked counts cannot exceed the total. Zero draws are supported. Limits are 10000 items and 1000 draws.")
heading("Compare methods across groups")
p("Open Probability and choose Compare grouped rates. Each row gives A successes, A total, B successes, B total for the same group. Totals must be positive integers and successes cannot exceed them.")
code("Grouped counts:\n9 10 80 100\n20 100 1 10")
p("Method A has rates 0.9 and 0.2; method B has 0.8 and 0.1. A is ahead in both groups, but its pooled rate is 29/110 (0.2636364) and B's is 81/110 (0.7363636). The app identifies this Simpson's reversal.")
p("Pooling weights shows each group's share of each method's total cases. The pooled rate is total successes divided by total cases, not the unweighted mean of percentages. Terminal shortcut: probability, then choose sampling or rates.")

page("Paper B tool map and syntax reference")
heading("Choose a tool for Paper B")
table([
    ["Questions", "App workflow", "Page"],
    ["4", "Array > Summarize data (median of halves)", "17"],
    ["7", "Probability > Sampling without replacement", "17"],
    ["8 and 10", "Probability > Discrete distribution or Two events", "12"],
    ["11", "Probability > Compare grouped rates", "17"],
    ["12 to 15", "Array > Inspect for numeric ranks; Solve for systems", "2"],
    ["16 and 17", "Gradient > Check a minimum point", "15"],
    ["18 and 20", "Gradient > Differentiate a matrix expression", "16"],
    ["19 and 23", "Regression > Fit from samples; disable linear bias", "3-4"],
    ["21 and 22", "Gradient > Function and derivatives > classify a score", "13"],
    ["24 and 25", "Regression > feature preview / primal and dual forms", "4 and 14"],
], [67, 334, 67], padding=3)
p("For question 12, substitute each proposed parameter into the numeric matrix and inspect rank. For question 20, evaluate trace(E'*E) using E's numeric matrix; the expression value is available in the detail panel. For question 22, Sign classification adds a collapsed Decision boundary panel: 2*x1-x2-1 = 0 becomes x2 = 2*x1-1. Conceptual statements still require interpretation.", "SmallText")
heading("Supported MATLAB notation")
table([
    ["Notation", "Meaning"],
    ["A = [1 2;3 4]; syms x [2 1] real", "Numeric matrix assignment; symbolic real vector declaration."],
    ["*, /, backslash, ^", "Matrix multiplication, division, and powers. Division uses square invertible matrix operands; powers are integer -12 to 12."],
    [".*, ./, .^; apostrophe or .'", "Elementwise arithmetic; transpose of real matrices."],
    ["trace, det, inv, transpose, diag", "Matrix functions; trace/det/inv require square matrices."],
    ["norm(v), norm(A,'fro'); sum(A,'all')", "Euclidean vector / Frobenius matrix norm; sum every entry. Plain sum(A) sums columns for a matrix."],
    ["eye, zeros, ones; A(1,2)", "Matrix constructors; one-based indexing. A single index uses column-major order."],
    ["sin, cos, tan, exp, log, sqrt, abs", "Real entrywise mathematical functions."],
], [206, 262], padding=3)
p("Each matrix is limited to 100 entries and each Jacobian to 5000 entries. This is an arithmetic syntax subset: loops, scripts, complex arithmetic, and general MATLAB execution are unsupported. Matrix definitions and numeric evaluation use separate boxes.", "SmallText")
code("python -m unittest run_tests test_general_tools test_paper_b_tools test_webui -v")
p("Practice reference: ee2211-four-midterm-practice-papers.pdf, Paper B, pages 6 to 10. No extra tabs were added; select the task you need and expand details when useful.", "SmallText")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#526170"))
    canvas.drawString(0.75 * inch, 0.47 * inch, "EE2211 Solver User Guide")
    canvas.drawRightString(7.75 * inch, 0.47 * inch, str(doc.page))
    canvas.restoreState()


with tempfile.NamedTemporaryFile(dir=OUTPUT.parent, suffix=".pdf", delete=False) as staged:
    staged_path = Path(staged.name)
doc = SimpleDocTemplate(
    str(staged_path), pagesize=letter,
    rightMargin=0.75 * inch, leftMargin=0.75 * inch,
    topMargin=0.6 * inch, bottomMargin=0.65 * inch,
    title="EE2211 Solver User Guide", author="EE2211 Solver",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
try:
    os.replace(staged_path, OUTPUT)
    print(OUTPUT)
except PermissionError:
    # Windows PDF viewers may hold the previous guide open without write sharing.
    updated = OUTPUT.with_stem(OUTPUT.stem + "_Copy")
    os.replace(staged_path, updated)
    print(f"The existing guide is open or locked. Saved the updated guide to {updated}")
