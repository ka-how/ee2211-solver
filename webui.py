import contextlib
import html
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import sympy as sp
from sklearn.preprocessing import PolynomialFeatures

from CommonUtils import (
    add_bias_column,
    calculate_closed_form_weights,
    calculate_ridge_weights,
    ensure_2d,
    format_number,
    parse_mixed_array,
    parse_numeric_array,
    parse_number,
    python_array_literal,
    squared_error_summary,
    validate_same_feature_count,
    validate_same_sample_count,
)
from DecisionTreeUtils import (
    CLASSIFICATION_MEASURES,
    classification_split_details,
    find_best_regression_split,
    regression_threshold_summary,
)
from GradientDescent import gradient_descent, infer_variables
from DataPreparation import PREPARATION_METHODS, descriptive_statistics, matrix_calculation, prepare_data
from FunctionUtils import analyze_functions, check_minimum, parse_functions
from MatrixCalculus import differentiate_matrix
from ProbabilityUtils import (
    bayes_update, compare_grouped_rates, density_probability, discrete_distribution,
    event_probabilities, format_probability_result, sampling_without_replacement, solve_probabilities,
)
from RegressionUtils import polynomial_features_details, regression_diagnostics, regression_shapes, solve_normal_equations
from KMeansUtils import k_means
from OneHotLinearClassification import prepare_class_targets, validate_onehot_targets
from pearson_correlation import pearson_correlation_details


MATRIX_HELP = (
    "Rows can be separated with semicolons or new lines. Examples: "
    "`1 2; 3 4`, `1, 2\\n3, 4`, or `[[1, 2], [3, 4]]`. Fractions such as `1/3` are accepted."
)
RIDGE_FORMS = ("auto", "primal form", "dual form")
STATE_FILE = Path(__file__).resolve().parent / "temp" / "webui_state.json"
STATE_VERSION = 1
DEFAULT_MAX_ALLOWED_CACHE = 50
CACHE_LIMIT_KEY = "max_allowed_cache"
PERSISTED_WIDGET_KEYS = (
    CACHE_LIMIT_KEY,
    "array_inspect_raw",
    "solve_equation",
    "solve_x_raw",
    "solve_y_raw",
    "reg_model_type",
    "reg_target_mode",
    "reg_regularization",
    "reg_predict_test",
    "reg_order",
    "reg_include_bias",
    "reg_ridge_lambda",
    "reg_ridge_form",
    "reg_x_raw",
    "reg_y_raw",
    "reg_x_test_raw",
    "reg_y_test_raw",
    "pearson_x_raw",
    "pearson_y_raw",
    "gradient_cost_expression",
    "gradient_initial_values",
    "gradient_learning_rate",
    "gradient_iterations",
    "gradient_tolerance",
    "tree_mode",
    "tree_counts_raw",
    "tree_measure",
    "tree_includes_parent",
    "tree_x_raw",
    "tree_y_raw",
    "tree_threshold",
    "tree_find_best",
    "kmeans_x_raw",
    "kmeans_centroids_raw",
    "kmeans_max_iterations",
    "array_mode", "matrix_operation", "matrix_a_raw", "matrix_b_raw",
    "prep_method", "prep_data", "prep_reference", "prep_supplied",
    "prep_inverse", "prep_range_low", "prep_range_high", "prep_missing_marker",
    "prep_first_minmax", "prep_second_minmax", "prep_first_zscore", "prep_second_zscore",
    "reg_workflow", "normal_regularization", "normal_lambda", "normal_gram_raw", "normal_rhs_raw", "normal_predict_raw",
    "features_count", "features_degree", "features_bias", "features_points_raw", "features_samples", "features_outputs",
    "gradient_mode", "function_expressions", "function_variables", "function_point", "function_interpretation",
    "probability_mode", "pmf_table", "pmf_comparison", "pmf_boundary", "pmf_upper",
    "bayes_table", "bayes_observed", "event_known_kind", "event_pa", "event_pb", "event_known",
    "density_expression", "density_support_low", "density_support_high", "density_query_low", "density_query_high",
    "stats_data", "stats_convention", "minimum_expression", "minimum_variables", "minimum_point",
    "matdiff_expression", "matdiff_definitions", "matdiff_variable", "matdiff_evaluation",
    "sampling_total", "sampling_marked", "sampling_draws", "sampling_comparison", "sampling_boundary", "sampling_upper",
    "rates_table",
    "probability_equations", "probability_equations_related",
)


def main():
    configure_page()
    ensure_state()

    render_workspace_header()

    tabs = st.tabs(
        [
            "Array",
            "Solve",
            "Regression",
            "Pearson",
            "Gradient",
            "Decision Tree",
            "K-means",
            "Probability",
            "Cache",
        ]
    )
    with tabs[0]:
        render_array_tab()
    with tabs[1]:
        render_solve_tab()
    with tabs[2]:
        render_regression_tab()
    with tabs[3]:
        render_pearson_tab()
    with tabs[4]:
        render_gradient_tab()
    with tabs[5]:
        render_decision_tree_tab()
    with tabs[6]:
        render_kmeans_tab()
    with tabs[7]:
        render_probability_tab()
    with tabs[8]:
        render_cache_tab()


def configure_page():
    st.set_page_config(page_title="Calculation workspace", page_icon="🧮", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        """
        <style>
        :root {
          color-scheme: dark;
          --solver-border: #293744;
          --solver-soft: #141e28;
          --workspace-muted: #9aabba;
          --workspace-accent: #6ee7c0;
        }
        [data-testid="stAppViewContainer"] {
          background: #0b1016;
          color: #e6edf3;
        }
        [data-testid="stHeader"] {
          background: #0b1016;
        }
        .block-container {
          padding: 3.5rem 2.5rem 4rem;
          max-width: 1240px;
        }
        .workspace-header {
          margin-bottom: 0.35rem;
        }
        .workspace-masthead {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 1rem;
          padding-bottom: 1.25rem;
          border-bottom: 1px solid var(--solver-border);
          color: var(--workspace-muted);
          font: 0.69rem 'Cascadia Code', Consolas, monospace;
          letter-spacing: 0.11em;
          text-transform: uppercase;
        }
        .workspace-identity {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          color: #e6edf3;
        }
        .workspace-mark {
          display: grid;
          grid-template-columns: repeat(2, 6px);
          gap: 4px;
        }
        .workspace-mark i {
          width: 6px;
          height: 6px;
          background: var(--workspace-accent);
        }
        .workspace-mark i:last-child {
          background: transparent;
          border: 1px solid var(--workspace-accent);
        }
        .workspace-hero {
          display: grid;
          grid-template-columns: minmax(0, 1fr) 260px;
          gap: 2rem;
          align-items: center;
          padding: 2rem 0 1.75rem;
        }
        .workspace-eyebrow {
          margin: 0 0 0.75rem;
          color: var(--workspace-accent);
          font: 0.68rem 'Cascadia Code', Consolas, monospace;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }
        .workspace-hero h1 {
          color: #f1f5f9;
          font-size: clamp(1.9rem, 3.4vw, 2.65rem);
          font-weight: 600;
          letter-spacing: -0.045em;
          line-height: 1.12;
          padding: 0;
          margin: 0 0 0.85rem;
        }
        .workspace-description {
          max-width: 440px;
          margin: 0;
          color: var(--workspace-muted);
          font-size: 0.91rem;
          line-height: 1.65;
        }
        .workspace-equation {
          padding: 1.1rem 1.3rem;
          border: 1px solid var(--solver-border);
          border-radius: 12px;
          background-color: #101923;
          background-image: radial-gradient(#2a3947 0.7px, transparent 0.7px);
          background-size: 14px 14px;
        }
        .workspace-equation-label {
          color: var(--workspace-muted);
          font: 0.6rem 'Cascadia Code', Consolas, monospace;
          letter-spacing: 0.08em;
        }
        .workspace-equation-body {
          display: flex;
          gap: 0.9rem;
          align-items: center;
          justify-content: center;
          padding: 1.1rem 0;
          color: var(--workspace-accent);
          font: 1.1rem 'Cascadia Code', Consolas, monospace;
        }
        .workspace-matrix {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 0.25rem 0.75rem;
          padding: 0.25rem 0.6rem;
          border-left: 2px solid #6ee7c080;
          border-right: 2px solid #6ee7c080;
          color: #dce7f1;
          font-size: 0.9rem;
        }
        .workspace-equation-footer {
          border-top: 1px solid var(--solver-border);
          padding-top: 0.65rem;
          color: var(--workspace-muted);
          text-align: center;
          font: 0.57rem 'Cascadia Code', Consolas, monospace;
          letter-spacing: 0.09em;
        }
        [role="tablist"] {
          gap: 0.25rem;
          padding: 0.4rem;
          border: 1px solid var(--solver-border);
          border-radius: 10px;
          background: #111923;
        }
        [role="tab"] {
          flex: 1 0 auto;
          height: 2.6rem;
          padding: 0.5rem 0.8rem;
          border-radius: 6px;
          color: #9aabba;
          transition: background 120ms, color 120ms;
        }
        [role="tab"] p {
          font-size: 0.82rem;
          font-weight: 500;
        }
        [role="tab"]:hover {
          background: #1c2935;
          color: #e6edf3;
        }
        [role="tab"][aria-selected="true"] {
          background: #1c3937;
          color: #85efce;
          box-shadow: inset 0 0 0 1px #355d55;
        }
        [data-baseweb="tab-highlight"], [data-baseweb="tab-border"],
        [data-testid="stTabs"] .react-aria-SelectionIndicator {
          display: none;
        }
        [role="tabpanel"] {
          padding-top: 1.65rem;
        }
        h2, h3, h4 {
          color: #e6edf3;
          letter-spacing: -0.025em;
        }
        h3 {
          font-size: 1.35rem;
          font-weight: 600;
        }
        h4 {
          font-size: 1.05rem;
        }
        [data-testid="stWidgetLabel"] p {
          color: #b7c6d3;
          font-size: 0.8rem;
        }
        [data-testid="stCaptionContainer"] {
          color: var(--workspace-muted);
        }
        [data-baseweb="textarea"], [data-baseweb="input"], [data-baseweb="select"] > div {
          border-color: var(--solver-border);
          border-radius: 7px;
        }
        [data-testid="stTextArea"] textarea {
          background: #0d151e;
          color: #dce7f1;
          font: 0.88rem/1.65 'Cascadia Code', Consolas, monospace;
          caret-color: var(--workspace-accent);
        }
        [data-testid="stButton"] button,
        [data-testid="stFormSubmitButton"] button {
          border-radius: 7px;
          font-weight: 500;
          min-height: 2.5rem;
        }
        [data-testid="stBaseButton-primaryFormSubmit"],
        [data-testid="stBaseButton-primaryFormSubmit"] p {
          color: #081b16;
        }
        button:focus-visible, [role="tab"]:focus-visible {
          outline: 2px solid var(--workspace-accent);
          outline-offset: 3px;
        }
        [data-testid="stExpander"] {
          border-color: var(--solver-border);
          border-radius: 7px;
          background: #101923;
        }
        div[data-testid="stMetric"] {
          border: 1px solid var(--solver-border);
          border-radius: 10px;
          padding: 0.8rem 0.9rem;
          background: var(--solver-soft);
          min-height: 0;
        }
        div[data-testid="stMetricLabel"] {
          color: var(--workspace-muted);
          font-size: 0.78rem;
          line-height: 1.15;
          overflow-wrap: anywhere;
        }
        div[data-testid="stMetricValue"] {
          color: #85efce;
          font-size: 1.05rem !important;
          line-height: 1.2 !important;
          overflow-wrap: anywhere;
          white-space: normal;
        }
        div[data-testid="stMetricValue"] > div {
          font-size: inherit !important;
          line-height: inherit !important;
        }
        div[data-testid="stForm"] {
          padding: 1.4rem;
          border-radius: 12px;
          border-color: var(--solver-border);
          background: #111923;
        }
        .solver-meta {
          color: var(--workspace-muted);
          font: 0.78rem/1.5 'Cascadia Code', Consolas, monospace;
          margin: -0.35rem 0 0.5rem 0;
        }
        .solver-chip {
          display: inline-block;
          border: 1px solid var(--solver-border);
          border-radius: 999px;
          padding: 0.1rem 0.45rem;
          margin-right: 0.25rem;
          font-size: 0.78rem;
          color: #b7c6d3;
          background: #1b2935;
        }
        div[data-testid="stLatex"] {
          overflow-x: auto;
          margin: 0.1rem 0 0.45rem 0;
        }
        div[data-testid="stLatex"] .katex-display {
          margin: 0.15rem 0;
        }
        div[data-testid="stLatex"] .katex,
        div[data-testid="stLatex"] .katex-display > .katex {
          font-size: 1.05rem !important;
          line-height: 1.2;
        }
        @media (max-width: 760px) {
          .block-container {
            padding: 3rem 1.1rem 2rem;
          }
          .workspace-hero {
            grid-template-columns: 1fr;
            padding: 1.5rem 0;
          }
          .workspace-equation, .workspace-topics {
            display: none;
          }
          .workspace-hero h1 {
            font-size: 2rem;
          }
          [role="tab"] {
            flex: 0 0 auto;
          }
          div[data-testid="stForm"] {
            padding: 1rem;
          }
        }
        @media (prefers-reduced-motion: reduce) {
          [role="tab"] {
            transition: none;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_header():
    st.markdown(
        """
        <header class="workspace-header">
          <div class="workspace-masthead">
            <div class="workspace-identity">
              <span class="workspace-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span>
              <span>Calculation workspace</span>
            </div>
            <span class="workspace-topics">Matrices / Models / Probability</span>
          </div>
          <div class="workspace-hero">
            <div>
              <p class="workspace-eyebrow">A space to work things out</p>
              <h1>Work through the numbers.</h1>
              <p class="workspace-description">Explore your data, test a model, or solve an equation.
              Choose a tool below to get started.</p>
            </div>
            <div class="workspace-equation" aria-hidden="true">
              <div class="workspace-equation-label">LINEAR SYSTEM / Xw = y</div>
              <div class="workspace-equation-body">
                <span class="workspace-matrix"><span>1</span><span>1</span><span>1</span><span>2</span></span>
                <span>w</span><span>=</span><span>y</span>
              </div>
              <div class="workspace-equation-footer">INPUT &nbsp; &rarr; &nbsp; COMPUTE &nbsp; &rarr; &nbsp; INSIGHT</div>
            </div>
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def ensure_state():
    if "_solver_state_loaded" not in st.session_state:
        load_persisted_state()
        st.session_state._solver_state_loaded = True
    if "matrix_cache" not in st.session_state:
        st.session_state.matrix_cache = []
    ensure_cache_limit()
    trim_matrix_cache()


def load_persisted_state():
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return

    widgets = payload.get("widgets", {})
    if isinstance(widgets, dict):
        for key in PERSISTED_WIDGET_KEYS:
            if key in widgets:
                st.session_state[key] = widgets[key]

    cache = payload.get("matrix_cache", [])
    if isinstance(cache, list):
        st.session_state.matrix_cache = [entry for entry in (deserialize_cache_entry(item) for item in cache) if entry]


def save_persisted_state():
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STATE_VERSION,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "widgets": persisted_widget_values(),
            "matrix_cache": [
                serialize_cache_entry(entry)
                for entry in st.session_state.get("matrix_cache", [])
            ],
        }
        temp_file = STATE_FILE.with_suffix(".tmp")
        temp_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_file.replace(STATE_FILE)
    except OSError:
        pass


def persisted_widget_values():
    return {key: json_safe_value(st.session_state[key]) for key in PERSISTED_WIDGET_KEYS if key in st.session_state}


def ensure_cache_limit():
    try:
        limit = int(st.session_state.get(CACHE_LIMIT_KEY, DEFAULT_MAX_ALLOWED_CACHE))
    except (TypeError, ValueError):
        limit = DEFAULT_MAX_ALLOWED_CACHE
    normalized = max(1, limit)
    if st.session_state.get(CACHE_LIMIT_KEY) != normalized:
        st.session_state[CACHE_LIMIT_KEY] = normalized


def max_allowed_cache():
    try:
        return max(1, int(st.session_state.get(CACHE_LIMIT_KEY, DEFAULT_MAX_ALLOWED_CACHE)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_ALLOWED_CACHE


def trim_matrix_cache():
    limit = max_allowed_cache()
    st.session_state.matrix_cache = st.session_state.get("matrix_cache", [])[:limit]


def on_cache_limit_change():
    trim_matrix_cache()
    save_persisted_state()


def serialize_cache_entry(entry):
    value = as_2d_any(entry.get("value", []))
    return {
        "id": str(entry.get("id", uuid.uuid4().hex)),
        "name": str(entry.get("name", "")),
        "source": str(entry.get("source", "")),
        "kind": str(entry.get("kind", "")),
        "shape": str(entry.get("shape", shape_text(value))),
        "value": json_safe_value(value),
        "text": matrix_plain_text(value),
        "created_at": str(entry.get("created_at", "")),
    }


def deserialize_cache_entry(entry):
    if not isinstance(entry, dict):
        return None
    try:
        value = as_2d_any(entry.get("value", []))
    except (TypeError, ValueError):
        return None
    return {
        "id": str(entry.get("id") or uuid.uuid4().hex),
        "name": str(entry.get("name", "")),
        "source": str(entry.get("source", "")),
        "kind": str(entry.get("kind", "")),
        "shape": str(entry.get("shape") or shape_text(value)),
        "value": value,
        "text": matrix_plain_text(value),
        "created_at": str(entry.get("created_at", "")),
    }


def json_safe_value(value):
    if isinstance(value, np.ndarray):
        return [[json_safe_value(cell) for cell in row] for row in as_2d_any(value).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    return value


def render_array_tab():
    mode = st.selectbox("Array task", ("Inspect", "Calculate", "Prepare data", "Summarize data"), key="array_mode")
    if mode == "Summarize data":
        render_statistics_panel()
        return
    if mode == "Calculate":
        render_matrix_calculator()
        return
    if mode == "Prepare data":
        render_preparation_panel()
        return
    st.subheader("Inspect an array")
    st.caption("Check dimensions, rank, transposes, and inverses in one place.")
    with st.form("array_form", border=True):
        array, _ = matrix_input("Array", "array_inspect", default="1 2; 3 4")
        submitted = st.form_submit_button("Inspect", type="primary")

    if submitted:
        save_persisted_state()
    if not submitted:
        return
    if array is None:
        st.error("Enter a valid numeric matrix first.")
        return

    record_matrix("array", "Array tab input", array, kind="input")
    st.markdown("#### Parsed Matrix")
    render_matrix("array", array)

    details, matrices = inspect_matrix(array)
    render_metrics(details)
    for name, value in matrices:
        st.markdown(f"#### {title_from_name(name)}")
        render_matrix(name, value)
        record_matrix(name, "Array inspection result", value, kind="output")


def render_matrix_calculator():
    st.subheader("Matrix and vector calculations")
    operation = st.selectbox("Operation", ("Dot product", "Matrix product", "Add", "Subtract", "Elementwise product", "Trace"), key="matrix_operation")
    with st.form("matrix_calculation_form"):
        columns = st.columns(2) if operation != "Trace" else [st.container()]
        with columns[0]:
            A, _ = matrix_input("A", "matrix_a", default="2;-1;3")
        B = None
        if operation != "Trace":
            with columns[1]:
                B, _ = matrix_input("B", "matrix_b", default="0;4;1")
        submitted = st.form_submit_button("Calculate", type="primary")
    if not submitted:
        return
    save_persisted_state()
    try:
        if A is None or (operation != "Trace" and B is None):
            raise ValueError("Enter all required matrices.")
        result = matrix_calculation(A, operation, B)
    except Exception as exc:
        st.error(str(exc))
        return
    render_matrix("result", result)
    record_matrix("matrix_result", operation, result, kind="output")


def render_preparation_panel():
    st.subheader("Prepare numeric data")
    method = st.selectbox("Transform", PREPARATION_METHODS, key="prep_method")
    supplied = False
    if method != "Mean imputation":
        supplied = st.checkbox("Use supplied feature parameters", key="prep_supplied")
    st.caption("Rows are samples. Use ? or NA for missing entries. Parameters are fitted per column from observed reference data. Z-scores use population standard deviation.")
    with st.form("preparation_form"):
        raw = st.text_area("Data", value="10;18;30", key="prep_data", help="For mean imputation, try 2;4;12;?. Fractions are accepted.")
        first_text = second_text = None
        if supplied:
            parameter_mode = "minmax" if method == "Min-max scaling" else "zscore"
            first_label, second_label = ("Feature minimums", "Feature maximums") if method == "Min-max scaling" else ("Feature means", "Feature standard deviations")
            columns = st.columns(2)
            with columns[0]:
                first_text = st.text_input(first_label, value="10" if method == "Min-max scaling" else "11", key=f"prep_first_{parameter_mode}")
            with columns[1]:
                second_text = st.text_input(second_label, value="30" if method == "Min-max scaling" else "3", key=f"prep_second_{parameter_mode}")
        reference_text = ""
        inverse = False
        low, high = 0.0, 1.0
        with st.expander("Reference and output settings", expanded=False):
            if not supplied:
                reference_text = st.text_area("Reference data (optional)", value="", key="prep_reference", help="Fit statistics on these training rows, then transform Data. Leave empty to fit on Data.")
            marker_text = st.text_input("Missing numeric marker (optional)", value="", key="prep_missing_marker", help="For example, -1 if that number represents missing data.")
            if method != "Mean imputation":
                inverse = st.checkbox("Reverse the transform", key="prep_inverse")
            if method == "Min-max scaling":
                columns = st.columns(2)
                low = columns[0].number_input("Output minimum", value=0.0, key="prep_range_low")
                high = columns[1].number_input("Output maximum", value=1.0, key="prep_range_high")
        submitted = st.form_submit_button("Transform", type="primary")
    if not submitted:
        return
    save_persisted_state()
    try:
        details = prepare_data(
            parse_numeric_array(raw, allow_missing=True), method,
            reference=parse_numeric_array(reference_text, allow_missing=True) if reference_text.strip() else None,
            first=parse_numeric_array(first_text) if first_text is not None else None,
            second=parse_numeric_array(second_text) if second_text is not None else None,
            output_range=(low, high), inverse=inverse,
            missing_marker=parse_number(marker_text) if marker_text.strip() else None,
        )
    except Exception as exc:
        st.error(str(exc))
        return
    render_matrix("prepared_data", details["result"])
    record_matrix("prepared_data", method, details["result"], kind="output")
    with st.expander("Parameters used", expanded=False):
        first_label, second_label = ("Minimum", "Maximum") if method == "Min-max scaling" else ("Mean", "Standard deviation")
        table = {"Feature": np.arange(1, details["result"].shape[1] + 1), first_label: details["first"], "Missing input entries": details["missing"]}
        if details["second"] is not None:
            table[second_label] = details["second"]
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        st.caption("Scaling does not clip values outside the reference range or make a distribution Gaussian.")


def render_model_shapes(shapes):
    st.dataframe(pd.DataFrame({"Matrix": ["Design X", "Weights W", "Primal X.T @ X", "Dual X @ X.T"],
                              "Shape": [f"{shapes[name][0]} x {shapes[name][1]}" for name in ("design", "weights", "primal", "dual")]}), hide_index=True, width="stretch")
    st.caption(f"Auto ridge uses {shapes['recommended_form']} based on these dimensions.")


def render_statistics_panel():
    st.subheader("Quartiles and data summary")
    with st.form("statistics_form"):
        raw = st.text_area("Observations", value="1;2;3;4;8;9;10;12", key="stats_data", help="One observation per row; multiple columns are summarized separately.")
        with st.expander("Quartile convention", expanded=False):
            convention = st.selectbox("Method", ("Median of halves", "Linear interpolation"), key="stats_convention")
            st.caption("Median of halves excludes the middle observation when the sample count is odd. For a single observation both quartiles equal that value.")
        submitted = st.form_submit_button("Summarize", type="primary")
    if not submitted:
        return
    save_persisted_state()
    try:
        details = descriptive_statistics(parse_numeric_array(raw), convention)
    except Exception as exc:
        st.error(str(exc))
        return
    names = {"minimum": "Minimum", "q1": "Q1", "median": "Median", "q3": "Q3", "maximum": "Maximum", "iqr": "IQR"}
    st.dataframe(pd.DataFrame({"Feature": np.arange(1, len(details["q1"])+1), **{label: details[key] for key, label in names.items()}}), hide_index=True, width="stretch")
    st.caption(f"{details['count']} observations per feature. Quartiles: {convention.lower()}.")
    summary = np.column_stack([details[key] for key in names])
    record_matrix("quartile_summary", "Columns: minimum, Q1, median, Q3, maximum, IQR", summary, kind="output")
    with st.expander("Mean, spread, and outlier fences", expanded=False):
        st.dataframe(pd.DataFrame({"Feature": np.arange(1, len(details["q1"])+1), "Mean": details["mean"], "Population std": details["population_std"], "Lower fence": details["lower_fence"], "Upper fence": details["upper_fence"]}), hide_index=True, width="stretch")
        st.caption("Fences are Q1 - 1.5 IQR and Q3 + 1.5 IQR.")


def render_normal_equations_panel():
    regularization = st.selectbox("Regularization", ("None", "Ridge"), key="normal_regularization")
    st.caption("Enter supplied X.T @ X and X.T @ Y directly. Ridge uses SSE + Lambda times squared coefficient norm and penalizes every coefficient.")
    with st.form("normal_equations_form"):
        columns = st.columns(2)
        with columns[0]:
            gram, _ = matrix_input("X.T @ X", "normal_gram", default="4 0;0 9")
        with columns[1]:
            rhs, _ = matrix_input("X.T @ Y", "normal_rhs", default="8;18")
        ridge_lambda = st.number_input("Lambda", min_value=0.0, value=1.0, key="normal_lambda") if regularization == "Ridge" else 0.0
        with st.expander("Optional prediction", expanded=False):
            prediction_rows, prediction_status = matrix_input("Model rows for prediction", "normal_predict", default="")
            st.caption("Include any bias or polynomial terms yourself, in the same order as the summary matrix columns.")
        submitted = st.form_submit_button("Solve", type="primary")
    if not submitted:
        return
    save_persisted_state()
    try:
        if gram is None or rhs is None:
            raise ValueError("Both summary matrices are required.")
        if prediction_status == "invalid":
            raise ValueError("Correct the optional prediction rows or leave them empty.")
        details = solve_normal_equations(gram, rhs, ridge_lambda)
        prediction = None
        if prediction_rows is not None:
            if prediction_rows.shape[1] != details["w"].shape[0]:
                raise ValueError("Prediction rows must have one column per coefficient.")
            prediction = prediction_rows @ details["w"]
    except Exception as exc:
        st.error(str(exc))
        return
    render_matrix("w", details["w"])
    record_matrix("summary_weights", "Normal equation summaries", details["w"], kind="trained matrix w")
    if prediction is not None:
        render_matrix("prediction", prediction)
        record_matrix("summary_prediction", "Normal equation prediction", prediction, kind="output")
    with st.expander("Equation check", expanded=False):
        render_matrix("regularized_gram", details["regularized_gram"])
        render_matrix("normal_equation_residual", details["normal_residual"])
        st.caption("Training SSE cannot be recovered from X.T @ X and X.T @ Y alone.")


def render_polynomial_features_panel():
    st.caption("Count and inspect full monomials of total degree up to the selected order. No training targets are needed.")
    with st.form("polynomial_features_form"):
        columns = st.columns(3)
        n_features = columns[0].number_input("Input features", min_value=1, max_value=100, value=3, step=1, key="features_count")
        degree = columns[1].number_input("Polynomial order", min_value=1, max_value=12, value=2, step=1, key="features_degree")
        include_bias = columns[2].checkbox("Include constant", value=True, key="features_bias")
        with st.expander("Evaluate points and check model sizes", expanded=False):
            points, point_status = matrix_input("Points (optional)", "features_points", default="")
            columns = st.columns(2)
            samples = columns[0].number_input("Training samples", min_value=1, value=9, step=1, key="features_samples")
            outputs = columns[1].number_input("Target outputs", min_value=1, value=1, step=1, key="features_outputs")
        submitted = st.form_submit_button("Preview", type="primary")
    if not submitted:
        return
    save_persisted_state()
    try:
        if point_status == "invalid":
            raise ValueError("Correct the optional points or leave them empty.")
        details = polynomial_features_details(int(n_features), int(degree), points, include_bias)
        shapes = regression_shapes(int(samples), details["count"], int(outputs))
    except Exception as exc:
        st.error(str(exc))
        return
    st.metric("Monomial coefficients", details["count"])
    st.dataframe(pd.DataFrame({"Column": np.arange(1, details["count"] + 1), "Monomial": details["names"]}), hide_index=True, width="stretch")
    if details["transformed"] is not None:
        render_matrix("polynomial_features", details["transformed"])
        record_matrix("polynomial_features", "Feature map preview", details["transformed"], kind="output")
    with st.expander("Matrix dimensions", expanded=False):
        render_model_shapes(shapes)


def render_function_panel():
    st.subheader("Functions and derivatives at a point")
    interpretation = st.selectbox("Interpret result as", ("Numeric values", "Sign classification", "Class scores"), key="function_interpretation")
    with st.form("function_analysis_form"):
        raw = st.text_area("Function outputs", value="x1**2 + 3*x1*x2 + 2*x2**2", key="function_expressions", help="Separate vector outputs with new lines or semicolons. Use * for multiplication and ** for powers.")
        with st.expander("Variable order", expanded=False):
            variable_text = st.text_input("Variables (optional)", value="", key="function_variables", help="Comma-separated names. Include unused inputs to keep the intended Jacobian shape; otherwise order is inferred alphabetically.")
        try:
            _, symbols = parse_functions(raw, variable_text.split(",") if variable_text.strip() else None)
            names = [str(symbol) for symbol in symbols]
        except Exception:
            names = []
        st.caption("Coordinate order: " + (", ".join(names) or "none; constants need no coordinates"))
        point_text = st.text_input("Point coordinates", value="1,-1", key="function_point")
        submitted = st.form_submit_button("Evaluate", type="primary")
    if not submitted:
        return
    save_persisted_state()
    try:
        details = analyze_functions(raw, parse_initial_values(point_text, len(names)), variable_text.split(",") if variable_text.strip() else None)
        if interpretation == "Sign classification" and details["value"].size != 1:
            raise ValueError("Sign classification needs exactly one scalar output.")
    except Exception as exc:
        st.error(str(exc))
        return
    render_matrix("function_value", details["value"])
    record_matrix("function_value", "Function evaluation", details["value"], kind="output")
    if interpretation == "Sign classification":
        label = int(np.sign(details["value"].item()))
        st.metric("Predicted label", label)
        record_matrix("function_class", "Score sign", [[label]], kind="output")
        with st.expander("Decision boundary", expanded=False):
            score = details["expressions"][0]
            st.latex(sp.latex(score) + " = 0")
            if score == 0:
                st.caption("The score is zero everywhere.")
            elif not score.free_symbols:
                st.caption("This nonzero constant score has no zero boundary.")
            else:
                symbol = sorted(score.free_symbols, key=lambda item: item.name)[-1]
                try:
                    if sp.diff(score, symbol, 2) == 0 and sp.diff(score, symbol) != 0:
                        solution = sp.solve(score, symbol)
                        if len(solution) == 1:
                            st.latex(sp.latex(symbol) + " = " + sp.latex(solution[0]))
                except (NotImplementedError, ValueError):
                    pass
    elif interpretation == "Class scores":
        label = int(np.argmax(details["value"]))
        st.metric("Predicted class index", label)
        st.caption("Indices start at 0, in output order. Ties select the first output; scores need not be probabilities.")
        record_matrix("function_class", "Largest score", [[label]], kind="output")
    if details["variables"]:
        name = "gradient" if details["gradient"] is not None else "jacobian"
        derivatives = details["jacobian_at_point"].T if name == "gradient" else details["jacobian_at_point"]
        render_matrix(name, derivatives)
        record_matrix(name, "Point derivatives", derivatives, kind="output")
        st.caption(f"Jacobian: {len(details['expressions'])} output rows x {len(details['variables'])} input columns.")
    with st.expander("Symbolic derivatives and function structure", expanded=False):
        st.latex("f = " + sp.latex(sp.Matrix(details["expressions"])))
        if details["variables"]:
            st.latex("J = " + sp.latex(details["jacobian"]))
            if details["hessian"] is not None:
                st.latex("H = " + sp.latex(details["hessian"]))
                render_matrix("hessian_at_point", details["hessian_at_point"])
        st.dataframe(pd.DataFrame({"Output": np.arange(1, len(details["dependence"]) + 1), "Dependence on inputs": details["dependence"]}), hide_index=True, width="stretch")


def render_solve_tab():
    st.subheader("Solve a linear system")
    st.caption("Find the weights, inspect the fitted values, and check the error.")
    with st.form("solve_form", border=True):
        equation = st.selectbox("Equation", ("Xw = y", "wX = y"), key="solve_equation")
        col_x, col_y = st.columns(2)
        with col_x:
            X, _ = matrix_input("X", "solve_x", default="1 1\n1 2\n1 3")
        with col_y:
            y, _ = matrix_input("y", "solve_y", default="2\n3\n4")
        submitted = st.form_submit_button("Solve", type="primary")

    if submitted:
        save_persisted_state()
    if not submitted:
        return
    if X is None or y is None:
        st.error("Both X and y are required.")
        return

    try:
        result = solve_linear_system(X, y, equation)
    except Exception as exc:
        st.error(str(exc))
        return

    record_matrix("X", f"Solve {equation} input", X, kind="input")
    record_matrix("y", f"Solve {equation} input", y, kind="input")
    record_matrix("w", f"Solve {equation} result", result["w"], kind="trained matrix w")
    record_matrix("y_hat", f"Solve {equation} fitted target", result["predicted"], kind="output")
    record_matrix("residual", f"Solve {equation} residual", result["residual"], kind="output")
    record_matrix("mse", f"Solve {equation} error", result["mse"], kind="output")

    st.markdown("#### System")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Type", result["system"])
    c2.metric("Equations", result["equations"])
    c3.metric("Unknowns", result["unknowns"])
    c4.metric("Solution", result["status"])
    c1, c2 = st.columns(2)
    c1.metric("rank(coefficients)", result["rank_coefficients"])
    c2.metric("rank(augmented)", result["rank_augmented"])

    st.markdown("#### Solution")
    render_matrix("w", result["w"])
    st.markdown("#### Fitted Target")
    render_matrix(r"\hat{y}", result["predicted"])
    st.markdown("#### Error")
    render_matrix("residual", result["residual"])
    left, right = st.columns(2)
    with left:
        render_matrix("standard error", result["standard_error"])
    with right:
        render_matrix("mse", result["mse"])


def render_regression_tab():
    st.subheader("Regression and classification")
    workflow = st.selectbox("Regression task", ("Fit from samples", "Use normal equation summaries", "Explore polynomial features"), key="reg_workflow")
    if workflow == "Use normal equation summaries":
        render_normal_equations_panel()
        return
    if workflow == "Explore polynomial features":
        render_polynomial_features_panel()
        return
    settings = st.columns(4)
    with settings[0]:
        model_type = st.selectbox("Model", ("Linear", "Polynomial"), key="reg_model_type")
    with settings[1]:
        target_mode = st.selectbox(
            "Y interpretation",
            ("Auto", "Numeric regression", "Sign classification", "One-hot classification"),
            key="reg_target_mode",
        )
    with settings[2]:
        regularization = st.selectbox("Regularization", ("None", "Ridge"), key="reg_regularization")
    with settings[3]:
        predict_test = st.checkbox("Predict X_test", value=False, key="reg_predict_test")

    order = 1
    include_bias = False
    ridge_lambda = 1.0
    ridge_form = RIDGE_FORMS[0]
    controls = st.columns(3)
    with controls[0]:
        if model_type == "Polynomial":
            order = st.number_input("Polynomial order", min_value=1, max_value=12, value=2, step=1, key="reg_order")
        else:
            include_bias = st.checkbox("Add bias for linear model", value=True, key="reg_include_bias")
    if regularization == "Ridge":
        with controls[1]:
            ridge_lambda = st.number_input(
                "Lambda",
                min_value=0.0,
                value=1.0,
                step=0.1,
                format="%.10g",
                key="reg_ridge_lambda",
            )
        with controls[2]:
            ridge_form = st.selectbox("Ridge form", RIDGE_FORMS, key="reg_ridge_form")

    with st.form("regression_form", border=True):
        col_x, col_y = st.columns(2)
        with col_x:
            X, _ = matrix_input("X", "reg_x", default="0\n1\n2\n3")
        with col_y:
            allow_mixed_y = target_mode in {"Auto", "One-hot classification"}
            y_default = "1\n3\n5\n7" if target_mode == "Numeric regression" else "low\nlow\nhigh\nhigh"
            Y, _ = matrix_input("Y", "reg_y", default=y_default, allow_mixed=allow_mixed_y)

        X_test = None
        Y_test = None
        if predict_test:
            test_col_x, test_col_y = st.columns(2)
            with test_col_x:
                X_test, _ = matrix_input("X_test", "reg_x_test", default="4\n5")
            with test_col_y:
                Y_test, y_test_status = matrix_input("Y_test (optional)", "reg_y_test", default="", allow_mixed=allow_mixed_y)

        submitted = st.form_submit_button("Fit", type="primary")

    if submitted:
        save_persisted_state()
    if not submitted:
        return
    if X is None or Y is None:
        st.error("X and Y are required.")
        return
    if predict_test and X_test is None:
        st.error("X_test is enabled but no valid test matrix was provided.")
        return
    if predict_test and y_test_status == "invalid":
        st.error("Correct Y_test or leave it empty before fitting.")
        return

    try:
        result = fit_regression_model(
            X=X,
            Y=Y,
            X_test=X_test,
            Y_test=Y_test,
            model_type=model_type,
            order=int(order),
            target_mode=target_mode,
            regularization=regularization,
            ridge_lambda=float(ridge_lambda),
            ridge_form=ridge_form,
            include_bias=bool(include_bias),
        )
    except Exception as exc:
        st.error(str(exc))
        return

    record_matrix("X", "Regression input", X, kind="input")
    record_matrix("Y", "Regression input", Y, kind="input")
    record_matrix(result["model_matrix_name"], "Regression model matrix", result["model_matrix"], kind="output")
    record_matrix("w", "Regression trained matrix w", result["w"], kind="trained matrix w")
    record_matrix("y_train_predicted", "Regression training prediction", result["train_pred"], kind="output")
    record_matrix("mse", "Regression training error", result["mse"], kind="output")
    if X_test is not None:
        record_matrix("X_test", "Regression prediction input", X_test, kind="input")
        record_matrix("y_predicted", "Regression test prediction", result["test_pred"], kind="output")
    if result["test_target_matrix"] is not None:
        record_matrix("Y_test", "Regression test target", Y_test, kind="input")
        record_matrix("test_mse", "Regression test error", result["test_mse"], kind="output")

    st.markdown("#### Model")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", result["model_label"])
    m2.metric("Target", result["target_mode"])
    m3.metric("System", result["system"])
    m4.metric("Form", result["form"])

    if result["class_labels"] is not None:
        st.markdown("#### Class Labels")
        st.write(", ".join(str(item) for item in result["class_labels"]))

    st.markdown("#### Model Matrix")
    render_matrix(result["model_matrix_name"], result["model_matrix"])
    if result["target_mode"] == "One-hot classification":
        st.markdown("#### Encoded Target")
        render_matrix("Y_encoded", result["target_matrix"])

    st.markdown("#### Weights")
    render_matrix("w", result["w"])
    st.markdown("#### Training Prediction")
    render_matrix(r"\hat{Y}_{train}", result["train_pred"])
    if result["train_class"] is not None:
        render_matrix("train_class", result["train_class"])

    left, right = st.columns(2)
    with left:
        render_matrix("square error", result["sse"])
    with right:
        render_matrix("mse", result["mse"])

    with st.expander("Normal equations and error checks", expanded=False):
        diagnostics = result["diagnostics"]
        render_metrics({"Total SSE": format_number(diagnostics["total_sse"]),
                        "SSE per sample across outputs": format_number(diagnostics["sse_per_sample"]),
                        "Ridge objective" if regularization == "Ridge" else "SSE objective": format_number(diagnostics["objective"])})
        render_matrix("X_transpose_X", diagnostics["gram"])
        render_matrix("X_transpose_Y", diagnostics["rhs"])
        render_matrix("residual_E", diagnostics["residual"])
        render_matrix("X_transpose_E", diagnostics["normal_residual"])
        render_matrix("objective_gradient", diagnostics["objective_gradient"])
        st.caption("At an optimum, the objective gradient is approximately zero. For ordinary least squares, X.T @ E is approximately zero; ridge also includes the penalty term.")
        render_model_shapes(diagnostics["shapes"])

    if X_test is not None:
        st.markdown("#### Test Prediction")
        render_matrix(r"\hat{Y}_{test}", result["test_pred"])
        if result["test_class"] is not None:
            render_matrix("test_class", result["test_class"])
        if result["test_target_matrix"] is not None:
            st.markdown("#### Test Error")
            if result["target_mode"] == "One-hot classification":
                render_matrix("Y_test_encoded", result["test_target_matrix"])
            left, right = st.columns(2)
            with left:
                render_matrix("test square error", result["test_sse"])
            with right:
                render_matrix("test mse", result["test_mse"])


def render_pearson_tab():
    st.subheader("Pearson correlation")
    st.caption("Measure the linear relationship between your variables.")
    with st.form("pearson_form", border=True):
        col_x, col_y = st.columns(2)
        with col_x:
            X, _ = matrix_input("X", "pearson_x", default="1 1\n2 1\n3 2\n4 2")
        with col_y:
            Y, _ = matrix_input("Y", "pearson_y", default="2 1\n4 1\n6 2\n8 2")
        submitted = st.form_submit_button("Calculate", type="primary")

    if submitted:
        save_persisted_state()
    if not submitted:
        return
    if X is None or Y is None:
        st.error("Both X and Y are required.")
        return

    try:
        details = pearson_details(X, Y)
    except Exception as exc:
        st.error(str(exc))
        return

    record_matrix("X", "Pearson input", X, kind="input")
    record_matrix("Y", "Pearson input", Y, kind="input")
    record_matrix("pearson", "Pearson result", details["pearson"], kind="output")

    st.markdown("#### Result")
    render_metrics(
        {
            "Observations": details["observations"],
            "X columns": details["x_variables"],
            "Y columns": details["y_variables"],
        }
    )
    render_matrix("pearson", details["pearson"])
    st.dataframe(pearson_frame(details["pearson"]), width="stretch")
    with st.expander("Intermediate values", expanded=True):
        for name in ("meanX", "stdX", "varX", "meanY", "stdY", "varY", "covariance"):
            render_matrix(name, details[name])


def render_gradient_tab():
    mode = st.selectbox("Gradient task", ("Gradient descent", "Function and derivatives at a point", "Check a minimum point", "Differentiate a matrix expression"), key="gradient_mode")
    if mode == "Check a minimum point":
        render_minimum_panel()
        return
    if mode == "Differentiate a matrix expression":
        render_matrix_derivative_panel()
        return
    if mode == "Function and derivatives at a point":
        render_function_panel()
        return
    st.subheader("Gradient descent")
    with st.form("gradient_form", border=True):
        cost_expression = st.text_input(
            "Cost function C(...)",
            value="x**2 + x*y**2",
            key="gradient_cost_expression",
        )
        inferred = infer_expression_variables(cost_expression)
        st.markdown(f'<div class="solver-meta">Variable order: {", ".join(inferred) or "none"}</div>', unsafe_allow_html=True)
        values_text = st.text_input("Initial values", value="3, 2", key="gradient_initial_values")
        controls = st.columns(3)
        with controls[0]:
            learning_rate = st.number_input(
                "Learning rate",
                min_value=0.0,
                value=0.2,
                step=0.05,
                format="%.10g",
                key="gradient_learning_rate",
            )
        with controls[1]:
            iterations = st.number_input("Iterations", min_value=1, max_value=10000, value=10, step=1, key="gradient_iterations")
        with controls[2]:
            tolerance_text = st.text_input("Tolerance", value="", key="gradient_tolerance")
        submitted = st.form_submit_button("Run", type="primary")

    if submitted:
        save_persisted_state()
    if not submitted:
        return
    if not cost_expression.strip():
        st.error("Cost function is required.")
        return

    try:
        variables = infer_variables(cost_expression)
        initial_values = parse_initial_values(values_text, expected=len(variables))
        tolerance = float(tolerance_text) if tolerance_text.strip() else None
        expression, gradient_expressions, history = gradient_descent(
            cost_expression,
            variables=variables,
            initial_values=initial_values,
            learning_rate=float(learning_rate),
            iterations=int(iterations),
            tolerance=tolerance,
        )
    except Exception as exc:
        st.error(str(exc))
        return

    final_values = history[-1]["values"].reshape(-1, 1)
    record_matrix("final_parameters", "Gradient descent result", final_values, kind="output")

    st.markdown("#### Function")
    st.latex(r"C = " + sp.latex(expression))
    for variable, gradient_expression in zip(variables, gradient_expressions):
        st.latex(rf"\frac{{\partial C}}{{\partial {latex_identifier(variable)}}} = {sp.latex(gradient_expression)}")

    st.markdown("#### History")
    st.dataframe(gradient_history_frame(history, variables), width="stretch", hide_index=True)
    st.markdown("#### Final Parameters")
    render_matrix("final_parameters", final_values)


def render_minimum_panel():
    st.subheader("Check a minimum point")
    st.caption("Check a supplied point of a scalar function. This tests unconstrained smooth interior points; a zero gradient alone does not prove a minimum.")
    with st.form("minimum_form"):
        raw = st.text_input("Function f", value="(x-3)^2+2", key="minimum_expression", help="Use ^ or ** for powers and * for multiplication. Multiple variables are supported.")
        with st.expander("Variable order", expanded=False):
            variable_text = st.text_input("Variables (optional)", value="", key="minimum_variables")
        try:
            _, symbols = parse_functions(raw, variable_text.split(",") if variable_text.strip() else None)
            names = [str(symbol) for symbol in symbols]
        except Exception:
            names = []
        st.caption("Coordinate order: " + (", ".join(names) or "specify variables for a constant function"))
        point = st.text_input("Candidate point", value="3", key="minimum_point", help="Comma-separated exact values, such as 1/3,0 or pi/2. Use * for multiplication.")
        submitted = st.form_submit_button("Check point", type="primary")
    if not submitted:
        return
    save_persisted_state()
    try:
        coordinates = [item.strip() for item in point.split(",") if item.strip()]
        details = check_minimum(raw, coordinates, variable_text.split(",") if variable_text.strip() else None)
    except Exception as exc:
        st.error(str(exc))
        return
    if details["is_minimum"] is True:
        st.success(details["classification"])
    elif details["is_minimum"] is False:
        st.info(details["classification"])
    else:
        st.warning("Inconclusive")
    st.write(details["reason"])
    st.metric("Function value at candidate", format_number(details["value"]))
    record_matrix("minimum_value", "Function at candidate point", [[details["value"]]], kind="output")
    with st.expander("Derivative checks", expanded=False):
        st.latex(r"\nabla f = " + sp.latex(details["gradient"]))
        st.latex("H = " + sp.latex(details["hessian"]))
        for key in ("gradient_at_point", "hessian_at_point"):
            if details[key] is not None:
                render_matrix(key, details[key])
                record_matrix("minimum_" + key, "Minimum point derivative check", details[key], kind="output")
        if details["eigenvalues"] is not None:
            st.caption("Hessian eigenvalues: " + ", ".join(format_number(value) for value in details["eigenvalues"]))
        st.caption("A local minimum is not automatically a global minimum. A global result is reported only when the tool has a convex quadratic certificate. Higher derivative checks for one variable stop at order 12.")


def render_matrix_derivative_panel():
    st.subheader("Differentiate a matrix expression")
    st.caption("MATLAB-style arithmetic for real scalars, vectors, and matrices. Scalar outputs give a gradient; matrix outputs give a Jacobian.")
    with st.form("matrix_derivative_form"):
        expression = st.text_area("Matrix expression", value="x'*A*x", key="matdiff_expression")
        definitions = st.text_area("Matrices and constants", value="A = [1 2;0 3]\nsyms x [2 1] real", key="matdiff_definitions", help="Numeric assignments, symbolic entries, or syms declarations. One definition per line or separated by semicolons.")
        variable = st.text_input("Differentiate with respect to", value="x", key="matdiff_variable")
        with st.expander("Evaluate at values (optional)", expanded=False):
            evaluation = st.text_area("Numeric assignments", value="", key="matdiff_evaluation", help="For example: x = [1;1]. Leave empty to keep the derivative symbolic.")
        with st.expander("Supported MATLAB syntax", expanded=False):
            st.code("x'*A*x\nnorm(X*w-y)^2 + lambda*(w'*w)\ntrace(W'*W)\nsum((X*W-Y).^2,'all')\nsyms A [2 2] real\nsyms lambda real", language="matlab")
            st.caption("Use *, /, \\, ^ for matrix operations; .*, ./, .^ for elementwise operations. Both ' and .' transpose real matrices. Functions: trace, norm, sum, det, inv, diag, transpose, eye, zeros, ones, sin, cos, tan, exp, log, sqrt, abs. Matrix norms require 'fro'; matrix / and \\ require square invertible operands. Indexing is one-based. No loops, scripts, complex arithmetic, or full MATLAB execution.")
        submitted = st.form_submit_button("Differentiate", type="primary")
    if not submitted:
        return
    save_persisted_state()
    try:
        details = differentiate_matrix(expression, definitions, variable, evaluation)
    except Exception as exc:
        st.error(str(exc))
        return
    derivative = details["gradient"] if details["gradient"] is not None else details["jacobian"]
    label = "Gradient" if details["gradient"] is not None else "Jacobian"
    st.markdown(f"#### {label}")
    st.latex(sp.latex(derivative))
    st.caption("Entry order: " + ", ".join(details["variables"]) + ". Jacobians flatten outputs and inputs column by column; a scalar gradient keeps the variable's shape.")
    if details["derivative_at_point"] is not None:
        render_matrix("derivative_at_point", details["derivative_at_point"])
        record_matrix("matrix_derivative", "Matrix expression derivative", details["derivative_at_point"], kind="output")
    elif evaluation.strip():
        st.caption("Some symbols still have no numeric value; the derivative remains symbolic.")
    with st.expander("Expanded expression, value, and Hessian", expanded=False):
        st.latex("f = " + sp.latex(details["value"]))
        if details["value_at_point"] is not None:
            render_matrix("expression_value", details["value_at_point"])
            record_matrix("matrix_expression_value", "Matrix expression value", details["value_at_point"], kind="output")
        if details["hessian"] is not None:
            st.latex("H = " + sp.latex(details["hessian"]))
            if details["hessian_at_point"] is not None:
                render_matrix("hessian_at_point", details["hessian_at_point"])


def render_decision_tree_tab():
    st.subheader("Decision trees")
    mode = st.selectbox("Mode", ("Classification impurity", "Regression MSE / split"), key="tree_mode")
    if mode == "Classification impurity":
        render_classification_tree_panel()
    else:
        render_regression_tree_panel()


def render_classification_tree_panel():
    st.caption("By default, enter the parent first and its disjoint child nodes below it. Only children contribute to weighted impurity. Uncheck the parent option to enter children only.")
    with st.form("classification_tree_form", border=True):
        class_counts, _ = matrix_input("Class counts by node", "tree_counts", default="8 5 5\n6 4 0\n2 1 5")
        includes_parent = st.checkbox("First row is the parent node", value=True, key="tree_includes_parent")
        measure = st.selectbox("Impurity measure", ("all", *CLASSIFICATION_MEASURES), key="tree_measure")
        submitted = st.form_submit_button("Calculate", type="primary")

    if submitted:
        save_persisted_state()
    if not submitted:
        return
    if class_counts is None:
        st.error("Class counts are required.")
        return

    try:
        measures = CLASSIFICATION_MEASURES if measure == "all" else (measure,)
        summaries = []
        for item in measures:
            details = classification_split_details(class_counts, item, includes_parent)
            summaries.append((item, details))
    except Exception as exc:
        st.error(str(exc))
        return

    record_matrix("class_counts", "Decision tree classification input", class_counts, kind="input")
    for item, details in summaries:
        record_matrix(f"{item}_per_node", "Decision tree impurity result", details["impurities"].reshape(-1, 1), kind="output")
        record_matrix(f"weighted_{item}_per_node", "Decision tree impurity result", details["weighted_contributions"].reshape(-1, 1), kind="output")
        record_matrix(f"weighted_{item}", "Decision tree impurity result", [[details["weighted_impurity"]]], kind="output")

    st.markdown("#### Class Counts")
    render_matrix("class_counts", class_counts)
    for item, details in summaries:
        st.markdown(f"#### {item.title()}")
        st.dataframe(pd.DataFrame({
            "Child node": np.arange(1, len(details["node_sizes"]) + 1),
            "Samples": details["node_sizes"],
            "Weight": details["weights"],
            "Impurity": details["impurities"],
            "Weighted contribution": details["weighted_contributions"],
        }), width="stretch", hide_index=True)
        render_matrix(f"weighted_{item}", [[details["weighted_impurity"]]])
        render_matrix(f"parent_{item}", [[details["parent_impurity"]]])
        render_matrix(f"{item}_gain", [[details["gain"]]])


def render_regression_tree_panel():
    st.caption("Split convention from the course book: left x < threshold, right x >= threshold. Enter one feature and one target per sample.")
    with st.form("regression_tree_form", border=True):
        col_x, col_y = st.columns(2)
        with col_x:
            X, _ = matrix_input("x values", "tree_x", default="0.2\n0.7\n1.8\n2.2\n3.7\n4.1")
        with col_y:
            Y, _ = matrix_input("y values", "tree_y", default="2.1\n1.5\n5.8\n6.1\n9.1\n9.5")
        controls = st.columns(2)
        with controls[0]:
            threshold_text = st.text_input("Decision threshold", value="", key="tree_threshold")
        with controls[1]:
            find_best = st.checkbox("Find best one-level split", value=True, key="tree_find_best")
        submitted = st.form_submit_button("Calculate", type="primary")

    if submitted:
        save_persisted_state()
    if not submitted:
        return
    if X is None or Y is None:
        st.error("Both x and y are required.")
        return

    if not threshold_text.strip() and not find_best:
        st.error("Enter a threshold or enable Find best one-level split.")
        return

    threshold = None
    if threshold_text.strip():
        try:
            threshold = float(threshold_text)
        except ValueError:
            st.error("Decision threshold must be numeric.")
            return

    record_matrix("x", "Decision tree regression input", X, kind="input")
    record_matrix("y", "Decision tree regression input", Y, kind="input")

    if threshold is not None:
        try:
            summary = regression_threshold_summary(X, Y, threshold)
        except Exception as exc:
            st.error(str(exc))
            return
        record_matrix("threshold_summary", "Decision tree threshold result", threshold_summary_matrix(summary), kind="output")
        st.markdown("#### Threshold Summary")
        render_threshold_summary(summary)

    if find_best:
        try:
            best, summaries = find_best_regression_split(X, Y)
        except Exception as exc:
            st.error(str(exc))
            return
        record_matrix("best_split", "Decision tree best split result", [[best["threshold"]], [best["overall_mse"]]], kind="output")
        st.markdown("#### Candidate Splits")
        st.dataframe(split_candidates_frame(summaries), width="stretch", hide_index=True)
        st.markdown("#### Best Split")
        render_threshold_summary(best)


def render_kmeans_tab():
    st.subheader("K-means clustering")
    st.caption("Follow how points settle into clusters from your initial centroids.")
    with st.form("kmeans_form", border=True):
        col_x, col_c = st.columns(2)
        with col_x:
            X, _ = matrix_input("Points X", "kmeans_x", default="1 1\n1.5 2\n3 4\n5 7\n3.5 5\n4.5 5\n3.5 4.5")
        with col_c:
            centroids, _ = matrix_input("Initial centroids", "kmeans_centroids", default="1 1\n5 7")
        max_iterations_text = st.text_input("Max iterations", value="", key="kmeans_max_iterations")
        submitted = st.form_submit_button("Run", type="primary")

    if submitted:
        save_persisted_state()
    if not submitted:
        return
    if X is None or centroids is None:
        st.error("Both point and centroid matrices are required.")
        return

    try:
        max_iterations = parse_optional_positive_int(max_iterations_text, "Max iterations")
        history = k_means(X, centroids, max_iterations=max_iterations)
    except Exception as exc:
        st.error(str(exc))
        return

    record_matrix("X", "K-means input", X, kind="input")
    record_matrix("initial_centroids", "K-means input", centroids, kind="input")
    if history:
        record_matrix("final_centroids", "K-means result", history[-1]["centroids"], kind="output")
        record_matrix("clusters", "K-means result", (history[-1]["assignments"] + 1).reshape(-1, 1), kind="output")

    st.markdown("#### Result")
    render_metrics(
        {
            "points": X.shape[0],
            "features": X.shape[1],
            "k": centroids.shape[0],
            "iterations": history[-1]["iteration"],
        }
    )
    for item in history:
        title = "Initial clusters" if item["iteration"] == 0 else f"Iteration {item['iteration']}"
        if item["converged"]:
            title += " - converged"
        with st.expander(title, expanded=True):
            render_metrics({"cost": format_number(item["cost"])})
            render_matrix("centroids", item["centroids"])
            st.dataframe(kmeans_iteration_frame(item["assignments"]), width="stretch", hide_index=True)


def _probability_table(raw):
    table = parse_numeric_array(raw, allow_missing=True)
    if table.shape[1] != 2:
        raise ValueError("Enter two values per row in the displayed column order.")
    return table


def render_probability_equations_panel():
    st.caption("Use events A, B, and C. Enter known probabilities and mark unknowns with ???. * means intersection, + means union, | means conditional probability, and ~ means complement. Independence is not assumed.")
    with st.form("probability_equations_form"):
        raw = st.text_area("Known probabilities and unknowns", value="a = 0.4\na*b = 0.1\na+b = 0.7\na|b = 0.25\nb = ???", key="probability_equations",
                           help="P(A) notation also works. Three-event examples: a*b*c, a+b+c, a|(b+c), ~c. Use one assignment per line; decimals, fractions and percentages are accepted.")
        related = st.checkbox("Show related probabilities", value=False, key="probability_equations_related")
        submitted = st.form_submit_button("Solve", type="primary")
    if not submitted:
        return
    save_persisted_state()
    try:
        solution = solve_probabilities(raw, include_related=related)
    except ValueError as exc:
        st.error(str(exc))
        return
    answers = solution["results"]
    st.dataframe(pd.DataFrame([{"Probability": answer["expression"], "Status": answer["status"].title(),
                               "Answer": format_probability_result(answer)} for answer in answers]),
                 hide_index=True, width="stretch")
    for answer in answers:
        if answer["value"] is not None and not answer["undefined_possible"]:
            record_matrix(answer["expression"], "Solved probability", [[float(answer["value"])]], kind="output")
    if solution["related"]:
        with st.expander("Related probabilities", expanded=True):
            st.dataframe(pd.DataFrame([{"Probability": answer["expression"], "Answer": format_probability_result(answer)}
                                       for answer in solution["related"]]), hide_index=True, width="stretch")
    with st.expander("Probability rules"):
        for identity in solution["identities"]:
            st.code(identity, language=None)
        st.caption("A*B is an event intersection, not multiplication of P(A) and P(B). Bounds use [ ] for included endpoints and ( ) for excluded endpoints. Fractions are treated exactly.")


def render_probability_tab():
    st.subheader("Probability")
    mode = st.selectbox("Probability task", ("Discrete distribution", "Bayes update", "Two events", "Solve from known probabilities", "Continuous density", "Sampling without replacement", "Compare grouped rates"), key="probability_mode")
    if mode == "Solve from known probabilities":
        render_probability_equations_panel()
        return
    if mode == "Sampling without replacement":
        render_sampling_panel()
        return
    if mode == "Compare grouped rates":
        render_rates_panel()
        return
    if mode == "Discrete distribution":
        comparison = st.selectbox("Event", (">=", ">", "<=", "<", "=", "between"), key="pmf_comparison")
        with st.form("discrete_probability_form"):
            raw = st.text_area("Outcome and probability", value="-2 1/4\n0 ?\n3 1/2", key="pmf_table", help="One outcome-probability pair per row. Use ? for at most one unknown probability.")
            boundary = st.text_input("Lower bound" if comparison == "between" else "Boundary", value="0", key="pmf_boundary")
            upper = st.text_input("Upper bound (inclusive)", value="3", key="pmf_upper") if comparison == "between" else None
            submitted = st.form_submit_button("Calculate", type="primary")
        if not submitted:
            return
        save_persisted_state()
        try:
            table = _probability_table(raw)
            details = discrete_distribution(table[:, 0], table[:, 1], comparison, parse_number(boundary), parse_number(upper) if upper is not None else None)
        except Exception as exc:
            st.error(str(exc))
            return
        render_metrics({"Event probability": format_number(details["event_probability"]), "Mean": format_number(details["mean"]), "Variance": format_number(details["variance"]), "Standard deviation": format_number(details["std"])})
        st.dataframe(pd.DataFrame({"Outcome": details["outcomes"], "Probability": details["probabilities"], "Inferred": details["unknown"]}), hide_index=True, width="stretch")
        record_matrix("distribution", "Completed probability distribution", np.column_stack((details["outcomes"], details["probabilities"])), kind="output")
        record_matrix("event_probability", "Discrete event", [[details["event_probability"]]], kind="output")
        with st.expander("Cumulative probabilities", expanded=False):
            unique = np.unique(details["outcomes"])
            st.dataframe(pd.DataFrame({"x": unique, "P(X <= x)": [np.sum(details["probabilities"][details["outcomes"] <= value]) for value in unique]}), hide_index=True, width="stretch")
    elif mode == "Bayes update":
        st.caption("Rows are mutually exclusive, exhaustive groups. Enter P(group) followed by P(event | group); one prior may be ?. Posterior groups keep the same row order.")
        with st.form("bayes_form"):
            raw = st.text_area("Prior and event likelihood", value="0.30 0.10\n0.70 0.04", key="bayes_table")
            observed = st.selectbox("Observation", ("Event occurred", "Event did not occur"), key="bayes_observed")
            submitted = st.form_submit_button("Calculate", type="primary")
        if not submitted:
            return
        save_persisted_state()
        try:
            table = _probability_table(raw)
            details = bayes_update(table[:, 0], table[:, 1], observed == "Event occurred")
        except Exception as exc:
            st.error(str(exc))
            return
        st.metric("Probability of observed evidence", format_number(details["evidence"]))
        st.dataframe(pd.DataFrame({"Group row": np.arange(1, len(details["priors"]) + 1), "Prior": details["priors"], "Observed likelihood": details["likelihoods"], "Joint probability": details["joint"], "Posterior": details["posterior"]}), hide_index=True, width="stretch")
        record_matrix("posterior", "Bayes update", details["posterior"].reshape(-1, 1), kind="output")
        st.caption("Posterior = prior times observed likelihood, divided by the total evidence probability.")
    elif mode == "Two events":
        known_kind = st.selectbox("Known probability", ("Intersection", "Union"), key="event_known_kind")
        with st.form("events_form"):
            columns = st.columns(3)
            p_a = columns[0].number_input("P(A)", min_value=0.0, max_value=1.0, value=0.5, key="event_pa")
            p_b = columns[1].number_input("P(B)", min_value=0.0, max_value=1.0, value=0.4, key="event_pb")
            known = columns[2].number_input(f"P(A {known_kind.lower()} B)", min_value=0.0, max_value=1.0, value=0.2, key="event_known")
            submitted = st.form_submit_button("Calculate", type="primary")
        if not submitted:
            return
        save_persisted_state()
        try:
            details = event_probabilities(p_a, p_b, known, known_kind.lower())
        except Exception as exc:
            st.error(str(exc))
            return
        render_metrics({"Intersection": format_number(details["intersection"]), "Union": format_number(details["union"]),
                        "P(A | B)": format_number(details["p_a_given_b"]) if details["p_a_given_b"] is not None else "undefined",
                        "P(B | A)": format_number(details["p_b_given_a"]) if details["p_b_given_a"] is not None else "undefined"})
        st.write(f"Independent: {'yes' if details['independent'] else 'no'}. Mutually exclusive: {'yes' if details['mutually_exclusive'] else 'no'}.")
        with st.expander("Joint table", expanded=False):
            st.dataframe(pd.DataFrame(details["joint_table"], index=["A", "not A"], columns=["B", "not B"]), width="stretch")
        record_matrix("joint_probability_table", "Two events", details["joint_table"], kind="output")
    else:
        st.caption("Integrate a continuous density on a finite support interval. The calculator verifies nonnegativity and total area 1 before returning a probability.")
        with st.form("density_form"):
            raw = st.text_input("Density f(x)", value="2*x", key="density_expression")
            columns = st.columns(2)
            support_low = columns[0].number_input("Support minimum", value=0.0, key="density_support_low")
            support_high = columns[1].number_input("Support maximum", value=1.0, key="density_support_high")
            query_low = columns[0].number_input("Query minimum", value=0.5, key="density_query_low")
            query_high = columns[1].number_input("Query maximum", value=1.0, key="density_query_high")
            submitted = st.form_submit_button("Integrate", type="primary")
        if not submitted:
            return
        save_persisted_state()
        try:
            details = density_probability(raw, support_low, support_high, query_low, query_high)
        except Exception as exc:
            st.error(str(exc))
            return
        st.latex("f(x) = " + sp.latex(details["expression"]))
        render_metrics({"Total area": format_number(details["total_area"]), "Interval probability": format_number(details["probability"])})
        st.caption("The density is zero outside the support. Interval probabilities are areas; the probability at one exact point is zero.")
        record_matrix("interval_probability", "Continuous density integral", [[details["probability"]]], kind="output")


def render_sampling_panel():
    st.caption("Draw uniformly without replacement from a finite bag. X counts the marked items drawn.")
    comparison = st.selectbox("Event", (">=", ">", "<=", "<", "=", "between"), key="sampling_comparison")
    with st.form("sampling_form"):
        cols = st.columns(3)
        total = cols[0].number_input("Total items", min_value=1, max_value=10000, value=6, step=1, key="sampling_total")
        marked = cols[1].number_input("Marked items", min_value=0, max_value=10000, value=2, step=1, key="sampling_marked")
        draws = cols[2].number_input("Number of draws", min_value=0, max_value=1000, value=2, step=1, key="sampling_draws")
        boundary = st.number_input("Lower count" if comparison == "between" else "Count boundary", min_value=0, value=1, step=1, key="sampling_boundary")
        upper = st.number_input("Upper count (inclusive)", min_value=0, value=2, step=1, key="sampling_upper") if comparison == "between" else None
        submitted = st.form_submit_button("Calculate", type="primary")
    if not submitted:
        return
    save_persisted_state()
    try:
        details = sampling_without_replacement(total, marked, draws, comparison, boundary, upper)
    except Exception as exc:
        st.error(str(exc))
        return
    st.metric("Event probability", format_number(details["event_probability"]))
    st.caption("Exact probability: " + str(details["exact_probability"]))
    record_matrix("sampling_probability", "Without-replacement event", [[details["event_probability"]]], kind="output")
    with st.expander("Count distribution and formula", expanded=False):
        st.latex(r"P(X=k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}")
        st.dataframe(pd.DataFrame({"Marked items drawn": details["outcomes"], "Probability": details["probabilities"]}), hide_index=True, width="stretch")


def render_rates_panel():
    st.caption("One row per group: A successes, A total, B successes, B total. Pooled rates use counts, so groups with different sizes have different weights.")
    with st.form("rates_form"):
        raw = st.text_area("Grouped counts", value="9 10 80 100\n20 100 1 10", key="rates_table")
        submitted = st.form_submit_button("Compare", type="primary")
    if not submitted:
        return
    save_persisted_state()
    try:
        details = compare_grouped_rates(parse_numeric_array(raw))
    except Exception as exc:
        st.error(str(exc))
        return
    st.dataframe(pd.DataFrame({"Group": np.arange(1, len(details["rates"])+1), "A rate": details["rates"][:, 0], "B rate": details["rates"][:, 1]}), hide_index=True, width="stretch")
    render_metrics({"Pooled A rate": format_number(details["pooled"][0]), "Pooled B rate": format_number(details["pooled"][1])})
    if details["reversal"]:
        st.info("Simpson's reversal: the method ahead in every group is behind overall.")
    else:
        st.caption("No reversal between all group comparisons and the pooled comparison.")
    record_matrix("grouped_rates", "Per-group A and B rates", details["rates"], kind="output")
    record_matrix("pooled_rates", "Pooled A and B rates", details["pooled"].reshape(1, -1), kind="output")
    with st.expander("Pooling weights", expanded=False):
        st.dataframe(pd.DataFrame({"Group": np.arange(1, len(details["rates"])+1), "A weight": details["weights"][:, 0], "B weight": details["weights"][:, 1]}), hide_index=True, width="stretch")
        st.caption("Pooled rate = total successes / total cases; it is not the unweighted average of group percentages.")


def render_cache_tab():
    st.subheader("Saved calculations")
    st.caption("Copy a previous result or reuse it as the input for your next calculation.")
    ensure_cache_limit()
    cache = st.session_state.matrix_cache
    top_left, top_middle, top_right = st.columns([1, 1.2, 2.8])
    with top_left:
        if st.button("Clear cache", disabled=not cache, width="stretch"):
            st.session_state.matrix_cache = []
            save_persisted_state()
            st.rerun()
    with top_middle:
        st.number_input(
            "Max allowed cache",
            min_value=1,
            max_value=1000,
            step=1,
            key=CACHE_LIMIT_KEY,
            on_change=on_cache_limit_change,
        )
    with top_right:
        st.markdown(
            f'<div class="solver-meta">{len(cache)} of {max_allowed_cache()} cached matrix item'
            f'{"s" if len(cache) != 1 else ""}</div>',
            unsafe_allow_html=True,
        )

    if not cache:
        st.info("No matrices have been cached yet.")
        return

    for entry in cache:
        with st.container(border=True):
            header_cols = st.columns([3, 2, 1])
            with header_cols[0]:
                st.markdown(f"#### {entry['name']}")
                st.markdown(
                    f'<span class="solver-chip">{entry["kind"]}</span>'
                    f'<span class="solver-chip">{entry["shape"]}</span>',
                    unsafe_allow_html=True,
                )
            with header_cols[1]:
                st.markdown(f'<div class="solver-meta">{html.escape(entry["source"])}</div>', unsafe_allow_html=True)
            with header_cols[2]:
                if st.button("Remove", key=f"remove_{entry['id']}", width="stretch"):
                    remove_cache_item(entry["id"])
                    st.rerun()

            render_matrix(entry["name"], entry["value"])
            with st.expander("Plain text"):
                st.code(entry["text"], language="python")


def matrix_input(label, key, default="", allow_mixed=False):
    raw_key = f"{key}_raw"
    if raw_key not in st.session_state:
        st.session_state[raw_key] = default

    raw = st.text_area(label, key=raw_key, help=MATRIX_HELP, height=110)
    value = None
    status = "empty"
    if raw.strip():
        try:
            value = parse_matrix(raw, allow_mixed=allow_mixed)
            status = "typed"
        except Exception as exc:
            st.warning(f"{label}: {exc}")
            status = "invalid"

    st.form_submit_button(
        f"Transpose {label}",
        key=f"{key}_transpose",
        disabled=value is None,
        on_click=transpose_matrix_input,
        args=(raw_key, allow_mixed),
        width="stretch",
    )

    if value is None:
        return None, status

    with st.expander(f"Preview {label}", expanded=False):
        render_matrix(label, value, copy=False)
    return value, status


def transpose_matrix_input(raw_key, allow_mixed):
    try:
        value = parse_matrix(st.session_state.get(raw_key, ""), allow_mixed=allow_mixed)
    except Exception:
        return
    st.session_state[raw_key] = matrix_editor_text(as_2d_any(value).T)
    save_persisted_state()


def parse_matrix(raw_text, allow_mixed=False):
    text = normalize_matrix_text(raw_text)
    parser = parse_mixed_array if allow_mixed else parse_numeric_array
    return parser(text)


def normalize_matrix_text(raw_text):
    text = raw_text.strip().replace("−", "-").replace("–", "-").replace("Ã¢Ë†â€™", "-")
    if not text:
        return text
    if text.startswith(("[", "(")):
        return " ".join(line.strip() for line in text.splitlines() if line.strip())
    return "; ".join(line.strip() for line in text.splitlines() if line.strip())


def inspect_matrix(array):
    matrix = numeric_matrix(array)
    rows, cols = matrix.shape
    rank = int(np.linalg.matrix_rank(matrix))
    is_square = rows == cols
    inverse_exists = is_square and rank == rows
    left_inverse_exists = rank == cols
    right_inverse_exists = rank == rows

    details = {
        "shape": f"{rows} x {cols}",
        "rank": rank,
        "determinant": format_number(np.linalg.det(matrix)) if is_square else "n/a",
        "inverse": "yes" if inverse_exists else "no",
        "left inverse": "yes" if left_inverse_exists else "no",
        "right inverse": "yes" if right_inverse_exists else "no",
    }
    matrices = []
    if inverse_exists:
        matrices.append(("inverse", np.linalg.inv(matrix)))
    if left_inverse_exists:
        matrices.append(("left_inverse", np.linalg.pinv(matrix)))
    if right_inverse_exists:
        matrices.append(("right_inverse", np.linalg.pinv(matrix)))
    return details, matrices


def solve_linear_system(X, y, equation):
    X = numeric_matrix(X)
    y = numeric_matrix(y)
    if equation == "Xw = y":
        validate_same_sample_count(X, y)
        coefficients = X
        target = y
        orient = lambda value: value
    else:
        if X.shape[1] != y.shape[1]:
            raise ValueError(f"Column count mismatch: X has {X.shape[1]} columns but y has {y.shape[1]} columns.")
        coefficients = X.T
        target = y.T
        orient = lambda value: value.T

    rows, variables = coefficients.shape
    rank_coefficients = int(np.linalg.matrix_rank(coefficients))
    rank_augmented = int(np.linalg.matrix_rank(np.hstack((coefficients, target))))
    is_consistent = rank_coefficients == rank_augmented
    solution, _, _, _ = np.linalg.lstsq(coefficients, target, rcond=None)
    predicted = coefficients @ solution
    residual = predicted - target
    mse = np.mean(np.square(residual), axis=0, keepdims=True)
    standard_error = np.sqrt(mse)

    if is_consistent and rank_coefficients == variables:
        status = "unique exact"
    elif is_consistent:
        status = "many exact"
    else:
        status = "least squares"

    return {
        "system": system_shape_description(coefficients),
        "equations": rows,
        "unknowns": variables,
        "rank_coefficients": rank_coefficients,
        "rank_augmented": rank_augmented,
        "status": status,
        "w": orient(solution),
        "predicted": orient(predicted),
        "residual": orient(residual),
        "standard_error": orient(standard_error),
        "mse": orient(mse),
    }


def fit_regression_model(
    X,
    Y,
    X_test,
    Y_test,
    model_type,
    order,
    target_mode,
    regularization,
    ridge_lambda,
    ridge_form,
    include_bias,
):
    X = numeric_matrix(X)
    validate_same_sample_count(X, as_2d_any(Y))

    model_matrix, poly, model_matrix_name, model_label = build_model_matrix(X, model_type, order, include_bias)
    resolved_target_mode = resolve_target_mode(Y, target_mode)
    target_matrix, class_labels = prepare_target_matrix(Y, resolved_target_mode)

    if regularization == "Ridge":
        system, resolved_form, w = calculate_ridge_weights(
            model_matrix,
            target_matrix,
            ridge_lambda=ridge_lambda,
            form=ridge_form,
        )
        form = f"{resolved_form}, lambda={format_number(ridge_lambda)}"
    else:
        system, w = calculate_closed_form_weights(model_matrix, target_matrix)
        form = "closed form"

    train_pred = model_matrix @ w
    sse, mse = squared_error_summary(target_matrix, train_pred)
    train_class = classify_predictions(train_pred, resolved_target_mode, class_labels)
    diagnostics = regression_diagnostics(model_matrix, target_matrix, w, ridge_lambda if regularization == "Ridge" else 0)

    test_pred = None
    test_class = None
    test_target_matrix = None
    test_sse = None
    test_mse = None
    if X_test is not None:
        X_test = numeric_matrix(X_test)
        validate_same_feature_count(X, X_test)
        X_test_model = transform_test_matrix(X_test, model_type, include_bias, poly)
        test_pred = X_test_model @ w # type: ignore
        test_class = classify_predictions(test_pred, resolved_target_mode, class_labels)
        if Y_test is not None:
            test_target_matrix = prepare_test_target_matrix(Y_test, resolved_target_mode, class_labels, target_matrix.shape[1])
            validate_same_sample_count(X_test, test_target_matrix)
            if test_target_matrix.shape[1] != test_pred.shape[1]:
                raise ValueError(
                    f"Test target output count mismatch: predictions have {test_pred.shape[1]} column(s) "
                    f"but Y_test has {test_target_matrix.shape[1]} column(s)."
                )
            test_sse, test_mse = squared_error_summary(test_target_matrix, test_pred)

    return {
        "model_label": model_label,
        "model_matrix_name": model_matrix_name,
        "model_matrix": model_matrix,
        "target_mode": resolved_target_mode,
        "target_matrix": target_matrix,
        "class_labels": class_labels,
        "system": system,
        "form": form,
        "w": w,
        "train_pred": train_pred,
        "train_class": train_class,
        "sse": np.asarray(sse).reshape(1, -1),
        "mse": np.asarray(mse).reshape(1, -1),
        "test_pred": test_pred,
        "test_class": test_class,
        "test_target_matrix": test_target_matrix,
        "test_sse": None if test_sse is None else np.asarray(test_sse).reshape(1, -1),
        "test_mse": None if test_mse is None else np.asarray(test_mse).reshape(1, -1),
        "diagnostics": diagnostics,
    }


def build_model_matrix(X, model_type, order, include_bias):
    if model_type == "Polynomial":
        poly = PolynomialFeatures(order)
        P = poly.fit_transform(X)
        return P, poly, "P", f"Polynomial order {order}"

    poly = None
    X_model = add_bias_column(X) if include_bias else X
    model_label = "Linear + bias" if include_bias else "Linear"
    return X_model, poly, "X_model", model_label


def transform_test_matrix(X_test, model_type, include_bias, poly):
    if model_type == "Polynomial":
        return poly.transform(X_test)
    return add_bias_column(X_test) if include_bias else X_test


def resolve_target_mode(Y, requested):
    if requested != "Auto":
        return requested

    arr = as_2d_any(Y)
    if contains_text(arr):
        return "One-hot classification"

    numeric = numeric_matrix(arr)
    unique = set(np.unique(numeric).tolist())
    if numeric.shape[1] == 1 and unique.issubset({-1.0, 1.0}):
        return "Sign classification"
    if numeric.shape[1] > 1 and np.all((numeric == 0) | (numeric == 1)) and np.all(np.sum(numeric, axis=1) == 1):
        return "One-hot classification"
    return "Numeric regression"


def prepare_target_matrix(Y, target_mode):
    if target_mode == "One-hot classification":
        return prepare_class_targets(Y)
    return numeric_matrix(Y), None


def prepare_test_target_matrix(Y_test, target_mode, class_labels, class_count=None):
    if target_mode != "One-hot classification":
        return numeric_matrix(Y_test)

    arr = as_2d_any(Y_test)
    if arr.shape[1] > 1:
        return validate_onehot_targets(arr)
    if class_labels is None:
        if class_count is None:
            raise ValueError("The training class count is required to encode test class indices.")
        indices = numeric_matrix(arr).reshape(-1)
        if np.any(indices != np.floor(indices)) or np.any(indices < 0) or np.any(indices >= class_count):
            raise ValueError("For one-hot training targets, test labels must be zero-based class column indices.")
        encoded = np.zeros((len(indices), class_count), dtype=float)
        encoded[np.arange(len(indices)), indices.astype(int)] = 1
        return encoded

    class_to_index = {label: index for index, label in enumerate(class_labels)}
    labels = arr.reshape(-1).tolist()
    encoded = np.zeros((len(labels), len(class_labels)), dtype=float)
    for row_index, label in enumerate(labels):
        if label not in class_to_index:
            raise ValueError(f"Y_test contains an unknown class label: {label!r}.")
        encoded[row_index, class_to_index[label]] = 1.0
    return encoded


def classify_predictions(prediction, target_mode, class_labels):
    if target_mode == "Sign classification":
        signs = np.sign(prediction)
        signs[np.isclose(signs, 0)] = 0
        return signs
    if target_mode == "One-hot classification":
        indices = np.argmax(prediction, axis=1).reshape(-1, 1)
        if class_labels is None:
            return indices
        labels = [class_labels[index] for index in indices.reshape(-1).tolist()]
        return np.asarray(labels, dtype=object).reshape(-1, 1)
    return None


def pearson_details(X, Y):
    return pearson_correlation_details(X, Y)


def infer_expression_variables(cost_expression):
    try:
        return infer_variables(cost_expression) if cost_expression.strip() else []
    except Exception:
        return []


def parse_initial_values(raw_text, expected):
    if expected == 0 and not raw_text.strip():
        return np.array([], dtype=float)
    parts = [part.strip() for part in raw_text.replace("−", "-").replace("–", "-").split(",")]
    if any(not part for part in parts):
        raise ValueError("Enter one initial value per variable, separated by commas.")
    if len(parts) != expected:
        raise ValueError(f"Expected {expected} initial value(s), got {len(parts)}.")
    return np.asarray([parse_number(part) for part in parts], dtype=float)


def parse_optional_positive_int(raw_text, label):
    text = raw_text.strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        raise ValueError(f"{label} must be a positive integer or left empty.") from None
    if value < 1:
        raise ValueError(f"{label} must be a positive integer or left empty.")
    return value


def gradient_history_frame(history, variables):
    rows = []
    for item in history:
        row = {"iteration": item["iteration"]}
        for variable, value in zip(variables, item["values"]):
            row[variable] = format_number(value)
        for variable, value in zip(variables, item["gradient"]):
            row[f"dC/d{variable}"] = "" if np.isnan(value) else format_number(value)
        row["cost"] = format_number(item["cost"])
        rows.append(row)
    return pd.DataFrame(rows)


def kmeans_iteration_frame(assignments):
    return pd.DataFrame(
        [
            {"point": index + 1, "cluster": int(cluster) + 1}
            for index, cluster in enumerate(np.asarray(assignments).reshape(-1).tolist())
        ]
    )


def render_threshold_summary(summary):
    metrics = {
        "threshold": summary["threshold"],
        "root mean": summary["root_mean"],
        "root mse": summary["root_mse"],
        "left mean": summary["left_mean"],
        "left mse": summary["left_mse"],
        "right mean": summary["right_mean"],
        "right mse": summary["right_mse"],
        "split mse": summary["overall_mse"],
    }
    render_metrics({key: format_number(value) for key, value in metrics.items()})
    render_matrix("summary", threshold_summary_matrix(summary))


def threshold_summary_matrix(summary):
    return np.asarray(
        [
            [summary["threshold"]],
            [summary["root_mse"]],
            [summary["left_mse"]],
            [summary["right_mse"]],
            [summary["overall_mse"]],
        ],
        dtype=float,
    )


def split_candidates_frame(summaries):
    return pd.DataFrame(
        [
            {
                "threshold": format_number(summary["threshold"]),
                "overall_mse": format_number(summary["overall_mse"]),
                "left_mse": format_number(summary["left_mse"]),
                "right_mse": format_number(summary["right_mse"]),
            }
            for summary in summaries
        ]
    )


def render_matrix(name, matrix, compact=False, show_table=False, copy=True):
    arr = as_2d_any(matrix)
    if compact:
        st.markdown(f'<div class="solver-meta">{name}: {shape_text(arr)}</div>', unsafe_allow_html=True)
    if copy:
        matrix_cols = st.columns([11, 1.4])
        with matrix_cols[0]:
            st.latex(matrix_latex(name, arr))
        with matrix_cols[1]:
            clipboard_button(matrix_plain_text(arr), "Copy", uuid.uuid4().hex)
    else:
        st.latex(matrix_latex(name, arr))
    if show_table:
        st.dataframe(matrix_frame(arr), width="stretch")


def render_metrics(items):
    columns = st.columns(min(4, max(1, len(items))))
    for index, (label, value) in enumerate(items.items()):
        columns[index % len(columns)].metric(label, str(value))


def pearson_frame(matrix):
    arr = np.asarray(as_2d_any(matrix), dtype=float)
    return pd.DataFrame(
        [[format_number(value) for value in row] for row in arr.tolist()],
        index=[f"X c{i + 1}" for i in range(arr.shape[0])],
        columns=[f"Y c{i + 1}" for i in range(arr.shape[1])],
    )


def matrix_latex(name, matrix):
    arr = as_2d_any(matrix)
    rows = []
    for row in arr.tolist():
        rows.append(" & ".join(latex_cell(value) for value in row))
    body = r" \\ ".join(rows) if rows else ""
    return rf"{latex_identifier(name)} = \begin{{bmatrix}} {body} \end{{bmatrix}}"


def latex_cell(value):
    if is_numeric_scalar(value):
        return format_number(float(value))
    text = str(value)
    escaped = (
        text.replace("\\", r"\textbackslash{}")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )
    return rf"\text{{{escaped}}}"


def latex_identifier(name):
    replacements = {
        r"\hat{y}": r"\hat{y}",
        r"\hat{Y}_{train}": r"\hat{Y}_{train}",
        r"\hat{Y}_{test}": r"\hat{Y}_{test}",
    }
    if name in replacements:
        return replacements[name]
    text = str(name)
    escaped = (
        text.replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace(" ", r"\ ")
    )
    if text.replace("_", "").isalnum():
        return escaped
    return rf"\mathrm{{{escaped}}}"


def matrix_frame(matrix):
    arr = as_2d_any(matrix)
    return pd.DataFrame(
        [[plain_cell(value) for value in row] for row in arr.tolist()],
        index=[f"r{i + 1}" for i in range(arr.shape[0])],
        columns=[f"c{i + 1}" for i in range(arr.shape[1])],
    )


def plain_cell(value):
    if is_numeric_scalar(value):
        return format_number(float(value))
    return str(value)


def title_from_name(name):
    return str(name).replace("_", " ").title()


def as_2d_any(value):
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("Input must be a nonempty two-dimensional matrix.")
    return arr


def numeric_matrix(value):
    return ensure_2d(value)


def contains_text(matrix):
    return any(isinstance(value, str) for value in as_2d_any(matrix).reshape(-1).tolist())


def is_numeric_scalar(value):
    try:
        float(value)
        return not isinstance(value, str)
    except (TypeError, ValueError):
        return False


def system_shape_description(matrix):
    rows, cols = matrix.shape
    if rows > cols:
        return "overdetermined"
    if rows < cols:
        return "underdetermined"
    return "evendetermined"


def record_matrix(name, source, value, kind):
    arr = as_2d_any(value)
    st.session_state.matrix_cache.insert(
        0,
        {
            "id": uuid.uuid4().hex,
            "name": str(name),
            "source": str(source),
            "kind": str(kind),
            "shape": shape_text(arr),
            "value": arr.copy(),
            "text": matrix_plain_text(arr),
            "created_at": time.strftime("%H:%M:%S"),
        },
    )
    trim_matrix_cache()
    save_persisted_state()


def remove_cache_item(entry_id):
    st.session_state.matrix_cache = [entry for entry in st.session_state.matrix_cache if entry["id"] != entry_id]
    save_persisted_state()


def cache_label(entry):
    return f"{entry['name']} | {entry['source']} | {entry['shape']}"


def shape_text(matrix):
    arr = as_2d_any(matrix)
    return f"{arr.shape[0]} x {arr.shape[1]}"


def matrix_editor_text(matrix):
    arr = as_2d_any(matrix)
    if all(is_numeric_scalar(value) for value in arr.reshape(-1).tolist()):
        return "\n".join(" ".join(repr(float(value)) for value in row) for row in arr.tolist())
    return json.dumps(arr.tolist(), ensure_ascii=False)


def matrix_plain_text(matrix):
    arr = as_2d_any(matrix)
    if all(is_numeric_scalar(value) for value in arr.reshape(-1).tolist()):
        return python_array_literal(np.asarray(arr, dtype=float))
    return json.dumps(arr.tolist(), ensure_ascii=False)


def clipboard_button(text, label, entry_id):
    button_id = f"copy_{entry_id}"
    payload = json.dumps(text).replace("</", "<\\/")
    reset_label = json.dumps(label).replace("</", "<\\/")
    safe_label = html.escape(label)
    st.iframe(
        f"""
        <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: transparent;
        }}
        #{button_id} {{
            width: 100%;
            height: 32px;
            box-sizing: border-box;
            border: 1px solid #355d55;
            border-radius: 6px;
            padding: 0 0.55rem;
            margin: 0;
            background: #1c3937;
            color: #85efce;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font: 13px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1;
            white-space: nowrap;
        }}
        #{button_id}:hover {{
            border-color: #6ee7c0;
            background: #244c45;
        }}
        #{button_id}:focus-visible {{
            outline: 2px solid #6ee7c0;
            outline-offset: -3px;
        }}
        </style>
        <button id="{button_id}">{safe_label}</button>
        <script>
        const button = document.getElementById("{button_id}");
        button.addEventListener("click", async () => {{
            try {{
                await navigator.clipboard.writeText({payload});
                const original = button.textContent;
                button.textContent = "Copied";
                setTimeout(() => button.textContent = original, 900);
            }} catch (error) {{
                button.textContent = "Copy failed";
                setTimeout(() => button.textContent = {reset_label}, 1200);
            }}
        }});
        </script>
        """,
        height=32,
    )


@contextlib.contextmanager
def quiet_stdout():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


if __name__ == "__main__":
    main()
