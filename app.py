from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.explainability import explain_model, explain_single_prediction
from src.predict import load_model, predict_for_row
from src.schema import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    ROOT_DIR,
)


st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="SP",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --canvas: #090c14;
        --surface: #101522;
        --surface-raised: #151c2c;
        --line: rgba(148, 163, 184, 0.16);
        --line-strong: rgba(148, 163, 184, 0.26);
        --text: #edf1fa;
        --muted: #9aa6ba;
        --accent: #8b83ff;
        --blue: #68a6ff;
        --positive: #67d9b0;
        --negative: #f08e9c;
        --font-ui: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 14% -10%, rgba(86, 79, 190, 0.15), transparent 36rem),
            var(--canvas);
        color: var(--text);
        font-family: var(--font-ui);
    }
    .block-container {
        max-width: 1620px;
        padding: 1.05rem clamp(1.1rem, 3vw, 3rem) 2rem;
    }
    h1, h2, h3, h4, p, label, li { color: var(--text); }
    .stApp h1 {
        padding: 0;
        margin: 0;
        font-size: clamp(1.45rem, 2vw, 1.8rem) !important;
        line-height: 1.25 !important;
        white-space: normal;
    }
    h2 { font-size: 1.35rem; letter-spacing: -0.02em; }
    h3 { font-size: 1.04rem; letter-spacing: -0.01em; }
    p, li, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="stVerticalBlock"] { gap: 0.68rem; }
    [data-testid="stRadio"] > label { display: none; }
    div[role="radiogroup"] {
        gap: 0.35rem;
        border-bottom: 1px solid var(--line);
        padding: 0.35rem 0 0.8rem;
    }
    div[role="radiogroup"] label {
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 0.42rem 0.86rem;
        color: var(--muted) !important;
        transition: background 140ms ease, border-color 140ms ease, color 140ms ease;
    }
    div[role="radiogroup"] label:hover {
        background: rgba(139, 131, 255, 0.08);
        border-color: var(--line);
        color: var(--text) !important;
    }
    div[role="radiogroup"] label:has(input:checked) {
        background: rgba(139, 131, 255, 0.14);
        border-color: rgba(139, 131, 255, 0.24);
        color: #dedbff !important;
    }
    div[role="radiogroup"] label > div:first-child { display: none; }
    [data-testid="stMetric"] {
        height: 100%;
        padding: 0.9rem 1.05rem;
        background: linear-gradient(150deg, rgba(23, 30, 48, 0.96), rgba(15, 20, 32, 0.94));
        border: 1px solid var(--line);
        border-radius: 13px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    [data-testid="stMetricLabel"] { color: #aeb9cc; font-size: 0.8rem; }
    [data-testid="stMetricValue"] { color: var(--text); font-size: 1.5rem; font-weight: 650; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem; }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
        background: rgba(16, 21, 34, 0.68);
    }
    [data-testid="stPlotlyChart"], [data-testid="stArrowVegaLiteChart"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        background: rgba(16, 21, 34, 0.62);
        padding: 0.45rem;
    }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        min-height: 2.65rem;
        background: #111827;
        border-color: var(--line-strong);
        border-radius: 9px;
    }
    [data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent);
    }
    div.stButton > button, div.stFormSubmitButton > button {
        min-height: 2.55rem;
        color: #f8f8ff;
        background: linear-gradient(115deg, #6259cb, #586dc9);
        border: 1px solid rgba(174, 166, 255, 0.25);
        border-radius: 8px;
        font-weight: 600;
        transition: transform 140ms ease, filter 140ms ease, border-color 140ms ease;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.08);
        border-color: rgba(190, 183, 255, 0.55);
    }
    div.stButton > button:focus-visible, div.stFormSubmitButton > button:focus-visible {
        outline: 2px solid #aaa3ff;
        outline-offset: 2px;
    }
    [data-testid="stExpander"] {
        background: rgba(16, 21, 34, 0.64);
        border: 1px solid var(--line);
        border-radius: 10px;
    }
    [data-testid="stAlert"] { border-radius: 10px; }
    .app-brand {
        display: flex;
        align-items: center;
        gap: 0.95rem;
        padding: 0.15rem 0 0.75rem;
    }
    .brand-mark {
        display: grid;
        width: 2.85rem;
        height: 2.85rem;
        place-items: center;
        flex: 0 0 auto;
        border: 1px solid rgba(153, 145, 255, 0.4);
        border-radius: 12px;
        color: #e2dfff;
        background:
            radial-gradient(circle at 25% 15%, rgba(187, 180, 255, 0.3), transparent 48%),
            linear-gradient(145deg, rgba(116, 105, 231, 0.34), rgba(64, 105, 178, 0.22));
        box-shadow: inset 0 1px rgba(255,255,255,0.08), 0 6px 18px rgba(54, 54, 130, 0.18);
        font-size: 0.9rem;
        font-weight: 800;
        letter-spacing: 0.08em;
    }
    .brand-kicker, .eyebrow {
        margin: 0 0 0.22rem;
        color: #aaa4f7 !important;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }
    .brand-name {
        color: #f0f2fb;
        font-size: 1.12rem;
        font-weight: 720;
        letter-spacing: -0.035em;
        line-height: 1.2;
    }
    .brand-subtitle { margin: 0.2rem 0 0; color: var(--muted) !important; font-size: 0.8rem; }
    .hero-panel {
        position: relative;
        overflow: hidden;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: clamp(1.35rem, 2.5vw, 2.1rem);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        background:
            radial-gradient(ellipse at 88% 20%, rgba(101, 111, 231, 0.21), transparent 25rem),
            radial-gradient(ellipse at 5% 115%, rgba(62, 108, 184, 0.13), transparent 27rem),
            linear-gradient(118deg, rgba(21, 28, 46, 0.99), rgba(12, 17, 29, 0.98));
        box-shadow: 0 16px 42px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        max-width: 64rem;
        margin: 0.35rem 0 0.55rem;
        color: var(--text);
        font-size: clamp(1.85rem, 2.7vw, 2.55rem);
        font-weight: 720;
        letter-spacing: -0.052em;
        line-height: 1.12;
    }
    .hero-copy { max-width: 62rem; margin: 0; color: #aeb8cb; font-size: 0.94rem; line-height: 1.6; }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.38rem 0.65rem;
        border: 1px solid rgba(240, 142, 156, 0.25);
        border-radius: 999px;
        color: #f2bdc5;
        background: rgba(135, 55, 70, 0.13);
        font-size: 0.72rem;
        font-weight: 600;
    }
    .status-dot { width: 0.42rem; height: 0.42rem; border-radius: 50%; background: #e9a0aa; }
    .section-heading {
        margin: 1.1rem 0 0.15rem;
        color: #f0f2fb;
        font-size: 1.08rem;
        font-weight: 670;
        letter-spacing: -0.02em;
    }
    .section-copy { margin: 0 0 0.65rem; color: #a0acc0; font-size: 0.82rem; line-height: 1.55; }
    .surface-card {
        height: 100%;
        min-height: 82px;
        padding: 0.9rem 1rem;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: linear-gradient(145deg, rgba(18, 24, 39, 0.92), rgba(14, 19, 31, 0.86));
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
    }
    .surface-title { margin: 0 0 0.35rem; color: #edf0fa; font-size: 0.94rem; font-weight: 650; letter-spacing: -0.01em; }
    .surface-copy { margin: 0; color: #a5b0c3; font-size: 0.79rem; line-height: 1.55; }
    .result-panel {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(220px, 0.72fr);
        align-items: center;
        gap: 1.25rem 2rem;
        padding: 1.35rem 1.55rem;
        border: 1px solid var(--result-border, rgba(103, 217, 176, 0.32));
        border-radius: 15px;
        background:
            radial-gradient(ellipse at 85% 12%, var(--result-glow, rgba(103, 217, 176, 0.1)), transparent 22rem),
            linear-gradient(120deg, rgba(20, 28, 40, 0.98), rgba(14, 20, 32, 0.96));
        box-shadow: 0 12px 34px rgba(0, 0, 0, 0.18);
    }
    .result-placed { --result-border: rgba(103, 217, 176, 0.36); --result-glow: rgba(103, 217, 176, 0.12); --result-color: #79e0bb; }
    .result-not-placed { --result-border: rgba(240, 142, 156, 0.34); --result-glow: rgba(240, 142, 156, 0.1); --result-color: #f39aa7; }
    .result-label { margin: 0; color: #a7b2c5; font-size: 0.72rem; font-weight: 650; letter-spacing: 0.12em; text-transform: uppercase; }
    .result-value { margin: 0.3rem 0 0.4rem; color: var(--result-color, var(--positive)); font-size: clamp(2rem, 3vw, 2.7rem); font-weight: 760; letter-spacing: -0.055em; line-height: 1; }
    .result-note { margin: 0; color: #aab5c8; font-size: 0.81rem; line-height: 1.5; }
    .probability-block { min-width: 0; }
    .probability-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; margin-bottom: 0.55rem; color: #c2cbda; font-size: 0.8rem; font-weight: 600; }
    .probability-value { color: var(--result-color, var(--positive)); font-size: 1.55rem; font-weight: 720; letter-spacing: -0.04em; white-space: nowrap; }
    .probability-track { height: 0.55rem; overflow: hidden; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 99px; background: rgba(6, 10, 18, 0.78); }
    .probability-fill { height: 100%; border-radius: inherit; background: linear-gradient(90deg, #7770e8, var(--result-color, var(--positive))); }
    .probability-caption { margin: 0.5rem 0 0; color: #8f9bb0; font-size: 0.72rem; line-height: 1.45; }
    .footer-note { padding-top: 1.1rem; color: #758197; font-size: 0.72rem; }
    @media (max-width: 760px) {
        .block-container { padding: 1rem 1rem 2rem; }
        div[role="radiogroup"] { flex-wrap: wrap; }
        .hero-panel { min-height: 0; padding: 1.1rem; }
        .hero-title { font-size: clamp(1.65rem, 7vw, 2.1rem); }
        .result-panel { grid-template-columns: 1fr; gap: 1rem; padding: 1.1rem; }
        .section-heading { margin-top: 0.9rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { transition-duration: 0.01ms !important; scroll-behavior: auto !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


SCREEN_OPTIONS = [
    "Overview",
    "Student assessment",
    "Model comparison",
    "Explainability",
    "About",
]


@st.cache_resource
def load_artifacts():
    model = load_model(str(MODEL_PATH)) if MODEL_PATH.exists() else None
    metric_file = ROOT_DIR / "reports" / "model_metrics.json"
    metrics = json.loads(metric_file.read_text(encoding="utf-8")) if metric_file.exists() else {}
    return model, metrics


@st.cache_data
def load_training_data() -> pd.DataFrame:
    return pd.read_csv(ROOT_DIR / "train.csv")


def navigate_to(screen: str) -> None:
    st.session_state["active_screen"] = screen


def render_brand() -> None:
    st.markdown(
        """
        <div class="app-brand">
            <div class="brand-mark">SP</div>
            <div>
                <div class="brand-kicker">Student outcome intelligence</div>
                <div class="brand-name">
                    Student Placement Predictor
                </div>
                <p class="brand-subtitle">Explainable analytics for an academic demonstration</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    render_brand()
    st.radio(
        "Main navigation",
        SCREEN_OPTIONS,
        horizontal=True,
        label_visibility="collapsed",
        key="active_screen",
    )


def section_intro(title: str, description: str) -> None:
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-copy">{description}</div>', unsafe_allow_html=True)


def panel(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="surface-card">
            <div class="surface-title">{title}</div>
            <p class="surface-copy">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def training_summary(train_df: pd.DataFrame) -> tuple[pd.Series, float]:
    distribution = train_df["Placement_Status"].value_counts().reindex(
        ["Placed", "Not Placed"], fill_value=0
    )
    placed_rate = float(distribution.get("Placed", 0) / len(train_df)) if len(train_df) else 0.0
    return distribution, placed_rate


def render_horizontal_bar_chart(
    data: pd.DataFrame,
    category_field: str,
    value_field: str,
    color: str,
    height: int = 280,
) -> None:
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "height": height,
        "mark": {"type": "bar", "cornerRadiusEnd": 4},
        "encoding": {
            "x": {
                "field": value_field,
                "type": "quantitative",
                "title": None,
                "axis": {
                    "grid": True,
                    "gridColor": "#273044",
                    "labelColor": "#9aa6ba",
                    "titleColor": "#9aa6ba",
                },
            },
            "x2": {"datum": 0},
            "y": {
                "field": category_field,
                "type": "nominal",
                "sort": "-x",
                "title": None,
                "axis": {
                    "labelColor": "#cbd5e1",
                    "ticks": False,
                    "domain": False,
                },
            },
            "color": {"value": color},
            "tooltip": [
                {"field": category_field, "type": "nominal", "title": category_field},
                {"field": value_field, "type": "quantitative", "title": value_field, "format": ",.3f"},
            ],
        },
        "config": {
            "view": {"stroke": None},
            "axis": {"labelFontSize": 12, "titleFontSize": 12},
        },
    }
    st.vega_lite_chart(data, spec=spec, width="stretch")


def render_overview(train_df: pd.DataFrame, metrics: dict, model) -> None:
    distribution, placed_rate = training_summary(train_df)
    selected_name = metrics.get("selected_model", {}).get("name", "Not available")
    test_metrics = metrics.get("selected_model", {}).get("test_metrics", {})
    test_accuracy = test_metrics.get("accuracy")

    st.markdown(
        """
        <section class="hero-panel">
            <div class="eyebrow">Placement intelligence · Machine learning demonstrator</div>
            <div class="hero-title">Read the signals behind student placement.</div>
            <p class="hero-copy">
                Assess a student profile, compare five classification approaches, and inspect
                explainable model outputs. Built for exploration—not real-world selection.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    stat_columns = st.columns(4)
    stat_columns[0].metric("Training records", f"{len(train_df):,}")
    stat_columns[1].metric("Placed in training data", f"{placed_rate:.1%}")
    stat_columns[2].metric("Selected model", selected_name)
    stat_columns[3].metric(
        "Held-out test accuracy",
        f"{test_accuracy:.3f}" if test_accuracy is not None else "Unavailable",
    )
    st.caption(
        "Dataset provenance has not been independently verified. The apparent synthetic/rule-generated labels "
        "make these metrics unsuitable as real-world performance estimates."
    )

    chart_column, note_column = st.columns([1.35, 0.9], gap="large")
    with chart_column:
        section_intro(
            "Placement outcome distribution",
            "Observed target counts in train.csv. This describes the dataset—not a live placement rate.",
        )
        distribution_frame = distribution.rename_axis("Outcome").rename("Students").reset_index()
        render_horizontal_bar_chart(
            distribution_frame,
            "Outcome",
            "Students",
            "#8278e8",
            height=230,
        )
    with note_column:
        section_intro("How this model works", "A compact, reproducible tabular ML workflow.")
        panel(
            "01 · Profile",
            "Enter academic, technical, experience, and aptitude attributes available in the dataset.",
        )
        panel(
            "02 · Predict",
            "The saved preprocessing-and-classifier pipeline returns a class and class probabilities.",
        )
        panel(
            "03 · Explain",
            "SHAP summarizes model contributions for global patterns and an individual submitted profile.",
        )

    section_intro("Explore the workspace", "Choose a workflow to continue.")
    action_columns = st.columns(3, gap="medium")
    actions = [
        ("Student assessment", "Create a profile assessment", "Student assessment"),
        ("Model comparison", "Review validation and test results", "Model comparison"),
        ("Explainability", "Inspect global model behavior", "Explainability"),
    ]
    for column, (title, description, screen) in zip(action_columns, actions):
        with column:
            panel(title, description)
            st.button(
                f"Open {title}",
                key=f"open_{screen}",
                on_click=navigate_to,
                args=(screen,),
                width="stretch",
                disabled=(screen == "Student assessment" and model is None),
            )


def build_input_form(train_df: pd.DataFrame) -> dict[str, object]:
    values: dict[str, object] = {}
    saved = st.session_state.get("last_assessment", {})
    with st.form("student_form"):
        left, right = st.columns(2, gap="large")
        with left:
            section_intro("Academic profile", "Core study information and academic record.")
            age = train_df["Age"]
            cgpa = train_df["CGPA"]
            values["Age"] = st.number_input(
                "Age",
                min_value=int(age.min()),
                max_value=int(age.max()),
                value=int(saved.get("Age", age.median())),
                help="Allowed range follows the observed training data.",
            )
            values["CGPA"] = st.number_input(
                "CGPA",
                min_value=float(cgpa.min()),
                max_value=float(cgpa.max()),
                value=float(saved.get("CGPA", round(cgpa.median(), 2))),
                step=0.01,
                help="Cumulative grade point average on the dataset's 10-point scale.",
            )
            for feature, label in (("Degree", "Degree"), ("Branch", "Branch")):
                choices = sorted(train_df[feature].dropna().astype(str).unique().tolist())
                default = saved.get(feature, choices[0])
                values[feature] = st.selectbox(
                    label, choices, index=choices.index(default) if default in choices else 0
                )

            gender_choices = sorted(
                train_df["Gender"].dropna().astype(str).unique().tolist()
            )
            default_gender = saved.get("Gender", gender_choices[0])
            values["Gender"] = st.selectbox(
                "Gender",
                gender_choices,
                index=gender_choices.index(default_gender)
                if default_gender in gender_choices
                else 0,
            )

            section_intro("Experience", "Practical experience and academic backlogs.")
            for feature, label in (("Internships", "Internships"), ("Backlogs", "Backlogs")):
                series = train_df[feature]
                values[feature] = st.number_input(
                    label,
                    min_value=int(series.min()),
                    max_value=int(series.max()),
                    value=int(saved.get(feature, series.median())),
                    help="Range follows observed values in train.csv.",
                )

        with right:
            section_intro("Technical skills", "Project work, coding ability, and certifications.")
            for feature, label in (
                ("Projects", "Projects"),
                ("Coding_Skills", "Coding skill rating (1–10)"),
                ("Certifications", "Certifications"),
            ):
                series = train_df[feature]
                values[feature] = st.number_input(
                    label,
                    min_value=int(series.min()),
                    max_value=int(series.max()),
                    value=int(saved.get(feature, series.median())),
                    help="Range follows observed values in train.csv.",
                )

            section_intro("Aptitude & communication", "Assessment score and soft-skill ratings.")
            for feature, label in (
                ("Aptitude_Test_Score", "Aptitude test score"),
                ("Communication_Skills", "Communication skill rating (1–10)"),
                ("Soft_Skills_Rating", "Soft skills rating (1–10)"),
            ):
                series = train_df[feature]
                values[feature] = st.number_input(
                    label,
                    min_value=int(series.min()),
                    max_value=int(series.max()),
                    value=int(series.median()),
                    help="Range follows observed values in train.csv.",
                )

        st.caption("Only fields present in the repository dataset are used. No student identifier is collected.")
        submitted = st.form_submit_button(
            "Generate assessment",
            type="primary",
            width="stretch",
        )
    return values if submitted else {}


def render_prediction_result(model, values: dict[str, object]) -> None:
    try:
        result = predict_for_row(model, values)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Please review the assessment fields: {exc}")
        return

    prediction = result["prediction"]
    predicted_probability = result["predicted_class_probability"]
    positive_probability = result["probability"]
    outcome_class = "result-placed" if prediction == "Placed" else "result-not-placed"
    st.markdown(
        f"""
        <div class="result-panel {outcome_class}">
            <div>
                <p class="result-label">Assessment outcome · model estimate</p>
                <div class="result-value">{prediction}</div>
                <p class="result-note">Predicted class from the saved placement pipeline.</p>
            </div>
            <div class="probability-block">
                <div class="probability-heading">
                    <span>Predicted-class probability</span>
                    <span class="probability-value">{predicted_probability:.1%}</span>
                </div>
                <div class="probability-track" role="img" aria-label="Model-estimated predicted-class probability">
                    <div class="probability-fill" style="width:{predicted_probability:.1%}"></div>
                </div>
                <p class="probability-caption">
                    Model-estimated probability, not a guarantee. Calibration has not been verified.
                    Probability of Placed: {positive_probability:.1%}.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    probability_columns = st.columns(2)
    probability_columns[0].metric(
        "Model-estimated probability · Placed",
        f"{result['probability_by_class']['Placed']:.1%}",
    )
    probability_columns[1].metric(
        "Model-estimated probability · Not Placed",
        f"{result['probability_by_class']['Not Placed']:.1%}",
    )
    st.caption(
        "Probabilities come from the classifier's fitted class ordering and are not verified as calibrated."
    )

    submitted_frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    background = load_training_data()[FEATURE_COLUMNS].sample(
        min(100, len(load_training_data())), random_state=17
    )
    try:
        explanation = explain_single_prediction(
            model, submitted_frame, background_df=background
        )
    except (ValueError, KeyError, RuntimeError, IndexError) as exc:
        st.warning(f"The prediction succeeded, but this explanation could not be generated: {exc}")
        return
    if explanation["status"] != "available":
        st.info(explanation.get("message", "SHAP is unavailable in this environment."))
        return

    section_intro(
        "Why this profile received this result",
        f"SHAP contributions for the submitted profile and predicted class "
        f"({explanation['explained_class']}). These describe model behavior, not causal effects.",
    )
    contribution_frame = pd.DataFrame(explanation["top_features"])
    contribution_frame["Direction"] = contribution_frame["contribution"].map(
        lambda value: "Supports class" if value > 0 else "Opposes class" if value < 0 else "Neutral"
    )
    contribution_frame = contribution_frame.rename(
        columns={
            "feature": "Transformed feature",
            "value": "Transformed input",
            "contribution": "SHAP contribution",
        }
    )
    st.dataframe(
        contribution_frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Transformed input": st.column_config.NumberColumn(format="%.3f"),
            "SHAP contribution": st.column_config.NumberColumn(format="%+.4f"),
        },
    )


def render_assessment(model, train_df: pd.DataFrame) -> None:
    section_intro(
        "Student assessment",
        "Enter a profile within the dataset's observed ranges. Inputs are processed by the saved fitted pipeline.",
    )
    if model is None:
        st.error("Model artifact is missing. Run `python scripts/train_models.py` before making an assessment.")
        return
    values = build_input_form(train_df)
    if values:
        st.session_state["last_assessment"] = values
    previous_assessment = st.session_state.get("last_assessment")
    if previous_assessment:
        render_prediction_result(model, previous_assessment)


def render_comparison(metrics: dict) -> None:
    section_intro(
        "Model comparison",
        "Five classifiers evaluated on the validation partition. The final selected model is reported separately on test.csv.",
    )
    if not metrics:
        st.info("Evaluation report not found. Run `python scripts/train_models.py` to generate it.")
        return

    st.caption(
        "Validation results on the repository's unverified synthetic/rule-generated dataset. "
        "Positive class: Placed."
    )
    rows = []
    confusion_rows = []
    for item in metrics["comparison"]:
        values = item["validation_metrics"]
        rows.append(
            {
                "Classifier": item["model"],
                "Accuracy": values["accuracy"],
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1 score": values["f1"],
                "ROC-AUC": values.get("roc_auc"),
            }
        )
        tn, fp = values["confusion_matrix"][0]
        fn, tp = values["confusion_matrix"][1]
        confusion_rows.append(
            {"Classifier": item["model"], "True negative": tn, "False positive": fp,
             "False negative": fn, "True positive": tp}
        )

    score_frame = pd.DataFrame(rows)
    st.dataframe(
        score_frame.style.format(
            {
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1 score": "{:.3f}",
                "ROC-AUC": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    chart_column, test_column = st.columns([1.15, 0.85], gap="large")
    with chart_column:
        section_intro("Validation score profile", "Compare core metrics across candidate classifiers.")
        chart_data = score_frame.melt(
            id_vars="Classifier",
            var_name="Metric",
            value_name="Score",
        )
        chart_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
            "height": 260,
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "encoding": {
                "x": {
                    "field": "Score",
                    "type": "quantitative",
                    "scale": {"domain": [0, 1]},
                    "axis": {
                        "title": "Metric score",
                        "format": ".0%",
                        "tickCount": 5,
                        "gridColor": "#273044",
                        "labelColor": "#9aa6ba",
                        "titleColor": "#9aa6ba",
                    },
                },
                "x2": {"datum": 0},
                "y": {
                    "field": "Classifier",
                    "type": "nominal",
                    "sort": "-x",
                    "axis": {"title": None, "labelColor": "#cbd5e1"},
                },
                "yOffset": {"field": "Metric"},
                "color": {
                    "field": "Metric",
                    "type": "nominal",
                    "scale": {
                        "range": ["#8077e8", "#5e9be6", "#50c4ad", "#b28add", "#6478bd"]
                    },
                    "legend": {"title": None, "labelColor": "#cbd5e1"},
                },
                "tooltip": [
                    {"field": "Classifier", "type": "nominal"},
                    {"field": "Metric", "type": "nominal"},
                    {"field": "Score", "type": "quantitative", "format": ".3f"},
                ],
            },
            "config": {
                "view": {"stroke": None},
                "axis": {"labelFontSize": 11, "titleFontSize": 12},
            },
        }
        st.vega_lite_chart(chart_data, spec=chart_spec, width="stretch")
    with test_column:
        selected = metrics.get("selected_model", {})
        test_metrics = selected.get("test_metrics", {})
        section_intro(
            "Final test evaluation",
            f"{selected.get('name', 'Selected classifier')} evaluated on test.csv after validation-based selection.",
        )
        test_columns = st.columns(2)
        for column, label, key in zip(
            test_columns,
            ["Accuracy", "F1 score"],
            ["accuracy", "f1"],
        ):
            column.metric(label, f"{test_metrics[key]:.3f}" if key in test_metrics else "Unavailable")
        st.caption(
            "Dataset provenance is unverified. Repeated runs recompute these test metrics; "
            "do not use them for model tuning."
        )

    section_intro("Validation confusion matrices", "Rows are actual; columns are predicted. Positive class is Placed.")
    st.dataframe(
        pd.DataFrame(confusion_rows),
        width="stretch",
        hide_index=True,
    )
    if test_metrics.get("confusion_matrix"):
        with st.expander("Selected model · held-out test confusion matrix"):
            st.dataframe(
                pd.DataFrame(
                    test_metrics["confusion_matrix"],
                    index=["Actual Not Placed", "Actual Placed"],
                    columns=["Predicted Not Placed", "Predicted Placed"],
                ),
                width="stretch",
            )


def render_explainability(model) -> None:
    section_intro(
        "Explainability",
        "Global and local SHAP views are computed from the selected saved pipeline. Attributions explain model behavior; they are not causal claims.",
    )
    if model is None:
        st.warning("Load the saved model artifact to compute explanations.")
        return
    train_features = load_training_data()[FEATURE_COLUMNS]
    explanation_sample = train_features.sample(
        min(100, len(train_features)), random_state=42
    )
    background_sample = train_features.sample(
        min(100, len(train_features)), random_state=17
    )
    try:
        explanation = explain_model(
            model, explanation_sample, background_sample
        )
    except (ValueError, KeyError, RuntimeError) as exc:
        st.error(f"Unable to calculate global SHAP values: {exc}")
        return
    if explanation["status"] != "available":
        st.info(explanation.get("message", "SHAP is unavailable in this environment."))
        return

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        section_intro(
            "Global feature contributions",
            f"Mean absolute SHAP contribution over {explanation['explained_rows']} reproducibly sampled training records.",
        )
        importance_frame = pd.DataFrame(explanation["global_top_features"]).rename(
            columns={"feature": "Transformed feature", "importance": "Mean |SHAP|"}
        )
        render_horizontal_bar_chart(
            importance_frame,
            "Transformed feature",
            "Mean |SHAP|",
            "#8278e8",
            height=320,
        )
    with right:
        st.markdown("<div style='height:1.45rem'></div>", unsafe_allow_html=True)
        st.dataframe(
            importance_frame,
            width="stretch",
            hide_index=True,
            column_config={
                "Mean |SHAP|": st.column_config.NumberColumn(format="%.4f")
            },
        )
        panel(
            "How to read this",
            "Larger mean absolute values indicate the feature has more influence on the selected model's outputs in this sample. Direction and impact vary by profile.",
        )

    section_intro(
        "Individual SHAP explanations",
        "A profile-specific explanation is generated after a student submits the assessment form.",
    )
    st.button(
        "Go to student assessment",
        on_click=navigate_to,
        args=("Student assessment",),
    )


def render_about() -> None:
    section_intro(
        "Project & responsible use",
        "An academic demonstration of a tabular machine-learning workflow for student placement outcomes.",
    )
    first, second = st.columns(2, gap="large")
    with first:
        panel(
            "Model workflow",
            "The pipeline applies numerical imputation/scaling and categorical encoding, compares Logistic Regression, Decision Tree, Random Forest, SVM, and XGBoost, selects using validation F1, and reports a separate final test evaluation.",
        )
        st.write("")
        panel(
            "Data provenance",
            "The origin and real-world validity of the dataset have not been independently verified. Existing notebook evidence and perfect separability are consistent with synthetic or rule-generated labels.",
        )
    with second:
        panel(
            "Intended use",
            "Use this interface to demonstrate feature handling, classification, evaluation, and explanation methods in an academic setting—not to decide placement eligibility or hiring outcomes.",
        )
        st.write("")
        panel(
            "Interpretation",
            "Displayed class probabilities are model estimates. Calibration has not been verified, and SHAP values describe learned model behavior rather than causal effects.",
        )
    st.warning(
        "Do not use these predictions for real-world student selection, recruitment, or placement decisions."
    )


model, metrics = load_artifacts()
train_data = load_training_data()
if "active_screen" not in st.session_state:
    st.session_state["active_screen"] = SCREEN_OPTIONS[0]

render_header()
active_screen = st.session_state["active_screen"]

if active_screen == "Overview":
    render_overview(train_data, metrics, model)
elif active_screen == "Student assessment":
    render_assessment(model, train_data)
elif active_screen == "Model comparison":
    render_comparison(metrics)
elif active_screen == "Explainability":
    render_explainability(model)
else:
    render_about()

st.markdown(
    '<div class="footer-note">Student Placement Predictor · Academic demonstration · Dataset provenance unverified</div>',
    unsafe_allow_html=True,
)
