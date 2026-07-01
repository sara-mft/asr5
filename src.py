"""
app.py — STT Benchmark Dashboard
=================================
Run with:
    streamlit run app.py -- --results results/ --config config/models.yaml

Or without a config:
    streamlit run app.py -- --results results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import difflib

from data_loader import (
    LoadResult,
    all_categories,
    enrich_with_metadata,
    load_model_metadata,
    load_results,
    parse_yaml_text,
)
from metrics_meta import (
    ALL_GROUPS,
    DEFAULT_PRIMARY_METRICS,
    DEFAULT_RADAR_METRICS,
    METRICS,
    METRICS_BY_GROUP,
    format_value,
    get,
    metric_direction,
    metric_label,
    normalize_series,
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="STT Benchmark Explorer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "# STT Benchmark Explorer\nInteractive dashboard for Speech-to-Text evaluation results.",
    },
)

# ─── Visual identity ──────────────────────────────────────────────────────────
# Palette: deep navy background feel, electric-cyan accent, slate greys.
# Signature element: the colour-coded rank "medal" badges in leaderboard rows.

# ─── Visual identity ──────────────────────────────────────────────────────────
# Palette: Crisp white background, slate text, and vibrant accents for legibility.

PALETTE = {
    "bg_page":     "#ffffff",   # pure white
    "bg_card":     "#ffffff",   # white cards 
    "bg_card2":    "#f8fafc",   # very light slate (for diffs/tooltips)
    "accent":      "#0284c7",   # deep sky blue
    "accent2":     "#7c3aed",   # vibrant violet
    "accent3":     "#059669",   # emerald (positive)
    "warn":        "#d97706",   # amber (medium)
    "danger":      "#dc2626",   # red (bad)
    "text_main":   "#0f172a",   # dark slate for main text
    "text_muted":  "#64748b",   # medium slate for secondary text
    "border":      "#e2e8f0",   # light grey border
}

# Plotly template shared by all charts
CHART_THEME = "plotly_white"

# Source-stable colour sequence for models (Plotly Vivid extended)
MODEL_COLORS = px.colors.qualitative.Bold + px.colors.qualitative.Vivid

def _model_color_map(models: list[str]) -> dict[str, str]:
    return {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(sorted(models))}


st.markdown(f"""
<style>
/* ── Global ── */
.stApp {{
    background-color: {PALETTE["bg_page"]};
    color: {PALETTE["text_main"]};
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {PALETTE["bg_card"]};
    border-right: 1px solid {PALETTE["border"]};
}}
[data-testid="stSidebar"] .stMarkdown h3 {{
    color: {PALETTE["accent"]};
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 1.5rem;
    margin-bottom: 0.25rem;
}}
[data-testid="stSidebar"] label {{
    color: {PALETTE["text_muted"]};
    font-size: 0.8rem;
}}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: {PALETTE["bg_card"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 10px;
    padding: 1rem 1.2rem;
}}
[data-testid="stMetric"] label {{
    color: {PALETTE["text_muted"]} !important;
    font-size: 0.72rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="stMetricValue"] {{
    color: {PALETTE["text_main"]} !important;
    font-size: 1.6rem !important;
    font-weight: 700;
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.78rem !important;
}}

/* ── Custom card ── */
.card {{
    background: {PALETTE["bg_card"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}}
.card-title {{
    color: {PALETTE["text_muted"]};
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}}
.card-value {{
    color: {PALETTE["text_main"]};
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
}}
.card-sub {{
    color: {PALETTE["text_muted"]};
    font-size: 0.78rem;
    margin-top: 0.25rem;
}}

/* ── Hero header ── */
.hero {{
    padding: 1.8rem 0 1.2rem 0;
    border-bottom: 1px solid {PALETTE["border"]};
    margin-bottom: 1.5rem;
}}
.hero h1 {{
    font-size: 2rem;
    font-weight: 800;
    color: {PALETTE["text_main"]};
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0;
}}
.hero .tagline {{
    color: {PALETTE["text_muted"]};
    font-size: 0.9rem;
    margin-top: 0.4rem;
}}
.accent-dot {{
    color: {PALETTE["accent"]};
}}

/* ── Badges ── */
/* ── Badges ── */
.badge {{
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0.1rem 0.15rem;
}}
.badge-cyan  {{ background: rgba(2, 132, 199, 0.1); color: {PALETTE["accent"]}; border: 1px solid rgba(2, 132, 199, 0.2); }}
.badge-violet{{ background: rgba(124, 58, 237, 0.1); color: {PALETTE["accent2"]}; border: 1px solid rgba(124, 58, 237, 0.2); }}
.badge-green {{ background: rgba(5, 150, 105, 0.1); color: {PALETTE["accent3"]}; border: 1px solid rgba(5, 150, 105, 0.2); }}
.badge-amber {{ background: rgba(217, 119, 6, 0.1); color: {PALETTE["warn"]}; border: 1px solid rgba(217, 119, 6, 0.2); }}
.badge-red   {{ background: rgba(220, 38, 38, 0.1); color: {PALETTE["danger"]}; border: 1px solid rgba(220, 38, 38, 0.2); }}
.badge-grey  {{ background: rgba(100, 116, 139, 0.1); color: {PALETTE["text_muted"]}; border: 1px solid rgba(100, 116, 139, 0.2); }}

/* ── Medal badges ── */
.medal-gold   {{ font-size: 1.1rem; }}
.medal-silver {{ font-size: 1.1rem; }}
.medal-bronze {{ font-size: 1.1rem; }}

/* ── Table tweaks ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {PALETTE["border"]};
    border-radius: 10px;
    overflow: hidden;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    border: 1px solid {PALETTE["border"]};
    border-radius: 10px;
    background: {PALETTE["bg_card"]};
}}
[data-testid="stExpanderDetails"] {{
    background: {PALETTE["bg_card"]};
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    gap: 0.5rem;
    border-bottom: 1px solid {PALETTE["border"]};
    padding-bottom: 0;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {PALETTE["text_muted"]};
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    padding: 0.5rem 1rem;
    text-transform: uppercase;
}}
.stTabs [aria-selected="true"] {{
    color: {PALETTE["accent"]} !important;
    border-bottom: 2px solid {PALETTE["accent"]} !important;
    background: transparent !important;
}}

/* ── Section divider ── */
.section-label {{
    color: {PALETTE["text_muted"]};
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.4rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.section-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {PALETTE["border"]};
}}

/* ── Transcript diff ── */
.diff-container {{
    background: {PALETTE["bg_card2"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 0.82rem;
    line-height: 1.7;
}}
.w-ins  {{ background: rgba(52,211,153,0.25); border-radius: 3px; padding: 0 2px; color: {PALETTE["accent3"]}; }}
.w-del  {{ background: rgba(248,113,113,0.25); border-radius: 3px; padding: 0 2px; color: {PALETTE["danger"]}; text-decoration: line-through; }}
.w-sub  {{ background: rgba(251,191,36,0.20); border-radius: 3px; padding: 0 2px; color: {PALETTE["warn"]}; }}

/* ── Info tooltip wrapper ── */
.tooltip-box {{
    background: {PALETTE["bg_card2"]};
    border-left: 3px solid {PALETTE["accent"]};
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    font-size: 0.8rem;
    color: {PALETTE["text_muted"]};
    margin-top: 0.5rem;
}}
</style>
""", unsafe_allow_html=True)


# ─── Argument parsing (allow `streamlit run app.py -- --results ...`) ─────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results", default="sample_results", help="Path to results root directory")
    parser.add_argument("--config", default=None, help="Path to model metadata YAML (optional)")
    # Streamlit passes its own flags before '--'; ignore them
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    return parser.parse_known_args(argv)[0]


_args = _parse_args()
RESULTS_ROOT = Path(_args.results)
CONFIG_YAML  = Path(_args.config) if _args.config else None


# ─── Data loading (cached) ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load(results_root: str) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    res: LoadResult = load_results(results_root)
    return res.summary_df, res.detail_df, res.issues


@st.cache_data(show_spinner=False)
def _load_meta_from_file(yaml_path: str | None):
    return load_model_metadata(yaml_path)


def _get_meta(yaml_text: str | None = None, yaml_path: str | None = None):
    if yaml_text:
        return parse_yaml_text(yaml_text)
    return _load_meta_from_file(yaml_path)


# ─── Shared chart helpers ─────────────────────────────────────────────────────

def _chart_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        template=CHART_THEME,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(size=13, color=PALETTE["text_muted"]), x=0),
        margin=dict(l=0, r=0, t=36 if title else 10, b=0),
        height=height,
        font=dict(family="Inter, Segoe UI, system-ui", color=PALETTE["text_main"]),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
        yaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
    )
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str | list[str],
              color_map: dict | None = None, barmode: str = "group",
              title: str = "", height: int = 400, x_label: str = "",
              y_label: str = "", sort_asc: bool | None = None,
              text_on_bars: bool = True) -> go.Figure:
    """Generic grouped/stacked bar."""
    if isinstance(y, str):
        y = [y]
    if sort_asc is not None:
        df = df.sort_values(y[0], ascending=sort_asc)
    fig = go.Figure()
    for col in y:
        colors = [color_map.get(v, MODEL_COLORS[0]) for v in df[x]] if color_map else None
        fig.add_trace(go.Bar(
            name=metric_label(col) if col in {m.key for m in METRICS} else col,
            x=df[x],
            y=df[col],
            marker_color=colors,
            text=[format_value(col, v) for v in df[col]] if text_on_bars else None,
            textposition="outside",
            textfont=dict(size=10),
        ))
    fig.update_layout(barmode=barmode, xaxis_title=x_label, yaxis_title=y_label)
    _chart_layout(fig, title=title, height=height)
    return fig


def scatter_chart(df: pd.DataFrame, x: str, y: str, color: str,
                  size: str | None = None, color_map: dict | None = None,
                  title: str = "", height: int = 440,
                  x_label: str = "", y_label: str = "") -> go.Figure:
    """Scatter / bubble chart."""
    fig = go.Figure()
    for label in df[color].unique():
        sub = df[df[color] == label]
        c = color_map.get(label) if color_map else None
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub[y],
            mode="markers+text",
            name=label,
            text=sub["display_name"] if "display_name" in sub.columns else sub[color],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                color=c,
                size=sub[size] * 60 + 8 if size else 12,
                opacity=0.85,
                line=dict(width=1, color=PALETTE["border"]),
            ),
        ))
    _chart_layout(fig, title=title, height=height)
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
    return fig

def radar_chart(df_sub: pd.DataFrame, df_global: pd.DataFrame, models: list[str],
                metrics: list[str], color_map: dict, title: str = "",
                height: int = 500) -> go.Figure:
    """Spider/radar chart with global normalization and raw value tooltips."""
    fig = go.Figure()
    theta = [metric_label(m) for m in metrics]
    theta_closed = theta + [theta[0]]
    
    for model in models:
        if model not in df_sub.index:
            continue
            
        values = []
        raw_texts = []
        for m in metrics:
            # Anchor min/max to the GLOBAL dataset, not just the subset
            global_vals = pd.to_numeric(df_global[m], errors="coerce")
            mn, mx = global_vals.min(), global_vals.max()
            
            raw = df_sub.loc[model, m]
            
            # Save the formatted raw string for the hover tooltip
            raw_texts.append(f"{metric_label(m)}: {format_value(m, raw)}")
            
            if pd.isna(raw) or pd.isna(mn) or mn == mx:
                values.append(0.0)
            elif metric_direction(m) == "lower":
                values.append(float(1 - (raw - mn) / (mx - mn)))
            else:
                values.append(float((raw - mn) / (mx - mn)))
                
        r_closed = values + [values[0]]
        text_closed = raw_texts + [raw_texts[0]]
        c = color_map.get(model, MODEL_COLORS[0])
        
        fig.add_trace(go.Scatterpolar(
            r=r_closed, theta=theta_closed, fill="toself",
            name=model,
            line=dict(color=c, width=2),
            fillcolor=c.replace("rgb", "rgba").replace(")", ", 0.12)") if c.startswith("rgb") else c,
            text=text_closed,
            # Custom tooltip showing exactly what the user wants to see
            hovertemplate="<b>%{name}</b><br>%{text}<br>Relative Score: %{r:.2f}<extra></extra>"
        ))
        
    _chart_layout(fig, title=title, height=height)
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                # Explicitly label the rings so users know what the center means
                tickvals=[0, 0.5, 1],
                ticktext=["Worst (Dataset)", "Average", "Best (Dataset)"],
                tickfont=dict(size=9, color=PALETTE["text_muted"]),
                gridcolor=PALETTE["border"],
                linecolor=PALETTE["border"],
            ),
            angularaxis=dict(
                gridcolor=PALETTE["border"],
                linecolor=PALETTE["border"],
                tickfont=dict(size=10, color=PALETTE["text_main"]),
            ),
        ),
    )
    return fig


def heatmap_chart(df_pivot: pd.DataFrame, title: str = "",
                  height: int = 400, higher_is_better: dict[str, bool] | None = None) -> go.Figure:
    """Metric × model normalised heatmap (green = better)."""
    if higher_is_better is None:
        higher_is_better = {}
    norm = pd.DataFrame(index=df_pivot.index, columns=df_pivot.columns, dtype=float)
    for col in df_pivot.columns:
        s = pd.to_numeric(df_pivot[col], errors="coerce")
        mn, mx = s.min(), s.max()
        if mn == mx or pd.isna(mn):
            norm[col] = 0.5
        elif higher_is_better.get(col, metric_direction(col) == "higher"):
            norm[col] = (s - mn) / (mx - mn)
        else:
            norm[col] = 1 - (s - mn) / (mx - mn)
    text_df = df_pivot.copy()
    for col in df_pivot.columns:
        text_df[col] = df_pivot[col].apply(lambda v: format_value(col, v))
    fig = go.Figure(data=go.Heatmap(
        z=norm.values.astype(float),
        x=[metric_label(c) for c in norm.columns],
        y=norm.index.tolist(),
        text=text_df.values,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale=[[0, "#f87171"], [0.5, "#fbbf24"], [1, "#34d399"]],
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            title="Score",
            tickvals=[0, 0.5, 1],
            ticktext=["Worst", "Mid", "Best"],
            tickfont=dict(size=9, color=PALETTE["text_muted"]),
            len=0.7,
            thickness=10,
        ),
    ))
    _chart_layout(fig, title=title, height=max(height, 60 * len(norm.index) + 80))
    fig.update_layout(
        xaxis=dict(tickfont=dict(size=10), side="top"),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
    )
    return fig


# ─── UI utility helpers ───────────────────────────────────────────────────────

def badge(label: str, style: str = "grey") -> str:
    return f'<span class="badge badge-{style}">{label}</span>'


def badges_html(labels: list[str]) -> str:
    styles = ["cyan", "violet", "green", "amber", "grey", "red"]
    out = ""
    for i, lbl in enumerate(labels):
        out += badge(lbl, styles[i % len(styles)])
    return out


def medal(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")


def direction_arrow(metric_key: str) -> str:
    return "↓ lower=better" if metric_direction(metric_key) == "lower" else "↑ higher=better"


def metric_tooltip(key: str) -> str:
    m = get(key)
    if m is None:
        return ""
    return f'<div class="tooltip-box">ℹ️ <b>{m.label}</b> — {m.description}</div>'


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)

def _agg_summary(df: pd.DataFrame, metric_keys: list[str], agg_fn: str | dict = "mean") -> pd.DataFrame:
    """
    Aggregate a summary DataFrame across datasets, grouped by model identity.
    """
    work = df.copy()
    work["_cats_key"] = work["categories"].apply(
        lambda x: tuple(x) if isinstance(x, list) else (str(x),) if x else ()
    )
    group_cols = ["model_key", "display_name", "source", "_cats_key"]
    
    num_keys = [k for k in metric_keys if k in work.columns
                and pd.api.types.is_numeric_dtype(work[k])]
    
    # NEW: Handle dictionary mapping for per-metric aggregation (min vs max)
    if isinstance(agg_fn, dict):
        # Filter the dict to only include valid numeric columns
        valid_agg_dict = {k: v for k, v in agg_fn.items() if k in num_keys}
        agg_df = work.groupby(group_cols, dropna=False).agg(valid_agg_dict).reset_index()
    else:
        agg_df = work.groupby(group_cols, dropna=False)[num_keys].agg(agg_fn).reset_index()
        
    agg_df["categories"] = agg_df["_cats_key"].apply(list)
    agg_df = agg_df.drop(columns=["_cats_key"])
    return agg_df

def _rank_df(summary_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Add per-metric ranks and a composite normalised score column."""
    df = summary_df.copy()
    score_cols = []
    for m in metrics:
        if m not in df.columns:
            continue
        normed = normalize_series(m, df[m])
        col = f"_norm_{m}"
        df[col] = normed
        score_cols.append(col)
    if score_cols:
        df["composite_score"] = df[score_cols].mean(axis=1)
    else:
        df["composite_score"] = float("nan")
    return df


# ─── Main sidebar (filters) ───────────────────────────────────────────────────

def render_sidebar(summary_df: pd.DataFrame, detail_df: pd.DataFrame, meta) -> dict:
    """Render sidebar controls; return a dict of selected filter values."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Results directory")
    st.sidebar.code(str(RESULTS_ROOT), language=None)

    # YAML upload
    st.sidebar.markdown("### 🗂 Model config (YAML)")
    uploaded_yaml = st.sidebar.file_uploader(
        "Upload models.yaml (optional)",
        type=["yaml", "yml"],
        help="Provides display names, categories, provider info.",
        label_visibility="collapsed",
    )
    yaml_text = None
    if uploaded_yaml is not None:
        yaml_text = uploaded_yaml.read().decode("utf-8")
        st.sidebar.success("✅ Config loaded")

    st.sidebar.markdown("### 🗃 Filters")

    datasets = sorted(summary_df["dataset"].unique())
    sources  = sorted(summary_df["source"].unique())

    sel_datasets = st.sidebar.multiselect(
        "Datasets", datasets, default=datasets,
        help="Include only data from the selected datasets.",
    )
    sel_sources = st.sidebar.multiselect(
        "Sources / Providers", sources, default=sources,
        help="Filter by STT provider / API source.",
    )

    # Enrich once for category list
    meta_inst = _get_meta(yaml_text, str(CONFIG_YAML) if CONFIG_YAML else None)
    enriched = enrich_with_metadata(summary_df, meta_inst)
    cats = all_categories(enriched)
    sel_categories: list[str] = []
    if cats:
        st.sidebar.markdown("### 🏷 Categories")
        sel_categories = st.sidebar.multiselect(
            "Model categories", cats, default=[],
            help="Leave empty to show all. Select one or more to filter.",
        )

    # Metric groups shown in the main views
    st.sidebar.markdown("### 📊 Metric groups")
    sel_groups = st.sidebar.multiselect(
        "Show groups", ALL_GROUPS, default=ALL_GROUPS,
        help="Controls which metric groups appear in tables and charts.",
    )

    return dict(
        sel_datasets=sel_datasets,
        sel_sources=sel_sources,
        sel_categories=sel_categories,
        sel_groups=sel_groups,
        yaml_text=yaml_text,
        meta=meta_inst,
    )


def apply_filters(summary_df: pd.DataFrame, detail_df: pd.DataFrame,
                  filters: dict, meta) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply sidebar filters and enrich both DataFrames with YAML metadata."""
    sel_datasets   = filters["sel_datasets"]
    sel_sources    = filters["sel_sources"]
    sel_categories = filters["sel_categories"]

    s_df = summary_df.copy()
    d_df = detail_df.copy()

    # 1. Apply Dataset & Source filters unconditionally.
    # If the user clears the multiselect, it becomes an empty list [], 
    # correctly resulting in an empty dataframe.
    s_df = s_df[s_df["dataset"].isin(sel_datasets)]
    d_df = d_df[d_df["dataset"].isin(sel_datasets)]
    
    s_df = s_df[s_df["source"].isin(sel_sources)]
    d_df = d_df[d_df["source"].isin(sel_sources)]

    # 2. Enrich with YAML metadata
    s_df = enrich_with_metadata(s_df, meta)
    d_df = enrich_with_metadata(d_df, meta)

    # 3. Category filter (this one remains conditional, because an empty 
    # selection here explicitly means "show all categories")
    if sel_categories:
        mask = s_df["categories"].apply(
            lambda cats: bool(set(cats or []) & set(sel_categories))
        )
        s_df = s_df[mask]
        keep_models = set(s_df["model_key"])
        d_df = d_df[d_df["model_key"].isin(keep_models)]

    return s_df, d_df

# ─── TAB 1: Overview ──────────────────────────────────────────────────────────

def tab_overview(s_df: pd.DataFrame, d_df: pd.DataFrame, filters: dict) -> None:
    if s_df.empty:
        st.warning("No data matches the current filters.")
        return

    n_models   = s_df["model_key"].nunique()
    n_datasets = s_df["dataset"].nunique()
    n_files    = int(s_df["num_files"].sum()) if "num_files" in s_df.columns else "—"
    
    # Safely calculate Best WER only if the column exists and has data
    best_wer, best_model_name = None, ""
    if "wer" in s_df.columns and not s_df["wer"].isna().all():
        best_idx = s_df["wer"].idxmin()
        best_wer = s_df.loc[best_idx, "wer"]
        best_model_name = s_df.loc[best_idx, "display_name"]

    # ── Hero KPIs ──
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Models evaluated", n_models)
    with c2:
        st.metric("Datasets", n_datasets)
    with c3:
        st.metric("Audio files", n_files)
    with c4:
        lbl = format_value("wer", best_wer) if best_wer is not None else "—"
        st.metric(
            "Best overall WER",
            lbl,
            delta=best_model_name if best_model_name else None,
            delta_color="off",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Model cards ──
    section_label("Model overview")
    sel_groups = filters.get("sel_groups", ALL_GROUPS)
    
    # ✅ Force core metrics to always be included so the cards never break
    base_keys = [m.key for g in sel_groups for m in METRICS_BY_GROUP.get(g, [])]
    metric_keys = list(set(base_keys + ["wer", "latency_sec", "cost_usd"]))
    metric_keys = [k for k in metric_keys if k in s_df.columns]

    # Aggregate across datasets (mean)
    agg_df = _agg_summary(s_df, metric_keys, agg_fn="mean")

    cols = st.columns(min(n_models, 3) if n_models > 0 else 1)

    for i, row in agg_df.iterrows():
        with cols[i % len(cols)]:
            cats_html = badges_html(row["categories"] or [])
            wer_v = format_value("wer", row.get("wer"))
            lat_v = format_value("latency_sec", row.get("latency_sec"))
            cost_v = format_value("cost_usd", row.get("cost_usd"))
            src_badge = badge(row["source"], "cyan")
            st.markdown(f"""
            <div class="card">
              <div class="card-title">{src_badge} {cats_html}</div>
              <div class="card-value">{row["display_name"]}</div>
              <div class="card-sub" style="margin-top:.6rem">
                <b style="color:{PALETTE['text_main']}">WER</b> {wer_v} &nbsp;
                <b style="color:{PALETTE['text_main']}">Latency</b> {lat_v}s &nbsp;
                <b style="color:{PALETTE['text_main']}">Cost</b> {cost_v}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── WER × Latency scatter ──
    if "wer" in agg_df.columns and "latency_sec" in agg_df.columns:
        section_label("Accuracy vs. Speed (Lower left is better)")
        
        plot_df = agg_df.dropna(subset=["wer", "latency_sec"]).copy()
        if not plot_df.empty:
            # Handle cost safely for bubble size
            has_cost = "cost_usd" in plot_df.columns and not plot_df["cost_usd"].isna().all()
            if has_cost:
                # Ensure we don't pass negative or zero sizes to Plotly
                plot_df["_bubble_size"] = plot_df["cost_usd"].fillna(0).clip(lower=0.0001)
            
            fig = px.scatter(
                plot_df, x="latency_sec", y="wer", color="source",
                size="_bubble_size" if has_cost else None,
                hover_name="display_name",
                text="display_name",
                labels={
                    "latency_sec": "Avg. Latency (seconds)", 
                    "wer": "Avg. WER (Word Error Rate)",
                    "source": "Source"
                },
                size_max=30 if has_cost else 10
            )
            
            fig.update_traces(
                textposition="top center",
                textfont=dict(size=10, color=PALETTE["text_muted"]),
                marker=dict(opacity=0.8, line=dict(width=1, color=PALETTE["bg_page"]))
            )
            
            fig = _chart_layout(fig, height=440)
            
            # Add ideal-corner annotation
            fig.add_annotation(
                x=plot_df["latency_sec"].min() * 0.9,
                y=plot_df["wer"].min() * 0.9,
                text="↙ Ideal Corner (Fast & Accurate)",
                showarrow=False,
                font=dict(size=10, color=PALETTE["accent3"]), # Green text
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Per-dataset WER bar ──
    if s_df["dataset"].nunique() > 1 and "wer" in s_df.columns:
        section_label("WER by model × dataset")
        pivot = s_df.pivot_table(index="display_name", columns="dataset", values="wer", aggfunc="mean")
        
        fig2 = go.Figure()
        global_color_map = _model_color_map(pivot.index.tolist())
        
        for dataset_col in pivot.columns:
            fig2.add_trace(go.Bar(
                name=dataset_col, x=pivot.index, y=pivot[dataset_col],
                text=[format_value("wer", v) for v in pivot[dataset_col]],
                textposition="outside",
                textfont=dict(size=10),
            ))
            
        fig2.update_layout(barmode="group")
        fig2 = _chart_layout(fig2, height=380)
        st.plotly_chart(fig2, use_container_width=True)

# ─── TAB 2: Leaderboard ───────────────────────────────────────────────────────

def tab_leaderboard(s_df: pd.DataFrame, d_df: pd.DataFrame, filters: dict) -> None:
    if s_df.empty:
        st.warning("No data matches the current filters.")
        return

    sel_groups  = filters.get("sel_groups", ALL_GROUPS)
    all_m_keys  = [m.key for g in sel_groups for m in METRICS_BY_GROUP.get(g, [])
                   if m.key in s_df.columns]
    avail_metrics = [k for k in all_m_keys if s_df[k].notna().any()]

    col_l, col_r = st.columns([3, 1])
    with col_r:
        section_label("Options")
        ranking_metrics = st.multiselect(
            "Ranking metrics",
            avail_metrics,
            default=[k for k in DEFAULT_PRIMARY_METRICS if k in avail_metrics],
            format_func=metric_label,
        )
        aggregate = st.selectbox(
            "Aggregate across datasets by",
            ["mean", "median", "best-run", "worst-run"], # ✅ Added "worst-run"
        )
        top_n = st.slider("Top N models", min_value=3, max_value=max(3, s_df["model_key"].nunique()), value=min(10, s_df["model_key"].nunique()))

    with col_l:
        if not ranking_metrics:
            st.info("Select at least one ranking metric on the right.")
            return

        # ✅ Added inverted logic for worst-run
        if aggregate == "best-run":
            agg_fn = {k: ("min" if metric_direction(k) == "lower" else "max") for k in avail_metrics}
        elif aggregate == "worst-run":
            agg_fn = {k: ("max" if metric_direction(k) == "lower" else "min") for k in avail_metrics}
        else:
            agg_fn = aggregate  # "mean" or "median"

        # Pass the string or the dictionary into the aggregator
        agg_df = _agg_summary(s_df, avail_metrics, agg_fn=agg_fn)




        ranked = _rank_df(agg_df, ranking_metrics).sort_values("composite_score", ascending=False).head(top_n).reset_index(drop=True)

        section_label("Leaderboard")
        
        display_df = ranked[["display_name", "source", "categories", "composite_score"] + ranking_metrics].copy()
        display_df.insert(0, "Rank", [medal(i+1) for i in range(len(display_df))])
        display_df["categories"] = display_df["categories"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        display_df["composite_score"] = display_df["composite_score"] * 100

        rename_map = {
            "display_name": "Model",
            "source": "Source",
            "categories": "Categories",
            "composite_score": "Composite Score (%)"
        }
        for m in ranking_metrics:
            dir_arrow = "↓" if metric_direction(m) == "lower" else "↑"
            rename_map[m] = f"{metric_label(m)} {dir_arrow}"
            
        display_df = display_df.rename(columns=rename_map)

        def highlight_podium(s, direction):
            ranks = s.rank(method='min', ascending=(direction == 'lower'))
            styles = []
            for r in ranks:
                if pd.isna(r):
                    styles.append('')
                elif r == 1:
                    styles.append(f'background-color: rgba(217, 119, 6, 0.15); font-weight: bold; color: {PALETTE["warn"]}')
                elif r == 2:
                    styles.append(f'background-color: rgba(100, 116, 139, 0.15); font-weight: bold; color: {PALETTE["text_muted"]}')
                elif r == 3:
                    styles.append(f'background-color: rgba(140, 99, 56, 0.15); font-weight: bold; color: #8c6338')
                else:
                    styles.append('')
            return styles

        styler = display_df.style
        styler = styler.apply(highlight_podium, direction="higher", subset=["Composite Score (%)"])
        
        for m in ranking_metrics:
            col_name = rename_map[m]
            styler = styler.apply(highlight_podium, direction=metric_direction(m), subset=[col_name])
            
        format_dict = {"Composite Score (%)": "{:.1f}%"}
        for m in ranking_metrics:
            fmt = get(m).format if get(m) else ".4f"
            format_dict[rename_map[m]] = f"{{:{fmt}}}"
            
        styler = styler.format(format_dict, na_rep="—")

        st.dataframe(styler, use_container_width=True, hide_index=True)

    # Generate color map based on ALL models, so it stays consistent between both charts
    global_color_map = _model_color_map(agg_df["display_name"].tolist())

    section_label("Composite score (normalised, higher = better)")
    fig = bar_chart(
        ranked, x="display_name", y="composite_score",
        color_map={row["display_name"]: global_color_map[row["display_name"]] for _, row in ranked.iterrows()},
        sort_asc=False, height=380, y_label="Composite score (0–1)",
    )
    st.plotly_chart(fig, use_container_width=True)

    section_label("Per-metric breakdown")
    sel_m = st.selectbox(
        "Metric",
        avail_metrics,
        format_func=metric_label,
        key="leaderboard_metric_select",
    )
    meta_obj = get(sel_m)
    if meta_obj:
        st.markdown(metric_tooltip(sel_m), unsafe_allow_html=True)
        
    plot_df = agg_df.dropna(subset=[sel_m]).sort_values(sel_m, ascending=(metric_direction(sel_m) == "lower"))

    fig2 = bar_chart(
        plot_df,
        x="display_name", y=sel_m,
        color_map={row["display_name"]: global_color_map[row["display_name"]] for _, row in plot_df.iterrows()},
        height=380,
        y_label=f"{metric_label(sel_m)} ({direction_arrow(sel_m)})",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ─── TAB 3: Compare ───────────────────────────────────────────────────────────

def tab_compare(s_df: pd.DataFrame, d_df: pd.DataFrame, filters: dict) -> None:
    if s_df.empty:
        st.warning("No data matches the current filters.")
        return

    all_models   = sorted(s_df["display_name"].unique())
    sel_groups   = filters.get("sel_groups", ALL_GROUPS)
    all_m_keys   = [m.key for g in sel_groups for m in METRICS_BY_GROUP.get(g, [])
                    if m.key in s_df.columns]
    avail_metrics = [k for k in all_m_keys if s_df[k].notna().any()]

    st.markdown("#### Select models to compare")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        sel_models = st.multiselect(
            "Models", all_models, default=all_models[:min(4, len(all_models))],
            help="Pick 2–8 models for a side-by-side comparison.",
            label_visibility="collapsed",
        )
    with col_b:
        sel_dataset = st.selectbox(
            "Dataset",
            ["All datasets (avg)"] + sorted(s_df["dataset"].unique()),
            key="compare_dataset",
        )

    if not sel_models:
        st.info("Select at least two models to compare.")
        return

    # Subset and set the global context bounds
    sub = s_df[s_df["display_name"].isin(sel_models)]
    context_df = s_df.copy() # NEW: Keep track of the background bounds
    
    if sel_dataset != "All datasets (avg)":
        sub = sub[sub["dataset"] == sel_dataset]
        context_df = context_df[context_df["dataset"] == sel_dataset] # ✅ Filter background bounds too
        
    sub_agg = _agg_summary(sub, avail_metrics, agg_fn="mean")
    sub_agg = sub_agg[sub_agg["display_name"].isin(sel_models)]
    color_map = _model_color_map(sel_models)

# ── Radar ──
    section_label("Spider chart (Outer edge = Best in dataset)")
    radar_cols = st.multiselect(
        "Radar metrics", avail_metrics,
        default=[k for k in DEFAULT_RADAR_METRICS if k in avail_metrics],
        format_func=metric_label, key="compare_radar",
    )
    
    if len(radar_cols) >= 3 and len(sel_models) >= 2:
        pivot = sub_agg.set_index("display_name")
        # Create a global aggregate to act as the true min/max anchor
        global_agg = _agg_summary(context_df, avail_metrics, agg_fn="mean").set_index("display_name")        
        fig_r = radar_chart(
            df_sub=pivot, 
            df_global=global_agg, # Pass the global dataset here
            models=sel_models, 
            metrics=radar_cols,
            color_map={m: color_map[m] for m in sel_models},
            title="", height=520,
        )
        st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.info("Select ≥ 3 radar metrics and ≥ 2 models to render the spider chart.")

    # ── Heatmap ──
    section_label("Metric heatmap (green = better within column)")
    heat_keys = st.multiselect(
        "Heatmap metrics", avail_metrics,
        default=avail_metrics[:min(8, len(avail_metrics))],
        format_func=metric_label, key="compare_heat",
    )
    if heat_keys and not sub_agg.empty:
        pivot_h = sub_agg.set_index("display_name")[heat_keys]
        fig_h = heatmap_chart(pivot_h, height=380,
                               higher_is_better={k: metric_direction(k) == "higher" for k in heat_keys})
        st.plotly_chart(fig_h, use_container_width=True)



# ─── TAB 4: Drill-down ────────────────────────────────────────────────────────
        
def compute_word_diff(ref_text: str, hyp_text: str) -> tuple[str, str]:
    """Compare two strings word-by-word and return HTML formatted with diff spans."""
    ref_words = ref_text.split()
    hyp_words = hyp_text.split()
    
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    ref_html, hyp_html = [], []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        r_chunk = " ".join(ref_words[i1:i2])
        h_chunk = " ".join(hyp_words[j1:j2])
        
        if tag == 'equal':
            if r_chunk: ref_html.append(r_chunk)
            if h_chunk: hyp_html.append(h_chunk)
        elif tag == 'delete':
            # Word is in ref, but missing in hyp (Deletions - Red strikethrough)
            if r_chunk: ref_html.append(f'<span class="w-del">{r_chunk}</span>')
        elif tag == 'insert':
            # Word is in hyp, but missing in ref (Insertions - Green)
            if h_chunk: hyp_html.append(f'<span class="w-ins">{h_chunk}</span>')
        elif tag == 'replace':
            # Substituted words (Amber)
            if r_chunk: ref_html.append(f'<span class="w-sub">{r_chunk}</span>')
            if h_chunk: hyp_html.append(f'<span class="w-sub">{h_chunk}</span>')
            
    return " ".join(ref_html), " ".join(hyp_html)

def tab_drilldown(s_df: pd.DataFrame, d_df: pd.DataFrame, filters: dict) -> None:
    if s_df.empty or d_df.empty:
        st.warning("No data matches the current filters.")
        return

    all_models = sorted(s_df["display_name"].unique())

    col_a, col_b = st.columns(2)
    with col_a:
        sel_model = st.selectbox("Model", all_models, key="drill_model")
    with col_b:
        sel_dataset = st.selectbox(
            "Dataset",
            sorted(s_df["dataset"].unique()),
            key="drill_dataset",
        )

    model_row = s_df[(s_df["display_name"] == sel_model) & (s_df["dataset"] == sel_dataset)]
    detail_rows = d_df[(d_df["display_name"] == sel_model) & (d_df["dataset"] == sel_dataset)]

    if model_row.empty:
        st.warning("No summary data for this combination.")
        return

    row = model_row.iloc[0]
    meta_cats_html = badges_html(row.get("categories") or [])
    src_badge = badge(row["source"], "cyan")
    prov_badge = badge(row.get("provider", row["source"]), "violet")

    # ── Model identity card ──
    st.markdown(f"""
    <div class="card">
      <div class="card-title">{src_badge} {prov_badge} {meta_cats_html}</div>
      <div class="card-value">{row["display_name"]}</div>
      <div class="card-sub">engine_id: <code>{row["engine_id"]}</code> &nbsp;·&nbsp; dataset: <b>{sel_dataset}</b></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary metrics ──
    sel_groups = filters.get("sel_groups", ALL_GROUPS)
    all_m_keys = [m.key for g in sel_groups for m in METRICS_BY_GROUP.get(g, [])
                  if m.key in s_df.columns and pd.notna(row.get(m.key))]

    for g in sel_groups:
        group_keys = [m.key for m in METRICS_BY_GROUP.get(g, []) if m.key in all_m_keys]
        if not group_keys:
            continue
        section_label(g)
        metric_cols = st.columns(min(len(group_keys), 4))
        for ci, mk in enumerate(group_keys):
            v = row.get(mk)
            if pd.isna(v) if isinstance(v, float) else v is None:
                continue
            # Compute relative rank
            all_vals = pd.to_numeric(s_df[s_df["dataset"] == sel_dataset][mk], errors="coerce").dropna()
            if len(all_vals) > 1:
                n_worse = int((all_vals > v).sum()) if metric_direction(mk) == "lower" else int((all_vals < v).sum())
                rank_txt = f"#{len(all_vals) - n_worse} of {len(all_vals)}"
            else:
                rank_txt = ""
            with metric_cols[ci % 4]:
                st.metric(
                    label=metric_label(mk),
                    value=format_value(mk, v),
                    delta=rank_txt,
                    delta_color="off",
                    help=get(mk).description if get(mk) else "",
                )

# ── Per-file breakdown ──
    if not detail_rows.empty:
        section_label("Correlation: Metric vs. Audio Duration")
        audio_files = sorted(detail_rows["audio_file"].unique())

        show_keys = [k for k in all_m_keys if detail_rows[k].notna().any()]

        if show_keys and len(audio_files) > 1:
            plot_df = detail_rows.copy()

            y_metric = st.selectbox(
                "Metric (Y-Axis)", show_keys, format_func=metric_label, key="drill_scatter_y"
            )

            if "audio_duration_sec" not in plot_df.columns or plot_df["audio_duration_sec"].isna().all():
                st.warning("Audio duration data is missing for this dataset. Scatter plot cannot be rendered.")
            else:
                # Use Plotly Express for a clean scatter plot
                fig = px.scatter(
                    plot_df, x="audio_duration_sec", y=y_metric,
                    hover_name="audio_file",
                    labels={
                        y_metric: f"{metric_label(y_metric)} ({direction_arrow(y_metric)})", 
                        "audio_duration_sec": "Audio Duration (seconds)"
                    },
                )
                
                # Apply our custom theme and marker styling
                fig.update_traces(
                    marker=dict(
                        size=10, color=PALETTE["accent"], opacity=0.85, 
                        line=dict(width=1, color=PALETTE["bg_page"])
                    )
                )
                fig = _chart_layout(fig, height=380)
                st.plotly_chart(fig, use_container_width=True)

        # ── Transcript viewer ──
        section_label("Transcript comparison")
        sel_file = st.selectbox("Audio file", audio_files, key="drill_audio_file")
        file_row = detail_rows[detail_rows["audio_file"] == sel_file].iloc[0]

        ref_raw  = str(file_row.get("reference", "") or "")
        hyp_raw  = str(file_row.get("hypothesis", "") or "")
        diar = str(file_row.get("diarization_transcript", "") or "")

        # ✅ Calculate the visual diff
        ref_html, hyp_html = compute_word_diff(ref_raw, hyp_raw)

        col_ref, col_hyp = st.columns(2)
        with col_ref:
            st.markdown("**Reference (Ground Truth)**")
            st.markdown(f'<div class="diff-container">{ref_html}</div>', unsafe_allow_html=True)
        with col_hyp:
            st.markdown("**Hypothesis (Model Output)**")
            st.markdown(f'<div class="diff-container">{hyp_html}</div>', unsafe_allow_html=True)

        # ── Error counts ──
        ins = file_row.get("insertions")
        dels = file_row.get("deletions")
        subs = file_row.get("substitutions")
        if any(v is not None for v in [ins, dels, subs]):
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                st.metric("Insertions",  int(ins)  if ins  is not None else "—")
            with ec2:
                st.metric("Deletions",   int(dels) if dels is not None else "—")
            with ec3:
                st.metric("Substitutions", int(subs) if subs is not None else "—")


# ─── TAB 5: Metrics guide ────────────────────────────────────────────────────

def tab_metrics_guide() -> None:
    st.markdown("## Metrics reference guide")
    st.markdown(
        "Every metric used in this dashboard is documented below — "
        "what it measures, how it's calculated, and how to interpret it."
    )
    for g in ALL_GROUPS:
        ms = METRICS_BY_GROUP.get(g, [])
        if not ms:
            continue
        section_label(g)
        for m in ms:
            dir_txt = "↓ **lower is better**" if m.direction == "lower" else "↑ **higher is better**"
            with st.expander(f"{m.label}  —  `{m.key}`  {dir_txt}"):
                st.markdown(m.description)
                kv_cols = st.columns(4)
                kv_cols[0].markdown(f"**Unit** `{m.unit or '—'}`")
                kv_cols[1].markdown(f"**Group** {m.group}")
                kv_cols[2].markdown(f"**Direction** {m.direction}")
                kv_cols[3].markdown(f"**Format** `{m.format}`")




# ─── Main entry point ─────────────────────────────────────────────────────────

def main() -> None:
    # ── Load data ──
    with st.spinner("Loading benchmark results…"):
        raw_s_df, raw_d_df, issues = _load(str(RESULTS_ROOT))

    # Show load errors unobtrusively
    if issues:
        with st.expander(f"⚠️ {len(issues)} loading issue(s)"):
            for iss in issues:
                st.warning(f"`{iss.file_path}` — {iss.message}")

    if raw_s_df.empty:
        st.error(
            f"No results found in **{RESULTS_ROOT}**. "
            "Check that the directory exists and contains `<dataset>/<source>_<dataset>.json` files."
        )
        st.stop()

    # ── Sidebar ──
    filters = render_sidebar(raw_s_df, raw_d_df, None)
    meta    = filters["meta"]
    s_df, d_df = apply_filters(raw_s_df, raw_d_df, filters, meta)

    # ── Header ──
    st.markdown(
        '<div class="hero">'
        '<h1>🎙️ STT Benchmark<span class="accent-dot"> Explorer</span></h1>'
        '<p class="tagline">Interactive benchmark dashboard for Speech-to-Text evaluation — '
        f'{raw_s_df["dataset"].nunique()} dataset(s) · '
        f'{raw_s_df["model_key"].nunique()} model(s) · '
        f'{raw_s_df["source"].nunique()} source(s)</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Tabs ──
    t1, t2, t3, t4, t5= st.tabs([
        "📊 Overview",
        "🏆 Leaderboard",
        "⚖️ Compare",
        "🔍 Drill-down",
        "📖 Metrics guide"
    ])

    with t1:
        tab_overview(s_df, d_df, filters)
    with t2:
        tab_leaderboard(s_df, d_df, filters)
    with t3:
        tab_compare(s_df, d_df, filters)
    with t4:
        tab_drilldown(s_df, d_df, filters)
    with t5:
        tab_metrics_guide()



if __name__ == "__main__":
    main()
