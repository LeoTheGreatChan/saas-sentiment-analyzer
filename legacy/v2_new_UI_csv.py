import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import pipeline

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH        = "uber_reviews.csv"
MODEL_NAME       = "distilbert-base-uncased-finetuned-sst-2-english"
SAMPLE_SIZE      = 200
ALL_VERSIONS     = "All Versions"
REQUIRED_COLUMNS = {"content", "reviewCreatedVersion", "at", "thumbsUpCount"}

SAMPLE_REVIEWS = [
    "The driver was fantastic, arrived in 2 minutes and the car was spotless.",
    "Charged me twice for the same trip and customer support is completely useless.",
    "App keeps crashing when I try to schedule a ride in advance. Very frustrating.",
]

# ── Page config — must be FIRST Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Pulse — Uber Product Insights",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
# Target every Streamlit wrapper that can show white so nothing bleeds through.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Full-app dark base ── */
html, body { background: #0b0d12 !important; }

/* All major Streamlit wrapper elements */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] > .main > .block-container,
section[data-testid="stSidebar"] ~ div,
.stMainBlockContainer,
.main,
.block-container {
    background-color: #0b0d12 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Catch-all for any white-background divs Streamlit injects */
div[data-testid="column"],
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"] {
    background: transparent !important;
}

.main .block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0f1117 !important;
    border-right: 1px solid #1e2433 !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #0f1117 !important;
    padding: 1.5rem 1rem !important;
}

/* ── Radio in sidebar ── */
[data-testid="stSidebar"] .stRadio label {
    color: #64748b !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    color: #64748b !important;
    font-size: 13px !important;
}

/* ── Selectbox ── */
/* Selectbox: label */
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"],
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    visibility: visible !important;
    display: block !important;
    opacity: 1 !important;
}
/* Selectbox: the clickable control box — use a lighter bg so it stands out */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] > div > div > div {
    background: #1e2433 !important;
    border: 1px solid #2d3a52 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
/* Selectbox: selected text and arrow */
[data-testid="stSelectbox"] > div > div span,
[data-testid="stSelectbox"] > div > div svg {
    color: #e2e8f0 !important;
    fill: #94a3b8 !important;
}
/* Selectbox: dropdown popover list */
[data-baseweb="popover"] ul,
[data-baseweb="menu"] {
    background: #1e2433 !important;
    border: 1px solid #2d3a52 !important;
}
[data-baseweb="popover"] li,
[data-baseweb="menu"] li {
    color: #e2e8f0 !important;
    background: transparent !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="menu"] li:hover {
    background: #2d3a52 !important;
}

/* ── Textarea ── */
[data-testid="stTextArea"] textarea {
    background: #0f1117 !important;
    border: 1px solid #1e2433 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
    resize: vertical !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
    outline: none !important;
}
[data-testid="stTextArea"] label { display: none !important; }

/* ── All buttons: base = ghost ── */
.stButton > button {
    background: #1a2030 !important;
    color: #94a3b8 !important;
    border: 1px solid #1e2433 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    padding: 0.35rem 0.9rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.15s, color 0.15s !important;
}
.stButton > button:hover {
    border-color: #3b82f6 !important;
    color: #e2e8f0 !important;
    background: #131720 !important;
}

/* Primary analyse button — identified by its container class we inject */
.analyze-btn-wrap .stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 0.55rem 1.4rem !important;
}
.analyze-btn-wrap .stButton > button:hover {
    opacity: 0.88 !important;
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    border: none !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: #131720 !important;
    border: 1px solid #1e2433 !important;
    color: #94a3b8 !important;
    font-size: 12px !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #3b82f6 !important;
    color: #e2e8f0 !important;
    background: #131720 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e2433 !important;
    gap: 0 !important;
    padding: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #64748b !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.65rem 1.25rem !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #e2e8f0 !important;
    border-bottom: 2px solid #3b82f6 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 1.25rem !important;
}

/* ── Plotly chart background ── */
.js-plotly-plot .plotly .bg { fill: #131720 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }
[data-testid="stDataFrame"] iframe { border-radius: 10px !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3b82f6 !important; }

/* ── Alerts / info boxes ── */
[data-testid="stAlert"] {
    background: #131720 !important;
    border: 1px solid #1e2433 !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
[data-testid="stHeader"] { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* ── Custom component classes ── */
.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 0.25rem 0 1.25rem; border-bottom: 1px solid #1e2433;
    margin-bottom: 1.25rem;
}
.sidebar-brand-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
    border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-size: 17px; flex-shrink: 0;
}
.sidebar-brand-text  { font-size: 15px; font-weight: 600; color: #f1f5f9; letter-spacing: -0.01em; }
.sidebar-brand-sub   { font-size: 10px; color: #475569; letter-spacing: 0.05em; text-transform: uppercase; margin-top: 1px; }
.sidebar-section-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #334155;
    margin-bottom: 0.5rem; padding: 0 2px;
}
.styled-divider { height: 1px; background: #1e2433; margin: 1.5rem 0; }

.page-header { margin-bottom: 1.75rem; padding-bottom: 1.5rem; border-bottom: 1px solid #1e2433; }
.page-title  { font-size: 24px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.03em; margin: 0 0 0.3rem; }
.page-subtitle { font-size: 13px; color: #64748b; line-height: 1.6; }

.metric-card {
    background: #131720; border: 1px solid #1e2433; border-radius: 12px;
    padding: 1.2rem 1.4rem; transition: border-color 0.2s;
}
.metric-card:hover { border-color: #2d3a52; }
.metric-label { font-size: 11px; font-weight: 600; color: #475569; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.55rem; }
.metric-value { font-size: 30px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.03em; line-height: 1; margin-bottom: 0.35rem; }
.metric-sub   { font-size: 11px; color: #334155; }
.metric-value.positive { color: #34d399; }
.metric-value.negative { color: #f87171; }
.metric-value.alert    { color: #fb923c; }
.metric-value.neutral  { color: #f1f5f9; }

.section-heading { font-size: 15px; font-weight: 600; color: #e2e8f0; letter-spacing: -0.01em; margin: 0 0 0.2rem; }
.section-sub     { font-size: 12px; color: #475569; margin-bottom: 0.75rem; }

.result-block {
    background: #0f1117; border: 1px solid #1e2433; border-radius: 10px;
    padding: 1rem 1.25rem; margin-top: 1rem;
}
.result-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 11px; border-radius: 999px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
}
.badge-positive { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
.badge-negative { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }
.result-score-bar-bg   { height: 3px; background: #1e2433; border-radius: 2px; margin-top: 0.7rem; }
.result-score-bar-fill { height: 3px; border-radius: 2px; }

.alert-row {
    background: #0f1117; border: 1px solid #1e2433; border-left: 3px solid #f87171;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin-bottom: 0.55rem;
}
.alert-review-text { font-size: 13px; color: #cbd5e1; line-height: 1.55; margin-bottom: 0.4rem; }
.alert-meta { font-size: 11px; color: #475569; display: flex; gap: 1.2rem; flex-wrap: wrap; }
</style>
""", unsafe_allow_html=True)


# ── Backend (preserved exactly) ───────────────────────────────────────────────
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model=MODEL_NAME)

sentiment_pipeline = load_sentiment_model()


def version_sort_key(version):
    try:
        return tuple(int(p) for p in str(version).split("."))
    except ValueError:
        return (0,)


def normalize_reviews(df):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    df = df.rename(columns={
        "content": "Review", "reviewCreatedVersion": "Version",
        "at": "Date", "thumbsUpCount": "Likes",
    })
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df          = df.dropna(subset=["Date"])
    df["Version"] = df["Version"].fillna("Unknown")
    df["Review"]  = df["Review"].fillna("")
    df["Likes"]   = pd.to_numeric(df["Likes"], errors="coerce").fillna(0).astype(int)
    return df.sort_values("Date", ascending=False).head(SAMPLE_SIZE)


def score_reviews(df):
    def analyze_text(text):
        result = sentiment_pipeline(str(text)[:512])[0]
        score  = result["score"] if result["label"] == "POSITIVE" else -result["score"]
        return pd.Series([score, result["label"].capitalize()])
    df = df.copy()
    df[["Score", "Sentiment"]] = df["Review"].apply(analyze_text)
    return df


def get_critical_alerts(df):
    return df[(df["Score"] < -0.6) | ((df["Sentiment"] == "Negative") & (df["Likes"] > 0))]


@st.cache_data
def get_processed_data():
    try:
        raw = pd.read_csv(DATA_PATH)
        return score_reviews(normalize_reviews(raw))
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return pd.DataFrame()


def score_single(text: str):
    result = sentiment_pipeline(str(text)[:512])[0]
    score  = result["score"] if result["label"] == "POSITIVE" else -result["score"]
    return score, result["label"].capitalize()


# ── Plotly helpers — explicit kwargs, no dict spread ─────────────────────────
PLOT_BG     = "#131720"
GRID_COLOR  = "#1e2433"
TICK_COLOR  = "#475569"
INTER       = "Inter, -apple-system, sans-serif"

def apply_dark_theme(fig, height=300):
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=INTER, color=TICK_COLOR, size=11),
        height=height,
        margin=dict(l=8, r=8, t=16, b=8),
        showlegend=False,
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
        tickfont=dict(size=10, color=TICK_COLOR), zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
        tickfont=dict(size=10, color=TICK_COLOR), zeroline=False,
    )
    return fig


# ── Load data ─────────────────────────────────────────────────────────────────
# Load without spinner wrapper — spinner can defer execution and leave
# version_options empty on first render, causing selectbox to silently hide.
df = get_processed_data()

# Build version list immediately after load, before any widgets render
raw_versions    = df["Version"].unique().tolist() if not df.empty else []
sorted_versions = sorted(raw_versions, key=version_sort_key, reverse=True)
# Always include at least two items so Streamlit never collapses the selectbox
version_options = [ALL_VERSIONS] + sorted_versions

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">⚡</div>
        <div>
            <div class="sidebar-brand-text">Pulse</div>
            <div class="sidebar-brand-sub">Uber Product Insights</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)
    active_nav = st.radio("nav", ["Overview", "Reviews", "Releases"],
                          label_visibility="collapsed", key="nav_radio")

    st.markdown('<div class="styled-divider" style="margin:1.5rem 0 0.75rem"></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#334155;line-height:1.7">'
        'DistilBERT · SST-2 · 200 reviews<br>'
        'saas-sentiment-analyzer.streamlit.app</div>',
        unsafe_allow_html=True,
    )


# ── Guard ─────────────────────────────────────────────────────────────────────
if df.empty:
    st.error("No data — ensure `uber_reviews.csv` is in the working directory.")
    st.stop()


# ── Version filter — main area (always visible, no CSS dependency) ────────────
filter_col, spacer = st.columns([2, 5])
with filter_col:
    selected_version = st.selectbox(
        "Filter by App Version",
        options=version_options,
        index=0,
        key="version_select",
    )

# ── Derived data ──────────────────────────────────────────────────────────────
filtered_df  = df[df["Version"] == selected_version] if selected_version != ALL_VERSIONS else df
critical_alerts = get_critical_alerts(filtered_df)

total      = len(filtered_df)
avg_score  = filtered_df["Score"].mean() if total else 0
alert_count = len(critical_alerts)
active_ver = selected_version if selected_version != ALL_VERSIONS else "Multiple"

score_cls = "positive" if avg_score >= 0.2 else ("negative" if avg_score < 0 else "neutral")
alert_cls = "alert" if alert_count > 0 else "neutral"


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">Product Insights Dashboard</div>
    <div class="page-subtitle">
        Sentiment intelligence across the latest 200 Uber reviews — track release health,
        surface critical issues, and analyse feedback in real time.
    </div>
</div>""", unsafe_allow_html=True)


# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4, gap="small")

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Sample</div>
        <div class="metric-value">{total}</div>
        <div class="metric-sub">Most recent reviews analysed</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Sentiment</div>
        <div class="metric-value {score_cls}">{avg_score:+.2f}</div>
        <div class="metric-sub">Mean score from -1 to +1</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Critical Alerts</div>
        <div class="metric-value {alert_cls}">{alert_count}</div>
        <div class="metric-sub">High-priority negative reviews</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active Version</div>
        <div class="metric-value neutral" style="font-size:18px;padding-top:5px">{active_ver}</div>
        <div class="metric-sub">Current filter selection</div>
    </div>""", unsafe_allow_html=True)


st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)


# ── Insight tabs ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading" style="margin-bottom:0.75rem">Explore Review Insights</div>',
            unsafe_allow_html=True)

tab_trends, tab_alerts, tab_explorer = st.tabs(
    ["Performance Trends", "Critical Alerts", "Review Explorer"]
)

# ── Tab 1 — Performance Trends ────────────────────────────────────────────────
with tab_trends:
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="section-sub">Sentiment by Version · App health by release (mean score)</div>',
                    unsafe_allow_html=True)
        v_data = filtered_df.groupby("Version")["Score"].mean().reset_index().sort_values("Score")
        bar_colors = ["#f87171" if s < 0 else "#34d399" for s in v_data["Score"]]

        fig_bar = go.Figure(go.Bar(
            x=v_data["Version"],
            y=v_data["Score"],
            marker_color=bar_colors,
            marker_line_width=0,
        ))
        apply_dark_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown('<div class="section-sub">Time-Based Trend · Average sentiment by hour of day</div>',
                    unsafe_allow_html=True)
        trend_df = filtered_df.assign(Day=filtered_df["Date"].dt.date)
        trend    = trend_df.groupby("Day")["Score"].mean().reset_index()

        if len(trend) > 1:
            fig_trend = go.Figure(go.Scatter(
                x=trend["Day"], y=trend["Score"],
                mode="lines+markers",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=5, color="#6366f1"),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.08)",
            ))
        else:
            hourly = filtered_df.groupby(filtered_df["Date"].dt.hour)["Score"].mean().reset_index()
            fig_trend = go.Figure(go.Bar(
                x=hourly["Date"], y=hourly["Score"],
                marker_color="#3b82f6", marker_line_width=0,
            ))

        apply_dark_theme(fig_trend)
        st.plotly_chart(fig_trend, use_container_width=True)


# ── Tab 2 — Critical Alerts ───────────────────────────────────────────────────
with tab_alerts:
    if critical_alerts.empty:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem">
            <div style="font-size:28px;margin-bottom:0.75rem">✓</div>
            <div style="font-size:14px;font-weight:600;color:#64748b">No critical alerts for this selection</div>
            <div style="font-size:12px;color:#334155;margin-top:0.35rem">All reviews within acceptable sentiment range</div>
        </div>""", unsafe_allow_html=True)
    else:
        n = len(critical_alerts)
        st.markdown(
            f'<div class="section-sub">{n} review{"s" if n != 1 else ""} with '
            f'very negative sentiment or high community agreement (Likes)</div>',
            unsafe_allow_html=True,
        )
        for _, row in critical_alerts.sort_values(["Likes", "Score"], ascending=[False, True]).iterrows():
            text    = str(row["Review"])
            trunc   = text[:300]
            ellipsis = "…" if len(text) > 300 else ""
            ver      = row.get("Version", "Unknown")
            likes    = int(row.get("Likes", 0))
            date_str = row["Date"].strftime("%b %d, %I:%M %p") if pd.notna(row["Date"]) else ""
            st.markdown(f"""
            <div class="alert-row">
                <div class="alert-review-text">{trunc}{ellipsis}</div>
                <div class="alert-meta">
                    <span>Score:&nbsp;<strong style="color:#f87171">{row['Score']:+.3f}</strong></span>
                    <span>Version:&nbsp;{ver}</span>
                    <span>Likes:&nbsp;{likes}</span>
                    <span>{date_str}</span>
                </div>
            </div>""", unsafe_allow_html=True)


# ── Tab 3 — Review Explorer ───────────────────────────────────────────────────
with tab_explorer:
    hdr_col, dl_col = st.columns([5, 1])
    with hdr_col:
        ver_label = selected_version if selected_version != ALL_VERSIONS else "all versions"
        st.markdown(
            f'<div class="section-sub">{ver_label} · {len(filtered_df)} reviews</div>',
            unsafe_allow_html=True,
        )
    with dl_col:
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export CSV", data=csv_bytes,
            file_name="uber_sentiment_export.csv", mime="text/csv",
        )

    display_df = filtered_df[["Date", "Version", "Review", "Sentiment", "Score", "Likes"]].copy()
    display_df["Date"]  = display_df["Date"].dt.strftime("%b %d, %I:%M %p")
    display_df["Score"] = display_df["Score"].map(lambda x: f"{x:+.3f}")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "Date":      st.column_config.TextColumn("Date",      width=135),
            "Version":   st.column_config.TextColumn("Version",   width=125),
            "Review":    st.column_config.TextColumn("Review",    width="large"),
            "Sentiment": st.column_config.TextColumn("Sentiment", width=95),
            "Score":     st.column_config.TextColumn("Score",     width=75),
            "Likes":     st.column_config.NumberColumn("Likes",   width=65),
        },
    )
