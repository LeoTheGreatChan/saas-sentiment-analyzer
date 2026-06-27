import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

DATA_PATH = "uber_reviews.csv"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
SAMPLE_SIZE = 200
ALL_VERSIONS = "All Versions"
REQUIRED_COLUMNS = {"content", "reviewCreatedVersion", "at", "thumbsUpCount"}

# 1. Page Configuration
st.set_page_config(page_title="Uber Insights", layout="wide")
st.title("🚗 Uber Product Insights Dashboard")


# 2. AI Model Loading (Cached)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model=MODEL_NAME)


sentiment_pipeline = load_sentiment_model()


def version_sort_key(version):
    """Parse version strings as integer tuples so 4.10 sorts after 4.9."""
    try:
        return tuple(int(part) for part in str(version).split("."))
    except ValueError:
        return (0,)


def normalize_reviews(df):
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.rename(
        columns={
            "content": "Review",
            "reviewCreatedVersion": "Version",
            "at": "Date",
            "thumbsUpCount": "Likes",
        }
    )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Version"] = df["Version"].fillna("Unknown")
    df["Review"] = df["Review"].fillna("")
    df["Likes"] = pd.to_numeric(df["Likes"], errors="coerce").fillna(0).astype(int)
    return df.sort_values("Date", ascending=False).head(SAMPLE_SIZE)


def score_reviews(df):
    def analyze_text(text):
        result = sentiment_pipeline(str(text)[:512])[0]
        score = result["score"] if result["label"] == "POSITIVE" else -result["score"]
        return pd.Series([score, result["label"].capitalize()])

    df = df.copy()
    df[["Score", "Sentiment"]] = df["Review"].apply(analyze_text)
    return df


def get_critical_alerts(df):
    return df[
        (df["Score"] < -0.6)
        | ((df["Sentiment"] == "Negative") & (df["Likes"] > 0))
    ]


# 3. Data Loading & AI Processing
@st.cache_data
def get_processed_data():
    try:
        raw_df = pd.read_csv(DATA_PATH)
        normalized_df = normalize_reviews(raw_df)
        return score_reviews(normalized_df)
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return pd.DataFrame()


with st.spinner(f"AI is analyzing {SAMPLE_SIZE} reviews..."):
    df = get_processed_data()

if not df.empty:
    # 4. Sidebar - version filter with correct semantic version sorting
    st.sidebar.header("Filter Analytics")

    raw_versions = df["Version"].unique().tolist()
    sorted_versions = sorted(raw_versions, key=version_sort_key, reverse=True)
    version_options = [ALL_VERSIONS] + sorted_versions

    selected_version = st.sidebar.selectbox(
        "Select App Version (Latest First)",
        version_options,
    )

    # 5. Filter dataset - used for metrics AND the explorer
    if selected_version != ALL_VERSIONS:
        filtered_df = df[df["Version"] == selected_version]
    else:
        filtered_df = df

    # 6. Top Metrics - all driven by the filtered dataset
    critical_alerts = get_critical_alerts(filtered_df)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Sample", len(filtered_df))
    m2.metric("Avg Sentiment", f"{filtered_df['Score'].mean():.2f}" if not filtered_df.empty else "-")
    m3.metric("Critical Alerts", len(critical_alerts) if not filtered_df.empty else 0)
    m4.metric("Active Version", selected_version if selected_version != ALL_VERSIONS else "Multiple")

    st.divider()

    # 7. Main view selector - charts always use the full dataset for context,
    #    but the critical alerts view respects the version filter
    st.subheader("Explore Review Insights")
    selected_view = st.segmented_control(
        "Choose dashboard view",
        options=["Performance Trends", "Critical Alerts"],
        selection_mode="single",
        default="Performance Trends",
        label_visibility="collapsed",
        width="stretch",
    )
    st.caption("Use the selector above to switch between trend charts and high-priority customer issues.")

    if selected_view == "Performance Trends":

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment by Version")
            v_data = df.groupby("Version")["Score"].mean().reset_index()
            fig_bar = px.bar(
                v_data,
                x="Version",
                y="Score",
                color="Score",
                color_continuous_scale="RdYlGn",
                title="App Health by Release",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.subheader("Time-Based Trend")
            trend_df = df.assign(Day=df["Date"].dt.date)
            trend = trend_df.groupby("Day")["Score"].mean().reset_index()
            if len(trend) > 1:
                fig_trend = px.line(trend, x="Day", y="Score", markers=True)
            else:
                hourly = df.groupby(df["Date"].dt.hour)["Score"].mean().reset_index()
                fig_trend = px.bar(
                    hourly,
                    x="Date",
                    y="Score",
                    labels={"Date": "Hour (24h)"},
                    color="Score",
                )
            st.plotly_chart(fig_trend, use_container_width=True)

    else:
        st.subheader("High-Priority Customer Issues")
        st.info("Showing reviews with very negative sentiment or high community agreement (Likes).")
        st.dataframe(
            critical_alerts.sort_values(by=["Likes", "Score"], ascending=[False, True]),
            use_container_width=True,
        )

    # 8. Detailed Review Explorer
    st.divider()
    st.subheader(f"📋 Review Explorer: {selected_version}")
    st.dataframe(
        filtered_df[["Date", "Version", "Review", "Sentiment", "Score", "Likes"]],
        use_container_width=True,
    )

    # 9. Download - exports the filtered dataset
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Full Dataset", data=csv, file_name="uber_sentiment_export.csv")
