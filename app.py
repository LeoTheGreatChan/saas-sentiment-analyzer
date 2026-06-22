import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(page_title="Uber Insights", layout="wide")
st.title("🚗 Uber Product Insights Dashboard")

# 2. AI Model Loading (Cached)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

# 3. Data Loading & AI Processing
@st.cache_data
def get_processed_data():
    try:
        df = pd.read_csv("uber_reviews.csv")
        df = df.rename(columns={'content': 'Review', 'appVersion': 'Version', 'at': 'Date', 'thumbsUpCount': 'Likes'})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Version'] = df['Version'].fillna('Unknown')

        # Increased to 200 reviews for better "Critical Alert" depth
        df = df.sort_values('Date', ascending=False).head(200)

        def analyze_text(text):
            result = sentiment_pipeline(str(text)[:512])[0]
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            return pd.Series([score, result['label'].capitalize()])

        df[['Score', 'Sentiment']] = df['Review'].apply(analyze_text)
        return df
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return pd.DataFrame()

with st.spinner("AI is analyzing 200 reviews..."):
    df = get_processed_data()

if not df.empty:
    # 4. Sidebar — version filter with correct semantic version sorting
    st.sidebar.header("Filter Analytics")

    def version_sort_key(v):
        """Parse version strings as integer tuples so 4.10 sorts after 4.9."""
        try:
            return tuple(int(x) for x in str(v).split('.'))
        except ValueError:
            return (0,)

    raw_versions = df['Version'].unique().tolist()
    sorted_versions = sorted(raw_versions, key=version_sort_key, reverse=True)

    version_options = ["All Versions"] + sorted_versions

    selected_version = st.sidebar.selectbox(
        "Select App Version (Latest First)",
        version_options
    )

    # 5. Filter dataset — used for metrics AND the explorer
    if selected_version != "All Versions":
        filtered_df = df[df['Version'] == selected_version]
    else:
        filtered_df = df

    # 6. Top Metrics — all driven by the filtered dataset
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Sample", len(filtered_df))
    m2.metric("Avg Sentiment", f"{filtered_df['Score'].mean():.2f}" if not filtered_df.empty else "—")
    m3.metric(
        "Critical Alerts",
        len(filtered_df[(filtered_df['Score'] < -0.6) | (filtered_df['Likes'] > 5)])
        if not filtered_df.empty else 0
    )
    m4.metric("Active Version", selected_version if selected_version != "All Versions" else "Multiple")

    st.divider()

    # 7. Layout with Tabs — charts always use the full dataset for context,
    #    but the critical alerts tab respects the version filter
    tab1, tab2 = st.tabs(["📊 Performance Trends", "⚠️ Critical Alerts"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment by Version")
            v_data = df.groupby('Version')['Score'].mean().reset_index()
            fig_bar = px.bar(v_data, x='Version', y='Score', color='Score',
                             color_continuous_scale='RdYlGn', title="App Health by Release")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.subheader("Time-Based Trend")
            df['Day'] = df['Date'].dt.date
            trend = df.groupby('Day')['Score'].mean().reset_index()
            if len(trend) > 1:
                fig_trend = px.line(trend, x='Day', y='Score', markers=True)
            else:
                hourly = df.groupby(df['Date'].dt.hour)['Score'].mean().reset_index()
                fig_trend = px.bar(hourly, x='Date', y='Score', labels={'Date': 'Hour (24h)'}, color='Score')
            st.plotly_chart(fig_trend, use_container_width=True)

    with tab2:
        st.subheader("High-Priority Customer Issues")
        st.info("Showing reviews with very negative sentiment or high community agreement (Likes).")
        crit_df = filtered_df[
            (filtered_df['Score'] < -0.6) |
            ((filtered_df['Sentiment'] == 'Negative') & (filtered_df['Likes'] > 0))
        ]
        st.dataframe(
            crit_df.sort_values(by=['Likes', 'Score'], ascending=[False, True]),
            use_container_width=True
        )

    # 8. Detailed Review Explorer
    st.divider()
    st.subheader(f"📋 Review Explorer: {selected_version}")
    st.dataframe(
        filtered_df[['Date', 'Version', 'Review', 'Sentiment', 'Score', 'Likes']],
        use_container_width=True
    )

    # 9. Download — exports the filtered dataset
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Full Dataset", data=csv, file_name='uber_sentiment_export.csv')
