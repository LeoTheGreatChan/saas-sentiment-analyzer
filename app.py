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
    # 4. Sidebar Drill-Down with Smart Sorting
    if not df.empty:
        st.sidebar.header("Filter Analytics")
        
        # Get unique versions and sort them in descending order (latest first)
        # This handles strings like '4.34' vs '4.4' correctly
        raw_versions = df['Version'].unique().tolist()
        sorted_versions = sorted(raw_versions, key=lambda x: str(x), reverse=True)
        
        version_options = ["All Versions"] + sorted_versions
        
        selected_version = st.sidebar.selectbox(
            "Select App Version (Latest First)", 
            version_options
        )
    
        # 5. Dynamic Filtering for the Explorer
        if selected_version != "All Versions":
            explorer_df = df[df['Version'] == selected_version]
        else:
            explorer_df = df

    # 6. Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Sample", len(df))
    m2.metric("Avg Sentiment", f"{df['Score'].mean():.2f}")
    m3.metric("Critical Alerts", len(df[(df['Score'] < -0.6) | (df['Likes'] > 5)]))
    m4.metric("Active Version", selected_version if selected_version != "All Versions" else "Multiple")

    st.divider()

    # 7. Layout with Tabs
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
        # Logic: Score below -0.6 (Very Angry) OR any negative review with Likes
        crit_df = df[(df['Score'] < -0.6) | ((df['Sentiment'] == 'Negative') & (df['Likes'] > 0))]
        st.dataframe(crit_df.sort_values(by=['Likes', 'Score'], ascending=[False, True]), use_container_width=True)

    # 8. Detailed Review Explorer
    st.divider()
    st.subheader(f"📋 Review Explorer: {selected_version}")
    st.dataframe(explorer_df[['Date', 'Version', 'Review', 'Sentiment', 'Score', 'Likes']], use_container_width=True)

    # 9. Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Full Dataset", data=csv, file_name='uber_sentiment_export.csv')
