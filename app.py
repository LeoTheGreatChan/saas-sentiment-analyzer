import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(page_title="Uber Insights", layout="wide")
st.title("🚗 Uber Product Insights Dashboard")
st.markdown("Use the filters below to analyze sentiment across app versions and time.")

# 2. AI Model Loading (Cached)
@st.cache_resource
def load_sentiment_model():
    # Minimalistic approach to avoid memory overhead
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

# 3. Data Loading & AI Processing
@st.cache_data
def get_processed_data():
    try:
        df = pd.read_csv("uber_reviews.csv")
        # Standardize Columns
        df = df.rename(columns={'content': 'Review', 'appVersion': 'Version', 'at': 'Date', 'thumbsUpCount': 'Likes'})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Version'] = df['Version'].fillna('Unknown')
        
        # Analyze 150 reviews (keeping it safe for cloud memory)
        df = df.sort_values('Date', ascending=False).head(150)

        def analyze_text(text):
            result = sentiment_pipeline(str(text)[:512])[0]
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            return pd.Series([score, result['label'].capitalize()])

        df[['Score', 'Sentiment']] = df['Review'].apply(analyze_text)
        return df
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return pd.DataFrame()

with st.spinner("AI is analyzing reviews..."):
    df = get_processed_data()

if not df.empty:
    # 4. Filter Sidebar (The Stable Interactivity)
    st.sidebar.header("Filter Analytics")
    versions = ["All"] + sorted(df['Version'].unique().tolist())
    selected_version = st.sidebar.selectbox("Select App Version to Drill Down", versions)

    # 5. Dynamic Filtering
    if selected_version != "All":
        display_df = df[df['Version'] == selected_version]
    else:
        display_df = df

    # 6. Top Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Reviews Shown", len(display_df))
    m2.metric("Avg Sentiment", f"{display_df['Score'].mean():.2f}")
    m3.metric("Negative Reviews", len(display_df[display_df['Sentiment'] == 'Negative']))

    st.divider()

    # 7. Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment by Version")
        v_data = df.groupby('Version')['Score'].mean().reset_index()
        fig_bar = px.bar(v_data, x='Version', y='Score', color='Score', 
                         color_continuous_scale='RdYlGn', title="Overall Version Health")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("Time-Based Trend")
        display_df['Day'] = display_df['Date'].dt.date
        trend = display_df.groupby('Day')['Score'].mean().reset_index()
        if len(trend) > 1:
            fig_trend = px.line(trend, x='Day', y='Score', markers=True)
        else:
            fig_trend = px.bar(display_df.groupby(display_df['Date'].dt.hour)['Score'].mean().reset_index(), 
                               x='Date', y='Score', labels={'Date': 'Hour (24h)'})
        st.plotly_chart(fig_trend, use_container_width=True)

    # 8. Detailed Review Explorer
    st.subheader(f"📋 Reviews: {selected_version}")
    st.dataframe(display_df[['Date', 'Version', 'Review', 'Sentiment', 'Score', 'Likes']], use_container_width=True)

    # 9. Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Full Analysis", data=csv, file_name='uber_analysis.csv')
