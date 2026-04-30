import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(page_title="Uber Product Insights", layout="wide")
st.title("🚗 Uber App: Strategic Sentiment Dashboard")
st.markdown("Analyze user feedback trends and drill down into specific app versions.")

# 2. AI Model Loading (Cached)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

# 3. Data Loading & AI Processing
@st.cache_data
def get_processed_data():
    try:
        # Load the Uber Dataset
        df = pd.read_csv("uber_reviews.csv")
        
        # Standardize and Clean
        df = df.rename(columns={'content': 'Review', 'appVersion': 'Version', 'at': 'Date', 'thumbsUpCount': 'Likes'})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Version'] = df['Version'].fillna('Unknown')
        
        # Take the 200 most recent for analysis
        df = df.sort_values('Date', ascending=False).head(200)

        def analyze_text(text):
            # Transformers have a 512 character limit
            result = sentiment_pipeline(str(text)[:512])[0]
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            return pd.Series([score, result['label'].capitalize()])

        df[['Score', 'Sentiment']] = df['Review'].apply(analyze_text)
        return df
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return pd.DataFrame()

with st.spinner("AI is analyzing 200 reviews..."):
    df = get_processed_data()

# 4. Top-Level Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Reviews", len(df))
m2.metric("Avg Sentiment", f"{df['Score'].mean():.2f}")
m3.metric("Negative Count", len(df[df['Sentiment'] == 'Negative']))
m4.metric("High Engagement", len(df[df['Likes'] > 2]))

st.divider()

# 5. Interactive Visualizations
selected_version = None

tab1, tab2 = st.tabs(["📊 Market Health", "⚠️ Critical Alerts"])

with tab1:
    left, right = st.columns(2)
    
    with left:
        st.subheader("Sentiment by Version")
        st.caption("💡 Click a bar to filter the table below")
        v_data = df.groupby('Version')['Score'].mean().reset_index().sort_values('Version')
        
        fig_bar = px.bar(v_data, x='Version', y='Score', color
