import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(page_title="Uber Product Insights", layout="wide")
st.title("🚗 Uber App: Strategic Sentiment Dashboard")
st.markdown("Analyzing real-time user feedback to identify version regressions and critical bugs.")

# 2. Optimized AI Model Loading
@st.cache_resource
def load_sentiment_model():
    # Loading the transformer model once and sharing it across all sessions
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

# 3. Data Loading & AI Processing (Increased to 200 Reviews)
@st.cache_data
def get_processed_data():
    try:
        # Load the Uber Dataset
        df = pd.read_csv("uber_reviews.csv")
        
        # Mapping Uber columns
        df = df.rename(columns={'content': 'Review', 'appVersion': 'Version', 'at': 'Date', 'thumbsUpCount': 'Likes'})
        df['Version'] = df['Version'].fillna('Unknown')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Increase Sample Size to 200 for better statistical significance
        df = df.sort_values('Date', ascending=False).head(200)

        # Batch AI Sentiment Analysis
        def analyze_text(text):
            # Transformers have a 512 character limit
            result = sentiment_pipeline(str(text)[:512])[0]
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            return pd.Series([score, result['label'].capitalize()])

        df[['Score', 'Sentiment']] = df['Review'].apply(analyze_text)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Run the heavy lifting
with st.spinner("AI is analyzing 200 reviews... this may take 30 seconds on the first run."):
    df = get_processed_data()

# 4. Top-Level Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Reviews Analyzed", len(df))
col2.metric("Avg Score", f"{df['Score'].mean():.2f}")
col3.metric("Negative Reviews", len(df[df['Sentiment'] == 'Negative']))
col4.metric("Avg Likes/Review", f"{df['Likes'].mean():.1f}")

st.divider()

# 5. Visualizations: The "PM Story"
tab1, tab2 = st.tabs(["📊 Market Health", "⚠️ Critical Alerts"])

with tab1:
    left, right = st.columns(2)
    
    with left:
        st.subheader("Sentiment by Version")
        # Showing how different releases are performing
        version_avg = df.groupby('Version')['Score'].mean().reset_index()
        fig_bar = px.bar(version_avg.sort_values('Version'), x='Version', y='Score', 
                         color='Score', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with right:
        st.subheader("Sentiment Trend Over Time")
        
        # 1. Ensure Date is the Index for resampling
        trend_data = df.copy().set_index('Date')
        
        # 2. Smart Resampling: 
        # If the reviews span multiple days, use 'D' (Daily)
        # If they are all from today, use 'H' (Hourly)
        days_span = (trend_data.index.max() - trend_data.index.min()).days
        freq = 'D' if days_span > 1 else 'H'
        
        # 3. Aggregate and Reset Index for Plotly
        trend_df = trend_data.resample(freq)['Score'].mean().reset_index()
        
        # 4. Handle missing periods (empty hours/days) to keep the line continuous
        trend_df = trend_df.dropna(subset=['Score'])
    
        if not trend_df.empty:
            fig_line = px.line(trend_df, x='Date', y='Score', 
                              title=f"Sentiment Trend ({'Daily' if freq=='D' else 'Hourly'})",
                              markers=True,
                              color_discrete_sequence=['#3498db'])
            # Add a horizontal line at 0 for "Neutral"
            fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Not enough date variety in this sample to show a trend line.")

with tab2:
    st.subheader("High-Priority Customer Pain Points")
    st.markdown("Reviews that are **highly negative** or have gathered **user agreement (Likes)**.")
    
    # NEW FILTER: Show if it's Very Negative OR if it has Likes
    # This ensures the table is rarely empty
    critical_df = df[
        (df['Score'] < -0.5) | 
        ((df['Sentiment'] == 'Negative') & (df['Likes'] > 0))
    ].sort_values(by=['Likes', 'Score'], ascending=[False, True])
    
    if not critical_df.empty:
        st.dataframe(critical_df[['Version', 'Review', 'Likes', 'Score']], use_container_width=True)
    else:
        st.info("No critical alerts found in this sample. Users seem relatively happy with the recent versions!")
        
# 6. Raw Data Search
with st.expander("🔍 Search All Reviews"):
    search = st.text_input("Filter by keyword (e.g., 'price', 'map', 'driver')")
    if search:
        st.dataframe(df[df['Review'].str.contains(search, case=False)])
    else:
        st.dataframe(df)

# 7. Download Data
st.divider()
st.subheader("📥 Export for Product Team")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Analyzed Reviews as CSV",
    data=csv,
    file_name='uber_sentiment_analysis.csv',
    mime='text/csv',
)
