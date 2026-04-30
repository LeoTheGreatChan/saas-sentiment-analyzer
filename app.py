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

# 3. Data Loading & AI Processing
@st.cache_data
def get_processed_data():
    try:
        # Load the Uber Dataset
        df = pd.read_csv("uber_reviews.csv")
        
        # Standardize Columns
        df = df.rename(columns={
            'content': 'Review', 
            'appVersion': 'Version', 
            'at': 'Date', 
            'thumbsUpCount': 'Likes'
        })
        
        # FORCE DATE CONVERSION: Essential for the Trend Line
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']) # Remove unreadable dates
        
        # Fill missing versions and take top 200
        df['Version'] = df['Version'].fillna('Unknown')
        df = df.sort_values('Date', ascending=False).head(200)

        # Batch AI Sentiment Analysis
        def analyze_text(text):
            # Limit to 512 chars for Transformer stability
            result = sentiment_pipeline(str(text)[:512])[0]
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            return pd.Series([score, result['label'].capitalize()])

        df[['Score', 'Sentiment']] = df['Review'].apply(analyze_text)
        return df
    except Exception as e:
        st.error(f"Critical Data Error: {e}")
        return pd.DataFrame()

# Run the processing
with st.spinner("AI is analyzing 200 reviews... this may take 30 seconds on the first run."):
    df = get_processed_data()

if not df.empty:
    # 4. Top-Level Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reviews Analyzed", len(df))
    col2.metric("Avg Score", f"{df['Score'].mean():.2f}")
    col3.metric("Negative Reviews", len(df[df['Sentiment'] == 'Negative']))
    col4.metric("Avg Likes/Review", f"{df['Likes'].mean():.1f}")

    st.divider()

    # 5. Visualizations
    tab1, tab2 = st.tabs(["📊 Market Health", "⚠️ Critical Alerts"])

    with tab1:
        left, right = st.columns(2)
        
        with left:
            st.subheader("Sentiment by Version")
            version_avg = df.groupby('Version')['Score'].mean().reset_index()
            fig_bar = px.bar(version_avg.sort_values('Version'), x='Version', y='Score', 
                             color='Score', color_continuous_scale='RdYlGn',
                             range_color=[-1, 1])
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with right:
            st.subheader("Sentiment Trend")
            # Group by day for stability (avoiding resample errors)
            df['Day'] = df['Date'].dt.date
            trend_df = df.groupby('Day')['Score'].mean().reset_index()

            if len(trend_df) > 1:
                fig_line = px.line(trend_df, x='Day', y='Score', markers=True)
                fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                    # If all reviews are from 1 day, show a Bar Chart by Hour
                    st.info("The 200 most recent reviews are from the same day. Showing hourly breakdown:")
                    
                    # Create an 'Hour' column for better labeling
                    df['Hour'] = df['Date'].dt.hour
                    
                    # Group data to ensure the chart is clean
                    hourly_df = df.groupby('Hour')['Score'].mean().reset_index()
                    
                    fig_hour = px.bar(
                        hourly_df, 
                        x='Hour', 
                        y='Score',
                        labels={
                            'Hour': 'Time of Day (24h Format)', 
                            'Score': 'Average Sentiment Score'
                        },
                        title="Hourly Sentiment Snapshot",
                        color='Score',
                        color_continuous_scale='RdYlGn'
                    )
                    
                    # Force the X-axis to show every hour clearly
                    fig_hour.update_layout(xaxis_tickmode='linear', xaxis_dtick=1)
                    
                    st.plotly_chart(fig_hour, use_container_width=True)
    with tab2:
        st.subheader("High-Priority Customer Pain Points")
        # Inclusion filter: Score below -0.5 OR any Negative review with likes
        critical_df = df[
            (df['Score'] < -0.5) | 
            ((df['Sentiment'] == 'Negative') & (df['Likes'] > 0))
        ].sort_values(by=['Likes', 'Score'], ascending=[False, True])
        
        if not critical_df.empty:
            st.dataframe(critical_df[['Version', 'Review', 'Likes', 'Score']], use_container_width=True)
        else:
            st.info("No critical alerts found. The current sample is relatively healthy!")

    # 6. Raw Data Search & Export
    with st.expander("🔍 Search & Export Raw Data"):
        search = st.text_input("Filter by keyword (e.g., 'price', 'map', 'driver')")
        filtered_df = df[df['Review'].str.contains(search, case=False)] if search else df
        st.dataframe(filtered_df)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Full Analysis as CSV", data=csv, file_name='uber_analysis.csv')

else:
    st.warning("Please ensure 'uber_reviews.csv' is uploaded to your repository.")
