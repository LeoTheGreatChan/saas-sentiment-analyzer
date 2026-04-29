import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob

# 1. Page Config
st.set_page_config(page_title="SaaS Product Insights", layout="wide")
st.title("📊 SaaS Sentiment & Feature Health Dashboard")
st.markdown("Analyzing user feedback to drive the product roadmap.")

# 2. Mock Data (This replaces the Kaggle CSV for now)
data = {
    'Review': [
        "Love the automation features!", "Mobile app crashes often", 
        "Pricing is too high for small teams", "Great UI/UX design",
        "The API documentation is confusing", "Customer support was slow"
    ],
    'Feature': ['Automation', 'Mobile', 'Pricing', 'UI/UX', 'API', 'Support'],
    'Date': pd.to_datetime(['2024-01-01', '2024-01-05', '2024-01-10', '2024-01-15', '2024-01-20', '2024-01-25'])
}
df = pd.DataFrame(data)

# 3. Sentiment Logic
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    # VADER gives a 'compound' score from -1 to 1
    score = analyzer.polarity_scores(text)['compound']
    return score

df['Score'] = df['Review'].apply(get_vader_sentiment)
df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral'))

# 4. Layout: Top Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", len(df))
col2.metric("Avg. Sentiment", f"{df['Score'].mean():.2f}")
col3.metric("Critical Alerts", len(df[df['Score'] < -0.3]))

st.divider()

# 5. Visualizations
left_chart, right_chart = st.columns(2)

with left_chart:
    st.subheader("Sentiment Distribution")
    fig_pie = px.pie(df, names='Sentiment', color='Sentiment', 
                     color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#95a5a6'})
    st.plotly_chart(fig_pie, use_container_width=True)

with right_chart:
    st.subheader("Sentiment by Feature")
    fig_bar = px.bar(df, x='Feature', y='Score', color='Sentiment')
    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Raw Feedback Analysis")
st.dataframe(df[['Date', 'Feature', 'Review', 'Sentiment']], use_container_width=True)
