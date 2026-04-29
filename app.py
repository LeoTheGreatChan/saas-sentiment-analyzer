import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 1. Page Config
st.set_page_config(page_title="SaaS Product Insights", layout="wide")
st.title("📊 SaaS Sentiment & Feature Health Dashboard")
st.markdown("Analyzing user feedback to drive the product roadmap.")

# 2. Setup Sentiment Engine with PM Tuning
analyzer = SentimentIntensityAnalyzer()

# Custom Lexicon: We manually tell the AI that these words are important for SaaS
'''new_words = {
    'crashes': -4.0,
    'slow': -2.0,
    'confusing': -2.0,
    'high': -1.5,
    'automation': 2.0
}
analyzer.lexicon.update(new_words)
'''
def categorize_feedback(text):
    score = analyzer.polarity_scores(text)['compound']
    # PM Thresholds:
    # Anything > 0.3 is clearly happy
    # Anything < 0.0 is a pain point (including "slow" or "crashes")
    # Anything in between is Neutral
    if score >= 0.3:
        return score, 'Positive'
    elif score < 0.0:
        return score, 'Negative'
    else:
        return score, 'Neutral'

# 3. Data Loading (Mock Data for now)
data = {
    'Review': [
        "Love the automation features!", 
        "Mobile app crashes often", 
        "Pricing is too high for small teams", 
        "Great UI/UX design",
        "The API documentation is confusing", 
        "Customer support was slow"
    ],
    'Feature': ['Automation', 'Mobile', 'Pricing', 'UI/UX', 'API', 'Support'],
    'Date': pd.to_datetime(['2024-01-01', '2024-01-05', '2024-01-10', '2024-01-15', '2024-01-20', '2024-01-25'])
}
df = pd.DataFrame(data)

# Apply the tuned sentiment logic
df[['Score', 'Sentiment']] = df['Review'].apply(lambda x: pd.Series(categorize_feedback(x)))

# 4. Layout: Top Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", len(df))
col2.metric("Avg. Sentiment Score", f"{df['Score'].mean():.2f}")
col3.metric("Critical Alerts", len(df[df['Sentiment'] == 'Negative']))

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
    # Sorting by score so the worst features jump out
    fig_bar = px.bar(df.sort_values('Score'), x='Feature', y='Score', color='Sentiment',
                     color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#95a5a6'})
    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Raw Feedback Analysis")
st.dataframe(df[['Date', 'Feature', 'Review', 'Sentiment', 'Score']], use_container_width=True)
