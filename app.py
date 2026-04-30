import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# 1. Page Config
st.set_page_config(page_title="SaaS Product Insights", layout="wide")
st.title("📊 SaaS Sentiment & Feature Health Dashboard")
st.markdown("Using Deep Learning (Transformers) to analyze user feedback automatically.")

# 2. Load the Automatic AI Model
# We use a 'cache' so the model only downloads once
@st.cache_resource
def load_model():
    # This model is specifically trained for sentiment analysis
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_model()

def get_automatic_sentiment(text):
    # The AI looks at the whole sentence context
    result = sentiment_pipeline(text)[0]
    label = result['label'] # 'POSITIVE' or 'NEGATIVE'
    score = result['score'] # Confidence percentage
    
    # Standardize score to a -1 to 1 scale for our charts
    final_score = score if label == 'POSITIVE' else -score
    return pd.Series([final_score, label.capitalize()])

# 3. Data Loading (Uber CSV)
@st.cache_data
def load_data():
    # Make sure your file on GitHub is named exactly 'uber_reviews.csv'
    try:
        df = pd.read_csv("uber_reviews.csv")
        
        # Mapping Uber columns to our dashboard logic
        # 'content' = Review text
        # 'appVersion' = The software version (our "Feature" tracker)
        # 'at' = The date/time
        df = df.rename(columns={
            'content': 'Review', 
            'appVersion': 'Feature', 
            'at': 'Date'
        })

        # Data Cleaning
        df['Feature'] = df['Feature'].fillna('General/Unknown')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # We take the most recent 50 reviews to keep the AI processing fast
        return df.sort_values('Date', ascending=False).head(50)
    
    except FileNotFoundError:
        st.error("CSV file not found. Please upload 'uber_reviews.csv' to your GitHub repo.")
        return pd.DataFrame()

df = load_data()
# Apply the automatic AI logic
df[['Score', 'Sentiment']] = df['Review'].apply(get_automatic_sentiment)

# 4. Layout: Top Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", len(df))
col2.metric("Avg. Sentiment Confidence", f"{df['Score'].abs().mean():.2f}")
col3.metric("Negative Issues", len(df[df['Sentiment'] == 'Negative']))

st.divider()

# 5. Visualizations
left_chart, right_chart = st.columns(2)

with left_chart:
    st.subheader("Sentiment Distribution")
    fig_pie = px.pie(df, names='Sentiment', color='Sentiment', 
                     color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c'})
    st.plotly_chart(fig_pie, use_container_width=True)

with right_chart:
    st.subheader("SaaS Health Score by Feature")
    # Sort so the worst problems are at the top
    fig_bar = px.bar(df.sort_values('Score'), x='Feature', y='Score', color='Sentiment',
                     color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c'})
    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Raw Feedback: Deep Learning Analysis")
st.dataframe(df[['Date', 'Feature', 'Review', 'Sentiment', 'Score']], use_container_width=True)

st.divider()
st.subheader("🚀 Live Sentiment Tester")
user_input = st.text_input("Type a review to test the AI's logic (e.g., 'The support was surprisingly helpful'):")

if user_input:
    # Run the same automatic AI logic on the user's text
    score, sentiment = get_automatic_sentiment(user_input)
    
    # Display the result in a cool way
    if sentiment == 'Positive':
        st.success(f"The AI thinks this is **{sentiment}** (Confidence: {score:.2f})")
    else:
        st.error(f"The AI thinks this is **{sentiment}** (Confidence: {score:.2f})")
