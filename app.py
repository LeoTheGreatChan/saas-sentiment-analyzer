import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Set page config
st.set_page_config(page_title="Uber Insights", layout="wide")

# --- 1. REAL DATA LOADING ---
@st.cache_data
def load_data():
    # 1. Change 'your_original_file.csv' to the exact name of your dataset file
    df = pd.read_csv("your_original_file.csv") 
    
    # 2. Automatically apply the metrics logic to your real data
    # (Ensuring 'sentiment_score' maps correctly to labels)
    if 'sentiment_label' not in df.columns:
        df['sentiment_label'] = df['sentiment_score'].apply(
            lambda x: 'Negative' if x < -0.1 else ('Positive' if x > 0.1 else 'Neutral')
        )
    
    # 3. Dynamically tag 'High' priority based on your real negative scores or high likes
    if 'priority' not in df.columns:
        # Adjust 'likes' to match your exact column name if it's called 'thumbs_up' or 'upvotes'
        likes_col = 'likes' if 'likes' in df.columns else df.columns[0] 
        df['priority'] = np.where((df['sentiment_label'] == 'Negative') | (df[likes_col] > 30), 'High', 'Normal')
        
    return df

df = load_data()

# --- 2. SIDEBAR FILTERS ---
st.sidebar.header("Filter Analytics")
all_versions = ["All Versions"] + sorted(df['version'].unique().tolist())
selected_version = st.sidebar.selectbox("Select App Version (Latest First)", all_versions)

# Apply reactive filtering
if selected_version == "All Versions":
    df_filtered = df
else:
    df_filtered = df[df['version'] == selected_version]

# --- 3. MAIN DASHBOARD HEADER ---
st.title("🚗 Uber Product Insights Dashboard")

# --- 4. DYNAMIC TOP METRICS (FIXED) ---
# Calculations run dynamically based on the 'df_filtered' variable
total_sample = len(df_filtered)

if total_sample > 0:
    avg_sentiment = df_filtered['sentiment_score'].mean()
    # Critical Alerts: items with High Priority AND Negative sentiment
    critical_alerts = len(df_filtered[(df_filtered['priority'] == 'High') & (df_filtered['sentiment_label'] == 'Negative')])
else:
    avg_sentiment = 0.0
    critical_alerts = 0

# Render KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Sample", value=f"{total_sample:,}")
with col2:
    st.metric(label="Avg Sentiment", value=f"{avg_sentiment:.2f}")
with col3:
    st.metric(label="Critical Alerts", value=critical_alerts)
with col4:
    st.metric(label="Active Version", value=selected_version)

st.markdown("---")

# --- 5. CHARTS & TABLES SIDE-BY-SIDE ---
main_col1, main_col2 = st.columns([1, 1.2]) # Gives the table slightly more room

with main_col1:
    st.write("### 📊 Sentiment by Version")
    version_chart = alt.Chart(df_filtered).mark_boxplot().encode(
        x=alt.X('sentiment_score:Q', title='Score'),
        y=alt.Y('version:N', title='App Version', sort='-x'),
        color=alt.Color('sentiment_label:N', legend=None)
    ).properties(height=350)
    st.altair_chart(version_chart, use_container_width=True)

with main_col2:
    st.write("### ⚠️ Critical Alerts (High-Priority)")
    
    # Filter explorer view data dynamically based on selection
    explorer_df = df_filtered[df_filtered['priority'] == 'High'].sort_values(by='sentiment_score')
    
    st.dataframe(
        explorer_df[['review_id', 'version', 'sentiment_label', 'sentiment_score', 'likes', 'review_text']], 
        use_container_width=True,
        hide_index=True,
        height=350 # Matches the chart height perfectly
    )

st.markdown("---")

# --- 6. TIME TREND & DOWNLOAD (Bottom Row) ---
bottom_col1, bottom_col2 = st.columns([2, 1])

with bottom_col1:
    st.write("### 🕒 Time-Based Trend")
    time_chart = alt.Chart(df_filtered).mark_circle(size=60).encode(
        x=alt.X('hour:Q', title='Hour (24h)'),
        y=alt.Y('sentiment_score:Q', title='Score'),
        color='sentiment_label:N',
        tooltip=['review_id', 'version', 'sentiment_score']
    ).properties(height=250)
    st.altair_chart(time_chart, use_container_width=True)

with bottom_col2:
    st.write("### 📥 Export Data")
    st.write("") # Padding
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Active Dataset (CSV)",
        data=csv,
        file_name=f"uber_insights_{selected_version.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True
    )
