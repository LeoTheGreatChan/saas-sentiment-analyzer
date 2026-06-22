import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Set page config
st.set_page_config(page_title="Uber Insights", layout="wide")

# --- 1. MOCK DATA GENERATION (To match your live dashboard structure) ---
@st.cache_data
def load_data():
    np.random.seed(42)
    versions = ["4.451.10003", "4.526.10000", "4.527.10000", "4.528.10000", 
                "4.532.10001", "4.537.10000", "4.541.10003", "4.547.10001", 
                "4.550.10001", "4.552.10000", "4.554.10001", "4.555.10003", 
                "4.556.10005", "Unknown"]
    
    data = {
        "review_id": [f"REV_{i}" for i in range(1, 201)],
        "version": np.random.choice(versions, 200),
        "sentiment_score": np.random.uniform(-1.0, 1.0, 200),
        "hour": np.random.randint(0, 24, 200),
        "likes": np.random.randint(0, 50, 200),
        "review_text": [f"Sample review text details for context tracking ID {i}..." for i in range(1, 201)]
    }
    df = pd.DataFrame(data)
    
    # Assign labels based on score matching the live setup
    df['sentiment_label'] = df['sentiment_score'].apply(lambda x: 'Negative' if x < -0.1 else ('Positive' if x > 0.1 else 'Neutral'))
    df['priority'] = np.where((df['sentiment_label'] == 'Negative') | (df['likes'] > 30), 'High', 'Normal')
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

# --- 5. CHARTS & VISUALIZATIONS ---
st.subheader("📊 Performance Trends")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.write("### Sentiment by Version")
    version_chart = alt.Chart(df_filtered).mark_box().encode(
        x=alt.X('sentiment_score:Q', title='Score'),
        y=alt.Y('version:N', title='App Version', sort='-x'),
        color=alt.Color('sentiment_label:N', legend=None)
    ).properties(height=300)
    st.altair_chart(version_chart, use_container_width=True)

with chart_col2:
    st.write("### Time-Based Trend")
    time_chart = alt.Chart(df_filtered).mark_circle(size=60).encode(
        x=alt.X('hour:Q', title='Hour (24h)'),
        y=alt.Y('sentiment_score:Q', title='Score'),
        color='sentiment_label:N',
        tooltip=['review_id', 'version', 'sentiment_score']
    ).properties(height=300)
    st.altair_chart(time_chart, use_container_width=True)

st.markdown("---")

# --- 6. REVIEW EXPLORER TABLE ---
st.subheader("⚠️ Critical Alerts")
st.write("### High-Priority Customer Issues")
st.caption("Showing reviews with very negative sentiment or high community agreement (Likes).")

# Filter explorer view data based on top-level interactive selections
explorer_df = df_filtered[df_filtered['priority'] == 'High'].sort_values
