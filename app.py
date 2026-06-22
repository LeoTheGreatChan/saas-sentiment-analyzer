import streamlit as st
import pandas as pd
import altair as alt

# 1. Load the data exactly as you had it originally
URL = "https://raw.githubusercontent.com/paimonscook/saas-sentiment-analyzer/main/uber_reviews.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(URL)
    # Ensure correct data types
    df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
    return df

df = load_data()

# 2. Sidebar Filter Configuration
st.sidebar.header("Filter Analytics")
all_versions = ["All Versions"] + sorted(df['version'].dropna().unique().tolist(), reverse=True)
selected_version = st.sidebar.selectbox("Select App Version (Latest First)", all_versions)

# Create the reactive filtered dataframe
if selected_version == "All Versions":
    df_filtered = df
else:
    df_filtered = df[df['version'] == selected_version]

# 3. Main Header
st.title("🚗 Uber Product Insights Dashboard")
st.markdown("---")

# 4. DYNAMIC TOP METRICS (Fixed to use df_filtered instead of global df)
total_sample = len(df_filtered)

if total_sample > 0:
    avg_sentiment = df_filtered['sentiment_score'].mean()
    # Count critical alerts using the filtered dataframe context
    critical_alerts = len(df_filtered[(df_filtered['priority'] == 'High') & (df_filtered['sentiment_label'] == 'Negative')])
else:
    avg_sentiment = 0.0
    critical_alerts = 0

# Render the KPI metric cards dynamically
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Sample", value=f"{total_sample:,}")
with col2:
    st.metric(label="Avg Sentiment", value=f"{avg_sentiment:.2f}")
with col3:
    st.metric(label="Critical Alerts", value=critical_alerts)

st.markdown("---")

# 5. Visualizations Layout (Using your exact original columns and df_filtered)
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Performance Trends")
    st.write("### Sentiment by Version")
    
    version_chart = alt.Chart(df_filtered).mark_boxplot().encode(
        x=alt.X('sentiment_score:Q', title='Score'),
        y=alt.Y('version:N', title='App Version', sort='-x'),
        color=alt.Color('sentiment_label:N', legend=None)
    ).properties(height=300)
    
    st.altair_chart(version_chart, use_container_width=True)

with col_right:
    st.subheader("⚠️ Critical Alerts")
    st.write("### High-Priority Customer Issues")
    st.caption("Showing reviews with very negative sentiment or high community agreement.")
    
    # Filter the explorer table based on your original 'review' column
    explorer_df = df_filtered[df_filtered['priority'] == 'High'].sort_values(by='sentiment_score')
    
    st.dataframe(
        explorer_df[['review_id', 'version', 'sentiment_label', 'sentiment_score', 'likes', 'review']], 
        use_container_width=True,
        hide_index=True,
        height=300
    )

st.markdown("---")

# 6. Bottom Row: Time Trend & Download Options
col_bottom_left, col_bottom_right = st.columns([2, 1])

with col_bottom_left:
    st.write("### 🕒 Time-Based Trend")
    time_chart = alt.Chart(df_filtered).mark_circle(size=60).encode(
        x=alt.X('hour:Q', title='Hour (24h)'),
        y=alt.Y('sentiment_score:Q', title='Score'),
        color='sentiment_label:N',
        tooltip=['review_id', 'version', 'sentiment_score']
    ).properties(height=250)
    
    st.altair_chart(time_chart, use_container_width=True)

with col_bottom_right:
    st.write("### 📥 Export Data")
    st.write("") # Layout spacer
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Active Dataset (CSV)",
        data=csv,
        file_name=f"uber_insights_{selected_version.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True
    )
