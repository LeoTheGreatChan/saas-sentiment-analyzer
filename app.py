import streamlit as st
import pandas as pd
import altair as alt

# 1. Load the data directly from the local folder directory (Bypasses HTTP/Network requests)
@st.cache_data
def load_data():
    df = pd.read_csv("uber_reviews.csv")
    df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
    return df

df = load_data()

# 2. Sidebar configuration matching your original list order
st.sidebar.header("Filter Analytics")
all_versions = ["All Versions"] + sorted(df['version'].dropna().unique().tolist(), reverse=True)
selected_version = st.sidebar.selectbox("Select App Version (Latest First)", all_versions)

# Create the filtered DataFrame matrix reactively
if selected_version == "All Versions":
    df_filtered = df
else:
    df_filtered = df[df['version'] == selected_version]

# 3. Main Dashboard Header
st.title("🚗 Uber Product Insights Dashboard")
st.markdown("---")

# 4. FIXED DYNAMIC METRICS (Recalculating using df_filtered)
total_sample = len(df_filtered)

if total_sample > 0:
    avg_sentiment = df_filtered['sentiment_score'].mean()
    critical_alerts = len(df_filtered[(df_filtered['priority'] == 'High') & (df_filtered['sentiment_label'] == 'Negative')])
else:
    avg_sentiment = 0.0
    critical_alerts = 0

# Render the metric blocks 
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Sample", value=f"{total_sample:,}")
with col2:
    st.metric(label="Avg Sentiment", value=f"{avg_sentiment:.2f}")
with col3:
    st.metric(label="Critical Alerts", value=critical_alerts)
with col4:
    st.metric(label="Active Version", value="Multiple" if selected_version == "All Versions" else selected_version)

st.markdown("---")

# 5. Charts and Review Table 
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📊 Performance Trends")
    st.write("### Sentiment by Version")
    
    version_chart = alt.Chart(df_filtered).mark_boxplot().encode(
        x=alt.X('sentiment_score:Q', title='Score'),
        y=alt.Y('version:N', title='App Version', sort='-x'),
        color=alt.Color('sentiment_label:N', legend=None)
    ).properties(height=300)
    st.altair_chart(version_chart, use_container_width=True)

with chart_col2:
    st.subheader("⚠️ Critical Alerts")
    st.write("### High-Priority Customer Issues")
    st.caption("Showing reviews with very negative sentiment or high community agreement (Likes).")
    
    explorer_df = df_filtered[df_filtered['priority'] == 'High'].sort_values(by='sentiment_score')
    
    st.write(f"📋 **Review Explorer:** {selected_version}")
    st.dataframe(
        explorer_df[['review_id', 'version', 'sentiment_label', 'sentiment_score', 'likes', 'review']], 
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# 6. Time Chart & Export Layout
bottom_col1, bottom_col2 = st.columns([2, 1])

with bottom_col1:
    st.write("### 🕒 Time-Based Trend")
    time_chart = alt.Chart(df_filtered).mark_circle(size=60).encode(
        x=alt.X('hour:Q', title='Hour (24h)'),
        y=alt.Y('sentiment_score:Q', title='Score'),
        color='sentiment_label:N',
        tooltip=['review_id', 'version', 'sentiment_score']
    ).properties(height=300)
    st.altair_chart(time_chart, use_container_width=True)

with bottom_col2:
    st.write("### 📥 Download Options")
    st.write("") # Spacer padding
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Active Dataset (CSV)",
        data=csv,
        file_name=f"uber_insights_{selected_version.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True
    )
