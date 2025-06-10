import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Set Streamlit config
st.set_page_config(page_title="Accident Data Dashboard", layout="wide")

# Title
st.title("🚧 Road Accident Data Analysis Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload Accident Dataset (CSV format)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Data Cleaning
    df.dropna(subset=["Location", "Severity"], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Sidebar filters
    st.sidebar.header("Filters")
    year_filter = st.sidebar.multiselect("Select Year", sorted(df['Year'].dropna().unique()), default=df['Year'].dropna().unique())
    severity_filter = st.sidebar.multiselect("Select Severity", df['Severity'].unique(), default=df['Severity'].unique())

    df_filtered = df[(df['Year'].isin(year_filter)) & (df['Severity'].isin(severity_filter))]

    # KPIs
    st.subheader("📊 Key Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Accidents", len(df_filtered))
    col2.metric("Fatal Accidents", df_filtered[df_filtered['Severity'] == "Fatal"].shape[0])
    col3.metric("Most Common Cause", df_filtered['Cause'].mode()[0] if 'Cause' in df.columns else "Not Available")

    # Accident Trend
    st.subheader("📈 Monthly Accident Trend")
    monthly_trend = df_filtered.groupby(['Year', 'Month']).size().reset_index(name='Count')
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=monthly_trend, x='Month', y='Count', hue='Year', marker='o', ax=ax)
    st.pyplot(fig)

    # Severity Chart
    st.subheader("🚨 Accident Severity Distribution")
    fig2, ax2 = plt.subplots()
    df_filtered['Severity'].value_counts().plot(kind='bar', color='tomato', ax=ax2)
    ax2.set_ylabel("Number of Accidents")
    st.pyplot(fig2)

    # Causes of Accidents
    if 'Cause' in df.columns:
        st.subheader("💥 Top Causes of Accidents")
        top_causes = df_filtered['Cause'].value_counts().head(10)
        st.bar_chart(top_causes)

    # High-risk locations
    st.subheader("📍 High-Risk Locations (Top 10)")
    if 'Location' in df.columns:
        risky_locations = df_filtered['Location'].value_counts().head(10)
        st.table(risky_locations)

    # Hotspot Clustering (if lat/lon present)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        st.subheader("🗺️ Accident Hotspots (via Clustering)")
        coords = df_filtered[['Latitude', 'Longitude']].dropna()
        if len(coords) >= 5:
            kmeans = KMeans(n_clusters=5, random_state=0).fit(coords)
            coords['Cluster'] = kmeans.labels_
            fig3, ax3 = plt.subplots()
            sns.scatterplot(data=coords, x='Longitude', y='Latitude', hue='Cluster', palette='Set1', ax=ax3)
            ax3.set_title("KMeans Clusters of Accident Locations")
            st.pyplot(fig3)
        else:
            st.warning("Not enough data points for clustering (need at least 5).")

else:
    st.info("👈 Please upload a CSV file to begin analysis.")

