# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 17:27:46 2025

@author: User
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📟 Flash Memory Health Anomaly Detection")

uploaded_file = st.file_uploader("Upload flash memory health CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    # Ensure timestamp is in datetime format
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # Select numerical features only for training
    features = ['EraseCycles', 'ProgramFailCount', 'EraseFailCount',
                'Temperature(C)', 'ReadLatency(ms)', 'WriteLatency(ms)', 'RetentionErrors']

    X = df[features].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    st.subheader("🔍 Training Isolation Forest Model")
    contamination = st.slider("Contamination (expected % of outliers)", 0.01, 0.2, 0.05, 0.01)
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    df['AnomalyScore'] = model.fit_predict(X_scaled)
    df['AnomalyValue'] = model.decision_function(X_scaled)
    df['IsAnomaly'] = df['AnomalyScore'] == -1

    st.success(f"Model trained. {df['IsAnomaly'].sum()} anomalies detected out of {len(df)} records.")

    st.subheader("📉 Anomaly Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['AnomalyValue'], bins=50, kde=True, ax=ax)
    threshold = np.percentile(df['AnomalyValue'], 100 * contamination)
    ax.axvline(threshold, color='red', linestyle='--', label='Anomaly Threshold')
    ax.set_title("Anomaly Score Distribution")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📈 Time Series of Key Metrics with Anomalies")
    if 'Timestamp' in df.columns:
        metric_options = ['EraseCycles', 'ProgramFailCount', 'EraseFailCount',
                          'Temperature(C)', 'ReadLatency(ms)', 'WriteLatency(ms)', 'RetentionErrors']
        selected_metric = st.selectbox("Select metric to visualize over time", metric_options)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(df['Timestamp'], df[selected_metric], label=selected_metric, color='blue')
        ax2.scatter(df[df['IsAnomaly']]['Timestamp'], df[df['IsAnomaly']][selected_metric],
                    color='red', label='Anomaly', zorder=5)
        ax2.set_title(f"{selected_metric} Over Time with Anomalies")
        ax2.set_xlabel("Timestamp")
        ax2.set_ylabel(selected_metric)
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.warning("Timestamp column not found or is invalid for time series visualization.")

    st.subheader("🚨 Anomalous Records")
    st.dataframe(df[df['IsAnomaly']].head(20))

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download Results as CSV", data=csv, file_name="anomaly_detection_results.csv", mime="text/csv")

else:
    st.info("Please upload a CSV file to begin.")
