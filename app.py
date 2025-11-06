# Save as app.py and run with:
#    streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.neural_network import MLPRegressor
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="ZTA Anomaly Dashboard", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data(path="zt_logs.csv"):
    df = pd.read_csv(path)
    if "is_anomaly" not in df.columns:
        raise RuntimeError("Dataset must include 'is_anomaly' column")
    return df

def feature_matrix(df, drop_cols=None):
    if drop_cols is None:
        drop_cols = []
    X = df.drop(columns=[c for c in (["is_anomaly"] + drop_cols) if c in df.columns], errors='ignore').copy()
    for c in X.select_dtypes(include="bool").columns:
        X[c] = X[c].astype(int)
    X_num = X.select_dtypes(include=[np.number])
    return X_num

def evaluate_model(y_true, y_pred, name="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {"name": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred Normal (0)", "Pred Anomaly (1)"],
        y=["True Normal (0)", "True Anomaly (1)"],
        colorscale="Blues",
        hovertemplate="(%{y}, %{x}) = %{z}<extra></extra>"
    ))
    fig.update_layout(title=title, margin=dict(l=0,r=0,t=30,b=0))
    return fig

# UI
st.title("AI-Powered Anomaly Detection — Zero Trust Dashboard (99th pct AE)")
st.markdown("Models trained on normal data. Autoencoder threshold = **99th percentile** of training reconstruction error.")

# Sidebar
st.sidebar.header("Controls")
data_path = st.sidebar.text_input("Path to features CSV", "zt_logs.csv")
simulate = st.sidebar.checkbox("Enable real-time simulation", value=False)
chunk_size = st.sidebar.number_input("Real-time batch size (rows)", min_value=1, max_value=1000, value=50)
interval = st.sidebar.number_input("Simulation interval (seconds)", min_value=0.5, max_value=30.0, value=3.0, step=0.5)
run_models_button = st.sidebar.button("Train & Evaluate Models")
page = st.sidebar.radio("Dashboard page", ["Overview", "Model Comparison", "Combined Decisions", "Real-time Feed"])

# Load
df = load_data(data_path)
st.sidebar.write(f"Rows: {len(df):,}  |  Anomalies: {int(df['is_anomaly'].sum()):,}")

# Feature prep
drop_cols = []
X = feature_matrix(df, drop_cols=drop_cols)
y = df["is_anomaly"].astype(int).values
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Overview page
if page == "Overview":
    st.header("Overview & Anomaly Trends")
    c1, c2 = st.columns([2,1])
    with c1:
        if "hour" in df.columns:
            agg = df.groupby("hour")["is_anomaly"].sum().reset_index()
            fig = px.line(agg, x="hour", y="is_anomaly", title="Anomalies by Hour", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        if {"lat","lon"}.issubset(df.columns):
            anoms = df[df["is_anomaly"]==1]
            if len(anoms)>0:
                fig2 = px.scatter_geo(anoms, lat="lat", lon="lon", hover_name=anoms.index.astype(str),
                                      title="Anomalous Events (Geolocation)", scope="world", height=400)
                st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.subheader("Dataset sample")
        st.dataframe(df.head(8), use_container_width=True)

# Model training & evaluation (runs when button clicked)
models_ready = False
if run_models_button:
    st.info("Training models... this may take a bit.")
    X_all_num = X.select_dtypes(include=[np.number])
    X_train = X_all_num[df["is_anomaly"] == 0].values
    contamination = max(0.001, df["is_anomaly"].mean())
    scaler = StandardScaler().fit(X_train)
    X_all_scaled = scaler.transform(X_all_num.values)

    # IsolationForest
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso.fit(scaler.transform(X_all_num[df["is_anomaly"] == 0].values))
    iso_preds = (iso.predict(X_all_scaled) == -1).astype(int)

    # Autoencoder (MLPRegressor approximation) with 99th percentile threshold
    input_dim = X_all_scaled.shape[1]
    mlp = MLPRegressor(hidden_layer_sizes=(max(8,input_dim//2), max(8,input_dim//4), max(8,input_dim//2)),
                       max_iter=200, random_state=42)
    mlp.fit(scaler.transform(X_all_num[df["is_anomaly"] == 0].values), scaler.transform(X_all_num[df["is_anomaly"] == 0].values))
    train_rec = mlp.predict(scaler.transform(X_all_num[df["is_anomaly"] == 0].values))
    train_mse = np.mean((scaler.transform(X_all_num[df["is_anomaly"] == 0].values) - train_rec)**2, axis=1)
    thr = np.percentile(train_mse, 99)  # <-- 99th percentile
    all_rec = mlp.predict(X_all_scaled)
    ae_preds = (np.mean((X_all_scaled - all_rec)**2, axis=1) > thr).astype(int)

    # One-Class SVM
    nu = min(0.5, max(0.001, contamination*1.2))
    oc_svm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
    oc_svm.fit(scaler.transform(X_all_num[df["is_anomaly"] == 0].values))
    svm_preds = (oc_svm.predict(X_all_scaled) == -1).astype(int)

    # Evaluate
    results = []
    for name, preds in [("IsolationForest", iso_preds), ("Autoencoder(99th)", ae_preds), ("OneClassSVM", svm_preds)]:
        res = evaluate_model(y, preds, name=name)
        results.append(res)

    res_df = pd.DataFrame([{"Model": r["name"], "Accuracy": r["accuracy"], "Precision": r["precision"],
                            "Recall": r["recall"], "F1": r["f1"]} for r in results])
    st.subheader("Model Metrics")
    st.dataframe(res_df.style.format({"Accuracy":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}"}), use_container_width=True)

    st.subheader("Confusion Matrices")
    cms = st.columns(3)
    for i, r in enumerate(results):
        with cms[i]:
            st.markdown(f"**{r['name']}**")
            fig_cm = plot_confusion_matrix(r["cm"], title=r["name"])
            st.plotly_chart(fig_cm, use_container_width=True)

    # Persist predictions to df for other pages
    df["iso_pred"] = iso_preds
    df["ae_pred"] = ae_preds
    df["svm_pred"] = svm_preds
    models_ready = True
    st.success("Models trained and evaluated (99th percentile used for autoencoder).")

# Model Comparison page
if page == "Model Comparison":
    st.header("Model Comparison — Per-event agreement & disagreements")
    if not models_ready:
        st.info("Train models first (click 'Train & Evaluate Models' in sidebar).")
    else:
        st.markdown("Top disagreements between models (where models disagree or disagree with label)")
        df["agreement_count"] = df[["iso_pred","ae_pred","svm_pred"]].sum(axis=1)
        disagreements = df[df["agreement_count"].isin([1,2])].copy()
        st.dataframe(disagreements.sort_values("agreement_count").head(200), use_container_width=True)
        # show heatmap of model decisions for a sample
        sample = df.sample(300, random_state=42).reset_index(drop=True)
        decs = sample[["iso_pred","ae_pred","svm_pred"]].T
        fig = px.imshow(decs, labels=dict(x="event index", y="model", color="decision"), title="Sample model decisions (0/1)")
        st.plotly_chart(fig, use_container_width=True)

# Combined Decisions page
if page == "Combined Decisions":
    st.header("Combined Decisions — Side-by-side per-event view")
    if not models_ready:
        st.info("Train models first.")
    else:
        cols = st.multiselect("Columns to show", options=["is_anomaly","iso_pred","ae_pred","svm_pred","hour","dayofweek","lat","lon"], default=["is_anomaly","iso_pred","ae_pred","svm_pred","hour"])
        view = df[cols].copy()
        # color rows where any model disagrees with label
        def highlight(row):
            if "is_anomaly" in row.index:
                lab = row["is_anomaly"]
                preds = [row.get("iso_pred",0), row.get("ae_pred",0), row.get("svm_pred",0)]
                if any(p != lab for p in preds):
                    return ["background-color: #ffd6d6"]*len(row)
            return [""]*len(row)
        st.dataframe(view.head(500), use_container_width=True)

# Real-time Feed page
if page == "Real-time Feed":
    st.header("Real-time Monitoring Simulation")
    st.markdown("Simulate streaming logs. Models will flag incoming events (if trained).")
    if "simulation_index" not in st.session_state:
        st.session_state.simulation_index = 0
        st.session_state.sim_running = False

    start = st.button("Start Simulation")
    stop = st.button("Stop Simulation")
    reset = st.button("Reset Simulation")

    if start:
        st.session_state.sim_running = True
    if stop:
        st.session_state.sim_running = False
    if reset:
        st.session_state.simulation_index = 0
        st.session_state.sim_running = False

    if st.session_state.sim_running:
        st.info("Simulation running. Press Stop to halt.")
        i = st.session_state.simulation_index
        batch = df.iloc[i:i+int(chunk_size)].copy()
        if models_ready:
            # recompute on-the-fly using scaler and models in memory (assumes they exist in this scope)
            Xb = feature_matrix(batch).select_dtypes(include=[np.number])
            Xb_scaled = scaler.transform(Xb.values)
            batch["iso_pred"] = (iso.predict(Xb_scaled) == -1).astype(int)
            batch["ae_pred"] = (np.mean((mlp.predict(Xb_scaled) - Xb_scaled)**2, axis=1) > thr).astype(int)
            batch["svm_pred"] = (oc_svm.predict(Xb_scaled) == -1).astype(int)
        st.dataframe(batch.head(200), use_container_width=True)
        st.session_state.simulation_index += int(chunk_size)
        time.sleep(float(interval))
        if st.session_state.simulation_index >= len(df):
            st.success("Simulation finished.")
            st.session_state.sim_running = False
    else:
        st.write("Simulation stopped. Start to begin streaming.")

st.markdown("---")
st.caption("Notes: Autoencoder uses 99th percentile threshold for anomaly decision. Adjust 'thr' or contamination to change sensitivity.")
