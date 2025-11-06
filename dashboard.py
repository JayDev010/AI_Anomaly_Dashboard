"""
Streamlit Dashboard for ML Anomaly Detection Results
Displays saved model results and allows predictions
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import tensorflow as tf
from tensorflow import keras

st.set_page_config(
    page_title="ML Anomaly Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .best-model {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(path="zt_logs_with_time_diff_and_anomaly.csv"):
    """Load the dataset"""
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return None

@st.cache_resource
def load_models():
    """Load saved models"""
    models = {}
    try:
        if os.path.exists('models/isolation_forest_model.pkl'):
            models['isolation_forest'] = joblib.load('models/isolation_forest_model.pkl')
        if os.path.exists('models/svm_model.pkl'):
            models['svm'] = joblib.load('models/svm_model.pkl')
        if os.path.exists('models/autoencoder_model.h5'):
            models['autoencoder'] = keras.models.load_model('models/autoencoder_model.h5')
        if os.path.exists('models/feature_scaler.pkl'):
            models['scaler'] = joblib.load('models/feature_scaler.pkl')
        if os.path.exists('models/feature_columns.json'):
            with open('models/feature_columns.json', 'r') as f:
                models['feature_cols'] = json.load(f)
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def load_results():
    """Load model results and metrics"""
    results = {}
    try:
        if os.path.exists('models/model_comparison_results.csv'):
            results['comparison'] = pd.read_csv('models/model_comparison_results.csv')
        if os.path.exists('models/model_metrics.json'):
            with open('models/model_metrics.json', 'r') as f:
                results['metrics'] = json.load(f)
        if os.path.exists('models/model_summary.json'):
            with open('models/model_summary.json', 'r') as f:
                results['summary'] = json.load(f)
        return results
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None

def plot_metrics_comparison(results_df):
    """Create comparison charts for model metrics"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    if 'ROC-AUC' in results_df.columns:
        metrics.append('ROC-AUC')
    
    fig = go.Figure()
    
    for metric in metrics:
        if metric in results_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df['Model'],
                y=results_df[metric],
                text=[f'{v:.3f}' for v in results_df[metric]],
                textposition='auto',
            ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def plot_confusion_matrix_plotly(y_true, y_pred, model_name):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Normal', 'Anomaly'],
        y=['Normal', 'Anomaly'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{model_name} - Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='True',
        height=400
    )
    
    return fig

# Main App
st.title("🤖 ML Anomaly Detection Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Model Results", "Model Comparison", "Predictions", "Data Overview"]
)

# Load data and models
df = load_data()
models = load_models()
results = load_results()

# Check if models are available
models_available = models is not None and len(models) > 0
results_available = results is not None and len(results) > 0

if not models_available:
    st.sidebar.warning("⚠️ Models not found. Please train and save models first.")
if not results_available:
    st.sidebar.warning("⚠️ Results not found. Please save model results first.")

# Model Results Page
if page == "Model Results":
    st.header("📈 Model Performance Results")
    
    if not results_available:
        st.error("Model results not found. Please run the notebook and save the models first.")
        st.info("""
        To save models, add this code to your notebook after training:
        ```python
        from save_models import save_models
        save_models(iso_forest, svm_model, autoencoder, scaler, feature_cols, 
                   results_df, iso_metrics, svm_metrics, ae_metrics)
        ```
        """)
    else:
        # Display summary
        if 'summary' in results:
            st.subheader("🏆 Best Models by Metric")
            summary = results['summary']
            
            cols = st.columns(len(summary))
            for idx, (metric, info) in enumerate(summary.items()):
                with cols[idx]:
                    metric_name = metric.replace('_', ' ').title()
                    st.metric(
                        label=f"Best {metric_name}",
                        value=info['model'],
                        delta=f"{info['value']:.4f}"
                    )
        
        # Display comparison table
        if 'comparison' in results:
            st.subheader("📊 Model Comparison Table")
            results_df = results['comparison']
            
            # Highlight best values
            def highlight_max(s):
                is_max = s == s.max()
                return ['background-color: #d4edda' if v else '' for v in is_max]
            
            styled_df = results_df.style.format({
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1-Score': '{:.4f}'
            }).apply(highlight_max, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
            
            if 'ROC-AUC' in results_df.columns:
                styled_df = styled_df.format({'ROC-AUC': '{:.4f}'}).apply(
                    highlight_max, subset=['ROC-AUC']
                )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Metrics comparison chart
            st.subheader("📈 Metrics Comparison Chart")
            fig = plot_metrics_comparison(results_df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display individual model metrics
        if 'metrics' in results:
            st.subheader("🔍 Detailed Model Metrics")
            
            metrics = results['metrics']
            model_names = {
                'isolation_forest': 'Isolation Forest',
                'svm': 'SVM',
                'autoencoder': 'AutoEncoder'
            }
            
            cols = st.columns(3)
            for idx, (model_key, model_name) in enumerate(model_names.items()):
                if model_key in metrics:
                    with cols[idx]:
                        st.markdown(f"### {model_name}")
                        model_metrics = metrics[model_key]
                        
                        for key, value in model_metrics.items():
                            if key != 'Model' and isinstance(value, (int, float)):
                                st.metric(
                                    label=key.replace('_', ' ').title(),
                                    value=f"{value:.4f}"
                                )

# Model Comparison Page
if page == "Model Comparison":
    st.header("⚖️ Model Comparison")
    
    if not results_available:
        st.error("Results not available. Please save model results first.")
    else:
        if 'comparison' in results:
            results_df = results['comparison']
            
            # Side-by-side comparison
            st.subheader("Metrics Side-by-Side")
            
            metrics_to_compare = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            if 'ROC-AUC' in results_df.columns:
                metrics_to_compare.append('ROC-AUC')
            
            for metric in metrics_to_compare:
                if metric in results_df.columns:
                    st.markdown(f"### {metric}")
                    fig = px.bar(
                        results_df,
                        x='Model',
                        y=metric,
                        text=metric,
                        title=f'{metric} Comparison',
                        color='Model',
                        color_discrete_map={
                            'Isolation Forest': '#1f77b4',
                            'SVM': '#ff7f0e',
                            'AutoEncoder': '#2ca02c'
                        }
                    )
                    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Summary insights
            st.subheader("💡 Key Insights")
            
            best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
            best_f1 = results_df.loc[results_df['F1-Score'].idxmax()]
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Best Overall Model (F1-Score):** {best_f1['Model']} ({best_f1['F1-Score']:.4f})")
            with col2:
                st.info(f"**Most Accurate Model:** {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")

# Predictions Page
if page == "Predictions":
    st.header("🔮 Make Predictions")
    
    if not models_available:
        st.error("Models not available. Please train and save models first.")
    else:
        if df is not None:
            st.subheader("Predict on Dataset")
            
            # Select sample size
            sample_size = st.slider("Number of samples to predict", 10, min(1000, len(df)), 100)
            
            if st.button("Generate Predictions"):
                sample_df = df.sample(n=sample_size, random_state=42)
                
                # Prepare features
                feature_cols = models.get('feature_cols', [])
                scaler = models.get('scaler')
                
                if scaler and feature_cols:
                    # Select only numeric features that exist
                    available_features = [col for col in feature_cols if col in sample_df.columns]
                    X_sample = sample_df[available_features].select_dtypes(include=[np.number])
                    
                    # Fill missing values
                    X_sample = X_sample.fillna(0)
                    
                    # Scale features
                    X_scaled = scaler.transform(X_sample)
                    
                    # Make predictions
                    predictions = {}
                    
                    if 'isolation_forest' in models:
                        iso_pred = models['isolation_forest'].predict(X_scaled)
                        predictions['Isolation Forest'] = (iso_pred == -1).astype(int)
                    
                    if 'svm' in models:
                        svm_pred = models['svm'].predict(X_scaled)
                        predictions['SVM'] = svm_pred
                    
                    if 'autoencoder' in models:
                        reconstructions = models['autoencoder'].predict(X_scaled, verbose=0)
                        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
                        # Use a threshold (you might want to save this from training)
                        threshold = np.percentile(mse, 95)  # Default threshold
                        predictions['AutoEncoder'] = (mse > threshold).astype(int)
                    
                    # Add predictions to dataframe
                    result_df = sample_df.copy()
                    for model_name, preds in predictions.items():
                        result_df[f'{model_name}_prediction'] = preds
                    
                    if 'is_anomaly' in result_df.columns:
                        result_df['ground_truth'] = result_df['is_anomaly']
                    
                    st.subheader("Prediction Results")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Show prediction statistics
                    st.subheader("Prediction Statistics")
                    pred_stats = pd.DataFrame({
                        'Model': list(predictions.keys()),
                        'Anomalies Detected': [pred.sum() for pred in predictions.values()],
                        'Anomaly Rate': [pred.mean() for pred in predictions.values()]
                    })
                    st.dataframe(pred_stats, use_container_width=True)
                else:
                    st.error("Feature columns or scaler not found in saved models.")

# Data Overview Page
if page == "Data Overview":
    st.header("📋 Data Overview")
    
    if df is not None:
        st.subheader("Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            if 'is_anomaly' in df.columns:
                st.metric("Anomalies", f"{df['is_anomaly'].sum():,}")
        with col3:
            if 'is_anomaly' in df.columns:
                anomaly_rate = df['is_anomaly'].mean() * 100
                st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        with col4:
            st.metric("Features", len(df.columns))
        
        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Anomaly distribution
        if 'is_anomaly' in df.columns:
            st.subheader("Anomaly Distribution")
            anomaly_counts = df['is_anomaly'].value_counts()
            fig = px.pie(
                values=anomaly_counts.values,
                names=['Normal', 'Anomaly'],
                title='Anomaly Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Dataset not found. Please ensure 'zt_logs_with_time_diff_and_anomaly.csv' exists.")

# Footer
st.markdown("---")
st.caption("ML Anomaly Detection Dashboard - Results from Isolation Forest, AutoEncoder, and SVM models")

