# Quick Start Guide - ML Anomaly Detection Dashboard

## Step 1: Save Models from Notebook

After training your models in the notebook, add this cell to save them:

```python
# Import the save function
from save_best_models import save_models_from_notebook

# Save all models and results
summary = save_models_from_notebook(
    iso_forest=iso_forest,
    svm_model=svm_model,
    autoencoder=autoencoder,
    scaler=scaler,
    feature_cols=feature_cols,
    results_df=results_df,
    iso_metrics=iso_metrics,
    svm_metrics=svm_metrics,
    ae_metrics=ae_metrics
)
```

This will create a `models/` directory with:
- `isolation_forest_model.pkl` - Best model (F1-Score: 0.6116)
- `svm_model.pkl` - Best Precision (0.7970) and ROC-AUC (0.5047)
- `autoencoder_model.h5` - AutoEncoder model
- `feature_scaler.pkl` - Feature scaler
- `feature_columns.json` - Feature column names
- `model_comparison_results.csv` - Comparison results
- `model_metrics.json` - Detailed metrics
- `model_summary.json` - Best models summary

## Step 2: Run Streamlit Dashboard

```bash
# Activate virtual environment
source venv/bin/activate

# Install streamlit and plotly if not already installed
pip install streamlit plotly

# Run the dashboard
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Dashboard Features

1. **Model Results** - View performance metrics and best models
2. **Model Comparison** - Compare all three models side-by-side
3. **Predictions** - Make predictions on new data
4. **Data Overview** - Explore the dataset

## Model Performance Summary

Based on your results:

- **Best Overall (F1-Score)**: Isolation Forest (0.6116)
- **Best Accuracy**: Isolation Forest (0.4981)
- **Best Precision**: SVM (0.7970)
- **Best Recall**: Isolation Forest (0.4966)
- **Best ROC-AUC**: SVM (0.5047)

## Troubleshooting

If models are not found:
1. Make sure you've run the save function in the notebook
2. Check that the `models/` directory exists
3. Verify all model files are present

If dashboard doesn't load:
1. Check that streamlit is installed: `pip install streamlit plotly`
2. Make sure you're in the correct directory
3. Verify the CSV file exists: `zt_logs_with_time_diff_and_anomaly.csv`

