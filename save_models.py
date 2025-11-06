"""
Script to save the best trained models from the ML notebook.
Run this after training models in the notebook.
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import os

def save_models(iso_forest, svm_model, autoencoder, scaler, feature_cols, results_df, 
                iso_metrics, svm_metrics, ae_metrics):
    """
    Save the trained models and their evaluation results.
    
    Parameters:
    - iso_forest: Trained Isolation Forest model
    - svm_model: Trained SVM model
    - autoencoder: Trained AutoEncoder model
    - scaler: Fitted StandardScaler
    - feature_cols: List of feature column names
    - results_df: DataFrame with model comparison results
    - iso_metrics, svm_metrics, ae_metrics: Individual model metrics
    """
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save models
    print("Saving models...")
    joblib.dump(iso_forest, 'models/isolation_forest_model.pkl')
    joblib.dump(svm_model, 'models/svm_model.pkl')
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    autoencoder.save('models/autoencoder_model.h5')
    
    # Save feature columns
    with open('models/feature_columns.json', 'w') as f:
        json.dump(feature_cols, f)
    
    # Save results
    results_df.to_csv('models/model_comparison_results.csv', index=False)
    
    # Save individual metrics
    metrics = {
        'isolation_forest': iso_metrics,
        'svm': svm_metrics,
        'autoencoder': ae_metrics
    }
    
    with open('models/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Save summary
    summary = {
        'best_accuracy': {
            'model': results_df.loc[results_df['Accuracy'].idxmax(), 'Model'],
            'value': float(results_df['Accuracy'].max())
        },
        'best_precision': {
            'model': results_df.loc[results_df['Precision'].idxmax(), 'Model'],
            'value': float(results_df['Precision'].max())
        },
        'best_recall': {
            'model': results_df.loc[results_df['Recall'].idxmax(), 'Model'],
            'value': float(results_df['Recall'].max())
        },
        'best_f1_score': {
            'model': results_df.loc[results_df['F1-Score'].idxmax(), 'Model'],
            'value': float(results_df['F1-Score'].max())
        }
    }
    
    if 'ROC-AUC' in results_df.columns:
        summary['best_roc_auc'] = {
            'model': results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model'],
            'value': float(results_df['ROC-AUC'].max())
        }
    
    with open('models/model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Models saved successfully!")
    print(f"   - Isolation Forest: models/isolation_forest_model.pkl")
    print(f"   - SVM: models/svm_model.pkl")
    print(f"   - AutoEncoder: models/autoencoder_model.h5")
    print(f"   - Scaler: models/feature_scaler.pkl")
    print(f"   - Results: models/model_comparison_results.csv")
    print(f"   - Metrics: models/model_metrics.json")
    print(f"   - Summary: models/model_summary.json")
    
    return summary

if __name__ == "__main__":
    print("This script should be run from the notebook after training models.")
    print("Copy the save_models() function call from the notebook.")

