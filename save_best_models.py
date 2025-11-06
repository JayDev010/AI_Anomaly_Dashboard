"""
Run this script from your notebook after training models to save them.
Copy and paste this code into a notebook cell, or run it as a script.
"""
import joblib
import pandas as pd
import numpy as np
import json
import os
from tensorflow import keras

def save_models_from_notebook(iso_forest, svm_model, autoencoder, scaler, 
                              feature_cols, results_df, iso_metrics, 
                              svm_metrics, ae_metrics):
    """
    Save all trained models and results.
    Call this function from your notebook after training.
    """
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    print("💾 Saving models and results...")
    
    # Save models
    joblib.dump(iso_forest, 'models/isolation_forest_model.pkl')
    print("   ✓ Isolation Forest saved")
    
    joblib.dump(svm_model, 'models/svm_model.pkl')
    print("   ✓ SVM saved")
    
    autoencoder.save('models/autoencoder_model.h5')
    print("   ✓ AutoEncoder saved")
    
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    print("   ✓ Scaler saved")
    
    # Save feature columns
    with open('models/feature_columns.json', 'w') as f:
        json.dump(feature_cols, f)
    print("   ✓ Feature columns saved")
    
    # Save results dataframe
    results_df.to_csv('models/model_comparison_results.csv', index=False)
    print("   ✓ Comparison results saved")
    
    # Save individual metrics
    metrics = {
        'isolation_forest': iso_metrics,
        'svm': svm_metrics,
        'autoencoder': ae_metrics
    }
    
    with open('models/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print("   ✓ Model metrics saved")
    
    # Create and save summary
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
    print("   ✓ Model summary saved")
    
    print("\n✅ All models and results saved successfully!")
    print("\n📊 Summary:")
    print(f"   Best Accuracy: {summary['best_accuracy']['model']} ({summary['best_accuracy']['value']:.4f})")
    print(f"   Best Precision: {summary['best_precision']['model']} ({summary['best_precision']['value']:.4f})")
    print(f"   Best Recall: {summary['best_recall']['model']} ({summary['best_recall']['value']:.4f})")
    print(f"   Best F1-Score: {summary['best_f1_score']['model']} ({summary['best_f1_score']['value']:.4f})")
    if 'best_roc_auc' in summary:
        print(f"   Best ROC-AUC: {summary['best_roc_auc']['model']} ({summary['best_roc_auc']['value']:.4f})")
    
    print("\n🚀 You can now run the Streamlit dashboard:")
    print("   streamlit run dashboard.py")
    
    return summary

# Example usage (uncomment and run in notebook):
"""
# After training all models and creating results_df, run:
from save_best_models import save_models_from_notebook

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
"""

