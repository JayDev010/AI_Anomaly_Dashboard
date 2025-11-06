"""
Example: How to save models from your notebook

Copy this code into a new cell in your notebook after training all models.
Make sure all variables (iso_forest, svm_model, autoencoder, scaler, etc.) are defined.
"""

# Example code to add to your notebook:

"""
# After training all models and evaluating, run this:

import joblib
import json
import os
from tensorflow import keras

# Create models directory
os.makedirs('models', exist_ok=True)

# Save models
print("Saving models...")
joblib.dump(iso_forest, 'models/isolation_forest_model.pkl')
joblib.dump(svm_model, 'models/svm_model.pkl')
autoencoder.save('models/autoencoder_model.h5')
joblib.dump(scaler, 'models/feature_scaler.pkl')

# Save feature columns
with open('models/feature_columns.json', 'w') as f:
    json.dump(feature_cols, f)

# Save results
results_df.to_csv('models/model_comparison_results.csv', index=False)

# Save metrics
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
print(f"Best F1-Score: {summary['best_f1_score']['model']} ({summary['best_f1_score']['value']:.4f})")
"""

