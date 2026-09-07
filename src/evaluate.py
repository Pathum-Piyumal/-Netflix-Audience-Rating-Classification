import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def evaluate_and_save_results(pipeline, X_test, y_test, figures_dir='reports/figures', results_dir='reports/results'):
    """
    Evaluates trained pipeline on test data, saves confusion matrix plot and classification metrics.
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    y_pred = pipeline.predict(X_test)
    
    # Calculate accuracy & F1 score
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # Save Metrics JSON
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'classification_report': report_dict
    }
    metrics_path = os.path.join(results_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics saved to {metrics_path}")
    
    # Plot & Save Confusion Matrix
    labels = sorted(list(y_test.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Rating Classification Confusion Matrix')
    plt.xlabel('Predicted Rating Category')
    plt.ylabel('True Rating Category')
    plt.tight_layout()
    
    cm_path = os.path.join(figures_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix heatmap saved to {cm_path}")
    
    return metrics
