import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import os

def load_results(results_path="results/cv_results.json"):
    with open(results_path, "r") as f:
        return json.load(f)

def plot_confusion_matrix(all_y_true, all_y_pred, output_path="results/confusion_matrix.png"):
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Control', 'Dementia'], 
                yticklabels=['Control', 'Dementia'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Aggregated Confusion Matrix (All Folds)')
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def plot_metrics_per_fold(folds, output_path="results/metrics_per_fold.png"):
    metrics_data = []
    
    for fold in folds:
        y_true = fold['y_true']
        y_pred = fold['y_pred']
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])
        
        metrics_data.append({
            "Fold": fold['fold'],
            "Metric": "Precision (Control)",
            "Value": precision[0]
        })
        metrics_data.append({
            "Fold": fold['fold'],
            "Metric": "Recall (Control)",
            "Value": recall[0]
        })
        metrics_data.append({
            "Fold": fold['fold'],
            "Metric": "Precision (Dementia)",
            "Value": precision[1]
        })
        metrics_data.append({
            "Fold": fold['fold'],
            "Metric": "Recall (Dementia)",
            "Value": recall[1]
        })
        metrics_data.append({
            "Fold": fold['fold'],
            "Metric": "Accuracy",
            "Value": fold['accuracy']
        })

    df_metrics = pd.DataFrame(metrics_data)
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_metrics, x="Fold", y="Value", hue="Metric")
    plt.ylim(0, 1.1)
    plt.title('Performance Metrics per Fold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Metrics per fold plot saved to {output_path}")

def main():
    if not os.path.exists("results/cv_results.json"):
        print("Error: results/cv_results.json not found. Run the pipeline first.")
        return

    data = load_results()
    folds = data['folds']
    
    all_y_true = []
    all_y_pred = []
    
    for fold in folds:
        all_y_true.extend(fold['y_true'])
        all_y_pred.extend(fold['y_pred'])
    
    os.makedirs("results", exist_ok=True)
    
    plot_confusion_matrix(all_y_true, all_y_pred)
    plot_metrics_per_fold(folds)
    
    print("\nGlobal Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=['Control', 'Dementia']))

if __name__ == "__main__":
    main()
