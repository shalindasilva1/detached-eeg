import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import os

def load_results(json_path="results/loso_results.json"):
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}. Please run the pipeline first.")
        return None
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_confusion_matrix(results, ax=None):
    y_true = [fold['y_true'] for fold in results['folds']]
    y_pred = [fold['y_pred'] for fold in results['folds']]
    
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=['Control', 'AD'], columns=['Control', 'AD'])
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues', ax=ax, annot_kws={"size": 16})
    ax.set_title('Subject-Level Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    return ax.figure

def plot_trial_accuracy(results, ax=None):
    df = pd.DataFrame(results['folds'])
    
    # Sort by true label so Controls and ADs are grouped
    df.sort_values(by=['y_true', 'trial_accuracy'], ascending=[True, False], inplace=True)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        
    colors = ['skyblue' if y == 0 else 'salmon' for y in df['y_true']]
    bars = ax.bar(df['subject'], df['trial_accuracy'] * 100, color=colors)
    
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='Random Chance')
    ax.set_title('Trial-Level Accuracy per Subject')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(range(len(df['subject'])))
    ax.set_xticklabels(df['subject'], rotation=45, ha='right')
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='skyblue', label='Control'),
                       Patch(facecolor='salmon', label='AD')]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    return ax.figure

def main():
    results = load_results()
    if not results:
        return
        
    os.makedirs('results/figures', exist_ok=True)
    
    print(f"Loaded results! Final Subject-Level Accuracy: {results['subject_level_accuracy']*100:.2f}%")
    print(f"Target Accuracy (Original Paper): {results.get('paper_target_accuracy', 0.8615)*100:.2f}%\n")
    
    # 1. Plot Confusion Matrix
    print("Generating Confusion Matrix...")
    fig_cm = plot_confusion_matrix(results)
    cm_path = "results/figures/confusion_matrix.png"
    fig_cm.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close(fig_cm)
    print(f"Saved to {cm_path}")
    
    # 2. Plot Trial Accuracy
    print("Generating Trial Accuracy Chart...")
    fig_acc = plot_trial_accuracy(results)
    acc_path = "results/figures/trial_accuracy_per_subject.png"
    fig_acc.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.close(fig_acc)
    print(f"Saved to {acc_path}")
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
