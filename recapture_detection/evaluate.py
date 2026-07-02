import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import numpy as np

def evaluate_pipeline(y_true, y_pred, y_probs, feature_names=None, model=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_probs)
    }
    
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    axs[0].plot(fpr, tpr, label=f'AUC = {metrics["ROC-AUC"]:.2f}')
    axs[0].plot([0, 1], [0, 1], 'k--')
    axs[0].set_title('ROC Curve')
    axs[0].legend()
    
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    axs[1].plot(rec, prec, label=f'F1 = {metrics["F1-Score"]:.2f}')
    axs[1].set_title('Precision-Recall Curve')
    axs[1].legend()
    plt.savefig('evaluation_curves.png')
    
    if hasattr(model, 'feature_importances_') and feature_names:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        plt.figure(figsize=(10,6))
        plt.bar(range(15), importances[indices])
        plt.xticks(range(15), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')