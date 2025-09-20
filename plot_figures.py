import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve, auc

os.makedirs("figures", exist_ok=True)

# ---------------- Load Data ----------------
with open("data/plot_data.json", "r") as f:
    plot_data = json.load(f)

y_true = np.array(plot_data["y_true"])
y_scores = np.array(plot_data["y_scores"])
feature_importances = np.array(plot_data["feature_importances"])

# ---------------- Binary or Multiclass? ----------------
if len(y_scores.shape) == 1:
    mode = "binary"
else:
    mode = "multiclass"

print(f"Detected mode: {mode}")

# ---------------- Precision-Recall ----------------
def plot_precision_recall(y_true, y_scores):
    if mode == "binary":
        # Pick one class as "positive"
        y_bin = (y_true == y_true[0]).astype(int)
        precision, recall, _ = precision_recall_curve(y_bin, y_scores)
        plt.figure()
        plt.plot(recall, precision, lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Binary)")
        plt.savefig("figures/precision_recall.png")
        plt.close()
    else:
        y_true_bin = label_binarize(y_true, classes=np.arange(y_scores.shape[1]))
        plt.figure()
        for i in range(y_scores.shape[1]):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
            plt.plot(recall, precision, lw=2, label=f"Class {i}")
        plt.legend()
        plt.title("Precision-Recall Curve (Multiclass)")
        plt.savefig("figures/precision_recall.png")
        plt.close()

# ---------------- ROC ----------------
def plot_roc(y_true, y_scores):
    if mode == "binary":
        y_bin = (y_true == y_true[0]).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_scores)
        plt.figure()
        plt.plot(fpr, tpr, lw=2)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Binary)")
        plt.savefig("figures/roc_curve.png")
        plt.close()
    else:
        y_true_bin = label_binarize(y_true, classes=np.arange(y_scores.shape[1]))
        plt.figure()
        for i in range(y_scores.shape[1]):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC={roc_auc:.2f})")
        plt.legend()
        plt.title("ROC Curve (Multiclass)")
        plt.savefig("figures/roc_curve.png")
        plt.close()

# ---------------- Feature Importances ----------------
def plot_feature_importances(feature_importances):
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(feature_importances)[::-1][:20]
    plt.bar(range(len(sorted_idx)), feature_importances[sorted_idx])
    plt.xticks(range(len(sorted_idx)), [f"F{i}" for i in sorted_idx], rotation=90)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig("figures/feature_importances.png")
    plt.close()

# ---------------- Run ----------------
plot_precision_recall(y_true, y_scores)
plot_roc(y_true, y_scores)
plot_feature_importances(feature_importances)

print("✅ Plots saved in 'figures/'")
