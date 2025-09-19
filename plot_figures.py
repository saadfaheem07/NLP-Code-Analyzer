import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

# Load the data
with open('data/plot_data.json', 'r') as f:
    plot_data = json.load(f)

y_true = plot_data['y_true']
y_scores = plot_data['y_scores']
feature_importances = plot_data['feature_importances']

# Plot Precision-Recall Curve
def plot_precision_recall_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

# Plot ROC Curve
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

# Plot Feature Importances
def plot_feature_importances(feature_importances):
    plt.figure()
    plt.bar(range(len(feature_importances)), feature_importances)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.show()

# Plot all figures
plot_precision_recall_curve(y_true, y_scores)
plot_roc_curve(y_true, y_scores)
plot_feature_importances(feature_importances)
