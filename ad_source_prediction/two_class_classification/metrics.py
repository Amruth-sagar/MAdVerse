import wandb
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

methods = {'accuracy':accuracy_score,
           'f1_score':f1_score,
           'precision':precision_score, 
           'recall': recall_score,
           'confusion_matrix': confusion_matrix}

def calculate_metrics(metrics, gt_labels, pred_labels):
    scores = {}
    for m in metrics:
        if m not in methods:
            raise ValueError('The given metric {0} is not the correct name. Retry by giving the correct name'.format(m))
        scores[m] = methods[m](y_true=gt_labels, y_pred=pred_labels)
    return scores

def PlotHeatMap(epoch,gt_labels, pred_labels, path):
    conf_matrix = confusion_matrix(gt_labels, pred_labels)
    conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt=".2f", xticklabels=sorted(set(pred_labels)), yticklabels=sorted(set(gt_labels)))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix Heatmap")
    plt.savefig(path+"heatmap_"+str(epoch)+".png", bbox_inches="tight")
    plt.show()
