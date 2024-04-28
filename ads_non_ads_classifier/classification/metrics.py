import wandb

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

methods = {'accuracy':accuracy_score,
           'f1_score':f1_score,
           'precision':precision_score, 
           'recall': recall_score}

def calculate_metrics(metrics, gt_labels, pred_labels):
    scores = {}
    for m in metrics:
        if m not in methods:
            raise ValueError('The given metric {0} is not the correct name. Retry by giving the correct name'.format(m))
        scores[m] = methods[m](y_true=gt_labels, y_pred=pred_labels)
    return scores
