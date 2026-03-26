
from sklearn.metrics import average_precision_score, roc_auc_score

def compute_metrics(y, p):
    return {
        "auc_pr": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p)
    }
