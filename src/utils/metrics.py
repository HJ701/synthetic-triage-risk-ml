from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

@dataclass(frozen=True)
class MetricReport:
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float

def classification_report(y_true, y_prob, y_pred) -> MetricReport:
    return MetricReport(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        f1=float(f1_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred)),
        recall=float(recall_score(y_true, y_pred))
    )