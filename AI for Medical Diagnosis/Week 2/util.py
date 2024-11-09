import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def get_true_pos(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 1))


def get_true_neg(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 0))


def get_false_neg(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 1))


def get_false_pos(y, pred, th=0.5):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 0))


def get_performance_metrics(y, pred, class_labels, tp=get_true_pos,
                            tn=get_true_neg, fp=get_false_pos,
                            fn=get_false_neg,
                            acc=None, prevalence=None, spec=None,
                            sens=None, ppv=None, npv=None, auc=None, f1=None,
                            thresholds=[]):
    if len(thresholds) != len(class_labels):
        thresholds = [0.5] * len(class_labels)

    columns = ["Disease", "TP", "TN", "FP", "FN", "Accuracy", "Prevalence",
               "Sensitivity", "Specificity", "PPV", "NPV", "AUC", "F1", "Threshold"]
    data = []

    for i, label in enumerate(class_labels):
        row = [
            label,
            round(tp(y[:, i], pred[:, i]), 3) if tp is not None else "Not Defined",
            round(tn(y[:, i], pred[:, i]), 3) if tn is not None else "Not Defined",
            round(fp(y[:, i], pred[:, i]), 3) if fp is not None else "Not Defined",
            round(fn(y[:, i], pred[:, i]), 3) if fn is not None else "Not Defined",
            round(acc(y[:, i], pred[:, i], thresholds[i]), 3) if acc is not None else "Not Defined",
            round(prevalence(y[:, i]), 3) if prevalence is not None else "Not Defined",
            round(sens(y[:, i], pred[:, i], thresholds[i]), 3) if sens is not None else "Not Defined",
            round(spec(y[:, i], pred[:, i], thresholds[i]), 3) if spec is not None else "Not Defined",
            round(ppv(y[:, i], pred[:, i], thresholds[i]), 3) if ppv is not None else "Not Defined",
            round(npv(y[:, i], pred[:, i], thresholds[i]), 3) if npv is not None else "Not Defined",
            round(auc(y[:, i], pred[:, i]), 3) if auc is not None else "Not Defined",
            round(f1(y[:, i], pred[:, i] > thresholds[i]), 3) if f1 is not None else "Not Defined",
            round(thresholds[i], 3)
        ]
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    df = df.set_index("Disease")
    return df


def print_confidence_intervals(class_labels, statistics):
    df = pd.DataFrame(columns=["Mean AUC (CI 5%-95%)"])
    for i in range(len(class_labels)):
        mean = statistics.mean(axis=1)[i]
        max_ = np.quantile(statistics, .95, axis=1)[i]
        min_ = np.quantile(statistics, .05, axis=1)[i]
        df.loc[class_labels[i]] = ["%.2f (%.2f-%.2f)" % (mean, min_, max_)]
    return df


def get_curve(gt, pred, target_names, curve='roc'):
    for i in range(len(target_names)):
        if curve == 'roc':
            curve_function = roc_curve
            auc_roc = roc_auc_score(gt[:, i], pred[:, i])
            label = target_names[i] + " AUC: %.3f " % auc_roc
            xlabel = "False positive rate"
            ylabel = "True positive rate"
            a, b, _ = curve_function(gt[:, i], pred[:, i])
            plt.figure(1, figsize=(7, 7))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(a, b, label=label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)
        elif curve == 'prc':
            precision, recall, _ = precision_recall_curve(gt[:, i], pred[:, i])
            average_precision = average_precision_score(gt[:, i], pred[:, i])
            label = target_names[i] + " Avg.: %.3f " % average_precision
            plt.figure(1, figsize=(7, 7))
            plt.step(recall, precision, where='post', label=label)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)
