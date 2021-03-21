import torch
from collections import Counter
from sklearn.metrics import classification_report
from sklearn import metrics

EPSILON = 1e-10

def accuracy(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    return torch.mean((y_pred == y_true).float()).numpy()

def precision(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    true_positives = Counter()
    predicted_positives = Counter()
    y_pred = y_pred.tolist() if torch.is_tensor(y_pred) else y_pred
    y_true = y_true.tolist() if torch.is_tensor(y_true) else y_true
    for true, pred in zip(y_true, y_pred):
        if pred == true:
            true_positives[true] += 1
        predicted_positives[pred] += 1
    classes = sorted(true_positives.keys())
    prec = dict()
    for i in classes:
        prec[i] = true_positives[i] / (predicted_positives[i] + EPSILON)

    return prec

def recall(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    true_positives = Counter()
    total = Counter()
    y_pred = y_pred.tolist() if torch.is_tensor(y_pred) else y_pred
    y_true = y_true.tolist() if torch.is_tensor(y_true) else y_true
    for true, pred in zip(y_true, y_pred):
        if pred == true:
            true_positives[true] += 1
        total[true] += 1
    classes = sorted(true_positives.keys())
    recall = dict()
    for i in classes:
        recall[i] = true_positives[i] / (total[i] + EPSILON)
    return recall


def f1Score(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    prec = precision(y_pred, y_true)
    rc= recall(y_pred, y_true)
    f1 = dict()
    for i in rc.keys():
        f1[i] = 2 * prec[i] * rc[i] / (prec[i] + rc[i])
    return f1

def mse(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    return torch.mean((y_true - y_pred)**2).numpy()

def mae(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    return torch.mean(torch.abs(y_true - y_pred))

def report(y_pred, y_true, logger):
    # logger.debug(classification_report(y_true, y_pred))
    f1_micro = metrics.f1_score(y_true, y_pred, average="micro")
    f1_macro = metrics.f1_score(y_true, y_pred, average="macro")
    logger.info("f1_micro = {} and f1_macro = {}".format(f1_micro, f1_macro))
    return f1_micro, f1_macro