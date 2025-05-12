''' Code        : Evaluation metrics
    Description : This file contains evaluation metrics'''

def iou(tp, fp, fn):
    iou = tp / (tp + fp + fn + 1e-12)  # Add a small epsilon to avoid division by zero
    return iou.item()

def precision(tp, fp):
    precision = tp / (tp + fp + 1e-12)  # Add a small epsilon to avoid division by zero
    return precision.item()

def recall(tp, fn):
    recall = tp / (tp + fn + 1e-12)  # Add a small epsilon to avoid division by zero
    return recall.item()

def f1score(precision,recall):
    f1score=(2*precision * recall)/(precision+recall + 1e-12)
    return f1score