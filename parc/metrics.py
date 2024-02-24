import numpy as np
from parc.utils import get_mode


def accuracy(y_data_true, y_data_pred, target=1):

    pred_to_true_dict = {}
    n_samples = len(y_data_pred)
    n_target = list(y_data_true).count(target)

    for k in range(n_samples):
        pred_to_true_dict.setdefault(y_data_pred[k], []).append(y_data_true[k])
    n_clusters = len(pred_to_true_dict)
    labels = list(sorted(pred_to_true_dict.keys()))
    error_count = []
    negative_labels = []
    positive_labels = []
    fp, fn, tp, tn, precision, recall, f1_score = 0, 0, 0, 0, 0, 0, 0

    for label in labels:
        targets = pred_to_true_dict[label]
        majority_val = get_mode(targets)
        if majority_val == target:
            print(f"cluster {label} has majority {target} with population {len(targets)}")
        if label == -1:
            len_unknown = len(targets)
            print('len unknown', len_unknown)
        elif (majority_val == target):
            positive_labels.append(label)
            fp = fp + len([e for e in targets if e != target])
            tp = tp + len([e for e in targets if e == target])
            list_error = [e for e in targets if e != majority_val]
            e_count = len(list_error)
            error_count.append(e_count)
        else:
            negative_labels.append(label)
            tn = tn + len([e for e in targets if e != target])
            fn = fn + len([e for e in targets if e == target])
            error_count.append(len([e for e in targets if e != majority_val]))

    number_clusters_for_target = len(positive_labels)
    error_rate = sum(error_count) / n_samples
    n_target = tp + fn
    tnr = tn / (n_samples - n_target)
    fnr = fn / n_target
    tpr = tp / n_target
    fpr = fp / (n_samples - n_target)

    if tp != 0 or fn != 0:
        recall = tp / (tp + fn)  # ability to find all positives
    if tp != 0 or fp != 0:
        precision = tp / (tp + fp)  # ability to not misclassify negatives as positives
    if precision != 0 or recall != 0:
        f1_score = precision * recall * 2 / (precision + recall)
    majority_truth_labels = np.empty((len(y_data_true), 1), dtype=object)

    for cluster_id in set(y_data_pred):
        cluster_i_loc = np.where(np.asarray(y_data_pred) == cluster_id)[0]
        y_data_true = np.asarray(y_data_true)
        majority_truth = get_mode(list(y_data_true[cluster_i_loc]))
        majority_truth_labels[cluster_i_loc] = majority_truth

    majority_truth_labels = list(majority_truth_labels.flatten())
    accuracy_val = [error_rate, f1_score, tnr, fnr, tpr, fpr, precision,
                    recall, n_clusters, n_target]

    return accuracy_val, majority_truth_labels, number_clusters_for_target
