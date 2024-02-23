import numpy as np
from parc.utils import get_mode


def accuracy(labels_true, labels_pred, onevsall=1):

    Index_dict = {}
    N = len(labels_pred)
    n_cancer = list(labels_true).count(onevsall)
    n_pbmc = N - n_cancer

    for k in range(N):
        Index_dict.setdefault(labels_pred[k], []).append(labels_true[k])
    num_groups = len(Index_dict)
    sorted_keys = list(sorted(Index_dict.keys()))
    error_count = []
    pbmc_labels = []
    thp1_labels = []
    fp, fn, tp, tn, precision, recall, f1_score = 0, 0, 0, 0, 0, 0, 0

    for kk in sorted_keys:
        vals = [t for t in Index_dict[kk]]
        majority_val = get_mode(vals)
        if majority_val == onevsall:
            print(f"cluster {kk} has majority {onevsall} with population {len(vals)}")
        if kk == -1:
            len_unknown = len(vals)
            print('len unknown', len_unknown)
        if (majority_val == onevsall) and (kk != -1):
            thp1_labels.append(kk)
            fp = fp + len([e for e in vals if e != onevsall])
            tp = tp + len([e for e in vals if e == onevsall])
            list_error = [e for e in vals if e != majority_val]
            e_count = len(list_error)
            error_count.append(e_count)
        elif (majority_val != onevsall) and (kk != -1):
            pbmc_labels.append(kk)
            tn = tn + len([e for e in vals if e != onevsall])
            fn = fn + len([e for e in vals if e == onevsall])
            error_count.append(len([e for e in vals if e != majority_val]))

    predict_class_array = np.array(labels_pred)
    labels_pred_array = np.array(labels_pred)
    number_clusters_for_target = len(thp1_labels)
    for cancer_class in thp1_labels:
        predict_class_array[labels_pred_array == cancer_class] = 1
    for benign_class in pbmc_labels:
        predict_class_array[labels_pred_array == benign_class] = 0
    predict_class_array.reshape((predict_class_array.shape[0], -1))
    error_rate = sum(error_count) / N
    n_target = tp + fn
    tnr = tn / n_pbmc
    fnr = fn / n_cancer
    tpr = tp / n_cancer
    fpr = fp / n_pbmc

    if tp != 0 or fn != 0:
        recall = tp / (tp + fn)  # ability to find all positives
    if tp != 0 or fp != 0:
        precision = tp / (tp + fp)  # ability to not misclassify negatives as positives
    if precision != 0 or recall != 0:
        f1_score = precision * recall * 2 / (precision + recall)
    majority_truth_labels = np.empty((len(labels_true), 1), dtype=object)

    for cluster_i in set(labels_pred):
        cluster_i_loc = np.where(np.asarray(labels_pred) == cluster_i)[0]
        labels_true = np.asarray(labels_true)
        majority_truth = get_mode(list(labels_true[cluster_i_loc]))
        majority_truth_labels[cluster_i_loc] = majority_truth

    majority_truth_labels = list(majority_truth_labels.flatten())
    accuracy_val = [error_rate, f1_score, tnr, fnr, tpr, fpr, precision,
                    recall, num_groups, n_target]

    return accuracy_val, predict_class_array, majority_truth_labels, number_clusters_for_target
