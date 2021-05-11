import numpy as np
from sklearn.metrics import auc
from sklearn.model_selection import KFold
from scipy import interpolate


def evaluate(distances, labels, num_folds=10):
    """

        :param distances: np array: pairwise distance of embeddings
        :param labels: np array: 1 means positive pair, o means negative pair
        :param num_folds: (int) Number of folds for KFold cross-validation, defaults to 10 folds
        :param fpr_target: (float) Target of false positive rate is 0.001
        :return:
        true_positive_rate: Mean value of all true positive rates across all cross validation folds for plotting
                             the Receiver operating characteristic (ROC) curve.
        false_positive_rate: Mean value of all false positive rates across all cross validation folds for plotting
                              the Receiver operating characteristic (ROC) curve.
        accuracy: Array of accuracy values per each fold in cross validation set.
        roc_auc: Area Under the Receiver operating characteristic (AUROC) metric.
        best_distances: Array of Euclidean distance values that had the best performing accuracy on the LFW dataset
                         per each fold in cross validation set.
        tpr: Array that contains True Acceptance Rate values per each fold in cross validation set
              when far (False Accept Rate) is set to a specific value.
        fpr: Array that contains False Acceptance Rate values per each fold in cross validation set.
        """
    thresholds_roc = np.arange(0.01, 2, 0.01)
    true_positive_rate, false_positive_rate, accuracy, best_distances = \
        calculate_roc(
            thresholds=thresholds_roc, distances=distances, labels=labels, nrof_folds=num_folds
        )

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Calculate validation rate
    # thresholds_val = np.arange(0, 8, 0.001)
    # val, val_std, far = calculate_val(thresholds_val, distances,
    #                                   labels, 1e-3, nrof_folds=num_folds)

    return true_positive_rate, false_positive_rate, accuracy, roc_auc, best_distances


def calculate_roc(thresholds, distances, labels, nrof_folds=10):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    best_distance = np.zeros((nrof_folds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, distances[train_set], labels[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 distances[test_set],
                                                                                                 labels[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], distances[test_set],
                                                      labels[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        best_distance[fold_idx] = thresholds[best_threshold_index]

    return tpr, fpr, accuracy, best_distance


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, distances, labels, far_target=1e-3, nrof_folds=10):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, distances[train_set], labels[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, distances[test_set], labels[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0, 0
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far
