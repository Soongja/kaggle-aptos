import numpy as np
from sklearn.metrics import cohen_kappa_score


def kappa(preds, labels, weights='quadratic'):
    return cohen_kappa_score(preds, labels, weights=weights)


def scale_threshold(preds):
    ths = [0.5, 1.5, 2.5, 3.5]

    for i, pred in enumerate(preds):
        if pred < ths[0]:
            preds[i] = 0
        elif pred >= ths[0] and pred < ths[1]:
            preds[i] = 1
        elif pred >= ths[1] and pred < ths[2]:
            preds[i] = 2
        elif pred >= ths[2] and pred < ths[3]:
            preds[i] = 3
        else:
            preds[i] = 4

    return preds


########################################################################################################################


def getOutputLabels(labels):
    labels = np.expand_dims(labels, axis=1)

    coef = [0.5, 1.5, 2.5, 3.5]

    labels_out = np.zeros((labels.shape[0], 5), dtype=labels.dtype)
    for i, pred in enumerate(labels):
        if pred < coef[0]:
            labels_out[i] = [1,0,0,0,0]
        elif pred >= coef[0] and pred < coef[1]:
            labels_out[i] = [0,1,0,0,0]
        elif pred >= coef[1] and pred < coef[2]:
            labels_out[i] = [0,0,1,0,0]
        elif pred >= coef[2] and pred < coef[3]:
            labels_out[i] = [0,0,0,1,0]
        else:
            labels_out[i] = [0,0,0,0,1]

    return labels_out


def batch_cohen_kappa_score(predicted, actual):
    actual = np.array(actual)
    actual = getOutputLabels(actual)

    predicted = np.array(predicted)
    predicted = getOutputLabels(predicted)

    batch_size = predicted.shape[0]
    total_score = 0.0
    for i in range(batch_size):
        total_score += cohen_kappa_score(actual[i], predicted[i], weights='quadratic')

    return total_score / batch_size


if __name__ == '__main__':
    preds = np.array([-1, 0.2, 4, 7, 2.5, 3.56])
    print(scale_threshold(preds))
