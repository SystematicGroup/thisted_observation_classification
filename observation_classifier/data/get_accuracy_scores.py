from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


def get_performance_scores(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    print('Accuracy: ', acc)
    print('F1-score, macro: ', f1score)
    print('Precision: ', precision)
    print('Recall: ', recall)

    return [acc, f1score, precision, recall]