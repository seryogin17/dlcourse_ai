def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    accuracy = sum(~(prediction ^ ground_truth)) / len(prediction)
    precision = sum(prediction & ground_truth) / sum(prediction)
    recall = sum(prediction & ground_truth) / sum(ground_truth)
    f1 = 2 * precision * recall / (precision + recall)
    
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    total = 0
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]:
            total += 1
    accuracy = total / len(prediction)

    return accuracy
