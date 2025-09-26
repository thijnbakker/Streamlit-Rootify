import tensorflow.keras.backend as K


def f1(y_true, y_pred):
    """
    Calculate the F1 score metric for Keras model training, combining precision and recall.

    Args:
        y_true (tensor): Ground truth binary labels.
        y_pred (tensor): Predicted binary labels (probabilities or logits).

    Returns:
        tensor: F1 score, harmonic mean of precision and recall.

    Helper functions:
        recall_m: Calculates recall as TP / (Actual Positives).
        precision_m: Calculates precision as TP / (Predicted Positives).

    Notes:
        - Uses Keras backend functions for tensor operations.
        - Adds epsilon to denominators to avoid division by zero.
    """

    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
