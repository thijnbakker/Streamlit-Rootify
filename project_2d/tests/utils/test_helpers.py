import pytest
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from src.npecage.utils.helpers import f1


@pytest.mark.parametrize("y_true, y_pred, expected_f1", [
    (np.array([[1, 0, 1, 0]]), np.array([[1, 0, 1, 0]]), 1.0),
    (np.array([[1, 0, 1, 0]]), np.array([[0, 1, 0, 1]]), 0.0),
    (np.array([[1, 1, 0, 0]]), np.array([[1, 0, 1, 0]]), 0.5),
])
def test_f1_score(y_true, y_pred, expected_f1):
    y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred_tf = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    f1_tensor = f1(y_true_tf, y_pred_tf)
    f1_value = K.eval(f1_tensor)

    assert np.isclose(f1_value, expected_f1, atol=1e-4), f"Expected {expected_f1}, got {f1_value}"
