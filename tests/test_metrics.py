import jax.numpy as jnp

from paxjaxlib import metrics


def test_accuracy():
    y_true = jnp.array([1, 1, 0, 0])
    y_pred = jnp.array([1, 0, 0, 1])

    assert metrics.accuracy(y_true, y_pred) == 0.5

    y_true_c = jnp.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    y_pred_c = jnp.array([[0, 1], [1, 0], [1, 0], [0, 1]])

    assert metrics.accuracy(y_true_c, y_pred_c) == 0.5


def test_precision():
    y_true = jnp.array([1, 0, 1, 1, 0, 0])
    y_pred = jnp.array([1, 1, 1, 0, 0, 1])

    # TP = 2, FP = 2
    # Precision = 2 / (2 + 2) = 0.5
    assert jnp.isclose(metrics.precision(y_true, y_pred), 0.5)


def test_recall():
    y_true = jnp.array([1, 0, 1, 1, 0, 0])
    y_pred = jnp.array([1, 1, 1, 0, 0, 1])

    # TP = 2, FN = 1
    # Recall = 2 / (2 + 1) = 2/3
    assert jnp.isclose(metrics.recall(y_true, y_pred), 2.0 / 3.0)


def test_f1_score():
    y_true = jnp.array([1, 0, 1, 1, 0, 0])
    y_pred = jnp.array([1, 1, 1, 0, 0, 1])

    precision = 0.5
    recall = 2.0 / 3.0
    expected_f1 = 2 * (precision * recall) / (precision + recall)

    assert jnp.isclose(metrics.f1_score(y_true, y_pred), expected_f1)
