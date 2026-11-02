import pandas as pd

from model_utils import classification_metrics, prepare_features


def test_prepare_features_drops_targets_and_aligns_one_hot_columns():
    train = pd.DataFrame({
        "id": [1, 2],
        "label": [0, 1],
        "attack_cat": ["Normal", "DoS"],
        "dur": [0.1, 0.2],
        "proto": ["tcp", "udp"],
        "service": ["http", "dns"],
    })
    train_features = prepare_features(train)
    schema = list(train_features.columns)

    test = pd.DataFrame({
        "id": [3],
        "label": [1],
        "attack_cat": ["Generic"],
        "dur": [0.3],
        "proto": ["icmp"],
        "service": ["ftp"],
    })
    aligned = prepare_features(test, schema)

    assert list(aligned.columns) == schema
    assert "label" not in aligned.columns
    assert "attack_cat" not in aligned.columns
    assert "id" not in aligned.columns
    assert aligned.shape == (1, len(schema))
    assert (aligned.iloc[0] == 0).all()


def test_classification_metrics_returns_binary_metrics():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])

    assert result["accuracy"] == 0.75
    assert result["precision"] == 2 / 3
    assert result["recall"] == 1.0
    assert round(result["f1"], 3) == 0.8
