import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from model_utils import classification_metrics, prepare_features


def main() -> None:
    train_path = Path(os.getenv("NIDS_TRAIN_PATH", "data/UNSW_NB15_training-set.csv"))
    test_path = Path(os.getenv("NIDS_TEST_PATH", "data/UNSW_NB15_testing-set.csv"))
    output_dir = Path(os.getenv("NIDS_MODEL_DIR", "models"))

    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {path}. Download UNSW-NB15 and place it under data/."
            )

    print("Loading UNSW-NB15 train and test splits...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if "label" not in train_df or "label" not in test_df:
        raise ValueError("Both UNSW-NB15 files must contain the label target column.")

    y_train = pd.to_numeric(train_df["label"], errors="raise").astype(int)
    y_test = pd.to_numeric(test_df["label"], errors="raise").astype(int)

    X_train = prepare_features(train_df)
    feature_columns = list(X_train.columns)
    X_test = prepare_features(test_df, feature_columns)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training K-Means...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)

    print("Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    print("Training Naive Bayes...")
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train_scaled, y_train)

    metrics = {
        "dataset": "UNSW-NB15",
        "evaluation_split": str(test_path),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": len(feature_columns),
        "models": {
            "KNN": classification_metrics(y_test, knn.predict(X_test_scaled)),
            "Naive Bayes": classification_metrics(y_test, naive_bayes.predict(X_test_scaled)),
        },
        "notes": [
            "KNN and Naive Bayes metrics use the supplied held-out test split.",
            "K-Means is unsupervised; cluster IDs are not inherently benign or malicious.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, output_dir / "kmeans_model.pkl")
    joblib.dump(knn, output_dir / "knn_model.pkl")
    joblib.dump(naive_bayes, output_dir / "naive_bayes_model.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")
    (output_dir / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))
    print(f"Model artifacts and evaluation metrics saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
