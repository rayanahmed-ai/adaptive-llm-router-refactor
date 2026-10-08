from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/home/sagemaker-user/adaptive-llm-routing"
)

LABEL_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "data"
    / "evaluation"
    / "v1_complexity_labels.json"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "v2_1"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CLASSIFIER_PATH = (
    MODEL_DIR
    / "classifier.joblib"
)

METADATA_PATH = (
    MODEL_DIR
    / "metadata.json"
)


# ============================================================
# LOAD LABEL DATA
# ============================================================

def load_examples():

    if not LABEL_PATH.exists():

        raise FileNotFoundError(
            f"Label file not found:\n"
            f"{LABEL_PATH}"
        )

    with open(
        LABEL_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        raw = json.load(file)


    examples = []


    # --------------------------------------------------------
    # LIST FORMAT
    # --------------------------------------------------------

    if isinstance(raw, list):

        for item in raw:

            if not isinstance(item, dict):
                continue

            query = (
                item.get("instruction")
                or item.get("query")
                or item.get("prompt")
            )

            complexity = (
                item.get("complexity")
                or item.get("label")
            )

            if (
                query
                and complexity in {
                    "simple",
                    "medium",
                    "complex",
                }
            ):

                examples.append(
                    (
                        str(query),
                        str(complexity),
                    )
                )


    # --------------------------------------------------------
    # DICTIONARY FORMAT
    # --------------------------------------------------------

    elif isinstance(raw, dict):

        for query, value in raw.items():

            if isinstance(value, dict):

                complexity = (
                    value.get("complexity")
                    or value.get("label")
                )

            else:

                complexity = value


            if complexity in {
                "simple",
                "medium",
                "complex",
            }:

                examples.append(
                    (
                        str(query),
                        str(complexity),
                    )
                )


    if not examples:

        raise ValueError(
            "No valid query → complexity "
            "examples found."
        )

    return examples


# ============================================================
# BUILD MODEL
# ============================================================

def main():

    print("=" * 100)
    print("BUILD V2 ARTIFACT FOR V7.1")
    print("=" * 100)


    examples = load_examples()


    queries = [
        item[0]
        for item in examples
    ]

    labels = [
        item[1]
        for item in examples
    ]


    print(
        f"\nTotal examples: "
        f"{len(queries)}"
    )


    distribution = Counter(
        labels
    )


    print("\nDistribution:")

    for label in [
        "simple",
        "medium",
        "complex",
    ]:

        print(
            f"  {label:<8}: "
            f"{distribution[label]}"
        )


    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    (
        train_queries,
        validation_queries,
        train_labels,
        validation_labels,
    ) = train_test_split(
        queries,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )


    print(
        f"\nTraining samples: "
        f"{len(train_queries)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_queries)}"
    )


    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    embedding_model_name = (
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )


    print(
        "\nLoading:"
    )

    print(
        embedding_model_name
    )


    encoder = SentenceTransformer(
        embedding_model_name
    )


    print(
        "\nGenerating training embeddings..."
    )


    X_train = encoder.encode(
        train_queries,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


    print(
        "\nGenerating validation embeddings..."
    )


    X_validation = encoder.encode(
        validation_queries,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )


    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )


    classifier.fit(
        X_train,
        train_labels,
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    predictions = classifier.predict(
        X_validation
    )


    accuracy = accuracy_score(
        validation_labels,
        predictions,
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "V2 VALIDATION"
    )

    print(
        "=" * 80
    )


    print(
        f"Accuracy: {accuracy:.4f}"
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    joblib.dump(
        classifier,
        CLASSIFIER_PATH,
    )


    metadata = {

        "version":
            "v2_1",

        "embedding_model":
            embedding_model_name,

        "classifier":
            "LogisticRegression",

        "training_examples":
            len(train_queries),

        "validation_examples":
            len(validation_queries),

        "validation_accuracy":
            float(accuracy),

        "classes":
            [
                "simple",
                "medium",
                "complex",
            ],
    }


    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


    print(
        "\nClassifier saved:"
    )

    print(
        CLASSIFIER_PATH
    )


    print(
        "\nMetadata saved:"
    )

    print(
        METADATA_PATH
    )


    print(
        "\nV2 artifact build complete."
    )


if __name__ == "__main__":

    main()