import os
import json
from collections import Counter

import torch
from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from transformers import pipeline


class BaselineRouter:
    """
    V1 Baseline Router

    Offline:
        Dolly dataset
            ↓
        Zero-shot complexity labeling
            ↓
        Balanced weak-label dataset

    Runtime:
        Query
            ↓
        TF-IDF
            ↓
        Logistic Regression
            ↓
        simple / medium / complex
            ↓
        small / medium / large model
    """

    def __init__(
        self,
        label_cache_path="data/evaluation/v1_complexity_labels.json",
        max_label_samples=3000,
    ):

        self.label_cache_path = label_cache_path
        self.max_label_samples = max_label_samples

        # ==================================================
        # TF-IDF
        # ==================================================

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )

        # ==================================================
        # Logistic Regression
        # ==================================================

        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

        # ==================================================
        # Complexity → Model
        # ==================================================

        self.routing_table = {
            "simple": "small_model",
            "medium": "medium_model",
            "complex": "large_model",
        }

        # ==================================================
        # Runtime cache
        # ==================================================

        self.classification_cache = {}

        # ==================================================
        # Offline labeler
        # ==================================================

        self.labeler = None

        # ==================================================
        # Training state
        # ==================================================

        self.is_trained = False

    # ======================================================
    # LOAD DOLLY
    # ======================================================

    def load_data(self):

        print("Loading Dolly dataset...")

        dataset = load_dataset(
            "databricks/databricks-dolly-15k",
            split="train",
        )

        print(
            f"Dataset size: {len(dataset)}"
        )

        print(
            f"Columns: {dataset.column_names}"
        )

        return dataset

    # ======================================================
    # LOAD ZERO-SHOT LABELER
    # ======================================================

    def load_labeler(self):

        if self.labeler is not None:
            return self.labeler

        print(
            "\nLoading zero-shot complexity labeler..."
        )

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required. "
                "Install torch in the Jupyter environment."
            ) from exc

        model_name = (
            "typeform/distilbert-base-uncased-mnli"
        )

        self.labeler = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1,
        )

        return self.labeler

    # ======================================================
    # GENERATE ONE COMPLEXITY LABEL
    # ======================================================

    def generate_complexity_label(
        self,
        query,
    ):

        labeler = self.load_labeler()

        # IMPORTANT:
        # We give the NLI model semantic descriptions
        # rather than bare words such as "simple".
        candidate_labels = [
            (
                "a straightforward request that can be "
                "answered with a short factual response "
                "or a simple transformation"
            ),
            (
                "a moderately difficult request that "
                "requires explanation, multiple concepts, "
                "comparison, summarization, or several steps"
            ),
            (
                "a highly complex request that requires "
                "deep reasoning, system design, extensive "
                "technical work, multiple constraints, "
                "or a multi-step solution"
            ),
        ]

        result = labeler(
            query,
            candidate_labels=candidate_labels,
            hypothesis_template=(
                "This user request is {}."
            ),
            multi_label=False,
        )

        best_label = result["labels"][0]

        confidence = float(
            result["scores"][0]
        )

        # Map the semantic descriptions back to
        # the three V1 complexity classes.
        label_to_complexity = {
            candidate_labels[0]: "simple",
            candidate_labels[1]: "medium",
            candidate_labels[2]: "complex",
        }

        complexity = label_to_complexity[
            best_label
        ]

        return complexity, confidence

    # ======================================================
    # GENERATE OFFLINE LABELS
    # ======================================================

    def generate_complexity_labels(
        self,
        dataset,
    ):

        # --------------------------------------------------
        # Existing cache
        # --------------------------------------------------

        if os.path.exists(
            self.label_cache_path
        ):

            print(
                "\nExisting V1 labels found."
            )

            print(
                f"Loading: "
                f"{self.label_cache_path}"
            )

            with open(
                self.label_cache_path,
                "r",
                encoding="utf-8",
            ) as file:

                labels = json.load(file)

            print(
                f"Loaded {len(labels)} labeled examples."
            )

            return labels

        # --------------------------------------------------
        # Directory
        # --------------------------------------------------

        directory = os.path.dirname(
            self.label_cache_path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        # --------------------------------------------------
        # Sample
        # --------------------------------------------------

        if self.max_label_samples is not None:

            sample_size = min(
                self.max_label_samples,
                len(dataset),
            )

            dataset = dataset.select(
                range(sample_size)
            )

        print(
            f"\nGenerating complexity labels "
            f"for {len(dataset)} examples..."
        )

        # Initialize ONCE.
        self.load_labeler()

        labeled_data = []

        # --------------------------------------------------
        # Label
        # --------------------------------------------------

        for index, row in enumerate(dataset):

            query = row["instruction"]

            if not query:
                continue

            try:

                complexity, confidence = (
                    self.generate_complexity_label(
                        query
                    )
                )

                labeled_data.append(
                    {
                        "instruction": query,
                        "complexity": complexity,
                        "confidence": confidence,
                    }
                )

            except Exception as error:

                print(
                    f"Labeling error at "
                    f"index {index}: {error}"
                )

                continue

            if (
                (index + 1) % 100
                == 0
            ):

                print(
                    f"Labeled "
                    f"{index + 1}/"
                    f"{len(dataset)}"
                )

        if not labeled_data:

            raise RuntimeError(
                "No complexity labels were generated."
            )

        # --------------------------------------------------
        # Check distribution
        # --------------------------------------------------

        distribution = Counter(
            row["complexity"]
            for row in labeled_data
        )

        print(
            "\nRaw weak-label distribution:"
        )

        for label in [
            "simple",
            "medium",
            "complex",
        ]:

            print(
                f"  {label:<8}: "
                f"{distribution[label]}"
            )

        # --------------------------------------------------
        # Balance classes
        # --------------------------------------------------

        balanced_data = (
            self._balance_labels(
                labeled_data
            )
        )

        print(
            "\nBalanced weak-label distribution:"
        )

        balanced_distribution = Counter(
            row["complexity"]
            for row in balanced_data
        )

        for label in [
            "simple",
            "medium",
            "complex",
        ]:

            print(
                f"  {label:<8}: "
                f"{balanced_distribution[label]}"
            )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        with open(
            self.label_cache_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                balanced_data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "\nSaved V1 labels to:"
        )

        print(
            self.label_cache_path
        )

        return balanced_data

    # ======================================================
    # BALANCE LABELS
    # ======================================================

    def _balance_labels(
        self,
        labeled_data,
    ):

        grouped = {
            "simple": [],
            "medium": [],
            "complex": [],
        }

        for row in labeled_data:

            label = row["complexity"]

            if label in grouped:

                grouped[label].append(row)

        available_classes = [
            rows
            for rows in grouped.values()
            if rows
        ]

        if len(available_classes) < 3:

            raise RuntimeError(
                "The zero-shot labeler did not "
                "produce all three complexity classes."
            )

        target_size = min(
            len(rows)
            for rows in grouped.values()
        )

        balanced = []

        for label in [
            "simple",
            "medium",
            "complex",
        ]:

            balanced.extend(
                grouped[label][:target_size]
            )

        return balanced

    # ======================================================
    # PREPARE DATA
    # ======================================================

    def prepare_data(
        self,
        dataset,
    ):

        labeled_data = (
            self.generate_complexity_labels(
                dataset
            )
        )

        queries = []
        labels = []

        for row in labeled_data:

            query = row["instruction"]
            complexity = row["complexity"]

            if not query:
                continue

            if complexity not in {
                "simple",
                "medium",
                "complex",
            }:

                continue

            queries.append(query)
            labels.append(complexity)

        print(
            f"\nPrepared examples: "
            f"{len(queries)}"
        )

        distribution = Counter(labels)

        print(
            "\nComplexity distribution:"
        )

        for label in [
            "simple",
            "medium",
            "complex",
        ]:

            print(
                f"  {label:<8}: "
                f"{distribution[label]}"
            )

        return queries, labels

    # ======================================================
    # TRAIN V1
    # ======================================================

    def fit(
        self,
        queries,
        labels,
    ):

        print(
            "\nTraining V1..."
        )

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            queries,
            labels,
            test_size=0.20,
            random_state=42,
            stratify=labels,
        )

        print(
            f"Training examples: "
            f"{len(X_train)}"
        )

        print(
            f"Validation examples: "
            f"{len(X_test)}"
        )

        # --------------------------------------------------
        # TF-IDF
        # --------------------------------------------------

        print(
            "\nFitting TF-IDF..."
        )

        X_train_tfidf = (
            self.vectorizer.fit_transform(
                X_train
            )
        )

        X_test_tfidf = (
            self.vectorizer.transform(
                X_test
            )
        )

        print(
            "TF-IDF shape:",
            X_train_tfidf.shape,
        )

        # --------------------------------------------------
        # Logistic Regression
        # --------------------------------------------------

        print(
            "\nTraining Logistic Regression..."
        )

        self.classifier.fit(
            X_train_tfidf,
            y_train,
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        predictions = (
            self.classifier.predict(
                X_test_tfidf
            )
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "V1 VALIDATION RESULTS"
        )

        print(
            "=" * 60
        )

        print(
            f"Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            "\nClassification Report:"
        )

        print(
            classification_report(
                y_test,
                predictions,
                labels=[
                    "simple",
                    "medium",
                    "complex",
                ],
                zero_division=0,
            )
        )

        print(
            "Confusion Matrix:"
        )

        print(
            confusion_matrix(
                y_test,
                predictions,
                labels=[
                    "simple",
                    "medium",
                    "complex",
                ],
            )
        )

        self.is_trained = True

        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "y_test": y_test,
        }

    # ======================================================
    # CLASSIFY
    # ======================================================

    def classify(
        self,
        query,
    ):

        if not self.is_trained:

            raise RuntimeError(
                "Router has not been trained. "
                "Call fit() first."
            )

        # --------------------------------------------------
        # Cache
        # --------------------------------------------------

        if query in self.classification_cache:

            result = (
                self.classification_cache[
                    query
                ].copy()
            )

            result["cached"] = True

            return result

        # --------------------------------------------------
        # TF-IDF
        # --------------------------------------------------

        vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        # --------------------------------------------------
        # Predict
        # --------------------------------------------------

        complexity = (
            self.classifier.predict(
                vector
            )[0]
        )

        # --------------------------------------------------
        # Confidence
        # --------------------------------------------------

        probabilities = (
            self.classifier.predict_proba(
                vector
            )[0]
        )

        confidence = float(
            max(probabilities)
        )

        result = {
            "complexity": complexity,
            "confidence": confidence,
        }

        self.classification_cache[
            query
        ] = result

        result = result.copy()

        result["cached"] = False

        return result

    # ======================================================
    # ROUTE
    # ======================================================

    def route_model(
        self,
        query,
    ):

        result = self.classify(
            query
        )

        complexity = result[
            "complexity"
        ]

        model = self.routing_table[
            complexity
        ]

        return {
            "query": query,
            "complexity": complexity,
            "confidence": result[
                "confidence"
            ],
            "model": model,
            "cached": result[
                "cached"
            ],
        }