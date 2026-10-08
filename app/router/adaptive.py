import os

import numpy as np



from sentence_transformers import SentenceTransformer



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import (

    accuracy_score,

    classification_report,

    confusion_matrix,

)



from app.router.baseline import BaselineRouter


from app.observability.logging import log_routing_decision





class AdaptiveRouterV2(BaselineRouter):

    """

    V2 Semantic Router



    V1:

        Query

          ↓

        TF-IDF

          ↓

        Logistic Regression

          ↓

        Complexity

          ↓

        Model



    V2:

        Query

          ↓

        Sentence Embedding

          ↓

        Logistic Regression

          ↓

        Complexity

          ↓

        Model



    V2 keeps:

        - V1 Dolly data

        - V1 complexity labels

        - V1 train/validation split

        - V1 routing table

        - V1 classification cache concept



    V2 changes:

        - TF-IDF → semantic embeddings

    """



    # ======================================================

    # INITIALIZATION

    # ======================================================



    def __init__(

        self,

       label_cache_path=(
    "notebooks/data/evaluation/"
    "v1_corrected_complexity_labels.json"
),

        max_label_samples=3000,

    ):



        # --------------------------------------------------

        # Reuse V1 initialization

        # --------------------------------------------------



        super().__init__(

            label_cache_path=label_cache_path,

            max_label_samples=max_label_samples,

        )



        # --------------------------------------------------

        # V2 embedding model

        # --------------------------------------------------



        self.embedding_model_name = (

            "sentence-transformers/"

            "all-MiniLM-L6-v2"

        )



        self.embedding_model = None



        # --------------------------------------------------

        # Classifier

        # --------------------------------------------------



        self.classifier = LogisticRegression(

            max_iter=1000,

            class_weight="balanced",

            random_state=42,

        )



        # --------------------------------------------------

        # Embedding configuration

        # --------------------------------------------------



        self.batch_size = 32



        # --------------------------------------------------

        # Version

        # --------------------------------------------------



        self.version = "V2"



    # ======================================================

    # LOAD EMBEDDING MODEL

    # ======================================================



    def load_embedding_model(self):



        if self.embedding_model is not None:

            return self.embedding_model



        print("\n" + "=" * 60)

        print("LOADING V2 EMBEDDING MODEL")

        print("=" * 60)



        print(

            f"\nModel: "

            f"{self.embedding_model_name}"

        )



        print(

            "Loading sentence-transformer..."

        )



        self.embedding_model = (

            SentenceTransformer(

                self.embedding_model_name

            )

        )



        print(

            "Embedding model loaded."

        )



        return self.embedding_model



    # ======================================================

    # ENCODE TEXT

    # ======================================================



    def encode(

        self,

        texts,

        show_progress=True,

    ):

        """

        Convert text into semantic embeddings.

        """



        model = (

            self.load_embedding_model()

        )



        embeddings = model.encode(

            texts,

            batch_size=self.batch_size,

            show_progress_bar=show_progress,

            convert_to_numpy=True,

            normalize_embeddings=True,

        )



        return embeddings



    # ======================================================

    # PREPARE DATA

    # ======================================================



    def prepare_v2_data(

        self,

        dataset,

    ):

        """

        Reuse the existing V1 labels.



        V2 does NOT create new complexity labels.

        """



        if not os.path.exists(

            self.label_cache_path

        ):



            raise FileNotFoundError(

                "\nV1 label file not found:\n"

                f"{self.label_cache_path}\n\n"

                "Make sure the V1 label file exists."

            )



        print(

            "\nUsing existing V1 complexity labels."

        )



        print(

            f"Label file: "

            f"{self.label_cache_path}"

        )



        # --------------------------------------------------

        # Reuse V1 data preparation

        # --------------------------------------------------



        queries, labels = (

            super().prepare_data(

                dataset

            )

        )



        print(

            f"\nV2 examples: "

            f"{len(queries)}"

        )



        return queries, labels



    # ======================================================

    # TRAIN V2

    # ======================================================



    def fit(

        self,

        queries,

        labels,

    ):

        """

        Train V2 using the same split as V1.

        """



        print("\n" + "=" * 60)

        print("TRAINING ADAPTIVE ROUTER V2")

        print("=" * 60)



        # --------------------------------------------------

        # Same train/validation split as V1

        # --------------------------------------------------



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

            f"\nTraining samples: "

            f"{len(X_train)}"

        )



        print(

            f"Validation samples: "

            f"{len(X_test)}"

        )



        # ==================================================

        # TRAINING EMBEDDINGS

        # ==================================================



        print(

            "\nGenerating training embeddings..."

        )



        X_train_embeddings = self.encode(

            X_train

        )



        print(

            "\nTraining embedding shape:"

        )



        print(

            X_train_embeddings.shape

        )



        # ==================================================

        # VALIDATION EMBEDDINGS

        # ==================================================



        print(

            "\nGenerating validation embeddings..."

        )



        X_test_embeddings = self.encode(

            X_test

        )



        print(

            "\nValidation embedding shape:"

        )



        print(

            X_test_embeddings.shape

        )



        # ==================================================

        # TRAIN CLASSIFIER

        # ==================================================



        print(

            "\nTraining Logistic Regression..."

        )



        self.classifier.fit(

            X_train_embeddings,

            y_train,

        )



        # ==================================================

        # VALIDATION

        # ==================================================



        predictions = (

            self.classifier.predict(

                X_test_embeddings

            )

        )



        accuracy = accuracy_score(

            y_test,

            predictions,

        )



        # ==================================================

        # RESULTS

        # ==================================================



        print("\n" + "=" * 60)

        print("V2 VALIDATION RESULTS")

        print("=" * 60)



        print(

            f"\nAccuracy: "

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

            "\nConfusion Matrix:"

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



        # ==================================================

        # TRAINING STATE

        # ==================================================



        self.is_trained = True



        return {

            "version": self.version,

            "accuracy": accuracy,

            "predictions": predictions,

            "actual": y_test,

        }



    # ======================================================

    # CLASSIFY

    # ======================================================



    def classify(

        self,

        query,

    ):

        """

        Classify a new query using V2 embeddings.

        """



        if not self.is_trained:



            raise RuntimeError(

                "V2 router must be trained "

                "before classification."

            )



        # --------------------------------------------------

        # CACHE

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

        # EMBED QUERY

        # --------------------------------------------------



        query_embedding = self.encode(

            [query],

            show_progress=False,

        )



        # --------------------------------------------------

        # PREDICT COMPLEXITY

        # --------------------------------------------------



        complexity = (

            self.classifier.predict(

                query_embedding

            )[0]

        )



        complexity = str(

            complexity

        )



        # --------------------------------------------------

        # PREDICTION PROBABILITIES

        # --------------------------------------------------



        probabilities = (

            self.classifier.predict_proba(

                query_embedding

            )[0]

        )



        probability_dict = {

            str(label): float(probability)

            for label, probability in zip(

                self.classifier.classes_,

                probabilities,

            )

        }



        # --------------------------------------------------

        # CONFIDENCE

        # --------------------------------------------------



        confidence = float(

            max(probabilities)

        )



        # --------------------------------------------------

        # RESULT

        # --------------------------------------------------



        result = {

            "version": self.version,

            "query": query,

            "complexity": complexity,

            "confidence": confidence,

            "probabilities": probability_dict,

            "cached": False,

        }



        # --------------------------------------------------

        # CACHE RESULT

        # --------------------------------------------------



        self.classification_cache[

            query

        ] = result.copy()



        return result



    # ======================================================

    # ROUTE MODEL

    # ======================================================



    def route_model(

        self,

        query,

    ):

        """

        Convert complexity into selected model.

        """



        result = self.classify(

            query

        )



        complexity = (

            result["complexity"]

        )



        selected_model = self.routing_table[
            complexity
        ]

        result["model"] = selected_model

        # --------------------------------------------------
        # V7.2 TELEMETRY
        # --------------------------------------------------
        # Record the routing decision here. The actual
        # Bedrock inference telemetry is recorded separately
        # by app/models/bedrock.py.

        try:
            log_routing_decision(
                prompt=query,
                complexity=complexity,
                confidence=result["confidence"],
                selected_model=selected_model,
            )
        except Exception as exc:
            # Telemetry must never break routing.
            print(
                f"V7.2 telemetry warning: {exc}"
            )

        return result
