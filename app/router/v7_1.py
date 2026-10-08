from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import boto3
import joblib

from sentence_transformers import SentenceTransformer

from app.models.v7_bedrock import (
    V7BedrockClient,
)


class V71Router:
    """
    V7.1

    Prompt
      ↓
    V2 complexity classifier
      ↓
    V6 DynamoDB state
      ↓
    adaptive model selection
      ↓
    Amazon Bedrock
      ↓
    real response

    IMPORTANT:
    V7.1 is READ-ONLY against V6 DynamoDB.
    """

    COMPLEXITIES = {
        "simple",
        "medium",
        "complex",
    }

    MODELS = (
        "small_model",
        "medium_model",
        "large_model",
    )


    def __init__(
        self,
        project_root=(
            "/home/sagemaker-user/"
            "adaptive-llm-routing"
        ),
        region="eu-north-1",
        dynamodb_table=(
            "adaptive-llm-router-v6-state-753176172779"
        ),
    ):

        self.project_root = Path(
            project_root
        )

        self.region = region

        self.dynamodb_table_name = (
            dynamodb_table
        )


        # ====================================================
        # DYNAMODB
        # ====================================================

        self.dynamodb = boto3.resource(
            "dynamodb",
            region_name=region,
        )

        self.table = (
            self.dynamodb.Table(
                dynamodb_table
            )
        )


        # ====================================================
        # BEDROCK
        # ====================================================

        self.bedrock = (
            V7BedrockClient(
                region=region
            )
        )


        # ====================================================
        # V2
        # ====================================================

        self._load_v2()


    # ========================================================
    # LOAD V2
    # ========================================================

    def _load_v2(self):

        model_dir = (
            self.project_root
            / "models"
            / "v2_1"
        )

        classifier_path = (
            model_dir
            / "classifier.joblib"
        )

        metadata_path = (
            model_dir
            / "metadata.json"
        )


        if not classifier_path.exists():

            raise FileNotFoundError(
                "V2 classifier artifact "
                "was not found:\n"
                f"{classifier_path}\n\n"
                "Run:\n"
                "python "
                "scripts/build_v2_artifact.py"
            )


        self.classifier = joblib.load(
            classifier_path
        )


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        embedding_model = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )


        if metadata_path.exists():

            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as file:

                metadata = json.load(
                    file
                )

            embedding_model = (
                metadata.get(
                    "embedding_model",
                    embedding_model,
                )
            )


        print(
            "\nLoading V2 encoder:"
        )

        print(
            embedding_model
        )


        self.encoder = (
            SentenceTransformer(
                embedding_model
            )
        )


    # ========================================================
    # V2 CLASSIFICATION
    # ========================================================

    def classify(
        self,
        prompt: str,
    ) -> Dict[str, Any]:

        if not isinstance(
            prompt,
            str,
        ):

            raise TypeError(
                "Prompt must be a string."
            )


        prompt = prompt.strip()


        if not prompt:

            raise ValueError(
                "Prompt cannot be empty."
            )


        embedding = (
            self.encoder.encode(
                [prompt],
                normalize_embeddings=True,
            )
        )


        complexity = str(
            self.classifier.predict(
                embedding
            )[0]
        )


        probabilities = (
            self.classifier.predict_proba(
                embedding
            )[0]
        )


        probability_dict = {

            str(label):
                float(probability)

            for label, probability
            in zip(
                self.classifier.classes_,
                probabilities,
            )
        }


        confidence = float(
            max(probabilities)
        )


        if complexity not in (
            self.COMPLEXITIES
        ):

            raise ValueError(
                f"Invalid V2 complexity: "
                f"{complexity}"
            )


        return {

            "complexity":
                complexity,

            "confidence":
                confidence,

            "probabilities":
                probability_dict,
        }


    # ========================================================
    # READ V6 EPSILON
    # ========================================================

    def get_epsilon(self):

        response = (
            self.table.get_item(
                Key={
                    "state_key": "CONFIG"
                }
            )
        )

        item = response.get(
            "Item"
        )


        if not item:

            return 0.05


        return float(
            item.get(
                "epsilon",
                0.05,
            )
        )


    # ========================================================
    # READ V6 STATE
    # ========================================================

    def get_state(
        self,
        complexity: str,
        model: str,
    ):

        response = (
            self.table.get_item(
                Key={
                    "state_key":
                        f"STATE#"
                        f"{complexity}#"
                        f"{model}"
                }
            )
        )


        item = response.get(
            "Item"
        )


        if not item:

            return {

                "reward":
                    0.0,

                "count":
                    0,
            }


        return {

            "reward":
                float(
                    item.get(
                        "estimated_reward",
                        0.0,
                    )
                ),

            "count":
                int(
                    item.get(
                        "count",
                        0,
                    )
                ),
        }


    # ========================================================
    # V6 MODEL SELECTION
    # ========================================================

    def select_model(
        self,
        complexity: str,
    ):

        epsilon = (
            self.get_epsilon()
        )


        states = {

            model:
                self.get_state(
                    complexity,
                    model,
                )

            for model in self.MODELS
        }


        # ----------------------------------------------------
        # EXPLORATION
        # ----------------------------------------------------

        if (
            random.random()
            < epsilon
        ):

            selected_model = (
                random.choice(
                    self.MODELS
                )
            )

            decision = (
                "EXPLORATION"
            )


        # ----------------------------------------------------
        # EXPLOITATION
        # ----------------------------------------------------

        else:

            selected_model = max(

                self.MODELS,

                key=lambda model:
                    states[
                        model
                    ][
                        "reward"
                    ]
            )

            decision = (
                "EXPLOITATION"
            )


        return {

            "selected_model":
                selected_model,

            "decision":
                decision,

            "epsilon":
                epsilon,

            "states":
                states,
        }


    # ========================================================
    # FULL V7.1 ROUTE
    # ========================================================

    def route(
        self,
        prompt: str,
    ):

        # ----------------------------------------------------
        # V2
        # ----------------------------------------------------

        classification = (
            self.classify(
                prompt
            )
        )


        # ----------------------------------------------------
        # V6
        # ----------------------------------------------------

        routing = (
            self.select_model(
                classification[
                    "complexity"
                ]
            )
        )


        # ----------------------------------------------------
        # BEDROCK
        # ----------------------------------------------------

        inference = (
            self.bedrock.invoke(
                logical_model=
                    routing[
                        "selected_model"
                    ],

                prompt=
                    prompt,
            )
        )


        # ----------------------------------------------------
        # FINAL TRACE
        # ----------------------------------------------------

        return {

            "version":
                "v7.1",

            "prompt":
                prompt,

            "complexity":
                classification[
                    "complexity"
                ],

            "confidence":
                classification[
                    "confidence"
                ],

            "probabilities":
                classification[
                    "probabilities"
                ],

            "selected_model":
                routing[
                    "selected_model"
                ],

            "decision":
                routing[
                    "decision"
                ],

            "epsilon":
                routing[
                    "epsilon"
                ],

            "v6_state":
                routing[
                    "states"
                ],

            "model_id":
                inference[
                    "model_id"
                ],

            "response":
                inference[
                    "response"
                ],

            "success":
                inference[
                    "success"
                ],

            "latency_ms":
                inference[
                    "latency_ms"
                ],

            "input_tokens":
                inference[
                    "input_tokens"
                ],

            "output_tokens":
                inference[
                    "output_tokens"
                ],

            "total_tokens":
                inference[
                    "total_tokens"
                ],

            "error":
                inference[
                    "error"
                ],
        }