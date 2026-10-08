from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    "/home/sagemaker-user/adaptive-llm-routing"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from app.router.v7_1 import (
    V71Router,
)


RESULT_DIR = (
    PROJECT_ROOT
    / "notebooks"
    / "data"
    / "evaluation"
    / "v7"
    / "v7_1"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RESULT_PATH = (
    RESULT_DIR
    / "v7_1_test_results.json"
)


PROMPTS = [

    "What is Python?",

    "What is the capital of France?",

    "Whats the nearest mall",

    "Explain recursion.",

    "Compare Kubernetes and Docker.",

    (
        "Write a Python function "
        "to reverse a linked list."
    ),

    (
        "Design a fraud detection "
        "system for a bank."
    ),

    (
        "Design a distributed system "
        "that can handle 100 million users."
    ),
]


def main():

    print("=" * 100)
    print("V7.1 END-TO-END TEST")
    print("=" * 100)


    router = V71Router()


    results = []


    for index, prompt in enumerate(
        PROMPTS,
        start=1,
    ):

        print("\n")
        print("-" * 100)

        print(
            f"TEST {index}"
        )

        print("-" * 100)


        print(
            "\nPrompt:",
            prompt
        )


        result = router.route(
            prompt
        )


        results.append(
            result
        )


        print(
            "\nV2 complexity:",
            result[
                "complexity"
            ]
        )


        print(
            "V2 confidence:",
            f"{result['confidence']:.4f}"
        )


        print(
            "\nV6 model:",
            result[
                "selected_model"
            ]
        )


        print(
            "Decision:",
            result[
                "decision"
            ]
        )


        print(
            "Epsilon:",
            f"{result['epsilon']:.4f}"
        )


        print(
            "\nBedrock:",
            result[
                "model_id"
            ]
        )


        print(
            "Success:",
            result[
                "success"
            ]
        )


        print(
            "Latency:",
            f"{result['latency_ms']:.2f} ms"
        )


        print(
            "Input tokens:",
            result[
                "input_tokens"
            ]
        )


        print(
            "Output tokens:",
            result[
                "output_tokens"
            ]
        )


        print(
            "\nResponse:"
        )


        print(
            result[
                "response"
            ]
        )


        if result["error"]:

            print(
                "\nERROR:"
            )

            print(
                result["error"]
            )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )


    print("\n")
    print("=" * 100)

    print(
        "V7.1 TEST COMPLETE"
    )

    print("=" * 100)


    print(
        "\nResults saved:"
    )

    print(
        RESULT_PATH
    )


if __name__ == "__main__":

    main()