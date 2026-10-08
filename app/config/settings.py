from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"
V8_CONFIG_PATH = PROJECT_ROOT / "config" / "v8.yaml"


def _env(
    name: str,
    default: str | None = None,
) -> str | None:

    value = os.getenv(name)

    if value is None:
        return default

    value = value.strip()

    return value if value else default


def load_yaml(
    path: Path,
) -> dict[str, Any]:

    if not path.exists():

        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        data = yaml.safe_load(file)

    if not isinstance(data, dict):

        raise ValueError(
            f"Expected YAML object in {path}"
        )

    return data


def load_models() -> dict[str, dict[str, Any]]:

    data = load_yaml(
        MODELS_CONFIG_PATH
    )

    models = data.get(
        "models",
        {},
    )

    if not isinstance(models, dict):

        raise ValueError(
            "config/models.yaml must contain "
            "'models'."
        )

    enabled = {}

    for logical_name, config in models.items():

        if not isinstance(
            config,
            dict,
        ):
            continue

        if not config.get(
            "enabled",
            True,
        ):
            continue

        model_id = config.get(
            "model_id"
        )

        if not model_id:
            raise ValueError(
                f"Missing model_id for "
                f"{logical_name}"
            )

        enabled[
            logical_name
        ] = config

    if not enabled:

        raise ValueError(
            "No enabled Bedrock models "
            "are configured."
        )

    return enabled


def load_v8_config() -> dict[str, Any]:

    return load_yaml(
        V8_CONFIG_PATH
    )


class Settings:

    PROJECT_NAME = "adaptive-llm-routing"

    AWS_REGION = _env(
        "AWS_REGION",
        "eu-north-1",
    )

    DYNAMODB_TABLE = _env(
        "DYNAMODB_TABLE",
        "adaptive-llm-router-state",
    )

    S3_BUCKET = _env(
        "S3_BUCKET"
    )

    S3_PREFIX = _env(
        "S3_PREFIX",
        "adaptive-llm-router",
    )

    ENVIRONMENT = _env(
        "ENVIRONMENT",
        "development",
    )

    LOG_LEVEL = _env(
        "LOG_LEVEL",
        "INFO",
    )

    @classmethod
    def validate(cls) -> None:

        if not cls.AWS_REGION:

            raise ValueError(
                "AWS_REGION is required."
            )

        if not cls.DYNAMODB_TABLE:

            raise ValueError(
                "DYNAMODB_TABLE is required."
            )


def get_enabled_models():

    return load_models()