from pathlib import Path
from typing import Dict

import yaml


class ModelRegistry:

    def __init__(
        self,
        config_path: str = "config/models.yaml",
    ):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Model configuration not found: "
                f"{self.config_path}"
            )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as f:
            config = yaml.safe_load(f) or {}

        self.models: Dict = config.get(
            "models",
            {},
        )

        if not self.models:
            raise ValueError(
                "No models found in config/models.yaml"
            )

    def resolve(
        self,
        model_alias: str,
    ) -> str:

        if model_alias not in self.models:
            raise KeyError(
                f"Unknown model alias: {model_alias}"
            )

        model_id = self.models[
            model_alias
        ].get("model_id")

        if not model_id:
            raise ValueError(
                f"No model_id configured for "
                f"{model_alias}"
            )

        return model_id

    def get(
        self,
        model_alias: str,
    ) -> Dict:

        if model_alias not in self.models:
            raise KeyError(
                f"Unknown model alias: {model_alias}"
            )

        return self.models[model_alias]


registry = ModelRegistry()
