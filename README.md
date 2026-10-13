# Adaptive LLM Router

An adaptive Large Language Model (LLM) routing system that selects among model tiers based on routing policy and available model-performance signals.

## Project Features

- Adaptive model routing using the V7.7 routing policy.
- Amazon Bedrock integration with Nova Micro, Nova Lite, and Nova Pro.
- V8.3 routing API.
- V8.4 GitHub issue-to-PR automation.
- File-action execution and automated test validation.

## Architecture

GitHub Issues trigger the V8.4 agent through GitHub Actions. The agent uses the routing API, invokes the selected model through the configured model registry, applies permitted file changes, runs configured tests, and opens a pull request when validation succeeds.

## Safety

The agent should restrict file operations to permitted repository paths and require validation before publishing changes.

## Testing

Run the API tests with:

python -m pytest -q tests/test_v8_api.py

## Project Status

This is an evolving project. Benchmark results and cost measurements should be reported with their evaluation conditions and limitations.

Requirements:
- Replace README.md completely with the content above.
- Edit README.md only.
- Do not modify application code, tests, configuration, dependencies, or routing behavior.
- Run the configured API tests.
- Open a pull request only if the README-only change succeeds and the tests pass.