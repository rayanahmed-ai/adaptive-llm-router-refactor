import boto3
import pytest
from fastapi.testclient import TestClient

from app.api_server import app


client = TestClient(app)


def _has_aws_credentials():
    try:
        return bool(
            boto3.session.Session().get_credentials()
        )
    except Exception:
        return False


HAS_AWS_CREDENTIALS = _has_aws_credentials()


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "adaptive-llm-routing"
    assert data["version"] == "V8.3"
    assert data["routing_version"] == "V7.7"
    assert data["aws_region"] == "eu-north-1"


def test_route_endpoint():
    response = client.post(
        "/api/route",
        json={"prompt": "What is Python?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["version"] == "V8.3"
    assert data["selected_model"] in {
        "small_model",
        "medium_model",
        "large_model",
    }
    assert data["decision_type"] in {
        "EXPLORATION",
        "EXPLOITATION",
    }
    assert data["candidate_models"] == [
        "small_model",
        "medium_model",
        "large_model",
    ]


def test_generate_endpoint():
    response = client.post(
        "/api/generate",
        json={
            "prompt": "What is Python?",
            "max_tokens": 32,
            "temperature": 0.1,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["selected_model"] in {
        "small_model",
        "medium_model",
        "large_model",
    }
    assert data["decision_type"] in {
        "EXPLORATION",
        "EXPLOITATION",
    }
    if not HAS_AWS_CREDENTIALS:
        # /api/generate still exercises the adaptive router, but the
        # Bedrock call itself cannot succeed without credentials.
        assert data["success"] is False
        assert data["response"] == ""
        return

    assert data["success"] is True
    assert data["response"]
