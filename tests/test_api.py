"""
API tests. They load the real model, so they need the full requirements installed:

    pip install -r requirements-dev.txt
    pytest -v tests/
"""

import base64
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = Path(__file__).resolve().parent / "samples"
TEST_KEY = "test-key-123"

os.environ["API_KEY"] = TEST_KEY
os.chdir(REPO_ROOT)  # app.py loads model files by relative path
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402

client = TestClient(app_module.app)
HEADERS = {"x-api-key": TEST_KEY}

# Language of each sample clip, from its filename
LANGUAGE_PREFIXES = {"hindi": "Hindi", "tamil": "Tamil", "english": "English",
                     "malayalam": "Malayalam", "telugu": "Telugu"}
SAMPLE_FILES = sorted(SAMPLES.glob("*/*.mp3"))


def language_of(path):
    return LANGUAGE_PREFIXES[path.name.split("_")[0].lower()]


def b64(path):
    return base64.b64encode(path.read_bytes()).decode()


def body(path=None, **overrides):
    path = path or SAMPLE_FILES[0]
    data = {"language": language_of(path), "audioFormat": "mp3", "audioBase64": b64(path)}
    data.update(overrides)
    return data


# ---------- Health ----------
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    assert r.json()["model_loaded"] is True


# ---------- Authentication ----------
def test_missing_api_key_is_rejected():
    r = client.post("/api/voice-detection", json=body())
    assert r.status_code == 401
    assert r.json()["status"] == "error"


def test_wrong_api_key_is_rejected():
    r = client.post("/api/voice-detection", json=body(), headers={"x-api-key": "wrong"})
    assert r.status_code == 401


def test_service_refuses_to_start_without_api_key():
    env = {k: v for k, v in os.environ.items() if k != "API_KEY"}
    result = subprocess.run([sys.executable, "-c", "import app"], cwd=REPO_ROOT, env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode != 0
    assert "API_KEY" in result.stderr


# ---------- Input validation ----------
def test_unsupported_language():
    r = client.post("/api/voice-detection", json=body(language="French"), headers=HEADERS)
    assert r.status_code == 400
    assert r.json()["status"] == "error"


def test_non_mp3_format():
    r = client.post("/api/voice-detection", json=body(audioFormat="wav"), headers=HEADERS)
    assert r.status_code == 400


def test_invalid_base64():
    r = client.post("/api/voice-detection", json=body(audioBase64="not base64!!"), headers=HEADERS)
    assert r.status_code == 400


def test_bytes_that_are_not_audio():
    junk = base64.b64encode(b"this is not an mp3 file" * 50).decode()
    r = client.post("/api/voice-detection", json=body(audioBase64=junk), headers=HEADERS)
    assert r.status_code == 400
    assert r.json()["status"] == "error"


def test_missing_field():
    data = body()
    del data["audioBase64"]
    r = client.post("/api/voice-detection", json=data, headers=HEADERS)
    assert r.status_code == 422


# ---------- Predictions ----------
@pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.name)
def test_prediction_response_format(path):
    r = client.post("/api/voice-detection", json=body(path), headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert set(data) == {"status", "language", "classification", "confidenceScore", "explanation"}
    assert data["status"] == "success"
    assert data["language"] == language_of(path)
    assert data["classification"] in {"AI_GENERATED", "HUMAN"}
    assert 0.5 <= data["confidenceScore"] <= 1.0
    assert data["explanation"]


def test_same_input_gives_same_result():
    first = client.post("/api/voice-detection", json=body(), headers=HEADERS).json()
    second = client.post("/api/voice-detection", json=body(), headers=HEADERS).json()
    assert first == second
