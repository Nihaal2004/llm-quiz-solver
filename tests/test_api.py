import os
os.environ["QUIZ_SECRET"] = "test-secret"
os.environ["DISABLE_WORKER"] = "1"  # do not launch Playwright

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("ok") is True

def test_invalid_json():
    r = client.post("/solve", data="not json")
    assert r.status_code == 400

def test_missing_fields():
    r = client.post("/solve", json={"email": "e"})
    assert r.status_code == 400

def test_wrong_secret():
    r = client.post("/solve", json={"email":"e","secret":"wrong","url":"https://x"})
    assert r.status_code == 403

def test_ok():
    r = client.post("/solve", json={"email":"e","secret":"test-secret","url":"https://x"})
    assert r.status_code == 200
    assert r.json()["status"] == "accepted"
