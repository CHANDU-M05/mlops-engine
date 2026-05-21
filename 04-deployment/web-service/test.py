"""
test.py — quick smoke test for the prediction service
Run: python test.py  (while uvicorn is running on port 9696)
"""

import requests

BASE = "http://localhost:9696"

def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    print("✅ health:", r.json())

def test_single():
    payload = {
        "PULocationID": "130",
        "DOLocationID": "205",
        "trip_distance": 3.5,
    }
    r = requests.post(f"{BASE}/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "duration_minutes" in data
    print("✅ single predict:", data)

def test_batch():
    payload = [
        {"PULocationID": "130", "DOLocationID": "205", "trip_distance": 3.5},
        {"PULocationID": "82",  "DOLocationID": "10",  "trip_distance": 1.2},
    ]
    r = requests.post(f"{BASE}/predict/batch", json=payload)
    assert r.status_code == 200
    print("✅ batch predict:", r.json())

if __name__ == "__main__":
    test_health()
    test_single()
    test_batch()
    print("\n All tests passed.")
