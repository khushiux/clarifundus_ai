import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from retinatrust.api.main import app

def test_api_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "model_loaded" in data

def test_api_predict_valid_image():
    with TestClient(app) as client:
        # Check if model is loaded
        health = client.get("/health").json()
        if not health["model_loaded"]:
            pytest.skip("Model not loaded for predict test.")

        # Create synthetic valid PNG image in memory
        img_arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img_bytes = io.BytesIO()
        Image.fromarray(img_arr).save(img_bytes, format="PNG")
        img_bytes.seek(0)

        response = client.post(
            "/predict",
            files={"file": ("test_fundus.png", img_bytes, "image/png")}
        )

        assert response.status_code == 200
        data = response.json()
        assert "predicted_grade" in data
        assert 0 <= data["predicted_grade"] <= 4
        assert "estimated_confidence" in data
        assert "class_probabilities" in data
        assert "disclaimer" in data

def test_api_predict_invalid_extension():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("document.txt", b"plain text", "text/plain")}
        )
        assert response.status_code == 400
        assert "unsupported file extension" in response.json()["detail"].lower()

def test_api_predict_pdf_report():
    with TestClient(app) as client:
        health = client.get("/health").json()
        if not health["model_loaded"]:
            pytest.skip("Model not loaded for predict report test.")

        img_arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img_bytes = io.BytesIO()
        Image.fromarray(img_arr).save(img_bytes, format="PNG")
        img_bytes.seek(0)

        response = client.post(
            "/predict/report",
            files={"file": ("test_fundus.png", img_bytes, "image/png")}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF")
