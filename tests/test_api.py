from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_prediction_endpoint():
    # Create a dummy image for testing
    from PIL import Image
    import io
    
    file = io.BytesIO()
    image = Image.new('RGB', (224, 224), color='red')
    image.save(file, 'jpeg')
    file.seek(0)
    
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", file, "image/jpeg")}
    )
    
    assert response.status_code == 200
    assert "result" in response.json()
    assert "confidence" in response.json()