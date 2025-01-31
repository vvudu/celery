import io 
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_upscale(client):
    with open("upscale/lama_300px.png", "rb") as img:
        response = client.post("/upscale", data={"file": img})
    assert response.status_code == 202
    assert "task_id" in response.json

def test_task_status(client):
    response = client.get("/tasks/fake_task_id")
    assert response.status_code in [202, 500]

def test_processed_image(client):
    response = client.get("/processed/fake_task_id.png")
    assert response.status_code == 404