from fastapi.testclient import TestClient
from api.api import app
from PIL import Image
import io

client = TestClient(app)

def create_dummy_image_bytes():
    """Crea una imagen falsa en memoria tal como pide el PDF (Pág 6)."""
    # 1. Crear imagen RGB
    img = Image.new('RGB', (100, 100), color='red')
    # 2. Buffer de bytes
    img_bytes = io.BytesIO()
    # 3. Guardar como JPEG
    img.save(img_bytes, format='JPEG')
    # 4. Volver al inicio del puntero
    img_bytes.seek(0)
    return img_bytes

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_endpoint():
    img_bytes = create_dummy_image_bytes()
    # Enviar la imagen como multipart/form-data
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_resize_endpoint():
    img_bytes = create_dummy_image_bytes()
    # Enviar imagen (files) y dimensiones (data)
    response = client.post(
        "/resize",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"width": "50", "height": "50"}
    )
    assert response.status_code == 200
    assert response.json()["new_size"] == [50, 50]
