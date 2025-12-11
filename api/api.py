from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from mylib.inference import OnnxInference
from PIL import Image
import io
import shutil
import os

app = FastAPI()

# Inicializamos el modelo
print("Cargando modelo ONNX...")
try:
    model = OnnxInference()
    print("Modelo cargado correctamente.")
except Exception as e: # pylint: disable=broad-exception-caught
    print(f"Aviso: No se pudo cargar el modelo (¿Estás en test?): {e}")
    model = None

@app.get("/")
def read_root():
    return {"message": "API de Clasificación de Mascotas (Lab 3)"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        return {"error": "El modelo no está cargado."}
        
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        prediction = model.predict(temp_filename)
        return {
            "filename": file.filename, 
            "prediction": prediction
        }
    except Exception as e: # pylint: disable=broad-exception-caught
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/resize")
async def resize_endpoint(
    file: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...)
):
    try:
        # Leer la imagen en memoria
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Redimensionar
        resized_image = image.resize((width, height))
        
        # Guardar en buffer
        buf = io.BytesIO()
        resized_image.save(buf, format="JPEG")
        
        return {
            "filename": file.filename,
            "width": width,
            "height": height,
            "status": "resized",
            "new_size": [width, height]
        }
    except Exception as e: # pylint: disable=broad-exception-caught
        raise HTTPException(status_code=500, detail=str(e)) from e
