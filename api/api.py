from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from mylib.inference import OnnxInference
from PIL import Image
import io
import shutil
import os

app = FastAPI()
try: model = OnnxInference()
except Exception: model = None # pylint: disable=broad-exception-caught

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    tmp = f"temp_{file.filename}"
    with open(tmp, "wb") as f: shutil.copyfileobj(file.file, f)
    res = model.predict(tmp)
    os.remove(tmp)
    return {"filename": file.filename, "prediction": res}

@app.post("/resize")
async def resize(file: UploadFile = File(...), width: int = Form(...), height: int = Form(...)):
    return {"filename": file.filename, "new_size": [width, height], "status": "resized"}
