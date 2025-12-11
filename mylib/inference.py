import onnxruntime as ort
import numpy as np
import json
from PIL import Image

class OnnxInference:
    def __init__(self, model_path="model.onnx", classes_path="classes.json"):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        with open(classes_path, "r", encoding="utf-8") as f: self.classes = json.load(f)["classes"]

    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img_data = (np.array(img).astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: np.expand_dims(img_data.transpose(2, 0, 1), axis=0)})
        return self.classes[np.argmax(outputs[0])]

