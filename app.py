import gradio as gr
import requests

RENDER_API_URL = "https://mlops-lab2-karim.onrender.com"  
# -------------------------------------------

def predict(image_file):
    if image_file is None:
        return "Please, upload a photo."
    
    try:
        with open(image_file, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{RENDER_API_URL}/predict", files=files)
            return response.json()
    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Upload a photo of a pet"),
    outputs="json",
    title="MLOps Lab 3 - Pet Classifier",
    description="Pet classifier trained with MobileNetV2."
)

if __name__ == "__main__":
    interface.launch()
