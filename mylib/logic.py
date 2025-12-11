import random
from PIL import Image

def predict_image_class(_image: Image.Image) -> str: 
    """
    Predice la clase de una imagen.
    Según el PDF, la clase se elige aleatoriamente entre un set fijo.
    """
    classes = ["dog", "cat", "car", "plane"]
    return random.choice(classes)

def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """
    Redimensiona una imagen al tamaño dado.
    """
    return image.resize((width, height))
