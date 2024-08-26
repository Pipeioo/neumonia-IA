"""
from cmath import log
from typing import Optional
from urllib.request import Request
from tensorflow.python.keras.layers import LSTM, Dense
from fastapi.middleware.cors import CORSMiddleware
"""
from fastapi import FastAPI
from PIL import Image, UnidentifiedImageError

app = FastAPI()

import keras
from keras._tf_keras.keras.preprocessing.image import img_to_array
"from keras.src.utils.image_utils import img_to_array"
from tensorflow.python.keras.models import load_model
from io import BytesIO
import numpy as np
import requests

@app.get("/")
async def root():
    return {"message": "Hello World"}

def procesoImagen(urlImagen):

    model = load_model('model/modelo.h5')

    """
    CODIGO SOLO PARA URL LOCAL:
    imagen = load_img(urlImagen, target_size=(150, 150))
    imagen_array2 = img_to_array(imagen)
    imagen_array2 = np.expand_dims(imagen_array2, axis=0)  # Añadir una dimensión para el batch
    imagen_array2 /= 255.0  # Escalar los valores de los píxeles
    """
    "CODIGO NUEVO QUE TOMA URL ONLINE"
    
    response = requests.get(urlImagen)
    imagen = Image.open(BytesIO(response.content))
    imagen = imagen.convert('RGB')

    imagen = imagen.resize((150, 150))  # Ajusta el tamaño según las dimensiones esperadas por tu modelo
    imagen_array = img_to_array(imagen)
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir una dimensión para el batch
    imagen_array /= 255.0  # Escalar los valores de los píxeles

    # Hacer la predicción
    prediccion = model.predict(imagen_array)

    return prediccion * 100

@app.get("/modeloIA")
async def get_image(imagen):
    prediccion = procesoImagen(imagen)
    return {"prediccion" : prediccion}

"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""

"""
@app.post("/rutamodelo/modelo")
async def predict(request: Request):
    data = await request.varchar()
    
    
    return {"item_id": item_id, "q": q}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0", port=8000, reload=True)

async def xdd():
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image)
    return image_bytes

---
"""