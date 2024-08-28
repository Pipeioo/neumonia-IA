"""
from cmath import log
from typing import Optional
from urllib.request import Request
from tensorflow.python.keras.layers import LSTM, Dense
from fastapi.middleware.cors import CORSMiddleware
"""
"""You should run it over following requirements

keras==2.2.4 tensorflow==1.15.0 pillow==7.0.0"""
import tensorflow as tf
from fastapi import FastAPI
from PIL import Image, UnidentifiedImageError
from keras._tf_keras.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
"from tensorflow.python.keras.models import load_model"
from io import BytesIO
import numpy as np
import requests

    

app = FastAPI()

@app.get("/")
async def root():
    return {"/modeloIA?imagen=(tuURL)"}

def procesoImagen(urlImagen):
    
    model = load_model('model/modelo4.h5')
    
    response = requests.get(urlImagen)
    imagen = Image.open(BytesIO(response.content))
    imagen = imagen.convert('RGB')

    imagen = imagen.resize((150, 150))  # Ajusta el tamaño según las dimensiones esperadas por tu modelo
    imagen_array = img_to_array(imagen)
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir una dimensión para el batch
    imagen_array /= 255.0  # Escalar los valores de los píxeles

    # Hacer la predicción

    prediccion = model.predict(imagen_array)[0][0]


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