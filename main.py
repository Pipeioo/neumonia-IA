from fastapi import FastAPI
from PIL import Image, UnidentifiedImageError
import keras
from keras._tf_keras.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model
from tensorflow.keras.models import load_model
from io import BytesIO
import numpy as np
import requests
import uvicorn
from os import getenv
from fastapi.middleware.cors import CORSMiddleware




app = FastAPI()

@app.get("/")
async def root():
    return {"/modeloIA?imagen=(tuURL)"}

    #TENGO QUE HOSTEAR LA IA EN ALGUN LADO ASI LA PUEDE LEVANTAR RENDER. GIT LFS

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
    return {prediccion}

if __name__ == "__main__":
    port = int(getenv("PORT", 8000))
    uvicorn.run("main:app",host="0.0.0.0",port=port,reload=True)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""
app.add_middleware(z
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/rutamodelo/modelo")
async def predict(request: Request):
    data = await request.varchar()
    
    
    return {"item_id": item_id, "q": q}
    

"""