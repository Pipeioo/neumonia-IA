"""
from typing import Optional
from urllib.request import Request
from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageFile
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import LSTM, Dense
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = load_model('modelo4.h5')

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/rutamodelo/modelo")
async def predict(request: Request):
    data = await request.varchar()
    
    
    return {"item_id": item_id, "q": q}

async def xdd():
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image)
    return image_bytes
    """

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import LSTM, Dense
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga el modelo de IA
model = load_model('modelo4.h5')

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/rutamodelo/modelo")
async def predict(file: UploadFile = File(...)):
    try:
        # Lee la imagen subida
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convertir imagen a numpy array
        image_np = np.array(image)

        # Realiza la predicción directamente con el modelo
        predictions = model.predict(image_np)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return {"predicted_class": int(predicted_class)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
