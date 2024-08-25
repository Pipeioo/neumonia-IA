"""
from cmath import log
from typing import Optional
from urllib.request import Request



from tensorflow.python.keras.layers import LSTM, Dense
from fastapi.middleware.cors import CORSMiddleware
"""
from fastapi import FastAPI
from PIL import Image, ImageFile
app = FastAPI()

"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""
import keras
"""from keras._tf_keras.keras.preprocessing.image import load_img"""
from tensorflow.python.keras.models import Sequential, load_model
from keras.src.utils.image_utils import load_img

@app.get("/")
async def root():
    return {"message": "Hello World"}

def procesoImagen(urlImagen):

    model = load_model('model/modelo.h5')
    imagen = load_img(urlImagen, target_size=(150, 150))
    imagen_array2 = img_to_array(imagen)
    imagen_array2 = np.expand_dims(imagen_array2, axis=0)  # Añadir una dimensión para el batch
    imagen_array2 /= 255.0  # Escalar los valores de los píxeles

    # Hacer la predicción
    prediccion = model.predict(imagen_array2)


    return prediccion * 100

@app.get("/modeloIA")
async def get_image(imagen):
    prediccion = procesoImagen(imagen)
    return {"prediccion" : prediccion}


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

from cmath import log
from distutils.log import Log
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

modelo = load_model('models/modelo4.h5')

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/rutamodelo/modelo")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        image_np = np.array(image)

        predictions = modelo.predict(image_np)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return {"predicted_class": int(predicted_class)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

"""