from typing import Optional
from urllib.request import Request
from fastapi import FastAPI, File, UploadFile
from PIL import Image
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


image_bytes = await file.read()
image = image.open(io.BytesIO(image_bytes))
image_np = np.array(image)