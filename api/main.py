from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras import models

app = FastAPI()
PRODMODEL = models.load_model("../models/1")
BETAMODEL = models.load_model("../models/2")
CLASS_NAME = ['Potato: Early_blight', 'Potato: Late_blight', 'Potato: healthy']

@app.get('/ping')
async def ping():
    return 'Hello world I am alive'

def imgToNumpy(data) -> np.ndarray:
    image =np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image = imgToNumpy(await file.read())
    image_batch=np.expand_dims(image, 0)
    prediction = PRODMODEL.predict(image_batch)
    index = np.argmax(prediction[0])
    confidence = np.max(prediction[0])*100
    predicted_class = CLASS_NAME[index]
    return {
        "Confidence": confidence, 
        "Class": predicted_class
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
    