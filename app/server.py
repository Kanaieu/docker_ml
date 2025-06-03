from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
import uvicorn
import os
import gdown

MODEL_PATH = "app/model.h5"
MODEL_ID = "1V1oPdLurrjk4so6PU8AE2DVuKpSZggQw"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

class_names = np.array(["Acne", "Actinic Keratosis", "Benign Tumors", "Bullous", "Candidiasis",
    "Drug Eruption", "Eczema", "Infestations/Bites", "Lichen", "Lupus",
    "Moles", "Psoriasis", "Rosacea", "Seborrheic Keratoses", "Skin Cancer",
    "Sun/Sunlight Damage", "Tinea", "Vascular Tumors", "Vasculitis", "Vitiligo",
    "Warts"])

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Welcome to DermaScan API"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    '''Predicts the skin condition from the provided image bytes.
    
    Args:
        image (bytes): The image bytes to be processed.
        
    Returns:
        dict: A dictionary containing the predicted class name.
    '''

    # Read image content
    contents = await image.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Make the prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    return {"prediction": class_names[predicted_class]}

port = int(os.environ.get("PORT", 10000))
