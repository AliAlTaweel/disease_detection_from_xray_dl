from fastapi import FastAPI, UploadFile, File, HTTPException
from app.services.model_service import ModelService
import os

app = FastAPI(title="Pneumonia Detection API")

# Initialize service (In production, use env variables for path)
MODEL_PATH = "models/pneumonia_model.v1.h5"
model_service = ModelService(MODEL_PATH)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    try:
        image_bytes = await file.read()
        results = model_service.predict(image_bytes)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))