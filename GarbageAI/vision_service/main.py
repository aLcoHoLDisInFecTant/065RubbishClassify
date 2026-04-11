from fastapi import FastAPI, File, UploadFile, HTTPException
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
from pydantic import BaseModel

app = FastAPI(title="Vision Service - Garbage Classification")

ONNX_MODEL_PATH = "d:/065创新/GarbageAI/vision_service/resnet50_garbage.onnx"
CLASSES_PATH = "d:/065创新/GarbageAI/vision_service/classes.txt"

# Load Model
try:
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name
    
    with open(CLASSES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"Failed to load model or classes: {e}")
    class_names = []

class PredictionResponse(BaseModel):
    label: str
    confidence: float

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        
        # Normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # NHWC to NCHW
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not class_names:
        raise HTTPException(status_code=500, detail="Model is not loaded properly.")
        
    try:
        contents = await file.read()
        input_data = preprocess_image(contents)
        
        # Inference
        outputs = session.run(None, {input_name: input_data})
        logits = outputs[0][0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        predicted_idx = np.argmax(probabilities)
        label = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        return {"label": label, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
