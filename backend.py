from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import io
import json
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Garbage Classification System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and configurations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = None
class_names = []
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
client = AsyncOpenAI(api_key=api_key, base_url=base_url) if api_key and api_key != "your_openai_api_key_here" else None

# Define CV transforms
cv_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.on_event("startup")
async def startup_event():
    global model, class_names
    try:
        # Load class names
        with open("class_names.json", "r") as f:
            class_names = json.load(f)
            
        # Initialize and load model
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
        
        if os.path.exists("best_model.pth"):
            model.load_state_dict(torch.load("best_model.pth", map_location=device))
        else:
            print("Warning: best_model.pth not found. Model is untrained.")
            
        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully with classes: {class_names}")
    except Exception as e:
        print(f"Error loading model: {e}")

class ClassificationResult(BaseModel):
    label: str
    confidence: float
    instructions: str
    upcycling: str

@app.post("/classify", response_model=ClassificationResult)
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    try:
        # 1. Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 2. Vision Inference
        input_tensor = cv_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        label = class_names[predicted_idx.item()]
        conf_value = confidence.item()
        
        if conf_value < 0.5:
            return ClassificationResult(
                label="Unknown",
                confidence=conf_value,
                instructions="The image is not clear enough. Please take another picture of the item.",
                upcycling="N/A"
            )
            
        # 3. LLM Inference
        prompt = f"""你是一个环保专家。用户上传的垃圾被系统识别为 {label}。
请提供2-3步简短、明确的回收指导，并提供一个创意升级改造（Upcycling）建议。
请以JSON格式返回：
{{
    "instructions": "回收指导文本...",
    "upcycling": "升级改造建议..."
}}"""
        
        if client:
            try:
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={ "type": "json_object" }
                )
                llm_result = json.loads(response.choices[0].message.content)
                instructions = llm_result.get("instructions", "No instructions generated.")
                upcycling = llm_result.get("upcycling", "No upcycling ideas generated.")
            except Exception as llm_err:
                print(f"LLM Error: {llm_err}")
                instructions = f"Standard recycling procedure for {label}. (Please check OpenAI API Key)"
                upcycling = "Consider reusing it creatively. (Please check OpenAI API Key)"
        else:
            instructions = f"Please configure your OpenAI API Key to get detailed instructions for {label}."
            upcycling = "Please configure your OpenAI API Key to get upcycling ideas."
            
        return ClassificationResult(
            label=label,
            confidence=conf_value,
            instructions=instructions,
            upcycling=upcycling
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
