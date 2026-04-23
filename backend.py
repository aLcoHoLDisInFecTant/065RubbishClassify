from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
import asyncio
import urllib.request
import urllib.error
import re
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

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
openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
use_hkbu_rest = bool(base_url and "/api/v0/rest/deployments/" in base_url)
client = (
    AsyncOpenAI(api_key=api_key, base_url=base_url)
    if api_key and api_key != "your_openai_api_key_here" and not use_hkbu_rest
    else None
)

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


def _extract_json_dict(text: str) -> dict:
    """Parse JSON from model output, even when wrapped by extra text."""
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No valid JSON found in LLM response")


def _hkbu_request_sync(prompt: str) -> dict:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "api-key": api_key or "",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


async def generate_recycling_text(prompt: str, label: str) -> tuple[str, str]:
    if use_hkbu_rest and api_key:
        try:
            hkbu_result = await asyncio.to_thread(_hkbu_request_sync, prompt)
            content = hkbu_result["choices"][0]["message"]["content"]
            parsed = _extract_json_dict(content)
            return (
                parsed.get("instructions", "No instructions generated."),
                parsed.get("upcycling", "No upcycling ideas generated."),
            )
        except Exception as llm_err:
            print(f"LLM Error: {llm_err}")
            return (
                f"Standard recycling procedure for {label}. (Please check HKBU API config)",
                "Consider reusing it creatively. (Please check HKBU API config)",
            )

    if client:
        try:
            response = await client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            parsed = _extract_json_dict(response.choices[0].message.content)
            return (
                parsed.get("instructions", "No instructions generated."),
                parsed.get("upcycling", "No upcycling ideas generated."),
            )
        except Exception as llm_err:
            print(f"LLM Error: {llm_err}")
            return (
                f"Standard recycling procedure for {label}. (Please check OpenAI API Key)",
                "Consider reusing it creatively. (Please check OpenAI API Key)",
            )

    return (
        f"Please configure your OpenAI API Key to get detailed instructions for {label}.",
        "Please configure your OpenAI API Key to get upcycling ideas.",
    )

@app.post("/classify", response_model=ClassificationResult)
async def classify_image(file: UploadFile = File(...), lang: str = Form("en")):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    lang = (lang or "en").lower()
    if lang not in {"en", "zh"}:
        lang = "en"
        
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
            low_conf_instructions = (
                "The image is not clear enough. Please take another picture of the item."
                if lang == "en"
                else "图片不够清晰，请重新拍摄后再试。"
            )
            low_conf_upcycling = "N/A" if lang == "en" else "不适用"
            return ClassificationResult(
                label="Unknown",
                confidence=conf_value,
                instructions=low_conf_instructions,
                upcycling=low_conf_upcycling
            )
            
        # 3. LLM Inference
        language_instruction = (
            "Please respond in English."
            if lang == "en"
            else "请使用简体中文回答。"
        )
        prompt = f"""You are an environmental expert. The uploaded item is classified as {label}.
Provide 2-3 short, clear recycling steps and one creative upcycling idea.
{language_instruction}
Return only valid JSON in this format:
{{
    "instructions": "recycling instructions text",
    "upcycling": "upcycling suggestion text"
}}"""
        
        instructions, upcycling = await generate_recycling_text(prompt, label)
            
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
