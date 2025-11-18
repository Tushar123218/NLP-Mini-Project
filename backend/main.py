# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from contextlib import asynccontextmanager
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"  
MAX_LEN = 256

emoji_map = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲",
}

tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    print(" Loading DistilBERT emotion model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")

    
    yield


    print("Shutting down — releasing model resources...")



app = FastAPI(
    title="Mental Health Sentiment Analyzer (Transformer)",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this to your frontend domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Request Model --------
class InputText(BaseModel):
    text: str


# -------- Routes --------
@app.get("/")
def root():
    return {"message": "Transformer-based Mental Health Analyzer is up!"}


@app.post("/analyze")
def analyze(input: InputText):
    if not input.text or not input.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")

    enc = tokenizer(
        input.text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}


    with torch.no_grad():
        logits = model(**enc).logits

    probs = F.softmax(logits, dim=-1)[0].cpu().tolist()
    pred_idx = int(torch.argmax(logits, dim=-1).cpu().item())
    pred_label = model.config.id2label[pred_idx]
    confidence = round(100 * probs[pred_idx], 2)

   
    prob_dict: Dict[str, float] = {
        model.config.id2label[i]: round(100 * probs[i], 2)
        for i in range(len(probs))
    }

    return {
        "prediction": pred_label,
        "emoji": emoji_map.get(pred_label, ""),
        "confidence": confidence,
        "probabilities": prob_dict,
    }
