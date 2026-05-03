from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import numpy as np
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

app = FastAPI(title="Multi-Modal Emotion Detector API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class TextInput(BaseModel):
    text: str

class DetectionResult(BaseModel):
    dominant: str
    confidence: float
    scores: Dict[str, float]

class FusionInput(BaseModel):
    text: Optional[DetectionResult] = None
    speech: Optional[DetectionResult] = None
    face: Optional[DetectionResult] = None
    video: Optional[DetectionResult] = None

# Emotions list
EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Emotion Detector API ready!"}

@app.get("/")
async def root():
    return {"message": "Multi-Modal Emotion Detection API. Visit /docs for API docs."}

@app.post("/detect/text", response_model=DetectionResult)
async def detect_text(input: TextInput):
    from .models.text_model import get_text_model
    model = get_text_model()
    return DetectionResult(**model.predict(input.text))

@app.post("/detect/speech")
async def detect_speech(audio: UploadFile = File(...)):
    from .models.speech_model import get_speech_model
    import librosa
    from io import BytesIO
    
    # Read audio bytes
    audio_bytes = await audio.read()
    y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)
    
    model = get_speech_model()
    result = model.predict(y)
    return JSONResponse(result)

@app.post("/detect/face")
async def detect_face(image_b64: str = Form(...)):
    from .models.face_model import get_face_model
    model = get_face_model()
    result = model.predict(image_b64)
    return JSONResponse(result)

@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    # TODO: Implement video model
    scores = {emo: np.random.uniform(0, 0.35) for emo in EMOTIONS}
    dominant = max(scores, key=scores.get)
    
    return JSONResponse(DetectionResult(dominant=dominant, confidence=scores[dominant], scores=scores).dict())

@app.post("/fusion")
async def fusion_analyze(fusion_input: Dict[str, Any]):
    from .models.fusion import get_fusion_engine
    engine = get_fusion_engine()
    result = engine.predict(fusion_input)
    return JSONResponse(result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
