import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List
import numpy as np
from ..utils.preprocess import preprocess_text

class TextEmotionModel:
    def __init__(self, model_name: str = \"bhadresh-savani/distilbert-base-uncased-emotion\"):
        \"\"\"Initialize pre-trained text emotion model.\"\"\"

        self.device = \"cuda\" if torch.cuda.is_available() else \"cpu\"
        print(f\"Loading text model on {self.device}...\")
        
        # Use pipeline for simplicity (auto-downloads)
        self.classifier = pipeline(
            \"text-classification\",
            model=model_name,
            tokenizer=model_name,
            device=0 if self.device == \"cuda\" else -1,
            return_all_scores=True
        )
        
        self.emotions = [
            \"anger\", \"fear\", \"joy\", \"love\", \"sadness\", 
            \"surprise\"  # Matches model output
        ]
        
        # Map to 7 standard
        self.emotion_map = {
            \"anger\": \"angry\",
            \"fear\": \"fearful\",
            \"joy\": \"happy\",
            \"love\": \"happy\",
            \"sadness\": \"sad\",
            \"surprise\": \"surprised\",
            \"neutral\": \"neutral\"  # Fallback
        }

    def predict(self, text: str) -> Dict[str, float]:
        \"\"\"Predict emotions for text. Returns normalized scores.\"\"\"

        processed_text = preprocess_text(text)
        
        # Get predictions
        results = self.classifier(processed_text)[0]
        
        # Extract scores
        scores = {self.emotion_map.get(r['label'], 'neutral'): r['score'] for r in results}
        
        # Standardize to 7 emotions
        std_scores = {emo: scores.get(emo, 0.0) for emo in self.emotions}
        std_scores['neutral'] = 1.0 - sum(std_scores.values())  # Remainder
        
        # Normalize
        total = sum(std_scores.values())
        if total > 0:
            std_scores = {k: v/total for k, v in std_scores.items()}
        
        dominant = max(std_scores, key=std_scores.get)
        confidence = std_scores[dominant]
        
        return {
            \"dominant\": dominant,
            \"confidence\": confidence,
            \"scores\": std_scores
        }

# Global instance
text_model = None

def get_text_model():
    global text_model
    if text_model is None:
        text_model = TextEmotionModel()
    return text_model
