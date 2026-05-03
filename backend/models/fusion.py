import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from typing import Dict, Any, List
import joblib
from ..utils.preprocess import preprocess_text  # For fusion features
import os

EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

class FusionEngine:
    def __init__(self, model_path: str = None):
        \"\"\"Multi-modal fusion using ML classifier on combined features.\"\"\"

        self.emotions = EMOTIONS
        self.modalities = ["text", "speech", "face", "video"]
        self.weights = {"text": 0.2, "speech": 0.25, "face": 0.35, "video": 0.18}
        
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            print(\"Creating fusion model...\")
            self.model = self._build_fusion_model()
            self._demo_fit()
    
    def _build_fusion_model(self):
        \"\"\"Random Forest for modality confidence fusion.\"\"\"

        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _demo_fit(self):
        \"\"\"Demo train fusion model.\"\"\"

        # Simulate multi-modal data
        X_demo = np.random.rand(1000, 28)  # 7 emotions * 4 modalities
        y_demo = np.random.randint(0, 7, 1000)
        self.model.fit(X_demo, y_demo)
        print(\"Fusion demo trained.\")

    def extract_fusion_features(self, results: Dict[str, Dict[str, Any]]) -> np.ndarray:
        \"\"\"Extract features from modality results.\"\"\"

        features = []
        for modality in self.modalities:
            if modality in results and results[modality]:
                scores = list(results[modality]['scores'].values())
                conf = results[modality]['confidence']
                # Weighted scores + conf as feature
                feat = np.array(scores) * conf * self.weights[modality]
                features.extend(feat)
            else:
                # Zero for missing
                features.extend([0.0] * 7)
        
        return np.array(features).reshape(1, -1)

    def predict(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        \"\"\"Fuse multi-modal results.\"\"\"

        # Simple weighted average (fallback)
        all_scores = np.zeros(len(self.emotions))
        total_weight = 0
        
        for modality, result in results.items():
            if result:
                w = self.weights.get(modality, 0.2)
                scores = np.array([result['scores'].get(emo, 0) for emo in self.emotions])
                all_scores += w * scores
                total_weight += w
        
        if total_weight > 0:
            all_scores /= total_weight
        
        # ML prediction
        features = self.extract_fusion_features(results)
        ml_pred = self.model.predict(features)[0]
        ml_scores = self.model.predict_proba(features)[0]
        
        # Ensemble: 70% weighted avg + 30% ML
        final_scores = 0.7 * all_scores + 0.3 * ml_scores
        
        dominant_idx = np.argmax(final_scores)
        dominant = self.emotions[dominant_idx]
        confidence = float(final_scores[dominant_idx])
        
        # Stability: variance of confidences
        stability = 1.0 - np.std(final_scores)
        
        scores_dict = {self.emotions[i]: float(final_scores[i]) for i in range(len(self.emotions))}
        
        return {
            \"dominant\": dominant,
            \"confidence\": confidence,
            \"scores\": scores_dict,
            \"stability\": float(stability)
        }

# Global instance
fusion_engine = None

def get_fusion_engine(model_path: str = \"models/fusion_model.pkl\"):
    global fusion_engine
    if fusion_engine is None:
        fusion_engine = FusionEngine(model_path)
    return fusion_engine
