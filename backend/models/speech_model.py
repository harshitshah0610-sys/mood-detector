import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from ..utils.preprocess import preprocess_audio_from_array
import pickle
import os

EMOTIONS_SPEECH = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

class SpeechEmotionModel:
    def __init__(self, model_path: str = None):
        \"\"\"Initialize speech emotion CNN-LSTM model.\"\"\"

        self.emotions = EMOTIONS_SPEECH
        self.input_shape = (100,)  # MFCC + features padded
        
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(\"Loaded pre-trained speech model.\")
        else:
            print(\"Creating lightweight speech model (placeholder trained)...\")
            self.model = self._build_model()
            self._demo_train()  # Placeholder training
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def _build_model(self) -> Sequential:
        \"\"\"Build CNN-LSTM for speech features.\"\"\"

        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=(self.input_shape[0], 1)),
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dense(len(self.emotions), activation='softmax')
        ])
        
        return model

    def _demo_train(self):
        \"\"\"Placeholder demo training (random data). Real use: RAVDESS dataset.\"\"\"

        # Generate demo data
        X = np.random.rand(1000, self.input_shape[0], 1)
        y = tf.keras.utils.to_categorical(np.random.randint(0, len(self.emotions), 1000), len(self.emotions))
        
        self.model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        print(\"Demo training complete. Use real RAVDESS dataset for production.\")

    def predict(self, audio_array: np.ndarray) -> dict:
        \"\"\"Predict emotion from audio features.\"\"\"

        # Preprocess
        features = preprocess_audio_from_array(audio_array, sr=22050)
        
        # Pad/reshape
        feature_padded = np.pad(features, (0, self.input_shape[0] - len(features)), 'constant')
        X = feature_padded.reshape(1, -1, 1)
        
        # Predict
        predictions = self.model.predict(X, verbose=0)[0]
        
        dominant_idx = np.argmax(predictions)
        dominant = self.emotions[dominant_idx]
        confidence = float(predictions[dominant_idx])
        
        scores = {emo: float(predictions[i]) for i, emo in enumerate(self.emotions)}
        
        return {
            \"dominant\": dominant,
            \"confidence\": confidence,
            \"scores\": scores
        }

# Global instance
speech_model = None

def get_speech_model(model_path: str = \"models/speech_model.h5\"):
    global speech_model
    if speech_model is None:
        speech_model = SpeechEmotionModel(model_path)
    return speech_model
