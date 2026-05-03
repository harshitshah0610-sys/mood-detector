import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from ..utils.preprocess import preprocess_image
import os

EMOTIONS_FACE = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

class FaceEmotionModel:
    def __init__(self, model_path: str = None):
        \"\"\"Initialize face emotion CNN model.\"\"\"

        self.emotions = EMOTIONS_FACE
        self.input_shape = (48, 48, 1)
        
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(\"Loaded pre-trained face model.\")
        else:
            print(\"Creating lightweight face model (FER2013 style)...\")
            self.model = self._build_model()
            self._demo_train()
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def _build_model(self):
        \"\"\"Build MiniXception-like CNN for 48x48 grayscale faces.\"\"\"

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            GlobalAveragePooling2D(),
            Dropout(0.25),
            
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(self.emotions), activation='softmax')
        ])
        
        return model

    def _demo_train(self):
        \"\"\"Demo training with random data. Real use: FER2013 dataset.\"""

        X = np.random.rand(1000, *self.input_shape)
        y = tf.keras.utils.to_categorical(np.random.randint(0, len(self.emotions), 1000), len(self.emotions))
        
        self.model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        print(\"Face model demo trained.\")

    def predict(self, image_b64: str) -> dict:
        \"\"\"Predict emotion from base64 image.\"\"\"

        img_array = preprocess_image(image_b64)
        X = np.expand_dims(img_array, axis=0)
        
        predictions = self.model.predict(X, verbose=0)[0]
        
        dominant_idx = np.argmax(predictions)
        dominant = self.emotions[dominant_idx]
        confidence = float(predictions[dominant_idx])
        
        scores = {self.emotions[i]: float(predictions[i]) for i in range(len(self.emotions))}
        
        return {
            \"dominant\": dominant,
            \"confidence\": confidence,
            \"scores\": scores
        }

# Global instance
face_model = None

def get_face_model(model_path: str = \"models/face_model.h5\"):
    global face_model
    if face_model is None:
        face_model = FaceEmotionModel(model_path)
    return face_model
