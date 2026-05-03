import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import librosa
import soundfile as sf
import cv2
import base64
from io import BytesIO
from PIL import Image
import torch
import os

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text: str) -> str:
    \"\"\"Clean and tokenize text for emotion detection.\"\"\"

    # Lowercase and remove special chars
    text = re.sub(r'[^a-zA-Z\\s]', '', text.lower())
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and stem
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

def preprocess_audio(audio_path: str, sr: int = 22050) -> np.ndarray:
    \"\"\"Extract MFCC features from audio.\"\"\"

    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Extract MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Aggregate (mean over time)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    # Add other features
    features = np.hstack([
        mfccs_mean,
        librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean(),
        librosa.feature.zero_crossing_rate(y)[0].mean(),
        librosa.feature.rms(y=y)[0].mean()
    ])
    
    return features

def audio_to_features(audio_data: bytes) -> np.ndarray:
    \"\"\"Process uploaded audio bytes to features.\"\"\"
 
    y, sr = librosa.load(BytesIO(audio_data), sr=22050)
    return preprocess_audio_from_array(y, sr)

def preprocess_audio_from_array(y: np.ndarray, sr: int) -> np.ndarray:
    \"\"\"MFCC + features from audio array.\"\"\"

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    return np.hstack([
        mfccs_mean,
        librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean(),
        librosa.feature.zero_crossing_rate(y)[0].mean(),
        librosa.feature.rms(y=y)[0].mean()
    ])

def preprocess_image(image_b64: str) -> np.ndarray:
    \"\"\"Preprocess base64 image for face detection.\"\"\"

    # Decode base64
    image_data = base64.b64decode(image_b64.split(',')[1])
    image = Image.open(BytesIO(image_data)).convert('RGB')
    
    # Resize to 48x48 (FER2013 standard)
    image = image.resize((48, 48))
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Grayscale if needed
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2)
    
    return img_array.reshape(48, 48, 1).astype(np.float32)

def detect_face_roi(image_b64: str) -> tuple:
    \"\"\"Detect face ROI from image (placeholder). Returns full image for now.\"\"\"

    img_array = preprocess_image(image_b64)
    return img_array, (0, 0, 48, 48)  # Full image ROI

def preprocess_video(video_path: str, frame_size: int = 48) -> List[np.ndarray]:
    \"\"\"Extract frames from video for emotion analysis.\"\"\"

    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize and preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (frame_size, frame_size))
        frame_norm = frame_resized.astype(np.float32) / 255.0
        
        frames.append(frame_norm)
    
    cap.release()
    return frames[::10]  # Every 10th frame for speed
