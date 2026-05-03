# Multi-Modal Emotion Detection System 🧠😊

A full-stack AI system that detects emotions from **text**, **speech**, **facial expressions**, and **video** using pre-trained ML models, with a fusion engine for consolidated reports.

## 🎯 Features
- **Multi-Modal**: Text (NLP), Speech (MFCC+CNN), Face (CNN), Video (frame analysis)
- **Fusion Engine**: Weighted combination of modalities
- **Real-time Dashboard**: Web UI with charts and reports
- **Easy Deploy**: Render/HuggingFace ready

## 🚀 Quick Start (Local)

1. **Clone & Setup**:
   ```
   git clone <repo>
   cd Mood Detector
   cd backend
   pip install -r requirements.txt
   ```

2. **Download Models** (first run):
   ```
   cd backend
   python -c \"from utils.datasets import download_models; download_models()\"
   ```

3. **Run Backend**:
   ```
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Open Frontend**:
   ```
   # Backend at http://localhost:8000
   # Open frontend/index.html in browser (connect to backend URL)
   ```

5. **Test API**:
   ```
   curl -X POST http://localhost:8000/detect/text -H \"Content-Type: application/json\" -d '{\"text\": \"I am so happy!\"}'
   ```

## 📁 Structure
```
backend/     # FastAPI + ML models
frontend/    # HTML/JS dashboard
models/      # Pre-trained weights
TODO.md      # Progress tracker
```

## 🛠️ Tech Stack
- **Backend**: FastAPI, TensorFlow, OpenCV, Librosa, Transformers
- **Frontend**: Vanilla JS, Chart.js, Media APIs
- **Models**: DistilBERT (text), CNN (speech/face/video)

## 🌐 Deployment (Render)
1. Push to GitHub
2. Render.com → New Web Service → Connect repo
3. Build: `pip install -r backend/requirements.txt`
4. Start: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
5. Frontend: Static site or same service /frontend/

**Frontend Static**: Host index.html/css/js on Netlify/Vercel, set API base URL.

## 🤖 Usage
- **UI**: Tabs for text/mic/webcam/upload → See live emotion pie chart + report
- **API Docs**: http://localhost:8000/docs (Swagger)

## 🔮 Future
- PyTorch conversion
- Video streaming
- User auth/history
- More emotions/context

## Issues?
Check TODO.md progress. Report bugs!

**License**: MIT
