// API Configuration
const API_BASE = 'http://localhost:8000'; // Change for deployment
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let chart;
let results = {
    text: null,
    speech: null,
    face: null,
    video: null
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initWebcam();
    initChart();
});

// Tab Switching
function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    event.target.classList.add('active');
    document.getElementById(tab + '-tab').classList.add('active');
}

// Charts
function initChart() {
    const ctx = document.getElementById('emotion-chart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Happy', 'Sad', 'Angry', 'Neutral', 'Fear', 'Surprise', 'Disgusted'],
            datasets: [{
                data: [0,0,0,0,0,0,0],
                backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#C9CBCF']
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { position: 'bottom' } }
        }
    });
}

function updateChart(emotions) {
    const data = Object.values(emotions);
    chart.data.datasets[0].data = data;
    chart.update();
}

// API Calls
async function postData(endpoint, data) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        alert('Backend not running? Start `uvicorn backend.app:app --reload`');
    }
}

// Text Detection
async function detectText() {
    const text = document.getElementById('text-input').value;
    if (!text) return alert('Enter text first!');
    
    const result = await postData('/detect/text', { text });
    results.text = result;
    document.getElementById('emotion-report').innerHTML = `Text: ${result.dominant} (${Math.round(result.confidence * 100)}%)`;
    updateChart(result.scores);
}

// Speech Detection
function toggleRecord() {
    if (!isRecording) {
        navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks);
                audioChunks = [];
                detectSpeech(audioBlob);
            };
            mediaRecorder.start();
            isRecording = true;
            document.getElementById('record-btn').textContent = 'Stop Recording';
        });
    } else {
        mediaRecorder.stop();
        isRecording = false;
        document.getElementById('record-btn').textContent = 'Record Speech';
    }
}

async function detectSpeech(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'speech.wav');
    
    const response = await fetch(`${API_BASE}/detect/speech`, {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    results.speech = result;
    document.getElementById('emotion-report').innerHTML += `<br>Speech: ${result.dominant}`;
    updateChart(result.scores);
}

// Face Detection (Webcam/Image)
function initWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        document.getElementById('webcam').srcObject = stream;
    }).catch(err => console.log('Webcam access denied'));
}

async function detectFace() {
    const canvas = document.getElementById('face-canvas');
    const video = document.getElementById('webcam');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    const imgData = canvas.toDataURL('image/jpeg');
    const result = await postData('/detect/face', { image: imgData });
    results.face = result;
    document.getElementById('emotion-report').innerHTML += `<br>Face: ${result.dominant}`;
    updateChart(result.scores);
}

// Video Detection
async function detectVideo() {
    const file = document.getElementById('video-upload').files[0];
    if (!file) return alert('Select video!');
    
    const formData = new FormData();
    formData.append('video', file);
    
    const response = await fetch(`${API_BASE}/detect/video`, {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    results.video = result;
    document.getElementById('emotion-report').innerHTML += `<br>Video: ${result.dominant}`;
    updateChart(result.scores);
}

// Fusion
async function fusionAnalyze() {
    const fusionResult = await postData('/fusion', results);
    updateChart(fusionResult.scores);
    document.getElementById('emotion-report').innerHTML = `
        <strong>Final: ${fusionResult.dominant} (${Math.round(fusionResult.confidence * 100)}%)</strong>
        <br>Stability: ${fusionResult.stability}/1.0
        <br>Modalities: Text(${results.text?.confidence||0}), Speech(${results.speech?.confidence||0}), Face(${results.face?.confidence||0}), Video(${results.video?.confidence||0})
    `;
}
