let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let timerInterval;
let seconds = 0;
let audioContext;
let analyser;
let canvasCtx;

const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const fileInfo = document.getElementById('fileInfo');
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const playBtn = document.getElementById('playBtn');
const timer = document.getElementById('timer');
const visualizer = document.getElementById('visualizer');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const emotionResult = document.getElementById('emotionResult');
const confidenceResult = document.getElementById('confidenceResult');
const resultAudio = document.getElementById('resultAudio');

// Initialize audio visualization
function initVisualizer() {
    canvasCtx = visualizer.getContext('2d');
    visualizer.width = visualizer.offsetWidth;
    visualizer.height = visualizer.offsetHeight;
}

// File upload handling
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#764ba2';
    uploadArea.style.background = '#eef1ff';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = '#667eea';
    uploadArea.style.background = '#f8f9ff';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#667eea';
    uploadArea.style.background = '#f8f9ff';
    
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.includes('audio')) {
        alert('Please upload an audio file');
        return;
    }

    fileInfo.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px;">
            <i class="fas fa-file-audio" style="color: #667eea; font-size: 1.5rem;"></i>
            <div>
                <strong>${file.name}</strong>
                <div style="color: #666; font-size: 0.9rem;">
                    ${(file.size / 1024 / 1024).toFixed(2)} MB • 
                    ${file.type}
                </div>
            </div>
        </div>
    `;
    fileInfo.classList.add('show');

    // Analyze the file
    analyzeAudio(file);
}

// Audio recording
recordBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);
playBtn.addEventListener('click', playRecording);

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        // Set up audio visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        drawVisualizer();

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const audioFile = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
            handleFile(audioFile);
            resultAudio.src = URL.createObjectURL(audioBlob);
        };

        audioChunks = [];
        mediaRecorder.start();
        isRecording = true;

        // Update UI
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        playBtn.disabled = true;

        // Start timer
        seconds = 0;
        updateTimer();
        timerInterval = setInterval(updateTimer, 1000);

    } catch (error) {
        alert('Error accessing microphone: ' + error.message);
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        isRecording = false;

        // Update UI
        recordBtn.disabled = false;
        stopBtn.disabled = true;
        playBtn.disabled = false;

        // Stop timer
        clearInterval(timerInterval);
    }
}

function playRecording() {
    resultAudio.play();
}

function updateTimer() {
    seconds++;
    const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
    const secs = (seconds % 60).toString().padStart(2, '0');
    timer.textContent = `${mins}:${secs}`;
}

function drawVisualizer() {
    if (!analyser || !isRecording) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    const width = visualizer.width;
    const height = visualizer.height;

    function draw() {
        requestAnimationFrame(draw);
        analyser.getByteTimeDomainData(dataArray);

        canvasCtx.fillStyle = '#1a1a2e';
        canvasCtx.fillRect(0, 0, width, height);

        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = '#667eea';
        canvasCtx.beginPath();

        const sliceWidth = width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * height / 2;

            if (i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        canvasCtx.lineTo(width, height / 2);
        canvasCtx.stroke();
    }

    draw();
}

// Analyze audio with backend
async function analyzeAudio(file) {
    loading.classList.add('active');
    results.style.display = 'none';

    const formData = new FormData();
    formData.append('audio', file);

    try {
        const response = await fetch('http://localhost:5000/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
    } catch (error) {
        alert('Error analyzing audio: ' + error.message);
    } finally {
        loading.classList.remove('active');
        results.style.display = 'block';
    }
}

function displayResults(data) {
    emotionResult.textContent = data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1);
    confidenceResult.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

    // Update emotion bars
    const emotions = ['anger', 'happy', 'sad'];
    const probabilities = data.probabilities || {
        anger: data.emotion === 'anger' ? data.confidence : 0.1,
        happy: data.emotion === 'happy' ? data.confidence : 0.1,
        sad: data.emotion === 'sad' ? data.confidence : 0.1
    };

    emotions.forEach(emotion => {
        const bar = document.querySelector(`.${emotion}-bar`);
        const percentElement = bar.parentElement.nextElementSibling;
        const percent = (probabilities[emotion] * 100).toFixed(1);
        
        setTimeout(() => {
            bar.style.width = `${percent}%`;
            bar.setAttribute('data-percent', percent);
            percentElement.textContent = `${percent}%`;
        }, 100);
    });

    // Add emotion-specific styling
    results.className = 'results';
    results.classList.add(`${data.emotion}-result`);
}

// Initialize
window.addEventListener('load', () => {
    initVisualizer();
    
    // Show initial message
    results.style.display = 'block';
});
