from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tempfile
import json
from pathlib import Path
from werkzeug.utils import secure_filename
import traceback
import numpy as np
import soundfile as sf
import librosa
from datetime import datetime

from config import config
from predictor import EmotionPredictor

app = Flask(__name__, template_folder='../frontend')
CORS(app)
# The code snippet you provided is setting up configurations for file uploads in a Flask web
# application:

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Initialize predictor
predictor = EmotionPredictor()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve frontend."""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    """Analyze audio for emotion detection."""
    try:
        # Check if file was uploaded
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'File type not allowed. Allowed types: {ALLOWED_EXTENSIONS}'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        temp_path = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(temp_path)
        
        # Convert to WAV if needed
        if not filename.lower().endswith('.wav'):
            converted_path = convert_to_wav(temp_path)
            if converted_path:
                temp_path.unlink()
                temp_path = converted_path
        
        # Predict emotion
        result = predictor.predict(temp_path)
        
        # Add metadata
        result['metadata'] = {
            'filename': file.filename,
            'timestamp': datetime.now().isoformat(),
            'file_size': os.path.getsize(temp_path),
            'processing_time': 'real-time'
        }
        
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()
        
        return jsonify(result)
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in analyze_audio: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple audio files."""
    try:
        if 'audio_files' not in request.files:
            return jsonify({'error': 'No audio files provided'}), 400
        
        files = request.files.getlist('audio_files')
        
        if not files:
            return jsonify({'error': 'No files selected'}), 400
        
        results = []
        temp_files = []
        
        for file in files:
            if file and allowed_file(file.filename):
                # Save to temp file
                filename = secure_filename(file.filename)
                temp_path = Path(app.config['UPLOAD_FOLDER']) / filename
                file.save(temp_path)
                temp_files.append(temp_path)
                
                # Convert to WAV if needed
                if not filename.lower().endswith('.wav'):
                    converted_path = convert_to_wav(temp_path)
                    if converted_path:
                        temp_path.unlink()
                        temp_path = converted_path
                        temp_files[-1] = temp_path
                
                # Predict
                result = predictor.predict(temp_path)
                result['filename'] = file.filename
                results.append(result)
        
        # Get statistics
        stats = predictor.get_emotion_statistics(results)
        
        # Clean up temp files
        for temp_path in temp_files:
            if temp_path.exists():
                temp_path.unlink()
        
        return jsonify({
            'success': True,
            'results': results,
            'statistics': stats,
            'total_files': len(results)
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in batch_analyze: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/realtime', methods=['POST'])
def realtime_analysis():
    """Real-time audio stream analysis."""
    try:
        # Get audio data from request
        if request.is_json:
            data = request.get_json()
            audio_data = data.get('audio_data')
            
            if not audio_data:
                return jsonify({'error': 'No audio data provided'}), 400
            
            # Convert base64 or array to numpy
            if isinstance(audio_data, list):
                audio_array = np.array(audio_data, dtype=np.float32)
            else:
                # Assume base64 encoded audio
                import base64
                audio_bytes = base64.b64decode(audio_data)
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Predict from array
            result = predictor.predict_from_array(audio_array)
            
            return jsonify(result)
        else:
            return jsonify({'error': 'JSON data expected'}), 400
            
    except Exception as e:
        print(f"Error in realtime_analysis: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': 'loaded',
        'device': str(predictor.device),
        'emotions': list(config.INDEX_TO_EMOTION.values()),
        'version': '2.0.0'
    })

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information."""
    total_params = sum(p.numel() for p in predictor.model.parameters())
    trainable_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
    
    return jsonify({
        'model_type': predictor.model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_shape': [1, 1, config.N_MELSPEC, 251],  # Example shape
        'output_classes': config.NUM_CLASSES,
        'emotion_mapping': config.INDEX_TO_EMOTION
    })

def convert_to_wav(input_path: Path) -> Optional[Path]:
    """Convert audio file to WAV format."""
    try:
        from pydub import AudioSegment
        
        output_path = input_path.with_suffix('.wav')
        
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)  # mono
        audio = audio.set_frame_rate(config.SAMPLE_RATE)
        audio.export(output_path, format='wav', parameters=[
            "-ac", "1",
            "-ar", str(config.SAMPLE_RATE),
            "-sample_fmt", "s16"
        ])
        
        return output_path
    except Exception as e:
        print(f"Conversion failed: {e}")
        return None

if __name__ == '__main__':
    print("🚀 Starting Advanced Speech Emotion Recognition API...")
    print(f"📁 Upload directory: {UPLOAD_FOLDER}")
    print("🌐 Server running at http://localhost:5000")
    print("📚 API Endpoints:")
    print("   POST /api/analyze        - Analyze single audio file")
    print("   POST /api/batch-analyze  - Analyze multiple audio files")
    print("   POST /api/realtime       - Real-time audio analysis")
    print("   GET  /api/health         - Health check")
    print("   GET  /api/model/info     - Model information")
    print("   GET  /                   - Web interface")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
