import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

from config import config
from model import AdvancedEmotionModel, LightweightEmotionModel
from feature_extractor import EnhancedFeatureExtractor

warnings.filterwarnings('ignore')


class EmotionPredictor:
    """Advanced emotion predictor with confidence scores and explanations."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_type: str = 'advanced'
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        if model_type == 'advanced':
            self.model = AdvancedEmotionModel().to(self.device)
        else:
            self.model = LightweightEmotionModel().to(self.device)
        
        # Load weights
        if model_path is None:
            model_path = config.MODEL_DIR / 'best_model.pth'
        
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ Model loaded from {model_path}")
        else:
            print("⚠️ Using randomly initialized model")
        
        self.model.eval()
        
        # Feature extractor
        self.feature_extractor = EnhancedFeatureExtractor()
        
        # Emotion colors for visualization
        self.emotion_colors = {
            'angry': '#FF6B6B',
            'disgust': '#94D82D',
            'fear': '#FFD93D',
            'happy': '#51CF66',
            'sad': '#5C7CFA',
            'surprise': '#FF922B',
            'neutral': '#868E96'
        }
        
        # Emotion descriptions
        self.emotion_descriptions = {
            'angry': 'Voice shows frustration, irritation or rage',
            'disgust': 'Voice expresses revulsion or strong disapproval',
            'fear': 'Voice indicates anxiety, worry or nervousness',
            'happy': 'Voice sounds cheerful, joyful or excited',
            'sad': 'Voice reflects sorrow, grief or depression',
            'surprise': 'Voice shows astonishment or amazement',
            'neutral': 'Voice is calm, balanced and without strong emotion'
        }
    
    def predict(self, audio_path: Path) -> Dict:
        """Predict emotion from audio file."""
        
        try:
            # Load and preprocess audio
            audio, _ = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
            
            # Extract features
            mel_spec = self.feature_extractor.prepare_cnn_input(audio)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                
                if isinstance(outputs, dict):
                    emotion_logits = outputs['emotion']
                    
                    # Get valence and arousal if available
                    if 'valence' in outputs:
                        valence = torch.softmax(outputs['valence'], dim=1)
                        arousal = torch.softmax(outputs['arousal'], dim=1)
                    else:
                        valence = arousal = None
                else:
                    emotion_logits = outputs
                    valence = arousal = None
                
                # Get probabilities
                probabilities = torch.softmax(emotion_logits, dim=1)[0]
                
                # Get top emotions
                top_probs, top_indices = torch.topk(probabilities, k=3)
                
                # Primary emotion
                primary_idx = top_indices[0].item()
                primary_emotion = config.INDEX_TO_EMOTION[primary_idx]
                primary_confidence = top_probs[0].item()
                
                # Secondary emotions
                secondary_emotions = []
                for i in range(1, min(3, len(top_indices))):
                    idx = top_indices[i].item()
                    emotion = config.INDEX_TO_EMOTION[idx]
                    confidence = top_probs[i].item()
                    secondary_emotions.append({
                        'emotion': emotion,
                        'confidence': float(confidence),
                        'color': self.emotion_colors[emotion]
                    })
                
                # Valence and arousal analysis
                if valence is not None:
                    valence_idx = torch.argmax(valence, dim=1)[0].item()
                    valence_label = ['Negative', 'Neutral', 'Positive'][valence_idx]
                    
                    arousal_idx = torch.argmax(arousal, dim=1)[0].item()
                    arousal_label = ['Low', 'Medium', 'High'][arousal_idx]
                else:
                    valence_label = arousal_label = None
                
                # Feature importance (if attention available)
                if isinstance(outputs, dict) and 'attention_weights' in outputs:
                    attention = outputs['attention_weights'][0].cpu().numpy()
                    attention_peaks = np.argsort(attention)[-5:]  # Top 5 attention points
                else:
                    attention = None
                    attention_peaks = []
                
                # Extract audio features for explanation
                features = self.feature_extractor.extract_all_features(audio)
                
                # Calculate audio statistics for explanation
                audio_stats = self._analyze_audio_features(features)
                
                # Build response
                response = {
                    'success': True,
                    'primary_emotion': {
                        'emotion': primary_emotion,
                        'confidence': float(primary_confidence),
                        'color': self.emotion_colors[primary_emotion],
                        'description': self.emotion_descriptions[primary_emotion]
                    },
                    'secondary_emotions': secondary_emotions,
                    'all_probabilities': {
                        config.INDEX_TO_EMOTION[i]: float(prob)
                        for i, prob in enumerate(probabilities.cpu().numpy())
                    },
                    'valence': valence_label,
                    'arousal': arousal_label,
                    'audio_analysis': audio_stats,
                    'attention_peaks': attention_peaks.tolist() if attention is not None else [],
                    'feature_shapes': {
                        k: v.shape for k, v in features.items()
                    }
                }
                
                return response
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'primary_emotion': {'emotion': 'unknown', 'confidence': 0.0}
            }
    
    def predict_from_array(self, audio_array: np.ndarray) -> Dict:
        """Predict emotion from numpy array."""
        # Save to temp file and predict
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = Path(f.name)
            sf.write(temp_path, audio_array, config.SAMPLE_RATE)
            
            try:
                result = self.predict(temp_path)
            finally:
                temp_path.unlink()
        
        return result
    
    def _analyze_audio_features(self, features: Dict) -> Dict:
        """Analyze audio features for explanation."""
        
        # Extract key statistics
        if 'prosodic' in features:
            prosodic = features['prosodic']
            
            # Pitch statistics
            if prosodic.shape[0] > 0:
                pitch = prosodic[0, :]
                pitch_mean = np.mean(pitch)
                pitch_std = np.std(pitch)
                pitch_range = np.max(pitch) - np.min(pitch)
            else:
                pitch_mean = pitch_std = pitch_range = 0
            
            # Energy statistics
            if prosodic.shape[0] > 1:
                energy = prosodic[1, :]
                energy_mean = np.mean(energy)
                energy_std = np.std(energy)
            else:
                energy_mean = energy_std = 0
            
            # Speaking rate approximation
            speaking_rate = len(features.get('mfcc', np.array([])).T) / 4.0  # Frames per second
        else:
            pitch_mean = pitch_std = pitch_range = 0
            energy_mean = energy_std = 0
            speaking_rate = 0
        
        # Spectral statistics
        if 'mel_spec' in features:
            mel_spec = features['mel_spec']
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(S=mel_spec))
        else:
            spectral_flatness = 0
        
        return {
            'pitch_statistics': {
                'mean': float(pitch_mean),
                'std': float(pitch_std),
                'range': float(pitch_range),
                'interpretation': 'High pitch variation may indicate emotional arousal'
            },
            'energy_statistics': {
                'mean': float(energy_mean),
                'std': float(energy_std),
                'interpretation': 'Energy variations correlate with emotional intensity'
            },
            'speaking_rate': {
                'rate': float(speaking_rate),
                'interpretation': 'Faster speaking may indicate excitement or anxiety'
            },
            'spectral_properties': {
                'flatness': float(spectral_flatness),
                'interpretation': 'Lower spectral flatness indicates more tonal/emotional speech'
            }
        }
    
    def batch_predict(self, audio_paths: List[Path]) -> List[Dict]:
        """Predict emotions for multiple audio files."""
        results = []
        
        for audio_path in audio_paths:
            result = self.predict(audio_path)
            result['filename'] = audio_path.name
            results.append(result)
        
        return results
    
    def get_emotion_statistics(self, predictions: List[Dict]) -> Dict:
        """Get statistics from multiple predictions."""
        
        emotions = []
        confidences = []
        
        for pred in predictions:
            if pred['success']:
                emotions.append(pred['primary_emotion']['emotion'])
                confidences.append(pred['primary_emotion']['confidence'])
        
        if not emotions:
            return {}
        
        # Calculate statistics
        unique_emotions, counts = np.unique(emotions, return_counts=True)
        emotion_counts = dict(zip(unique_emotions, counts))
        
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'total_samples': len(emotions),
            'emotion_distribution': emotion_counts,
            'average_confidence': float(avg_confidence),
            'dominant_emotion': dominant_emotion,
            'emotion_variance': np.var(confidences) if confidences else 0
        }
