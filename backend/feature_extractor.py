import numpy as np
import librosa
import librosa.display
from typing import Tuple, Optional, Dict, Union
from pathlib import Path
import warnings
from config import config

warnings.filterwarnings('ignore', category=UserWarning)


class EnhancedFeatureExtractor:
    """
    Enhanced feature extractor combining multiple representations.
    """
    
    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive feature set for emotion recognition.
        
        Returns:
            Dictionary containing:
            - mel_spec: Mel spectrogram for CNN
            - mfcc: MFCC features
            - prosodic: Prosodic features (pitch, energy, etc.)
            - spectral: Spectral features
            - chroma: Chroma features for tonality
        """
        features = {}
        
        # 1. Mel spectrogram (for CNN)
        features['mel_spec'] = self.extract_mel_spectrogram(audio)
        
        # 2. MFCC features
        features['mfcc'] = self.extract_mfcc(audio)
        
        # 3. Prosodic features
        features['prosodic'] = self.extract_prosodic_features(audio)
        
        # 4. Spectral features
        features['spectral'] = self.extract_spectral_features(audio)
        
        # 5. Chroma features
        features['chroma'] = self.extract_chroma_features(audio)
        
        return features
    
    def extract_mel_spectrogram(self, audio: np.ndarray, 
                               n_mels: int = config.N_MELSPEC) -> np.ndarray:
        """Extract log-mel spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=n_mels,
            hop_length=config.HOP_LENGTH,
            n_fft=config.N_FFT
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel
    
    def extract_mfcc(self, audio: np.ndarray, 
                    n_mfcc: int = config.N_MFCC) -> np.ndarray:
        """Extract MFCCs with deltas."""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            hop_length=config.HOP_LENGTH,
            n_fft=config.N_FFT
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        mfcc_all = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        return mfcc_all
    
    def extract_prosodic_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract prosodic features (pitch, energy, etc.)."""
        features = []
        
        # Fundamental frequency
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate,
            hop_length=config.HOP_LENGTH
        )
        f0 = np.nan_to_num(f0)
        features.append(f0)
        
        # Energy
        rms = librosa.feature.rms(y=audio, hop_length=config.HOP_LENGTH)[0]
        features.append(rms)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=config.HOP_LENGTH)[0]
        features.append(zcr)
        
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=config.HOP_LENGTH)[0]
        features.append(centroid)
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=config.HOP_LENGTH)[0]
        features.append(rolloff)
        
        # Align lengths
        min_len = min(len(f) for f in features)
        prosodic = np.vstack([f[:min_len] for f in features])
        
        return prosodic
    
    def extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral features."""
        stft = np.abs(librosa.stft(audio, hop_length=config.HOP_LENGTH, n_fft=config.N_FFT))
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sample_rate, hop_length=config.HOP_LENGTH)
        
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(
            y=audio, hop_length=config.HOP_LENGTH)
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, hop_length=config.HOP_LENGTH)
        
        # Combine
        spectral_features = np.vstack([
            contrast,
            flatness,
            bandwidth
        ])
        
        return spectral_features
    
    def extract_chroma_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features."""
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate, hop_length=config.HOP_LENGTH)
        chroma_cens = librosa.feature.chroma_cens(
            y=audio, sr=self.sample_rate, hop_length=config.HOP_LENGTH)
        
        chroma_features = np.vstack([chroma, chroma_cens])
        return chroma_features
    
    def prepare_cnn_input(self, audio: np.ndarray) -> np.ndarray:
        """Prepare input for CNN model."""
        mel_spec = self.extract_mel_spectrogram(audio)
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # Add channel dimension
        mel_spec = mel_spec[np.newaxis, ...]  # (1, n_mels, time)
        
        return mel_spec
    
    def prepare_lstm_input(self, audio: np.ndarray) -> np.ndarray:
        """Prepare features for LSTM model."""
        all_features = self.extract_all_features(audio)
        
        # Combine relevant features
        mfcc = all_features['mfcc']
        prosodic = all_features['prosodic']
        
        # Align time dimension
        min_time = min(mfcc.shape[1], prosodic.shape[1])
        combined = np.vstack([
            mfcc[:, :min_time],
            prosodic[:, :min_time]
        ])
        
        # Transpose to (time, features)
        combined = combined.T
        
        return combined
    
    def extract_from_file(self, audio_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Extract features from audio file."""
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Pad/trim to target duration
        target_length = int(config.TARGET_DURATION * self.sample_rate)
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            pad_len = target_length - len(audio)
            audio = np.pad(audio, (0, pad_len), mode='constant')
        
        return self.extract_all_features(audio)
