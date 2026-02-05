import numpy as np
from typing import Tuple, Optional, Callable, List
import librosa
from config import config

class AudioAugmentor:
    """Audio augmentation pipeline - Your implementation enhanced."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        noise_prob: float = 0.5,
        noise_snr_range: Tuple[float, float] = (5, 20),
        pitch_shift_prob: float = 0.3,
        pitch_shift_range: Tuple[float, float] = (-3, 3),
        time_stretch_prob: float = 0.3,
        time_stretch_range: Tuple[float, float] = (0.8, 1.2),
        volume_prob: float = 0.3,
        volume_range: Tuple[float, float] = (0.7, 1.3)
    ):
        self.sample_rate = sample_rate
        self.noise_prob = noise_prob
        self.noise_snr_range = noise_snr_range
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range
        self.volume_prob = volume_prob
        self.volume_range = volume_range
        
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        augmented = audio.copy()
        
        if np.random.random() < self.noise_prob:
            augmented = self.add_noise(augmented)
            
        if np.random.random() < self.pitch_shift_prob:
            augmented = self.pitch_shift(augmented)
            
        if np.random.random() < self.time_stretch_prob:
            augmented = self.time_stretch(augmented)
            
        if np.random.random() < self.volume_prob:
            augmented = self.volume_change(augmented)
            
        return augmented
    
    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        snr = np.random.uniform(*self.noise_snr_range)
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise.astype(audio.dtype)
    
    def pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        semitones = np.random.uniform(*self.pitch_shift_range)
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=semitones)
    
    def time_stretch(self, audio: np.ndarray) -> np.ndarray:
        rate = np.random.uniform(*self.time_stretch_range)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def volume_change(self, audio: np.ndarray) -> np.ndarray:
        factor = np.random.uniform(*self.volume_range)
        return audio * factor


class SpecAugment:
    """Spectrogram augmentation."""
    
    def __init__(
        self,
        freq_mask_param: int = 30,
        time_mask_param: int = 50,
        n_freq_masks: int = 2,
        n_time_masks: int = 2
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        
    def __call__(self, spectrogram: np.ndarray) -> np.ndarray:
        augmented = spectrogram.copy()
        time_steps, freq_channels = augmented.shape
        
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, min(self.freq_mask_param, freq_channels))
            f0 = np.random.randint(0, freq_channels - f)
            augmented[:, f0:f0 + f] = 0
            
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, min(self.time_mask_param, time_steps))
            t0 = np.random.randint(0, time_steps - t)
            augmented[t0:t0 + t, :] = 0
            
        return augmented


class EmotionAugmentationPipeline:
    """Complete augmentation pipeline for emotion recognition."""
    
    def __init__(self, augment_prob: float = 0.7):
        self.augment_prob = augment_prob
        self.audio_augmentor = AudioAugmentor(sample_rate=config.SAMPLE_RATE)
        self.spec_augmentor = SpecAugment()
        
    def augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio augmentations."""
        if np.random.random() < self.augment_prob:
            return self.audio_augmentor(audio)
        return audio
    
    def augment_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Apply spectrogram augmentations."""
        if np.random.random() < self.augment_prob:
            return self.spec_augmentor(spectrogram)
        return spectrogram


def get_default_augmentor(sample_rate: int = 16000) -> AudioAugmentor:
    """Get default augmentor."""
    return AudioAugmentor(
        sample_rate=sample_rate,
        noise_prob=0.5,
        noise_snr_range=(5, 20),
        pitch_shift_prob=0.3,
        pitch_shift_range=(-3, 3),
        time_stretch_prob=0.3,
        time_stretch_range=(0.8, 1.2),
        volume_prob=0.3,
        volume_range=(0.7, 1.3)
    )
