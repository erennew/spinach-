import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

from config import config
from augmentation import EmotionAugmentationPipeline
from feature_extractor import EnhancedFeatureExtractor


class EmotionDataset(Dataset):
    """Dataset for emotion recognition."""
    
    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        augment: bool = False,
        use_cache: bool = True
    ):
        self.data_dir = data_dir
        self.split = split
        self.augment = augment and split == 'train'
        self.use_cache = use_cache
        
        self.feature_extractor = EnhancedFeatureExtractor()
        self.augmentor = EmotionAugmentationPipeline() if self.augment else None
        
        # Load dataset
        self.samples = self._load_dataset()
        
        if use_cache:
            self._cache_features()
    
    def _load_dataset(self) -> List[Dict]:
        """Load dataset based on structure."""
        samples = []
        
        # Support multiple dataset structures
        dataset_formats = [
            self._load_ravdess_format,
            self._load_crema_format,
            self._load_tess_format,
            self._load_custom_format
        ]
        
        for format_loader in dataset_formats:
            samples = format_loader()
            if samples:
                break
        
        if not samples:
            raise ValueError(f"No data found in {self.data_dir}")
        
        # Split data
        if self.split in ['train', 'val', 'test']:
            samples = self._split_data(samples)
        
        return samples
    
    def _load_ravdess_format(self) -> List[Dict]:
        """Load RAVDESS dataset format."""
        samples = []
        
        for actor_dir in self.data_dir.glob('Actor_*'):
            for audio_file in actor_dir.glob('*.wav'):
                # Parse filename: 03-01-06-01-02-01-12.wav
                # 03 = modality (voice=03)
                # 01 = vocal channel (speech=01)
                # 06 = emotion (06=sadness)
                # 01 = emotional intensity (normal=01, strong=02)
                # 02 = statement ("kids"=01, "dogs"=02)
                # 01 = repetition (first=01, second=02)
                # 12 = actor (12th actor)
                
                parts = audio_file.stem.split('-')
                if len(parts) >= 3:
                    emotion_code = int(parts[2])
                    emotion = self._map_ravdess_emotion(emotion_code)
                    
                    if emotion in config.EMOTION_MAP:
                        samples.append({
                            'path': audio_file,
                            'emotion': emotion,
                            'emotion_id': config.EMOTION_MAP[emotion],
                            'actor': parts[6] if len(parts) > 6 else 'unknown'
                        })
        
        return samples
    
    def _load_crema_format(self) -> List[Dict]:
        """Load CREMA-D dataset format."""
        samples = []
        
        for audio_file in self.data_dir.glob('*.wav'):
            # Format: 1001_DFA_ANG_XX.wav
            # 1001 = Actor ID
            # DFA = Sentence
            # ANG = Emotion (ANG, DIS, FEA, HAP, NEU, SAD)
            # XX = Intensity (LO, MD, HI, XX)
            
            filename = audio_file.stem
            parts = filename.split('_')
            
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = self._map_crema_emotion(emotion_code)
                
                if emotion in config.EMOTION_MAP:
                    samples.append({
                        'path': audio_file,
                        'emotion': emotion,
                        'emotion_id': config.EMOTION_MAP[emotion],
                        'actor': parts[0]
                    })
        
        return samples
    
    def _load_tess_format(self) -> List[Dict]:
        """Load TESS dataset format."""
        samples = []
        
        for audio_file in self.data_dir.glob('*.wav'):
            # Format: OAF_back_angry.wav
            # OAF = Actor (OAF, YAF)
            # back = Sentence
            # angry = Emotion
            
            filename = audio_file.stem
            parts = filename.split('_')
            
            if len(parts) >= 3:
                emotion = parts[-1].lower()
                
                if emotion in config.EMOTION_MAP:
                    samples.append({
                        'path': audio_file,
                        'emotion': emotion,
                        'emotion_id': config.EMOTION_MAP[emotion],
                        'actor': parts[0]
                    })
        
        return samples
    
    def _load_custom_format(self) -> List[Dict]:
        """Load custom dataset structure."""
        samples = []
        
        # Structure: emotion/audio_files.wav
        for emotion_dir in self.data_dir.iterdir():
            if emotion_dir.is_dir():
                emotion = emotion_dir.name.lower()
                
                if emotion in config.EMOTION_MAP:
                    for audio_file in emotion_dir.glob('*.wav'):
                        samples.append({
                            'path': audio_file,
                            'emotion': emotion,
                            'emotion_id': config.EMOTION_MAP[emotion],
                            'actor': 'unknown'
                        })
        
        return samples
    
    def _map_ravdess_emotion(self, code: int) -> str:
        """Map RAVDESS emotion code to emotion name."""
        mapping = {
            1: 'neutral',
            2: 'neutral',  # Calm -> Neutral
            3: 'happy',
            4: 'sad',
            5: 'angry',
            6: 'fear',
            7: 'disgust',
            8: 'surprise'
        }
        return mapping.get(code, 'neutral')
    
    def _map_crema_emotion(self, code: str) -> str:
        """Map CREMA-D emotion code to emotion name."""
        mapping = {
            'ANG': 'angry',
            'DIS': 'disgust',
            'FEA': 'fear',
            'HAP': 'happy',
            'NEU': 'neutral',
            'SAD': 'sad'
        }
        return mapping.get(code, 'neutral')
    
    def _split_data(self, samples: List[Dict]) -> List[Dict]:
        """Split data into train/val/test."""
        np.random.shuffle(samples)
        
        if self.split == 'train':
            return samples[:int(0.7 * len(samples))]
        elif self.split == 'val':
            return samples[int(0.7 * len(samples)):int(0.85 * len(samples))]
        else:  # test
            return samples[int(0.85 * len(samples)):]
    
    def _cache_features(self):
        """Cache extracted features for faster loading."""
        cache_dir = self.data_dir / '.cache' / self.split
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, sample in tqdm(enumerate(self.samples), desc="Caching features"):
            cache_path = cache_dir / f"{sample['path'].stem}.npy"
            
            if not cache_path.exists():
                # Extract and cache features
                audio, _ = librosa.load(sample['path'], sr=config.SAMPLE_RATE, mono=True)
                features = self.feature_extractor.prepare_cnn_input(audio)
                np.save(cache_path, features)
            
            # Update path to cache
            sample['cache_path'] = cache_path
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.use_cache:
            # Load cached features
            features = np.load(sample['cache_path'])
        else:
            # Extract features on the fly
            audio, _ = librosa.load(sample['path'], sr=config.SAMPLE_RATE, mono=True)
            
            if self.augment and self.augmentor:
                audio = self.augmentor.augment_audio(audio)
            
            features = self.feature_extractor.prepare_cnn_input(audio)
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        label = torch.tensor(sample['emotion_id'], dtype=torch.long)
        
        return features, label


def create_data_loaders(
    data_dir: Path,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    
    train_dataset = EmotionDataset(data_dir, split='train', augment=True)
    val_dataset = EmotionDataset(data_dir, split='val', augment=False)
    test_dataset = EmotionDataset(data_dir, split='test', augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def load_emotion_datasets(
    dataset_names: List[str] = None
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load multiple emotion datasets.
    
    Args:
        dataset_names: List of dataset names to load
        
    Returns:
        Combined train, val, test samples
    """
    if dataset_names is None:
        dataset_names = ['RAVDESS', 'CREMA', 'TESS', 'SAVEE', 'EMODB']
    
    all_train = []
    all_val = []
    all_test = []
    
    for dataset_name in dataset_names:
        dataset_path = config.DATASET_DIR / dataset_name
        
        if dataset_path.exists():
            print(f"Loading {dataset_name}...")
            
            dataset = EmotionDataset(dataset_path, split='all', augment=False)
            
            # Split
            samples = dataset.samples
            np.random.shuffle(samples)
            
            train_split = int(0.7 * len(samples))
            val_split = int(0.85 * len(samples))
            
            all_train.extend(samples[:train_split])
            all_val.extend(samples[train_split:val_split])
            all_test.extend(samples[val_split:])
    
    return all_train, all_val, all_test
