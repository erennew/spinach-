import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Config:
    """Configuration for the emotion recognition system."""
    
    # Audio parameters
    SAMPLE_RATE: int = 16000
    TARGET_DURATION: float = 4.0  # seconds
    HOP_LENGTH: int = 512
    N_FFT: int = 2048
    
    # Feature extraction
    N_MFCC: int = 40
    N_MELS: int = 128
    N_MELSPEC: int = 64  # For CNN input
    
    # Model parameters
    NUM_CLASSES: int = 7  # 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 100
    
    # Augmentation
    AUGMENT_PROB: float = 0.7
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATASET_DIR: Path = BASE_DIR / "datasets"
    MODEL_DIR: Path = BASE_DIR / "backend" / "pretrained_models"
    CHECKPOINT_DIR: Path = BASE_DIR / "backend" / "checkpoints"
    
    # Emotions mapping
    EMOTION_MAP: Dict[str, int] = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }
    
    # Reverse mapping
    INDEX_TO_EMOTION: Dict[int, str] = {v: k for k, v in EMOTION_MAP.items()}
    
    def __post_init__(self):
        """Create directories."""
        self.MODEL_DIR.mkdir(exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.DATASET_DIR.mkdir(exist_ok=True)

# Global config instance
config = Config()
