import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class AttentionLayer(nn.Module):
    """Attention mechanism for emotion recognition."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch, seq_len, hidden_size)
        attention_scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
        return context, attention_weights


class MultiScaleCNN(nn.Module):
    """Multi-scale CNN for capturing different frequency patterns."""
    
    def __init__(self, in_channels: int = 1):
        super().__init__()
        
        # Branch 1: Small kernel (local patterns)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        
        # Branch 2: Medium kernel
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        
        # Branch 3: Large kernel (global patterns)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(7, 7), padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(7, 7), padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # Concatenate along channel dimension
        out = torch.cat([b1, b2, b3], dim=1)
        return out


class AdvancedEmotionModel(nn.Module):
    """Advanced CNN-LSTM model with attention for emotion recognition."""
    
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        
        # Multi-scale CNN
        self.cnn = MultiScaleCNN(in_channels=1)
        
        # Dimension calculation after multi-scale CNN
        # Assuming input: (batch, 1, 64, 251) for 4 seconds audio
        cnn_output_channels = 32 * 3  # 3 branches
        
        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=cnn_output_channels * 32,  # After pooling
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size=256)  # 128 * 2 for bidirectional
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
        )
        
        # Emotion-specific classifiers
        self.valence_classifier = nn.Linear(256, 3)  # Positive, Neutral, Negative
        self.arousal_classifier = nn.Linear(256, 3)  # High, Medium, Low
        
    def forward(self, x: torch.Tensor) -> dict:
        # x shape: (batch, 1, n_mels, time)
        
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch, channels, freq, time)
        
        # Frequency pooling
        cnn_out = torch.mean(cnn_out, dim=2)  # (batch, channels, time)
        
        # Prepare for LSTM
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, time, channels)
        
        # LSTM processing
        lstm_out, _ = self.lstm1(cnn_out)  # (batch, time, hidden*2)
        
        # Attention pooling
        context, attention_weights = self.attention(lstm_out)
        
        # Main emotion classification
        emotion_logits = self.fc(context)
        
        # Additional outputs for interpretability
        valence = self.valence_classifier(context)
        arousal = self.arousal_classifier(context)
        
        return {
            'emotion': emotion_logits,
            'valence': valence,
            'arousal': arousal,
            'attention_weights': attention_weights
        }


class LightweightEmotionModel(nn.Module):
    """Lightweight model for deployment."""
    
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
