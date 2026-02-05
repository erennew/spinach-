import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os
from pathlib import Path
from tqdm import tqdm
from model import LightweightEmotionModel
from feature_extractor import EnhancedFeatureExtractor

# Configuration
SAMPLE_RATE = 16000
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001

# Emotion mapping - ADJUST BASED ON YOUR DATASET!
EMOTIONS = ['angry', 'happy', 'sad', 'neutral', 'surprise', 'fear', 'disgust']
EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTIONS)}

class VoiceDataset(Dataset):
    """Dataset for your voice files."""
    
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.extractor = EnhancedFeatureExtractor()
        
        # Load all voice files
        for emotion in EMOTIONS:
            emotion_dir = self.root_dir / emotion
            if emotion_dir.exists():
                for file in emotion_dir.glob('*.wav'):
                    self.samples.append({
                        'path': file,
                        'emotion': emotion,
                        'emotion_id': EMOTION_TO_ID[emotion]
                    })
        
        print(f"✅ Loaded {len(self.samples)} voice samples")
        print(f"Emotion distribution:")
        for emotion in EMOTIONS:
            count = sum(1 for s in self.samples if s['emotion'] == emotion)
            if count > 0:
                print(f"  {emotion}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and preprocess audio
        audio, _ = librosa.load(sample['path'], sr=SAMPLE_RATE, mono=True)
        
        # Extract features for CNN
        features = self.extractor.prepare_cnn_input(audio)
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        label = torch.tensor(sample['emotion_id'], dtype=torch.long)
        
        return features, label

def train_model():
    """Train model with your voice data."""
    
    print("="*60)
    print("TRAINING WITH YOUR VOICE DATASET")
    print("="*60)
    
    # 1. Load dataset
    dataset = VoiceDataset('../../datasets/my_voices')  # Adjust path as needed
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 2. Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightweightEmotionModel(num_classes=len(EMOTIONS)).to(device)
    
    # 3. Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training loop
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100.*train_correct/train_total
            })
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  💾 Saved best model (Acc: {val_acc:.2f}%)")
        
        print("-" * 50)
    
    print("="*60)
    print(f"TRAINING COMPLETE! Best validation accuracy: {best_val_acc:.2f}%")
    print("Model saved as 'best_model.pth'")
    print("="*60)
    
    return model

if __name__ == "__main__":
    train_model()
