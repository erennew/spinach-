import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import os
from pathlib import Path
import warnings
from typing import Dict, Tuple, Optional

from config import config
from model import AdvancedEmotionModel
from dataset_loader import create_data_loaders

warnings.filterwarnings('ignore')


class EmotionTrainer:
    """Advanced trainer for emotion recognition."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader = None,
        use_wandb: bool = False
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.use_wandb = use_wandb
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer with different learning rates
        self.optimizer = optim.AdamW(
            [
                {'params': self.model.cnn.parameters(), 'lr': config.LEARNING_RATE},
                {'params': self.model.lstm1.parameters(), 'lr': config.LEARNING_RATE * 0.5},
                {'params': self.model.fc.parameters(), 'lr': config.LEARNING_RATE * 0.1}
            ],
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Early stopping
        self.best_val_acc = 0.0
        self.patience = 10
        self.counter = 0
        
        # Metrics tracking
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        # Initialize wandb
        if use_wandb:
            wandb.init(project="speech-emotion-recognition", config=vars(config))
            wandb.watch(self.model)
    
    def train_epoch(self) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            
            if isinstance(outputs, dict):
                loss = self.criterion(outputs['emotion'], targets)
            else:
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            if isinstance(outputs, dict):
                _, predicted = outputs['emotion'].max(1)
            else:
                _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            acc = 100. * correct / total
            progress_bar.set_postfix(loss=loss.item(), acc=acc)
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train_batch_loss': loss.item(),
                    'train_batch_acc': acc
                })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float, Dict]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Confusion matrix
        num_classes = config.NUM_CLASSES
        confusion = torch.zeros(num_classes, num_classes)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for data, targets in progress_bar:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                
                if isinstance(outputs, dict):
                    loss = self.criterion(outputs['emotion'], targets)
                    _, predicted = outputs['emotion'].max(1)
                else:
                    loss = self.criterion(outputs, targets)
                    _, predicted = outputs.max(1)
                
                total_loss += loss.item()
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update confusion matrix
                for t, p in zip(targets.cpu().view(-1), predicted.cpu().view(-1)):
                    confusion[t.long(), p.long()] += 1
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        # Calculate per-class metrics
        class_accuracies = confusion.diag() / confusion.sum(1)
        
        # Calculate F1 score
        precision = confusion.diag() / (confusion.sum(0) + 1e-8)
        recall = confusion.diag() / (confusion.sum(1) + 1e-8)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        macro_f1 = f1_scores.mean().item()
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1_score': macro_f1,
            'class_accuracies': class_accuracies.numpy(),
            'confusion_matrix': confusion.numpy(),
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets)
        }
        
        return epoch_loss, epoch_acc, metrics
    
    def train(self, num_epochs: int = config.EPOCHS):
        """Main training loop."""
        
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_metrics['f1_score'])
            
            # Print results
            print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_metrics['f1_score']:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_metrics['f1_score'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
                
                # Log confusion matrix
                wandb.log({
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=val_metrics['targets'],
                        preds=val_metrics['predictions'],
                        class_names=list(config.INDEX_TO_EMOTION.values())
                    )
                })
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"🔥 New best model saved with accuracy: {val_acc:.2f}%")
            else:
                self.counter += 1
            
            # Early stopping
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
    
    def test(self) -> Dict:
        """Test the model."""
        if self.test_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        num_classes = config.NUM_CLASSES
        confusion = torch.zeros(num_classes, num_classes)
        
        all_probs = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc="Testing")
            
            for data, targets in progress_bar:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                
                if isinstance(outputs, dict):
                    loss = self.criterion(outputs['emotion'], targets)
                    probs = torch.softmax(outputs['emotion'], dim=1)
                    _, predicted = outputs['emotion'].max(1)
                else:
                    loss = self.criterion(outputs, targets)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)
                
                total_loss += loss.item()
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update confusion matrix
                for t, p in zip(targets.cpu().view(-1), predicted.cpu().view(-1)):
                    confusion[t.long(), p.long()] += 1
                
                all_probs.extend(probs.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_loss = total_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        
        # Calculate per-class metrics
        class_accuracies = confusion.diag() / confusion.sum(1)
        
        # Calculate F1 score
        precision = confusion.diag() / (confusion.sum(0) + 1e-8)
        recall = confusion.diag() / (confusion.sum(1) + 1e-8)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        macro_f1 = f1_scores.mean().item()
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_f1_score': macro_f1,
            'class_accuracies': class_accuracies.numpy(),
            'confusion_matrix': confusion.numpy(),
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probs)
        }
        
        print(f"\n{'='*50}")
        print(f"Test Results:")
        print(f"{'='*50}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Test F1 Score: {macro_f1:.4f}")
        
        # Print per-class accuracy
        print(f"\nPer-class accuracy:")
        for i, acc in enumerate(class_accuracies.numpy()):
            emotion = config.INDEX_TO_EMOTION[i]
            print(f"  {emotion:10s}: {acc:.2%}")
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if is_best:
            checkpoint_path = config.CHECKPOINT_DIR / 'best_model.pth'
        else:
            checkpoint_path = config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_acc = checkpoint['best_val_acc']
            self.history = checkpoint['history']
            print(f"Checkpoint loaded from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")


def train_full_pipeline(
    data_dir: Path,
    model_name: str = 'advanced',
    use_wandb: bool = False
):
    """Complete training pipeline."""
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(data_dir)
    
    # Create model
    if model_name == 'advanced':
        model = AdvancedEmotionModel()
    else:
        from model import LightweightEmotionModel
        model = LightweightEmotionModel()
    
    # Create trainer
    trainer = EmotionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        use_wandb=use_wandb
    )
    
    # Train
    trainer.train()
    
    # Test
    test_metrics = trainer.test()
    
    return trainer, test_metrics
