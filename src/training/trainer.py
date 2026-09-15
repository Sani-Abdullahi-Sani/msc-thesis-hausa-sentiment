import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any
import json
import numpy as np

class Trainer:
    """Enhanced trainer class for research experiments with robust metrics tracking"""

    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, criterion,
                 device, config, output_dir, track_every=5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.output_dir = output_dir
        self.track_every = track_every

        # Training state
        self.current_epoch = 0
        self.best_val_f1 = -1.0
        self.best_epoch = -1
        self.training_history = []
        self.metrics_history = []  # For CSV tracking
        
        # Research tracking
        self.total_epochs = config.get('epochs', 50)
        self.early_stopping_patience = config.get('early_stopping_patience', 5)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        print(f"📈 Trainer initialized for research experiment:")
        print(f"   💾 Output: {output_dir}")
        print(f"   📊 Track every: {track_every} epochs")
        print(f"   🎯 Max epochs: {self.total_epochs}")
        print(f"   ⏰ Early stopping patience: {self.early_stopping_patience}")

    def train_epoch(self):
        """Train for one epoch with detailed progress tracking"""
        self.model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        num_batches = len(self.train_loader)

        # Enhanced progress bar
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch + 1}/{self.total_epochs}",
            leave=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(input_ids, attention_mask)

            if hasattr(self.model, 'multilabel') and self.model.multilabel:
                loss = self.criterion(logits, labels)
                preds = (torch.sigmoid(logits) > 0.5).float()
            else:
                loss = self.criterion(logits, labels)
                preds = torch.argmax(logits, dim=1)

            # Check for NaN loss (training instability indicator)
            if torch.isnan(loss):
                print(f"⚠️  WARNING: NaN loss detected at batch {batch_idx}")
                print(f"   🔍 This may indicate training instability")
                return float('inf'), 0.0, 0.0

            loss.backward()

            # Gradient clipping with monitoring
            if 'gradient_clip' in self.config:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip']
                )
                # Monitor gradient explosion
                if grad_norm > 10.0:
                    print(f"⚠️  High gradient norm: {grad_norm:.2f}")

            self.optimizer.step()

            epoch_loss += loss.item()
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

            # Update progress bar with current metrics
            current_loss = epoch_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'batch': f'{batch_idx + 1}/{num_batches}'
            })

            # Periodic batch reporting for debugging
            if batch_idx > 0 and batch_idx % 100 == 0:
                print(f"   📊 Batch {batch_idx}/{num_batches}: loss={loss.item():.4f}")

        # Calculate epoch metrics
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        if hasattr(self.model, 'multilabel') and self.model.multilabel:
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        else:
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        avg_loss = epoch_loss / len(self.train_loader)
        
        print(f"   🏋️  Training - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")
        
        return avg_loss, accuracy, f1

    def evaluate(self, dataloader, split_name="Validation"):
        """Enhanced evaluation with detailed metrics"""
        self.model.eval()
        epoch_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f"Evaluating {split_name}", leave=False)
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids, attention_mask)

                if hasattr(self.model, 'multilabel') and self.model.multilabel:
                    loss = self.criterion(logits, labels)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                else:
                    loss = self.criterion(logits, labels)
                    preds = torch.argmax(logits, dim=1)

                epoch_loss += loss.item()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

                # Update progress
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate comprehensive metrics
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        if hasattr(self.model, 'multilabel') and self.model.multilabel:
            accuracy = accuracy_score(all_labels, all_preds)
            macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
            per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

            results = {
                'loss': epoch_loss / len(dataloader),
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'micro_f1': micro_f1,
                'per_class_f1': per_class_f1.tolist()
            }
        else:
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            results = {
                'loss': epoch_loss / len(dataloader),
                'accuracy': accuracy,
                'macro_f1': f1
            }

        return results

    def save_checkpoint(self, epoch, is_best=False):
        """Enhanced checkpoint saving with metadata"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch,
            'training_history': self.training_history,
            'config': self.config,
            'total_epochs': self.total_epochs
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"   🏆 NEW BEST MODEL! F1: {self.best_val_f1:.4f} (epoch {epoch + 1})")

    def load_checkpoint(self, checkpoint_path):
        """Enhanced checkpoint loading with validation"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_f1 = checkpoint['best_val_f1']
        self.best_epoch = checkpoint['best_epoch']
        self.training_history = checkpoint['training_history']

        print(f"📥 Checkpoint loaded from epoch {self.current_epoch}")
        print(f"   🏆 Best F1 so far: {self.best_val_f1:.4f}")

    def save_metrics_csv(self):
        """Enhanced CSV saving with proper formatting"""
        if not self.metrics_history:
            return

        # Prepare data for numpy array with consistent format
        metrics_data = []
        for entry in self.metrics_history:
            row = [
                float(entry['epoch']),
                float(entry['train_loss']),
                float(entry['val_loss']),
                float(entry['train_acc']),
                float(entry['val_acc']),
                float(entry['train_f1']),
                float(entry['test_f1'])
            ]
            metrics_data.append(row)

        # Convert to numpy array
        metrics_array = np.array(metrics_data)

        # Save using np.savetxt with proper formatting
        csv_file = os.path.join(self.output_dir, 'training_metrics.csv')
        header = 'epoch,train_loss,val_loss,train_acc,val_acc,train_f1,test_f1'

        np.savetxt(csv_file, metrics_array, delimiter=',',
                   header=header, comments='', fmt='%.6f')

        print(f"   💾 Metrics saved: {csv_file} ({len(metrics_data)} data points)")

    def check_training_stability(self, train_loss, val_loss):
        """Check for training stability issues"""
        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"⚠️  NaN detected in losses!")
            return False
            
        if train_loss > 10.0 or val_loss > 10.0:
            print(f"⚠️  Very high losses detected - possible training instability")
            
        return True

    def train(self):
        """Enhanced main training loop with comprehensive monitoring"""
        patience_counter = 0
        
        print(f"\n🚀 STARTING RESEARCH TRAINING")
        print(f"=" * 60)
        print(f"🎯 Target: {self.total_epochs} epochs (patience: {self.early_stopping_patience})")
        print(f"📊 Tracking metrics every {self.track_every} epochs")
        print(f"💾 Results → {self.output_dir}")

        for epoch in range(self.current_epoch, self.total_epochs):
            self.current_epoch = epoch

            print(f"\n{'='*60}")
            print(f"🎯 EPOCH {epoch + 1}/{self.total_epochs}")
            print(f"{'='*60}")

            # Training phase
            train_loss, train_acc, train_f1 = self.train_epoch()
            
            # Check for training instability
            if not self.check_training_stability(train_loss, 0):
                print(f"❌ Training unstable - stopping early")
                break

            # Validation phase
            print(f"   🔍 Evaluating...")
            val_results = self.evaluate(self.val_loader, "Validation")
            val_loss = val_results['loss']
            val_acc = val_results['accuracy']
            val_f1 = val_results['macro_f1']

            # Test evaluation (for research tracking)
            test_results = self.evaluate(self.test_loader, "Test")
            test_f1 = test_results['macro_f1']

            # Log comprehensive results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'test_f1': test_f1
            }

            self.training_history.append(epoch_results)

            # CSV tracking at specified intervals
            if (epoch + 1) % self.track_every == 0:
                self.metrics_history.append(epoch_results)
                self.save_metrics_csv()
                print(f"   📈 Metrics logged at epoch {epoch + 1}")

            # Display epoch summary
            print(f"\n📊 EPOCH {epoch + 1} SUMMARY:")
            print(f"   🏋️  Train → Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"   🔍 Val   → Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"   🎯 Test  → F1: {test_f1:.4f}")

            # Best model tracking
            is_best = False
            if val_f1 > self.best_val_f1:
                improvement = val_f1 - self.best_val_f1
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                patience_counter = 0
                is_best = True
                print(f"   🆕 NEW BEST! F1: {val_f1:.4f} (+{improvement:.4f})")
            else:
                patience_counter += 1
                remaining_patience = self.early_stopping_patience - patience_counter
                print(f"   📉 No improvement (patience: {patience_counter}/{self.early_stopping_patience}, {remaining_patience} left)")

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Early stopping check
            if patience_counter >= self.early_stopping_patience:
                print(f"\n⏹️  EARLY STOPPING TRIGGERED!")
                print(f"   📊 No improvement for {self.early_stopping_patience} epochs")
                print(f"   🏆 Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch + 1}")
                break

        # Final cleanup and saves
        if epoch_results not in self.metrics_history:
            self.metrics_history.append(epoch_results)
        self.save_metrics_csv()

        # Save complete training history
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        # Training completion summary
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETED!")
        print(f"{'='*60}")
        print(f"🏆 Best validation F1: {self.best_val_f1:.4f} (epoch {self.best_epoch + 1})")
        print(f"📊 Total epochs: {self.current_epoch + 1}/{self.total_epochs}")
        print(f"💾 Training history: {history_path}")
        print(f"📈 CSV metrics: {os.path.join(self.output_dir, 'training_metrics.csv')}")
        
        if patience_counter >= self.early_stopping_patience:
            print(f"⏰ Stopped early (patience exhausted)")
        else:
            print(f"🎯 Completed full training")

        # Load best model for final evaluation
        best_model_path = os.path.join(self.output_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"🏆 Loaded best model for final evaluation")

        return self.training_history
