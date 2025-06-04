import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

class Trainer:
    def __init__(self, model, data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.data = data
        self.device = device
        
        # Move data to device
        self.data['graph']['X'] = self.data['graph']['X'].to(device)
        self.data['graph']['y'] = self.data['graph']['y'].to(device)
        self.data['graph']['utility']['A'] = self.data['graph']['utility']['A'].to(device)
        self.data['train_test_val']['idx_train'] = self.data['train_test_val']['idx_train'].to(device)
        self.data['train_test_val']['idx_val'] = self.data['train_test_val']['idx_val'].to(device)
        self.data['train_test_val']['idx_test'] = self.data['train_test_val']['idx_test'].to(device)
        
        # Apply weight decay only to the first layer (match GCN paper)
        if hasattr(self.model, 'layers') and isinstance(self.model.layers, torch.nn.ModuleList):
            param_groups = [
                {'params': self.model.layers[0].parameters(), 'weight_decay': 5e-4},
                {'params': [p for l in list(self.model.layers)[1:] for p in l.parameters()], 'weight_decay': 0}
            ]
            self.optimizer = Adam(param_groups, lr=0.01)
        else:
            self.optimizer = Adam([
                {'params': self.model.conv1.parameters(), 'weight_decay': 5e-4},
                {'params': self.model.conv2.parameters(), 'weight_decay': 0}
            ], lr=0.01)
        
        # Initialize metrics tracking
        self.metrics_history = {
            'train': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
            'val': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        }
        
        # Create results directory if it doesn't exist
        os.makedirs('result/stage_5_result/training_curves', exist_ok=True)
    
    def calculate_metrics(self, predictions, labels):
        """Calculate all required metrics"""
        pred_labels = predictions.argmax(dim=1)
        accuracy = (pred_labels == labels).float().mean()
        precision = precision_score(labels.cpu(), pred_labels.cpu(), average='macro', zero_division=0)
        recall = recall_score(labels.cpu(), pred_labels.cpu(), average='macro', zero_division=0)
        f1 = f1_score(labels.cpu(), pred_labels.cpu(), average='macro', zero_division=0)
        return {
            'accuracy': accuracy.item(),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data['graph']['X'], self.data['graph']['utility']['A'])
        loss = F.nll_loss(out[self.data['train_test_val']['idx_train']], 
                         self.data['graph']['y'][self.data['train_test_val']['idx_train']])
        loss.backward()
        self.optimizer.step()
        metrics = self.calculate_metrics(
            out[self.data['train_test_val']['idx_train']],
            self.data['graph']['y'][self.data['train_test_val']['idx_train']]
        )
        metrics['loss'] = loss.item()
        return metrics
    
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data['graph']['X'], self.data['graph']['utility']['A'])
            loss = F.nll_loss(out[self.data['train_test_val']['idx_val']], 
                             self.data['graph']['y'][self.data['train_test_val']['idx_val']])
            metrics = self.calculate_metrics(
                out[self.data['train_test_val']['idx_val']],
                self.data['graph']['y'][self.data['train_test_val']['idx_val']]
            )
            metrics['loss'] = loss.item()
        return metrics
    
    def plot_training_convergence(self, dataset_name):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        epochs = range(len(self.metrics_history['train']['loss']))
        ax1.plot(epochs, self.metrics_history['train']['loss'], label='Train Loss')
        ax1.plot(epochs, self.metrics_history['val']['loss'], label='Val Loss')
        ax1.set_title(f'{dataset_name} - Training Convergence')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(epochs, self.metrics_history['train']['accuracy'], label='Accuracy')
        ax2.plot(epochs, self.metrics_history['train']['precision'], label='Precision')
        ax2.plot(epochs, self.metrics_history['train']['recall'], label='Recall')
        ax2.plot(epochs, self.metrics_history['train']['f1'], label='F1')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig(f'result/stage_5_result/training_curves/{dataset_name}_convergence.png')
        plt.close()
    
    def train(self, dataset_name, epochs=200, patience=50):
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
                self.metrics_history['train'][metric].append(train_metrics[metric])
                self.metrics_history['val'][metric].append(val_metrics[metric])
            print(f"\nEpoch {epoch}:")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Accuracy: {train_metrics['accuracy']:.4f}, "
                  f"Precision: {train_metrics['precision']:.4f}, "
                  f"Recall: {train_metrics['recall']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.4f}, "
                  f"Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save(self.model.state_dict(), f'result/stage_5_result/best_model_{dataset_name}.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
        self.plot_training_convergence(dataset_name)
        self.model.load_state_dict(torch.load(f'result/stage_5_result/best_model_{dataset_name}.pt'))
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data['graph']['X'], self.data['graph']['utility']['A'])
            test_metrics = self.calculate_metrics(
                out[self.data['train_test_val']['idx_test']],
                self.data['graph']['y'][self.data['train_test_val']['idx_test']]
            )
        print("\nFinal Test Results:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1: {test_metrics['f1']:.4f}")
        return test_metrics 