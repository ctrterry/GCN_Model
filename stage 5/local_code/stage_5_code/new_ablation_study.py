import sys
import os
import torch
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.append(os.path.dirname(__file__))
from model_gcn import GCN
from train_eval import Trainer
from dataset_loader import DatasetLoader

def build_gcn(num_features, num_classes, hidden_channels, depth, dropout=0.6):
    """
    Dynamically build a GCN with variable depth and hidden dimensions.
    """
    import torch.nn as nn
    from torch_geometric.nn import GCNConv
    class DeepGCN(torch.nn.Module):
        def __init__(self, num_features, num_classes, hidden_channels, depth, dropout):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(GCNConv(num_features, hidden_channels))
            for _ in range(depth - 2):
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
            self.layers.append(GCNConv(hidden_channels, num_classes))
            self.dropout = dropout
        def forward(self, x, edge_index):
            for i, conv in enumerate(self.layers):
                x = conv(x, edge_index)
                if i < len(self.layers) - 1:
                    x = torch.relu(x)
                    x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            return torch.nn.functional.log_softmax(x, dim=1)
    return DeepGCN(num_features, num_classes, hidden_channels, depth, dropout)

def ablation_study(depths, hidden_dims, dataset_name='citeseer', dropout=0.6, epochs=50):
    # Load data
    data = DatasetLoader(dataset_name).load_data()
    num_features = data['graph']['X'].shape[1]
    num_classes = int(data['graph']['y'].max().item()) + 1

    results_depth = []
    results_dim = []

    # Ablation on depth (fix hidden_dim=16)
    for depth in depths:
        print(f"\nTesting depth={depth}, hidden_dim=16")
        model = build_gcn(num_features, num_classes, 16, depth, dropout)
        trainer = Trainer(model, data)
        test_metrics = trainer.train(dataset_name, epochs=epochs, patience=10)
        results_depth.append((depth, test_metrics['accuracy']))

    # Ablation on hidden_dim (fix depth=2)
    for hidden_dim in hidden_dims:
        print(f"\nTesting depth=2, hidden_dim={hidden_dim}")
        model = build_gcn(num_features, num_classes, hidden_dim, 2, dropout)
        trainer = Trainer(model, data)
        test_metrics = trainer.train(dataset_name, epochs=epochs, patience=10)
        results_dim.append((hidden_dim, test_metrics['accuracy']))

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([d for d, _ in results_depth], [a for _, a in results_depth], marker='o')
    plt.title('Depth vs Accuracy')
    plt.xlabel('Depth (# layers)')
    plt.ylabel('Test Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([h for h, _ in results_dim], [a for _, a in results_dim], marker='o')
    plt.title('Hidden Dim vs Accuracy')
    plt.xlabel('Hidden Dimension')
    plt.ylabel('Test Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'result/stage_5_result/ablation_depth_dim.png')
    plt.show()

if __name__ == '__main__':
    depths = [2, 3, 4]
    hidden_dims = [8, 16, 32, 64]
    ablation_study(depths, hidden_dims) 