import sys
import os
import torch
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.append(os.path.dirname(__file__))
from train_eval import Trainer
from dataset_loader import DatasetLoader

def build_gcn(num_features, num_classes, hidden_channels, depth, dropout=0.5):
    import torch.nn as nn
    from torch_geometric.nn import GCNConv
    class DeepGCN(torch.nn.Module):
        def __init__(self, num_features, num_classes, hidden_channels, depth, dropout):
            super().__init__()
            self.layers = nn.ModuleList()
            if depth == 1:
                self.layers.append(GCNConv(num_features, num_classes))
            else:
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

def ablation_depth_all_datasets(depths, datasets=['citeseer', 'cora', 'pubmed'], hidden_channels=16, dropout=0.5, epochs=50):
    results = {}
    for dataset_name in datasets:
        data = DatasetLoader(dataset_name).load_data()
        num_features = data.x.shape[1]
        num_classes = int(data.y.max().item()) + 1
        accs = []
        for depth in depths:
            print(f"\nTesting {dataset_name} depth={depth}")
            model = build_gcn(num_features, num_classes, hidden_channels=hidden_channels, depth=depth, dropout=dropout)
            trainer = Trainer(model, {'graph': {'X': data.x, 'y': data.y, 'utility': {'A': data.edge_index}},
                                      'train_test_val': {'idx_train': data.train_mask.nonzero(as_tuple=True)[0],
                                                         'idx_val': data.val_mask.nonzero(as_tuple=True)[0],
                                                         'idx_test': data.test_mask.nonzero(as_tuple=True)[0]}})
            test_metrics = trainer.train(dataset_name, epochs=epochs, patience=10)
            accs.append(test_metrics['accuracy'])
        results[dataset_name] = accs

    # Plotting
    plt.figure(figsize=(18, 6))
    for i, dataset_name in enumerate(datasets):
        plt.subplot(1, 3, i+1)
        accs = results[dataset_name]
        best_idx = int(torch.tensor(accs).argmax())
        plt.plot(depths, accs, marker='o', label='Test', color='b')
        plt.scatter([depths[best_idx]], [accs[best_idx]], color='red', label=f'Best: {depths[best_idx]} (Acc={accs[best_idx]:.4f})')
        plt.title(dataset_name.capitalize())
        plt.xlabel('Number of layers')
        plt.ylabel('Accuracy')
        plt.ylim(0.5, 1.0)
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig('result/stage_5_result/ablation_depth_all_datasets.png')
    plt.show()

if __name__ == '__main__':
    depths = list(range(1, 11))
    ablation_depth_all_datasets(depths) 