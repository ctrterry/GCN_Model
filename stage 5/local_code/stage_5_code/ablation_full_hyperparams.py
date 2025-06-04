import sys
import os
import torch
import matplotlib.pyplot as plt

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

def ablation_one_param(param_name, param_values, fixed_params, dataset_name, epochs=50):
    data = DatasetLoader(dataset_name).load_data()
    num_features = data.x.shape[1]
    num_classes = int(data.y.max().item()) + 1
    accs = []
    for val in param_values:
        params = fixed_params.copy()
        params[param_name] = val
        model = build_gcn(num_features, num_classes, hidden_channels=params['hidden_channels'], depth=params['depth'], dropout=params['dropout'])
        trainer = Trainer(
            model,
            {'graph': {'X': data.x, 'y': data.y, 'utility': {'A': data.edge_index}},
             'train_test_val': {'idx_train': data.train_mask.nonzero(as_tuple=True)[0],
                                'idx_val': data.val_mask.nonzero(as_tuple=True)[0],
                                'idx_test': data.test_mask.nonzero(as_tuple=True)[0]}},
        )
        # Set optimizer hyperparams
        for g in trainer.optimizer.param_groups:
            if 'weight_decay' in g:
                g['weight_decay'] = params['weight_decay']
            if 'lr' in g:
                g['lr'] = params['lr']
        test_metrics = trainer.train(dataset_name, epochs=epochs, patience=10)
        accs.append(test_metrics['accuracy'])
    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(param_values, accs, marker='o')
    best_idx = int(torch.tensor(accs).argmax())
    plt.scatter([param_values[best_idx]], [accs[best_idx]], color='red', label=f'Best: {param_values[best_idx]} (Acc={accs[best_idx]:.4f})')
    plt.title(f'{dataset_name.capitalize()} - {param_name} ablation')
    plt.xlabel(param_name)
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('result/stage_5_result/ablation_full', exist_ok=True)
    plt.savefig(f'result/stage_5_result/ablation_full/{dataset_name}_{param_name}_ablation.png')
    plt.close()
    return param_values[best_idx], accs[best_idx]

def full_ablation(datasets=['citeseer', 'cora', 'pubmed'], epochs=50):
    # Hyperparameter search spaces
    dropout_rates = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
    depths = [2, 3, 4]
    hidden_units = [8, 16, 32, 64]
    lrs = [0.001, 0.005, 0.01, 0.05]
    weight_decays = [0, 1e-5, 5e-4, 1e-3]
    summary = {}
    for dataset in datasets:
        print(f'\n===== {dataset.upper()} =====')
        # Default/fixed values
        params = {
            'dropout': 0.5,
            'depth': 2,
            'hidden_channels': 16,
            'lr': 0.01,
            'weight_decay': 5e-4
        }
        # 1. Dropout
        best_dropout, _ = ablation_one_param('dropout', dropout_rates, params, dataset, epochs)
        params['dropout'] = best_dropout
        # 2. Depth
        best_depth, _ = ablation_one_param('depth', depths, params, dataset, epochs)
        params['depth'] = best_depth
        # 3. Hidden units
        best_hidden, _ = ablation_one_param('hidden_channels', hidden_units, params, dataset, epochs)
        params['hidden_channels'] = best_hidden
        # 4. Learning rate
        best_lr, _ = ablation_one_param('lr', lrs, params, dataset, epochs)
        params['lr'] = best_lr
        # 5. Weight decay
        best_wd, best_acc = ablation_one_param('weight_decay', weight_decays, params, dataset, epochs)
        params['weight_decay'] = best_wd
        summary[dataset] = {
            'dropout': best_dropout,
            'depth': best_depth,
            'hidden_channels': best_hidden,
            'lr': best_lr,
            'weight_decay': best_wd,
            'accuracy': best_acc
        }
    # Print summary table
    print('\n===== Hyperparameter Ablation Summary =====')
    print(f"{'Dataset':<10} {'Dropout':<8} {'Depth':<6} {'Hidden':<8} {'LR':<8} {'WeightDecay':<12} {'Accuracy':<8}")
    for dataset in datasets:
        s = summary[dataset]
        print(f"{dataset:<10} {s['dropout']:<8} {s['depth']:<6} {s['hidden_channels']:<8} {s['lr']:<8} {s['weight_decay']:<12} {s['accuracy']:<8.4f}")

if __name__ == '__main__':
    full_ablation() 