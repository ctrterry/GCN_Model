import sys
import os
import torch
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.append(os.path.dirname(__file__))
from model_gcn import GCN
from train_eval import Trainer
from dataset_loader import DatasetLoader

def ablation_dropout_all_datasets(dropout_rates, datasets=['citeseer', 'cora', 'pubmed'], hidden_channels=16, epochs=50):
    results = {}
    for dataset_name in datasets:
        data = DatasetLoader(dataset_name).load_data()
        num_features = data.x.shape[1]
        num_classes = int(data.y.max().item()) + 1
        accs = []
        for dropout in dropout_rates:
            print(f"\nTesting {dataset_name} dropout={dropout}")
            model = GCN(num_features, num_classes, hidden_channels=hidden_channels, dropout=dropout)
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
        plt.plot(dropout_rates, accs, marker='o', label='Test', color='b')
        plt.scatter([dropout_rates[best_idx]], [accs[best_idx]], color='red', label=f'Best: {dropout_rates[best_idx]} (Acc={accs[best_idx]:.4f})')
        plt.title(dataset_name.capitalize())
        plt.xlabel('Dropout Rate')
        plt.ylabel('Accuracy')
        plt.ylim(0.5, 1.0)
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig('result/stage_5_result/ablation_dropout_all_datasets.png')
    plt.show()

if __name__ == '__main__':
    dropout_rates = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
    ablation_dropout_all_datasets(dropout_rates) 