import sys
import os
import torch

# Add the local_code directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../local_code'))

from stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from stage_5_code.model_gcn import GCN
from stage_5_code.train_eval import Trainer

def main():
    # List of datasets to process
    datasets = ['cora', 'pubmed', 'citeseer']
    
    # Best hyperparameters from ablation study
    best_hyperparams = {
        'citeseer': {'dropout': 0.8, 'hidden_channels': 64, 'lr': 0.05, 'weight_decay': 0.001}, # 0.7230
        'cora':     {'dropout': 0.7, 'hidden_channels': 64, 'lr': 0.01, 'weight_decay': 0.0005}, # 0.8210
        'pubmed':   {'dropout': 0.2, 'hidden_channels': 32, 'lr': 0.005, 'weight_decay': 0.0005}, # 0.7670
    }
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Processing {dataset_name} dataset")
        print(f"{'='*50}\n")
        
        # Load dataset
        dataset_loader = Dataset_Loader(dName=dataset_name)
        data = dataset_loader.load()
        
        # Initialize model with best hyperparameters
        num_features = data['graph']['X'].shape[1]
        num_classes = len(torch.unique(data['graph']['y']))
        params = best_hyperparams[dataset_name]
        model = GCN(num_features=num_features, num_classes=num_classes, hidden_channels=params['hidden_channels'], dropout=params['dropout'])
        
        # Initialize trainer
        trainer = Trainer(model, data)
        # Set optimizer hyperparameters
        for g in trainer.optimizer.param_groups:
            if 'weight_decay' in g:
                g['weight_decay'] = params['weight_decay']
            if 'lr' in g:
                g['lr'] = params['lr']
        
        # Train model
        test_metrics = trainer.train(dataset_name)
        
        print(f"\nCompleted training on {dataset_name}")
        print(f"Final test metrics:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1: {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    main() 