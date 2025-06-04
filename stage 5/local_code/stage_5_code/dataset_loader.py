import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

class DatasetLoader:
    def __init__(self, dataset_name, root='data/stage_5_data'):
        """
        Initialize dataset loader
        Args:
            dataset_name (str): Name of the dataset ('cora', 'pubmed', or 'citeseer')
            root (str): Root directory where data is stored
        """
        self.dataset_name = dataset_name
        self.root = root
        self.dataset = None
        self.data = None
        
    def normalize_features(self, x):
        """Row-wise feature normalization"""
        row_sum = x.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1  # Avoid division by zero
        return x / row_sum
        
    def load_data(self):
        """Load and preprocess the dataset"""
        # Load dataset with built-in feature normalization
        self.dataset = Planetoid(
            root=self.root,
            name=self.dataset_name,
            transform=NormalizeFeatures()
        )
        self.data = self.dataset[0]
        
        # Apply additional row-wise normalization
        self.data.x = self.normalize_features(self.data.x)
        
        return self.data
    
    def get_data(self):
        """Get the loaded data"""
        if self.data is None:
            self.load_data()
        return self.data 