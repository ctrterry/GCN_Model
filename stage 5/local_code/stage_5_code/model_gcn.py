import torch
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16, dropout=0.5):
        """
        Initialize GCN model
        Args:
            num_features (int): Number of input features
            num_classes (int): Number of output classes
            hidden_channels (int): Number of hidden channels
            dropout (float): Dropout rate (default: 0.6, optimal for Citeseer)
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        """
        Forward pass
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Graph connectivity
        """
        # First Graph Convolution Layer
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        # Second Graph Convolution Layer
        x = self.conv2(x, edge_index)
        
        return torch.nn.functional.log_softmax(x, dim=1) 