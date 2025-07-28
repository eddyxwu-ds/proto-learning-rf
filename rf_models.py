import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RFNet(nn.Module):
    """
    1D CNN for RF signal classification with prototype learning.
    Designed for complex signals converted to magnitude and phase channels.
    """
    def __init__(self, num_hidden_units=2, num_classes=8, scale=2):
        super(RFNet, self).__init__()
        self.scale = scale
        
        # 1D Convolutional layers for signal processing
        # Input: [batch, 2, signal_length] (magnitude and phase channels)
        
        # First conv block
        self.conv1_1 = nn.Conv1d(2, 32, kernel_size=7, padding=3)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size=7, padding=3)
        self.prelu1_2 = nn.PReLU()
        
        # Second conv block
        self.conv2_1 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        
        # Third conv block
        self.conv3_1 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.prelu3_2 = nn.PReLU()
        
        # Global average pooling and final linear layer
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128, num_hidden_units)
        
        # DCE loss for prototype learning
        self.dce = dce_loss(num_classes, num_hidden_units)
    
    def forward(self, x):
        # x shape: [batch, 2, signal_length]
        
        # First conv block
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool1d(x, 2)
        
        # Second conv block
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool1d(x, 2)
        
        # Third conv block
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool1d(x, 2)
        
        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1)  # [batch, 128, 1]
        x = x.squeeze(-1)  # [batch, 128]
        
        # Final feature extraction
        x1 = self.preluip1(self.ip1(x))  # [batch, num_hidden_units]
        
        # DCE loss computation
        centers, x = self.dce(x1)
        
        # Output probabilities
        output = F.log_softmax(self.scale * x, dim=1)
        
        return x1, centers, x, output

class dce_loss(torch.nn.Module):
    """
    Distance-based Cross Entropy loss for prototype learning.
    Same as in the original MNIST implementation.
    """
    def __init__(self, n_classes, feat_dim, init_weight=True):
        super(dce_loss, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.feat_dim, self.n_classes), requires_grad=True)
        if init_weight:
            self.__init_weight()
    
    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)
    
    def forward(self, x):
        # x shape: [batch, feat_dim]
        # centers shape: [feat_dim, n_classes]
        
        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(self.centers, 2), 0, keepdim=True)
        features_into_centers = 2 * torch.matmul(x, self.centers)
        dist = features_square + centers_square - features_into_centers
        
        return self.centers, -dist

def regularization(features, centers, labels):
    """
    Regularization term to pull features closer to their class centers.
    """
    distance = (features - torch.t(centers)[labels])
    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)
    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]
    return distance 