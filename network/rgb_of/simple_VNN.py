from config.logger import setup_logger
from config.dataloaders.dataset import VideoDataset
import torch
import torch.nn as nn
import os

# Set environment variable for CUDA allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setup logger
logger = setup_logger()

class SimpleVNN(nn.Module):
    """A simple 3D CNN for video classification."""
    def __init__(self, num_classes, num_ch=3):
        super(SimpleVNN, self).__init__()
        self.conv1 = nn.Conv3d(num_ch, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # Adjust fc input size based on input shape (3, 16, 112, 112)
        self.fc = nn.Linear(32 * 4 * 28 * 28, num_classes)  # After two poolings: 16/4=4, 112/4=28
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        logger.debug(f"Input shape: {x.shape}")
        x = torch.relu(self.bn1(self.conv1(x)))
        logger.debug(f"After conv1: {x.shape}, mean: {x.mean():.4f}, std: {x.std():.4f}")
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        logger.debug(f"After conv2: {x.shape}, mean: {x.mean():.4f}, std: {x.std():.4f}")
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logger.debug(f"After flatten: {x.shape}")
        x = self.dropout(x)
        x = self.fc(x)
        logger.debug(f"Output shape: {x.shape}")
        return x
