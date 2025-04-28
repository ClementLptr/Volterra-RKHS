import torch
import torch.nn as nn
from typing import Tuple

class CNN5Layers(nn.Module):
    """
    Convolutional Neural Network (CNN) with 5 convolutional layers for classification tasks.
    """
    def __init__(
        self, 
        num_classes: int, 
        input_shape: Tuple[int, int, int] = (16, 112, 112),
        num_channels: int = 3, 
        dropout_rate: float = 0.3,
        pretrained: bool = False
    ) -> None:
        super(CNN5Layers, self).__init__()
        
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        
        # Définition des 5 couches convolutionnelles
        self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        # Couches de pooling - réduire la profondeur de pooling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # Couches entièrement connectées (FC)
        # On calcule la dimension de l'entrée après le pooling
        self.fc_input_dim = 512 * (input_shape[0] // 8) * (input_shape[1] // 8) * (input_shape[2] // 8)  # Reduce division factor
        
        self.fc1 = nn.Linear(self.fc_input_dim, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Passer par les couches convolutionnelles et de pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = torch.relu(self.conv5(x))  # Remove pooling after last conv layer
        
        # Aplatir les sorties pour passer à la couche entièrement connectée
        x = x.view(-1, self.fc_input_dim)
        
        # Passer par les couches entièrement connectées
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
