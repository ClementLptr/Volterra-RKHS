import torch
import torch.nn as nn
import torch.nn.functional as F
from config.logger import setup_logger

logger = setup_logger()

class VNN_F(nn.Module):
    def __init__(self, num_classes, num_ch=3, pretrained=False):
        """
        Initialize the VNN_F model.

        Args:
            num_classes (int): Number of output classes for the classification task.
            num_ch (int, optional): Number of input channels (default is 3).
            pretrained (bool, optional): Whether to use pretrained weights (default is False).
        """
        super(VNN_F, self).__init__()

        # Define constants for output channels
        self.Q0 = 2
        self.Q1 = 2
        self.Q1_red = 2
        self.Q2 = 2

        self.nch_out0 = 96
        self.nch_out1 = 256
        self.nch_out1_red = 96
        self.nch_out2 = 192

        # Define Layers
        # First Layer
        self.conv10 = nn.Conv3d(num_ch, self.nch_out0, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.bn10 = nn.BatchNorm3d(self.nch_out0)
        self.conv20 = nn.Conv3d(num_ch, 2*self.Q0*self.nch_out0, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.bn20 = nn.BatchNorm3d(self.nch_out0)

        # Second Layer
        self.conv11 = nn.Conv3d(self.nch_out0, self.nch_out1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn11 = nn.BatchNorm3d(self.nch_out1)
        self.conv21 = nn.Conv3d(self.nch_out0, 2*self.Q1*self.nch_out1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.bn21 = nn.BatchNorm3d(self.nch_out1)

        self.fc8 = nn.Linear(200704, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights of the convolutional, batch norm, and linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _volterra_kernel_approximation(self, tensor1, tensor2, num_terms, num_channels_out):
        """
        Approximates the Volterra kernel by combining pairwise multiplicative interactions 
        between two input tensors.

        Args:
            tensor1 (torch.Tensor): First input tensor, shape (batch_size, channels, depth, height, width).
            tensor2 (torch.Tensor): Second input tensor, shape (batch_size, 2*num_terms*num_channels_out, depth, height, width).
            num_terms (int): Number of multiplicative terms in the kernel approximation.
            num_channels_out (int): Number of output channels for the final result.

        Returns:
            torch.Tensor: Approximated tensor, shape (batch_size, num_channels_out, depth, height, width).
        """
        tensor_mul = torch.mul(tensor2[:, 0:num_terms * num_channels_out, :, :, :], 
                              tensor2[:, num_terms * num_channels_out:2 * num_terms * num_channels_out, :, :, :])
        
        tensor_add = torch.zeros_like(tensor1)
        
        for q in range(num_terms):
            tensor_add = torch.add(tensor_add, tensor_mul[:, (q * num_channels_out):((q * num_channels_out) + num_channels_out), :, :, :])
                
        return tensor_add

    def forward(self, x, activation=False):
        """
        Forward pass through the network.

        Args:
            x (Tensor): The input tensor.
            activation (bool, optional): If True, return intermediate activations (default is False).

        Returns:
            Tensor: The output logits.
        """
        # Layer 1
        x10 = self.conv10(x)
        x10 = self.bn10(x10)
        x20 = self.conv20(x)
        x20_add = self._volterra_kernel_approximation(x10, x20, self.Q0, self.nch_out0)
        x20_add = self.bn20(x20_add)
        
        # Debug prints to check shapes
        logger.debug(f"x10 shape: {x10.shape}")
        logger.debug(f"x20 shape: {x20.shape}")
        logger.debug(f"x20_add shape: {x20_add.shape}")
        
        x = torch.add(x10, x20_add)

        # Layer 2
        x11 = self.conv11(x)
        x11 = self.bn11(x11)
        x21 = self.conv21(x)
        x21_add = self._volterra_kernel_approximation(x11, x21, self.Q1, self.nch_out1)
        x21_add = self.bn21(x21_add)
        
        # Debug prints to check shapes
        logger.debug(f"x11 shape: {x11.shape}")
        logger.debug(f"x21 shape: {x21.shape}")
        logger.debug(f"x21_add shape: {x21_add.shape}")

        x = self.pool1(torch.add(x11, x21_add))

        # Flatten and pass through fully connected layer
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        logger.debug(f"x shape: {x.shape}")
        logits = self.fc8(x)

        return logits

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and batch norm layers of the net.
    """
    b = [model.conv10, model.bn10, model.conv20, model.bn20, model.conv11, model.bn11, model.conv21, model.bn21]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k
                
def get_10x_lr_params(model):
    """
    Generator for parameters with 10x learning rate (fully connected layer).
    """
    for param in model.fc8.parameters():
        if param.requires_grad:
            yield param

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    model = VNN_F(num_classes=101, pretrained=False)
    outputs = model(inputs)
    logger.debug(outputs.size())