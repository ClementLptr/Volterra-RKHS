import torch
from config.logger import setup_logger
from network.base.base_vnn import BaseVNN
from torch import nn

logger = setup_logger()

class VNN_F(BaseVNN):
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

        # Third Layer
        # self.conv11_red = nn.Conv3d(self.nch_out1, self.nch_out1_red, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        # self.bn11_red = nn.BatchNorm3d(self.nch_out1_red)
        # self.conv21_red = nn.Conv3d(self.nch_out1, 2*self.Q1_red*self.nch_out1_red, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        # self.bn21_red = nn.BatchNorm3d(self.nch_out1_red)

        # # Fourth Layer
        # self.conv12 = nn.Conv3d(self.nch_out1_red, self.nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.bn12 = nn.BatchNorm3d(self.nch_out2)
        # self.conv22 = nn.Conv3d(self.nch_out1_red, 2*self.Q2*self.nch_out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # self.bn22 = nn.BatchNorm3d(self.nch_out2)
        
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
        x21 = self.conv21(x)
        x21_add = self._volterra_kernel_approximation(x11, x21, self.Q1, self.nch_out1)
        x21_add = self.bn21(x21_add)
        
        # Debug prints to check shapes
        logger.debug(f"x11 shape: {x11.shape}")
        logger.debug(f"x21 shape: {x21.shape}")
        logger.debug(f"x21_add shape: {x21_add.shape}")

        x = self.pool1(torch.add(x11, x21_add))

        # # Layer 3
        # x11_red = self.conv11_red(x)
        # x21_red = self.conv21_red(x)
        # x21_red_add = self._volterra_kernel_approximation(x11_red, x21_red, self.Q1_red, self.nch_out1_red)
        # x21_red_add = self.bn21_red(x21_red_add)
        
        # # Debug prints to check shapes
        # logger.debug(f"x11_red shape: {x11_red.shape}")
        # logger.debug(f"x21_red shape: {x21_red.shape}")
        # logger.debug(f"x21_red_add shape: {x21_red_add.shape}")
        
        # x = torch.add(x11_red, x21_red_add)

        # # Layer 4
        # x12 = self.conv12(x)
        # x22 = self.conv22(x)
        # x22_add = self._volterra_kernel_approximation(x12, x22, self.Q2, self.nch_out2)
        # x22_add = self.bn22(x22_add)
        
        # # Debug prints to check shapes
        # logger.debug(f"x12 shape: {x12.shape}")
        # logger.debug(f"x22 shape: {x22.shape}")
        # logger.debug(f"x22_add shape: {x22_add.shape}")
        
        # x = self.pool2(torch.add(x12, x22_add))

        # Flatten and pass through fully connected layer
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        logger.debug(f"x shape: {x.shape}")
        logits = self.fc8(x)

        return logits

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [ model.conv10, model.bn10, model.conv20, model.bn20, model.conv11, model.bn11, model.conv21, model.bn21] #, model.conv11_red, model.bn11_red, model.conv21_red, model.bn21_red, model.conv11, model.bn12, model.conv22, model.bn22] # model.conv10, model.bn10, model.conv20, model.bn20, 
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
