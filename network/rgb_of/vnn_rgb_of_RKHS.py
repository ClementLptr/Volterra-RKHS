import torch
import torch.nn as nn
from config.logger import setup_logger
from typing import Tuple

logger = setup_logger()

class RKHS_VNN(nn.Module):
    """
    Reproducing Kernel Hilbert Space Volterra Neural Network with k-means projectors.
    """
    def __init__(
        self, 
        num_classes: int, 
        num_projector: int = 512,
        num_ch: int = 3, 
        input_shape: Tuple[int, int, int] = (16, 112, 112),
        dropout_rate: float = 0.5,
        pretrained: bool = False,
        use_rff: bool = True,
        pre_determined_projectors: dict = None,
    ) -> None:
        super(RKHS_VNN, self).__init__()
        self.num_projector = num_projector
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.use_rff = use_rff
        self.pre_determined_projectors = pre_determined_projectors or {}
        logger.debug("Initializing architecture configuration.")
        
        # Configuration des 5 couches
        self.architecture_config = {
            'layer1': {'out_channels': 24, 'kernel_size': 3, 'stride': 1},
            'layer2': {'out_channels': 48, 'kernel_size': 3, 'stride': 1},
            'layer3': {'out_channels': 96, 'kernel_size': 3, 'stride': 1},
            'layer4': {'out_channels': 192, 'kernel_size': 3, 'stride': 1},
            'layer5': {'out_channels': 384, 'kernel_size': 3, 'stride': 1},
        }
        
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        last_output = self.architecture_config['layer5']['out_channels']
        self.depth, self.height, self.width = input_shape
        height_out = self.height // (2 ** 5)
        width_out = self.width // (2 ** 5)
        depth_out = self.depth
        
        logger.debug(f"Expected output shape after pool: ({last_output}, {depth_out}, {height_out}, {width_out})")
        fc_input_dim = last_output * depth_out * height_out * width_out
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.num_classes),
        )

        self._initialize_layers()
        self._initialize_weights()
        
        if pretrained:
            logger.debug("Loading pretrained weights.")
            self._load_pretrained_weights()

    def _initialize_layers(self) -> None:
        self.alpha_params = nn.ParameterDict()
        self.bn_layers = nn.ModuleDict()
        self.dropout_layers = nn.ModuleDict()

        for i in range(1, len(self.architecture_config) + 1):
            out_channels = self.architecture_config[f'layer{i}']['out_channels']
            alpha_shape = (self.num_projector, out_channels)
            self.alpha_params[f'alpha_{i}'] = nn.Parameter(
                nn.init.kaiming_normal_(torch.empty(alpha_shape), mode='fan_out', nonlinearity='relu')
            )
            self.bn_layers[f'bn_{i}'] = nn.BatchNorm3d(out_channels)
            # self.dropout_layers[f'dropout_{i}'] = nn.Dropout3d(p=self.dropout_rate)

    def _poly_kernel(self, x: torch.Tensor, projectors: torch.Tensor, degree: int, layer: int) -> torch.Tensor:
        alpha = self.alpha_params[f'alpha_{layer}']
        x_padded = torch.nn.functional.pad(x, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        batch_size, channels, depth, height, width = x_padded.shape
        patches = x_padded.unfold(2, 3, 1).unfold(3, 3, 1).unfold(4, 3, 1)
        patches = patches.contiguous().reshape(batch_size, depth-2, height-2, width-2, channels * 27)
        projectors = projectors.reshape(projectors.shape[0], -1)
        weighted_sum = torch.einsum('bdhwk,pk->bdhwp', patches, projectors)
        output = torch.einsum('bdhwp,po->bdhwo', weighted_sum, alpha)
        output = torch.clamp(1 + output, min=0)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug("Starting forward pass.")
        for layer in range(1, len(self.architecture_config) + 1):
            x = self._poly_kernel(x, self.pre_determined_projectors[layer], degree=2, layer=layer)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)
            x = self.bn_layers[f'bn_{layer}'](x)
            # x = self.dropout_layers[f'dropout_{layer}'](x)
            logger.debug(f"Shape after layer {layer}: {x.shape}")
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def forward_up_to_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Forward pass up to a specific layer for projector computation."""
        logger.debug(f"Forward pass up to layer {layer_idx}.")
        for layer in range(1, layer_idx + 1):
            x = self._poly_kernel(x, self.pre_determined_projectors[layer], degree=2, layer=layer)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)
            x = self.bn_layers[f'bn_{layer}'](x)
            x = self.dropout_layers[f'dropout_{layer}'](x)
        return x

    def _initialize_weights(self) -> None:
        logger.debug("Initializing weights for all layers.")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def get_1x_lr_params(model: RKHS_VNN):
    for layer in model.bn_layers.values():
        for param in layer.parameters():
            if param.requires_grad:
                yield param
    for alpha in model.alpha_params.values():
        if alpha.requires_grad:
            yield alpha
    
    for param in model.classifier.parameters():
        if param.requires_grad:
            yield param
            
if __name__ == '__main__':
    logger.debug("Initializing the RKHS_VNN model.")
    model = RKHS_VNN(
        num_classes=10,
        num_projector=15,
        num_ch=3,
        input_shape=(16, 112, 112),
        dropout_rate=0.5,
        pretrained=False,
        pre_determined_projectors={
            1: torch.randn(15, 3, 3, 3, 3),
            2: torch.randn(15, 24, 3, 3, 3),
            3: torch.randn(15, 48, 3, 3, 3),
            4: torch.randn(15, 96, 3, 3, 3),
            5: torch.randn(15, 192, 3, 3, 3),
        }
    )
    assert isinstance(model, RKHS_VNN)
    logger.info("RKHS_VNN initialized successfully.")
    x = torch.randn(2, 3, 16, 112, 112)
    output = model(x)
    assert output.shape == (2, 10)
    logger.info("Forward pass successful.")