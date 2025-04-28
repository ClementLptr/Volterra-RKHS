import torch
import torch.nn as nn
from config.logger import setup_logger
from typing import Tuple

logger = setup_logger()

class RKHS_VNN(nn.Module):
    """
    Reproducing Kernel Hilbert Space Volterra Neural Network with k-means projectors, convolutional blocks, and residual connections.
    Optimized for reduced parameter count.
    """
    def __init__(
        self, 
        num_classes: int, 
        num_projector: int = 8,
        num_ch: int = 3, 
        input_shape: Tuple[int, int, int] = (16, 112, 112),
        dropout_rate: float = 0.3,
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
            'layer1': {'out_channels': 16, 'kernel_size': 3, 'stride': 1},
            'layer2': {'out_channels': 24, 'kernel_size': 3, 'stride': 1},
            'layer3': {'out_channels': 48, 'kernel_size': 3, 'stride': 1},
            'layer4': {'out_channels': 64, 'kernel_size': 3, 'stride': 1},
            'layer5': {'out_channels': 128, 'kernel_size': 3, 'stride': 1},
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
            nn.Linear(fc_input_dim, 256),
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
        self.conv_layers = nn.ModuleDict()
        self.residual_convs = nn.ModuleDict()  # Pour les convolutions 1x1x1 des connexions résiduelles

        in_channels = self.input_shape[0]
        for i in range(1, len(self.architecture_config) + 1):
            out_channels = self.architecture_config[f'layer{i}']['out_channels']
            kernel_size = self.architecture_config[f'layer{i}']['kernel_size']
            stride = self.architecture_config[f'layer{i}']['stride']

            # Seulement pour les 3 premières couches RKHS
            if i <= 3:
                alpha_shape = (self.num_projector, out_channels)
                self.alpha_params[f'alpha_{i}'] = nn.Parameter(
                    nn.init.kaiming_normal_(torch.empty(alpha_shape), mode='fan_out', nonlinearity='relu')
                )

            
            # Couche de convolution séparable en profondeur
            self.conv_layers[f'conv_{i}'] = nn.Sequential(
                nn.Conv3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=out_channels,
                ),
                nn.Conv3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                )
            )
            
            # Convolution 1x1x1 pour connexion résiduelle si nécessaire
            if in_channels != out_channels:
                self.residual_convs[f'res_{i}'] = nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False
                )
            
            # Couche de normalisation
            self.bn_layers[f'bn_{i}'] = nn.BatchNorm3d(out_channels)
            
            in_channels = out_channels

    def update_projectors(self, projectors: torch.Tensor, layer: int) -> None:
        if layer in self.pre_determined_projectors:
            self.pre_determined_projectors[layer] = projectors
            logger.debug(f"Updated projectors for layer {layer}.")
        else:
            logger.warning(f"Layer {layer} does not exist in pre_determined_projectors.")
        
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
            residual = x  # Pour la connexion résiduelle

            if layer <= 3:
                # Couches RKHS
                x = self._poly_kernel(x, self.pre_determined_projectors[layer], degree=2, layer=layer)
                x = x.permute(0, 4, 1, 2, 3)
            else:
                # Couches 4 et 5 : convolutions standard
                x = self.conv_layers[f'conv_{layer}'](x)

            if f'res_{layer}' in self.residual_convs:
                residual = self.residual_convs[f'res_{layer}'](residual)

            if residual.shape == x.shape:
                x = x + residual

            x = self.pool(x)
            x = self.bn_layers[f'bn_{layer}'](x)
            logger.debug(f"Shape after layer {layer}: {x.shape}")

        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

    
    def forward_up_to_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        logger.debug(f"Forward pass up to layer {layer_idx}.")
        for layer in range(1, layer_idx + 1):
            residual = x
            x = self._poly_kernel(x, self.pre_determined_projectors[layer], degree=2, layer=layer)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.conv_layers[f'conv_{layer}'](x)
            
            if f'res_{layer}' in self.residual_convs:
                residual = self.residual_convs[f'res_{layer}'](residual)
            
            if residual.shape == x.shape:
                x = x + residual
            
            x = self.pool(x)
            x = self.bn_layers[f'bn_{layer}'](x)
        return x

    def _initialize_weights(self) -> None:
        logger.debug("Initializing weights for all layers.")
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
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
    for layer in model.conv_layers.values():
        for param in layer.parameters():
            if param.requires_grad:
                yield param
    for layer in model.residual_convs.values():
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
        num_projector=8,
        num_ch=3,
        input_shape=(16, 112, 112),
        dropout_rate=0.3,
        pretrained=False,
                pre_determined_projectors={
            1: torch.randn(8, 3, 3, 3, 3),
            2: torch.randn(8, 16, 3, 3, 3),
            3: torch.randn(8, 24, 3, 3, 3),
            # 4 et 5 supprimés car plus utilisés
        }

    )
    assert isinstance(model, RKHS_VNN)
    logger.info("RKHS_VNN initialized successfully.")
    x = torch.randn(2, 3, 16, 112, 112)
    output = model(x)
    assert output.shape == (2, 10)
    logger.info("Forward pass successful.")