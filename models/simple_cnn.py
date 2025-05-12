

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------------------------
# Set seed for reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
class simple(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super(simple, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)  # Dilated Conv
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=2,dilation=2)  # Dilated Conv
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)  # 1x1 conv for segmentation

        # Initialize weights
        self.initialize_weights()
        self.sigmoid = nn.Sigmoid() 
    def forward(self, x):
        x = x[:, [0, 1, 2, 3,4,5,6,7,8], :, :]  # Keep the first 5 channels (RGB, NDWI, Slope)

        x1 = torch.nn.functional.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x2 = torch.nn.functional.leaky_relu(self.bn2(self.conv2(x1)), negative_slope=0.01)

        x3 = torch.nn.functional.leaky_relu(self.bn3(self.conv3(x2)), negative_slope=0.01)
        x4 = torch.nn.functional.leaky_relu(self.bn4(self.conv4(x3)), negative_slope=0.01)

        x5 = torch.nn.functional.leaky_relu(self.bn5(self.conv5(x4)), negative_slope=0.01)
        x6 = torch.nn.functional.leaky_relu(self.bn6(self.conv6(x5)), negative_slope=0.01)

        x_out = self.output_layer(x6)
        x = self.sigmoid(x_out)
        return x  # Sigmoid is applied in the loss function

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
class M31(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(M31, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)  # Dilation = 2
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4)  # Dilation = 4
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=8, dilation=8)  # Dilation = 8
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)  # 1x1 conv for segmentation

        # Initialize weights
        self.initialize_weights()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Select specific bands
        x = x[:, [0, 1, 2, 3, 4, 5], :, :]
        
        x1 = torch.nn.functional.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x2 = torch.nn.functional.leaky_relu(self.bn2(self.conv2(x1)), negative_slope=0.01)

        x3 = torch.nn.functional.leaky_relu(self.bn3(self.conv3(x2)), negative_slope=0.01)
        x4 = torch.nn.functional.leaky_relu(self.bn4(self.conv4(x3)), negative_slope=0.01)
        x5 = torch.nn.functional.leaky_relu(self.bn5(self.conv5(x4)), negative_slope=0.01)
        
        x6 = torch.nn.functional.leaky_relu(self.bn6(self.conv6(x5)), negative_slope=0.01)

        x_out = self.output_layer(x6)
        x = self.sigmoid(x_out)
        
        return x  # Sigmoid applied for binary segmentation

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module for capturing multi-scale context."""
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global context
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)  

        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        x5 = self.global_avg_pool(x)
        x5 = self.conv5(x5)
        x5 = F.interpolate(x5, size=x.shape[2:], mode="bilinear", align_corners=False)  # Upsample

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)  # Concatenate multi-scale features
        x = self.final_conv(x)
        return x

class M3(nn.Module):
    """Simplified DeepLabV3 model using ASPP."""
    def __init__(self, in_channels=5, out_channels=1):
        super(M3, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(128, 128)
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        x = x[:, [0, 1, 2, 3,4], :, :]
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.final_conv(x)
        return x  # Apply sigmoid in loss function if binary segmentation

## Deeplab with more layers
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module for capturing multi-scale context."""
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global context
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)  

        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        x5 = self.global_avg_pool(x)
        x5 = self.conv5(x5)
        x5 = F.interpolate(x5, size=x.shape[2:], mode="bilinear", align_corners=False)  # Upsample

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)  # Concatenate multi-scale features
        x = self.final_conv(x)
        return x

class M4(nn.Module):
    """Modified DeepLabV3 model using ASPP with an enhanced backbone."""
    def __init__(self, in_channels=5, out_channels=1):
        super(M4, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(256, 128)
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        x = x[:, :5, :, :]  # Select first 5 channels
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.final_conv(x)
        return x  # Apply sigmoid in loss function if binary segmentation




######DeeplabVit
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

class M5(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(M5, self).__init__()

        # Load ViT-Base pretrained on ImageNet
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)

        # Modify first conv layer to accept 5 channels
        original_conv = self.vit.conv_proj
        new_conv = nn.Conv2d(5, 768, kernel_size=16, stride=16)

        # Copy RGB weights and initialize new bands using Kaiming initialization
        new_conv.weight.data[:, :3, :, :] = original_conv.weight.data
        nn.init.kaiming_normal_(new_conv.weight.data[:, 3:, :, :])  # Kaiming init for NDWI & extra band

        self.vit.conv_proj = new_conv  # Replace original conv layer

        # Remove the classification head
        self.vit.heads = nn.Identity()

        # ASPP Module (Atrous Spatial Pyramid Pooling)
        self.aspp = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=12, dilation=12),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=18, dilation=18),
            nn.ReLU(),
        )

        # Segmentation Head
        self.segmentation_head = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Ensure input has 5 channels
        x = x[:, :5, :, :]  # Select first 5 channels (handles cases where extra channels exist)

        # Resize input to 224x224 for ViT
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Extract features using ViT
        x = self.vit(x)  # Expected output: (batch, 197, 768)

        # Ensure ViT outputs all patch tokens
        if x.dim() == 2 and x.shape[1] == 768:
            raise ValueError(f"ViT returned unexpected shape {x.shape}. Possible issue with ViT token settings.")

        # Remove CLS token if present
        if x.shape[1] == 197:
            x = x[:, 1:, :]  # Remove CLS token, now (batch, 196, 768)

        # Ensure correct shape
        if x.shape[1] != 196:
            raise ValueError(f"Unexpected ViT output shape: {x.shape}, expected (batch, 196, 768)")

        # Reshape to (batch, 768, 14, 14) for CNN processing
        x = x.permute(0, 2, 1).view(x.shape[0], 768, 14, 14)

        # ASPP multi-scale feature extraction
        x = self.aspp(x)

        # Convert to segmentation map
        x = self.segmentation_head(x)

        # Resize output to 128x128
        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)

        return x




# # Example usage
# model = ViTDeepLab(num_classes=1, pretrained=True)
# print(model)
