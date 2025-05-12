import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # For pre-trained Swin Transformer


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)

    def forward(self, x, skip=None):
      
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
         
            
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class SwinBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(SwinBackbone, self).__init__()
        # Load Swin Transformer with pre-trained weights
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, features_only=True)

        # Retrieve channel dimensions for each stage
        self.feature_channels = self.model.feature_info.channels()

    def forward(self, x):
        # Extract feature maps
        features = self.model(x)
        # Convert NHWC to NCHW for compatibility with PyTorch operations
        features = [f.permute(0, 3, 1, 2) for f in features]
        return features  # List of feature maps from different stages


class NAU_Net(nn.Module):
    def __init__(self, n_class):
        super(NAU_Net, self).__init__()

        self.backbone = SwinBackbone(pretrained=True)
        feature_channels = self.backbone.feature_channels

        # Adjust the decoder blocks to match 4 feature maps
        self.decode_block3 = DecoderBlock(feature_channels[-1], feature_channels[-2], 256)
        self.decode_block2 = DecoderBlock(256, feature_channels[-3], 128)
        self.decode_block1 = DecoderBlock(128, feature_channels[-4], 64)
        self.decode_block0 = DecoderBlock(64, 64, 64)
        self.decode_block_f = DecoderBlock(64, 32, 32)

        self.shallow = Conv2dReLU(3, 32, kernel_size=3, padding=1)
        self.conv_last2 = nn.Conv2d(32, n_class, kernel_size=3, padding=1)

        # Learnable parameters for NDWI attention mechanism
        self.alpha0 = nn.Parameter(torch.ones(1))
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))
        self.beita0 = nn.Parameter(torch.zeros(1))
        self.beita1 = nn.Parameter(torch.zeros(1))
        self.beita2 = nn.Parameter(torch.zeros(1))
        self.beita3 = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        # Shallow feature map from RGB bands
        input = F.interpolate(input, size=(224, 224), mode="bicubic", align_corners=False)
        layer_shallow = self.shallow(input[:, [0, 1, 2], :, :])
        
        # Feature maps from Swin Transformer
        features = self.backbone(input[:, [0, 1, 2], :, :])
        layer0, layer1, layer2, layer3 = features
        
        # Debugging shapes
      

        # NDWI (Normalized Difference Water Index)
        ndi = input[:, 3:4, :, :]
        
        # Decode block 3
        ndi3 = F.avg_pool2d(ndi, kernel_size=[16, 16])
      
        ndi3 = F.interpolate(ndi3, size=layer2.shape[2:], mode='nearest')  # Align with layer3

        layer2 = self.alpha0 * ndi3 * layer2 + self.beita0 * layer2

        x = self.decode_block3(layer3, layer2)

        # Decode block 2
        ndi2 = F.avg_pool2d(ndi, kernel_size=[8, 8])

        ndi2 = F.interpolate(ndi2, size=layer1.shape[2:], mode='nearest')  # Align with layer2
 
        layer1 = self.alpha1 * ndi2 * layer1 + self.beita1 * layer1

        x = self.decode_block2(x, layer1)
    
        # Decode block 1
        ndi1 = F.avg_pool2d(ndi, kernel_size=[4, 4])
        ndi1 = F.interpolate(ndi1, size=layer0.shape[2:], mode='nearest')  # Align with layer1
        layer0 = self.alpha2 * ndi1 * layer0 + self.beita2 * layer0
        x = self.decode_block1(x, layer0)

        # # Decode block 0
        # ndi0 = F.avg_pool2d(ndi, kernel_size=[2, 2])
        # ndi0 = F.interpolate(ndi0, size=layer0.shape[2:], mode='nearest')  # Align with layer0
        # layer0 = self.alpha3 * ndi0 * layer0 + self.beita3 * layer0
        # x = self.decode_block0(x, layer0)

        # Final decoding
        # Final decoding
        x = F.interpolate(x, size=layer_shallow.shape[2:], mode="nearest")  # Align spatial dimensions
        x = self.decode_block_f(x, layer_shallow)
  
        out1 = self.conv_last2(x)
        out1 = F.interpolate(out1, size=(128, 128), mode="bicubic", align_corners=False)
        out1 = torch.sigmoid(out1)
        return out1


