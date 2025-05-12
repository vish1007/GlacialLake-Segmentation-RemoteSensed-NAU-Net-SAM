import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F
import random
import numpy as np
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
class ndwi_attention(nn.Module):
    def __init__(self, pooling_kernel):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.pooling_layer = nn.AvgPool2d(kernel_size = pooling_kernel)
    
    def forward(self, lateral_feature, input_ndwi):
        x = self.pooling_layer(input_ndwi[:, 0:1, :, :])  # NDWI
        y = self.pooling_layer(input_ndwi[:, 1:2, :, :])  # NDSI

        x = self.alpha * lateral_feature * x + self.gamma * lateral_feature * y +self.beta * lateral_feature
        return x

class FPNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample_1 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
                                        nn.GroupNorm(num_groups = 8, num_channels = 256),
                                        nn.ReLU(),
                                        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),
                                        nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
                                        nn.GroupNorm(num_groups = 8, num_channels = 256),
                                        nn.ReLU(),
                                        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),
                                        nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
                                        nn.GroupNorm(num_groups = 8, num_channels = 128),
                                        nn.ReLU(),
                                        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True))
        self.upsample_2 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
                                        nn.GroupNorm(num_groups = 8, num_channels = 256),
                                        nn.ReLU(),
                                        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),
                                        nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
                                        nn.GroupNorm(num_groups = 8, num_channels = 128),
                                        nn.ReLU(),
                                        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),)
        self.upsample_3 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
                                        nn.GroupNorm(num_groups = 8, num_channels = 128),
                                        nn.ReLU(),
                                        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True))
        self.upsample_4 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
                                        nn.GroupNorm(num_groups = 8, num_channels = 128),
                                        nn.ReLU())
        self.upsample_5 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 1, stride = 1, padding = 0),
                                        nn.Upsample(scale_factor = 4, mode = 'bilinear', align_corners = True)
                                    )
        
    def forward(self, P2, P3, P4, P5):
        mask_1 = self.upsample_1(P5)
        mask_2 = self.upsample_2(P4)
        mask_3 = self.upsample_3(P3)
        mask_4 = self.upsample_4(P2)
        sum_mask = mask_1 + mask_2 + mask_3 + mask_4
        final_mask = self.upsample_5(sum_mask)
        return final_mask



class FPN_block(nn.Module):
    def __init__(self, in_channels, out_channels = 256, top_block = False):
        super().__init__()
        self.lateral_connection = nn.Conv2d(in_channels= in_channels, out_channels = out_channels,
                                            kernel_size = 1, stride = 1, padding = 0)
        self.feature_output = nn.Conv2d(in_channels = out_channels, out_channels = out_channels,
                                        kernel_size = 3, stride = 1, padding = 1)
        self.top_block = top_block
        
    def forward(self, backbone_feature, pyramid_feature):
        x = self.lateral_connection(backbone_feature)
        if not self.top_block:
            x += F.interpolate(pyramid_feature, scale_factor = 2, mode = "bilinear")
        output = self.feature_output(x)
        return x, output
    


class ConvBNReLU(nn.Module):
    def  __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels= in_channels, out_channels = out_channels,
                              kernel_size = kernel_size, stride = stride, padding = padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, input):
        x = self.conv(input)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x 
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv_1 = ConvBNReLU(in_channels = in_channels + skip_channels, out_channels = in_channels + skip_channels,
                                 kernel_size = 3, stride = 1, padding = 1)
        self.conv_2 = ConvBNReLU(in_channels = in_channels + skip_channels, out_channels = out_channels,
                                 kernel_size = 3, stride = 1, padding = 1)
        # print(in_channels + skip_channels)
        # print(out_channels)
        
    def forward(self, input, skip = None):
        # print(input.shape, skip.shape)
        input = F.interpolate(input, scale_factor = 2, mode = "nearest")
        if skip is not None:
            x = torch.cat([input, skip], dim = 1)
        # print(x.shape)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class BottomUp(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_layers = list(resnet50model.children())

        # Modify first convolution layer to accept 4 channels
        old_weights = self.resnet_layers[0].weight  # Shape: (64, 3, 7, 7)
        new_conv = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Copy pretrained RGB weights safely
        new_conv.weight[:, :3, :, :].data.copy_(old_weights.detach())  

        # Initialize NDWI (4th channel) with Kaiming initialization
        nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        # Replace first convolution layer in ResNet
        self.resnet_layers[0] = new_conv

        # Define pyramid layers
        self.pyramid_layer_1 = nn.Sequential(*self.resnet_layers[:3])  # Includes modified conv1
        self.pyramid_layer_2 = nn.Sequential(*self.resnet_layers[3:5])
        self.pyramid_layer_3 = self.resnet_layers[5]
        self.pyramid_layer_4 = self.resnet_layers[6]
        self.pyramid_layer_5 = self.resnet_layers[7]

    def forward(self, input):
        feature_1 = self.pyramid_layer_1(input)
        feature_2 = self.pyramid_layer_2(feature_1)
        feature_3 = self.pyramid_layer_3(feature_2)
        feature_4 = self.pyramid_layer_4(feature_3)
        feature_5 = self.pyramid_layer_5(feature_4)
        return feature_2, feature_3, feature_4, feature_5

    
class TopDown(nn.Module):
    def __init__(self, in_channels = None, out_channels= 256):
        super().__init__()
        self.pyramid_block_1 = FPN_block(in_channels = 2048, out_channels = out_channels,top_block = True)
        self.pyramid_block_2 = FPN_block(in_channels = 1024, out_channels = out_channels)
        self.pyramid_block_3 = FPN_block(in_channels = 512, out_channels = out_channels)
        self.pyramid_block_4 = FPN_block(in_channels = 256, out_channels = out_channels)
        self.na_attention_1 = ndwi_attention(pooling_kernel = 4)
        self.na_attention_2 = ndwi_attention(pooling_kernel = 8)
        self.na_attention_3 = ndwi_attention(pooling_kernel = 16)

    def forward(self, C2, C3, C4, C5, ndwi):
        x, P5 = self.pyramid_block_1(C5, None)
        x, P4 = self.pyramid_block_2(C4, x)
        P4 = self.na_attention_3(P4,ndwi)
        x, P3 = self.pyramid_block_3(C3, x)
        P3 = self.na_attention_2(P3,ndwi)
        x, P2 = self.pyramid_block_4(C2, x)
        P2 = self.na_attention_1(P2, ndwi)
        return P2, P3, P4, P5

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_block_1 = DecoderBlock(in_channels = 256,skip_channels = 256, out_channels = 256)
        self.decoder_block_2 = DecoderBlock(in_channels = 256,skip_channels = 256, out_channels = 256)
        self.decoder_block_3 = DecoderBlock(in_channels = 256,skip_channels = 256, out_channels = 256)
        self.decoder_block_4 = nn.ConvTranspose2d(in_channels = 256, out_channels =64, stride = 2, kernel_size= 4, padding = 1)
        self.decoder_block_5 = nn.ConvTranspose2d(in_channels = 64, out_channels = 1, stride = 2, kernel_size= 4, padding = 1)

    def forward(self, P2, P3, P4, P5):
        out = self.decoder_block_1(P5, P4)
        out = self.decoder_block_2(out, P3)
        out = self.decoder_block_3(out, P2)
        out = self.decoder_block_4(out)
        out = self.decoder_block_5(out)
        out = torch.sigmoid(out)
        return out
    

import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

class FPNSegemer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottomup_path = BottomUp()
        self.topdown_path = TopDown()
        self.decoder = FPNDecoder()
    
    def forward(self, input_image):
        # Resize input from 128x128 to 256x256
        # input_image = F.interpolate(input_image, size=(256, 256), mode="bilinear", align_corners=False)
        
        C2, C3, C4, C5 = self.bottomup_path(input_image[:, [0, 1, 2,3,4,5,8], :, :])
        P2, P3, P4, P5 = self.topdown_path(C2, C3, C4, C5, input_image[:, 6:8, :, :])
        mask = self.decoder(P2, P3, P4, P5)
        
        # Resize output from 256x256 back to 128x128
        # mask = F.interpolate(mask, size=(128, 128), mode="bilinear", align_corners=False)
        
        return torch.sigmoid(mask)  # Apply sigmoid here if using BCELoss

