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

# ---------------------------
# Try to import InPlaceABN
# ---------------------------
try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None

# ---------------------------
# Conv2d + BN + ReLU Block
# ---------------------------
class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()
        elif use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)

# ---------------------------
# Decoder Block
# ---------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, attention_type=None):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# ---------------------------
# Base Model Naunet Definition
# ---------------------------

class Base_NAU_Net(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Initialize resnet50
        resnet50_weights_path ='/home/user/Documents/2017/practice/weights/resnet50-0676ba61.pth'
        resnet50model = models.resnet50(pretrained = True)
        resnet50model.load_state_dict(torch.load(resnet50_weights_path))

        # Set base layers
        self.base_layers = list(resnet50model.children())                
        self.layer0 = nn.Sequential(*self.base_layers[:3]) 
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]         
        self.layer3 = self.base_layers[6]         
        self.layer4 = self.base_layers[7] 

        self.decode_block3 = DecoderBlock(2048,1024,256)
        self.decode_block2 = DecoderBlock(256,512,128)
        self.decode_block1 = DecoderBlock(128,256,64)

        self.decode_block0 = DecoderBlock(64,64,64)
        self.decode_block_f = DecoderBlock(64,32,32)

        self.shallow = Conv2dReLU(3,32,3,1)

        self.conv_last2 = nn.Conv2d(32,n_class,3,1,1)
        
        # Initilize parameters 
        self.alpha0 = nn.Parameter(torch.ones(1))
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))
        self.beita0 = nn.Parameter(torch.zeros(1))
        self.beita1 = nn.Parameter(torch.zeros(1))
        self.beita2 = nn.Parameter(torch.zeros(1))
        self.beita3 = nn.Parameter(torch.zeros(1))
        
    def forward(self, input):
        #input = F.interpolate(input, size=(512, 512), mode="bicubic", align_corners=False)
        # self.slopeAttention=SlopeAttention()
        # rgb_attended=self.slopeAttention(input[:,[0,1,2,3,4,5],:,:])
        inp=input[:,[0,1,2],:,:]
        layer_shallow = self.shallow(inp)

        layer0 = self.layer0(inp)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        ndi = input[:, 6:7, :, :] 
        # Extract slope
        # s = input[:, 4:5, :, :]  # Shape: (batch_size, 1, height, width)

        # Find the maximum value in the tensor `s` along spatial dimensions (height, width)
        # Keep dimensions so that broadcasting works
        #s_max = s.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # Max over height and width

        # Safely normalize by adding a small epsilon to avoid division by zero
        # epsilon = 1e-8
        #ndi = s / (s_max + epsilon)
        # s = input[:,4:5, :, :]  # Extract slope
        # ndwi=input[:,3:4,:,:]
         # Extract NDWI
        # s = input[:, 4:5, :, :]  
        # epsilon = 1e-6  # Small constant for numerical stability
        # valid_mask = s > epsilon  # Mask where slope is greater than zero
        # ndi = torch.where(valid_mask, ndwi / (s + epsilon), ndwi)
        # ndi3 = F.avg_pool2d(ndi,kernel_size = [16,16])
        # layer3 = self.alpha0*ndi3*layer3 + self.beita0*layer3  # No NA block is applied to layer3 in the updated NAU-Net architecture.
        x = self.decode_block3(layer4,layer3)
        
        ndi2 = F.avg_pool2d(ndi,kernel_size = [8,8])
        # ndwi2=F.avg_pool2d(ndwi,kernel_size=[8,8])
        layer2 = self.alpha1*ndi2*layer2 + self.beita1*layer2
        x = self.decode_block2(x,layer2)

        ndi1 = F.avg_pool2d(ndi,kernel_size = [4,4])
        # ndwi1 = F.avg_pool2d(ndwi,kernel_size = [4,4])
        layer1 = self.alpha2*ndi1*layer1+ self.beita2*layer1      
        x = self.decode_block1(x,layer1)
 
        ndi0 = F.avg_pool2d(ndi,kernel_size = [2,2])
        # ndwi0 = F.avg_pool2d(ndwi,kernel_size = [2,2])                                                      
        layer0 = self.alpha3*ndi0*layer0 + self.beita3*layer0
        x = self.decode_block0(x,layer0) 
        x = self.decode_block_f(x,layer_shallow)
        out1 = self.conv_last2(x)
       # out1 = F.interpolate(out1, size=(128, 128), mode="nearest")
        out1 = torch.sigmoid(out1)  # new line added by me
        return out1


# ---------------------------
# NAU_Net Definition
# ---------------------------
class NAU_Net(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Load pretrained ResNet50
        resnet50_weights_path = '/home/user/Documents/2017/practice/weights/resnet50-0676ba61.pth'
        resnet50model = models.resnet50(pretrained=True)
        resnet50model.load_state_dict(torch.load(resnet50_weights_path))

        # Modify input to accept 5 channels (RGB + slope + NDSI/NDWI etc.)
        old_conv1 = resnet50model.conv1
        self.new_conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        with torch.no_grad():
            self.new_conv1.weight[:, :3, :, :] = old_conv1.weight  # R,G,B
            torch.nn.init.kaiming_normal_(self.new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        resnet50model.conv1 = self.new_conv1

        # Encoder
        self.base_layers = list(resnet50model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        # Decoder
        self.decode_block3 = DecoderBlock(2048, 1024, 256)
        self.decode_block2 = DecoderBlock(256, 512, 128)
        self.decode_block1 = DecoderBlock(128, 256, 64)
        self.decode_block0 = DecoderBlock(64, 64, 64)
        self.decode_block_f = DecoderBlock(64, 32, 32)

        self.shallow = Conv2dReLU(4, 32, 3, 1)

        self.conv_last2 = nn.Conv2d(32, n_class, 3, 1, 1)

        # Trainable parameters for attention
        self.alpha0 = nn.Parameter(torch.ones(1))
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))
        self.beita0 = nn.Parameter(torch.zeros(1))
        self.beita1 = nn.Parameter(torch.zeros(1))
        self.beita2 = nn.Parameter(torch.zeros(1))
        self.beita3 = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        slope_normalized = input[:, 8:9, :, :]  # Slope info
        ndi = input[:, 6:7, :, :]  # NDWI or similar
        # ndi = (ndi + 1.0) / 2.0
        
        # ndsi = (ndsi_batch + 1.0) / 2.0
        input_with_slope = torch.cat([input[:, [0, 1, 2], :, :], slope_normalized], dim=1)
        layer_shallow = self.shallow(input_with_slope)
        layer0 = self.layer0(input_with_slope)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # Decoder with slope and NDWI attention
        x = self.decode_block3(layer4, layer3)

        ndi2 = F.avg_pool2d(ndi, kernel_size=[8, 8])
        layer2 = self.alpha1 * ndi2 * layer2 + self.beita1 * layer2
        x = self.decode_block2(x, layer2)

        ndi1 = F.avg_pool2d(ndi, kernel_size=[4, 4])
        layer1 = self.alpha2 * ndi1 * layer1 + self.beita2 * layer1
        x = self.decode_block1(x, layer1)

        ndi0 = F.avg_pool2d(ndi, kernel_size=[2, 2])
        layer0 = self.alpha3 * ndi0 * layer0 + self.beita3 * layer0
        x = self.decode_block0(x, layer0)

        x = self.decode_block_f(x, layer_shallow)
        out1 = self.conv_last2(x)

        out1 = torch.sigmoid(out1)  # Binary segmentation output
        return out1
############################################################################# Variants_slope_aware_naunet_model######################################################################

# ---------------------------
# NAU_Net_slope_aware_ndwi_ndsi_nir Definition
# ---------------------------

class NAU_Net_slope_aware_ndwi_ndsi_nir(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Initialize resnet50
        resnet50_weights_path = '/home/user/Documents/2017/practice/weights/resnet50-0676ba61.pth'
        resnet50model = models.resnet50(pretrained=True)
        resnet50model.load_state_dict(torch.load(resnet50_weights_path))

        # Adjust the first convolutional layer to accept 4 channels
        old_conv1 = resnet50model.conv1
        self.new_conv1 = nn.Conv2d(
            in_channels=5,  # Adjusted to 4 channels
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        # Copy weights for the first 3 channels from the old convolutional layer
        with torch.no_grad():
            self.new_conv1.weight[:, :3, :, :] = old_conv1.weight  # Copy weights for R, G, B channels
            torch.nn.init.kaiming_normal_(self.new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        resnet50model.conv1 = self.new_conv1

        # Set base layers
        self.base_layers = list(resnet50model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.decode_block3 = DecoderBlock(2048, 1024, 256)
        self.decode_block2 = DecoderBlock(256, 512, 128)
        self.decode_block1 = DecoderBlock(128, 256, 64)
        self.decode_block0 = DecoderBlock(64, 64, 64)
        self.decode_block_f = DecoderBlock(64, 32, 32)

        self.shallow = Conv2dReLU(5, 32, 3, 1)
        self.conv_last2 = nn.Conv2d(32, n_class, 3, 1, 1)

        # Initialize parameters for attention
        self.alpha0 = nn.Parameter(torch.ones(1))
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))

        self.beita0 = nn.Parameter(torch.zeros(1))
        self.beita1 = nn.Parameter(torch.zeros(1))
        self.beita2 = nn.Parameter(torch.zeros(1))
        self.beita3 = nn.Parameter(torch.zeros(1))

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        slope_normalized = input[:,8:9, :, :]
        ndi = input[:, 6:7, :, :]
        ndsi = input[:, 7:8, :, :]
       

        # # Compute per-image min and max
        # ndsi_min = ndsi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndsi_max = ndsi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndsi = (ndsi_batch - ndsi_min) / (ndsi_max - ndsi_min + 1e-8) 

        # # Compute per-image min and max
        # ndwi_min = ndwi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndwi_max = ndwi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndi = (ndwi_batch - ndwi_min) / (ndwi_max - ndwi_min + 1e-8) 

        input_with_slope = torch.cat([input[:, [0, 1, 2,3], :, :], slope_normalized], dim=1)
        layer_shallow = self.shallow(input_with_slope)
        layer0 = self.layer0(input_with_slope)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.decode_block3(layer4, layer3)

        ndi2 = F.avg_pool2d(ndi, kernel_size=[8, 8])
        ndsi2 = F.avg_pool2d(ndsi, kernel_size=[8, 8])
        attention2 = self.alpha1 * ndi2 * layer2 + self.beita1 * ndsi2 * layer2 + self.gamma1 * layer2
        x = self.decode_block2(x, attention2)

        ndi1 = F.avg_pool2d(ndi, kernel_size=[4, 4])
        ndsi1 = F.avg_pool2d(ndsi, kernel_size=[4, 4])
        attention1 = self.alpha2 * ndi1 * layer1 + self.beita2 * ndsi1 * layer1 + self.gamma2 * layer1
        x = self.decode_block1(x, attention1)

        ndi0 = F.avg_pool2d(ndi, kernel_size=[2, 2])
        ndsi0 = F.avg_pool2d(ndsi, kernel_size=[2, 2])
        attention0 = self.alpha3 * ndi0 * layer0 + self.beita3 * ndsi0 * layer0 + self.gamma3 * layer0
        x = self.decode_block0(x, attention0)

        x = self.decode_block_f(x, layer_shallow)
        out1 = self.conv_last2(x)
        out1 = torch.sigmoid(out1)

        return out1
    
##################################################################################33
# ---------------------------
# NAU_Net_slope_aware_ndwi_ndsi_swir1 Definition
# ---------------------------

class NAU_Net_slope_aware_ndwi_ndsi_swir1(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Initialize resnet50
        resnet50_weights_path = '/home/user/Documents/2017/practice/weights/resnet50-0676ba61.pth'
        resnet50model = models.resnet50(pretrained=True)
        resnet50model.load_state_dict(torch.load(resnet50_weights_path))

        # Adjust the first convolutional layer to accept 4 channels
        old_conv1 = resnet50model.conv1
        self.new_conv1 = nn.Conv2d(
            in_channels=5,  # Adjusted to 4 channels
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        # Copy weights for the first 3 channels from the old convolutional layer
        with torch.no_grad():
            self.new_conv1.weight[:, :3, :, :] = old_conv1.weight  # Copy weights for R, G, B channels
            torch.nn.init.kaiming_normal_(self.new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        resnet50model.conv1 = self.new_conv1

        # Set base layers
        self.base_layers = list(resnet50model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.decode_block3 = DecoderBlock(2048, 1024, 256)
        self.decode_block2 = DecoderBlock(256, 512, 128)
        self.decode_block1 = DecoderBlock(128, 256, 64)
        self.decode_block0 = DecoderBlock(64, 64, 64)
        self.decode_block_f = DecoderBlock(64, 32, 32)

        self.shallow = Conv2dReLU(5, 32, 3, 1)
        self.conv_last2 = nn.Conv2d(32, n_class, 3, 1, 1)

        # Initialize parameters for attention
        self.alpha0 = nn.Parameter(torch.ones(1))
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))

        self.beita0 = nn.Parameter(torch.zeros(1))
        self.beita1 = nn.Parameter(torch.zeros(1))
        self.beita2 = nn.Parameter(torch.zeros(1))
        self.beita3 = nn.Parameter(torch.zeros(1))

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        slope_normalized = input[:,8:9, :, :]
        ndi = input[:, 6:7, :, :]
        ndsi = input[:, 7:8, :, :]
       

        # # Compute per-image min and max
        # ndsi_min = ndsi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndsi_max = ndsi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndsi = (ndsi_batch - ndsi_min) / (ndsi_max - ndsi_min + 1e-8) 

        # # Compute per-image min and max
        # ndwi_min = ndwi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndwi_max = ndwi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndi = (ndwi_batch - ndwi_min) / (ndwi_max - ndwi_min + 1e-8) 

        input_with_slope = torch.cat([input[:, [0, 1, 2,4], :, :], slope_normalized], dim=1)
        layer_shallow = self.shallow(input_with_slope)
        layer0 = self.layer0(input_with_slope)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.decode_block3(layer4, layer3)

        ndi2 = F.avg_pool2d(ndi, kernel_size=[8, 8])
        ndsi2 = F.avg_pool2d(ndsi, kernel_size=[8, 8])
        attention2 = self.alpha1 * ndi2 * layer2 + self.beita1 * ndsi2 * layer2 + self.gamma1 * layer2
        x = self.decode_block2(x, attention2)

        ndi1 = F.avg_pool2d(ndi, kernel_size=[4, 4])
        ndsi1 = F.avg_pool2d(ndsi, kernel_size=[4, 4])
        attention1 = self.alpha2 * ndi1 * layer1 + self.beita2 * ndsi1 * layer1 + self.gamma2 * layer1
        x = self.decode_block1(x, attention1)

        ndi0 = F.avg_pool2d(ndi, kernel_size=[2, 2])
        ndsi0 = F.avg_pool2d(ndsi, kernel_size=[2, 2])
        attention0 = self.alpha3 * ndi0 * layer0 + self.beita3 * ndsi0 * layer0 + self.gamma3 * layer0
        x = self.decode_block0(x, attention0)

        x = self.decode_block_f(x, layer_shallow)
        out1 = self.conv_last2(x)
        out1 = torch.sigmoid(out1)

        return out1
    #####################################################################

# ---------------------------
# NAU_Net_slope_aware_ndwi_ndsi_swir2 Definition
# ---------------------------

class NAU_Net_slope_aware_ndwi_ndsi_swir2(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Initialize resnet50
        resnet50_weights_path = '/home/user/Documents/2017/practice/weights/resnet50-0676ba61.pth'
        resnet50model = models.resnet50(pretrained=True)
        resnet50model.load_state_dict(torch.load(resnet50_weights_path))

        # Adjust the first convolutional layer to accept 4 channels
        old_conv1 = resnet50model.conv1
        self.new_conv1 = nn.Conv2d(
            in_channels=5,  # Adjusted to 4 channels
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        # Copy weights for the first 3 channels from the old convolutional layer
        with torch.no_grad():
            self.new_conv1.weight[:, :3, :, :] = old_conv1.weight  # Copy weights for R, G, B channels
            torch.nn.init.kaiming_normal_(self.new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        resnet50model.conv1 = self.new_conv1

        # Set base layers
        self.base_layers = list(resnet50model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.decode_block3 = DecoderBlock(2048, 1024, 256)
        self.decode_block2 = DecoderBlock(256, 512, 128)
        self.decode_block1 = DecoderBlock(128, 256, 64)
        self.decode_block0 = DecoderBlock(64, 64, 64)
        self.decode_block_f = DecoderBlock(64, 32, 32)

        self.shallow = Conv2dReLU(5, 32, 3, 1)
        self.conv_last2 = nn.Conv2d(32, n_class, 3, 1, 1)

        # Initialize parameters for attention
        self.alpha0 = nn.Parameter(torch.ones(1))
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))

        self.beita0 = nn.Parameter(torch.zeros(1))
        self.beita1 = nn.Parameter(torch.zeros(1))
        self.beita2 = nn.Parameter(torch.zeros(1))
        self.beita3 = nn.Parameter(torch.zeros(1))

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        slope_normalized = input[:,8:9, :, :]
        ndi = input[:, 6:7, :, :]
        ndsi = input[:, 7:8, :, :]
       

        # # Compute per-image min and max
        # ndsi_min = ndsi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndsi_max = ndsi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndsi = (ndsi_batch - ndsi_min) / (ndsi_max - ndsi_min + 1e-8) 

        # # Compute per-image min and max
        # ndwi_min = ndwi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndwi_max = ndwi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndi = (ndwi_batch - ndwi_min) / (ndwi_max - ndwi_min + 1e-8) 

        input_with_slope = torch.cat([input[:, [0, 1, 2,5], :, :], slope_normalized], dim=1)
        layer_shallow = self.shallow(input_with_slope)
        layer0 = self.layer0(input_with_slope)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.decode_block3(layer4, layer3)

        ndi2 = F.avg_pool2d(ndi, kernel_size=[8, 8])
        ndsi2 = F.avg_pool2d(ndsi, kernel_size=[8, 8])
        attention2 = self.alpha1 * ndi2 * layer2 + self.beita1 * ndsi2 * layer2 + self.gamma1 * layer2
        x = self.decode_block2(x, attention2)

        ndi1 = F.avg_pool2d(ndi, kernel_size=[4, 4])
        ndsi1 = F.avg_pool2d(ndsi, kernel_size=[4, 4])
        attention1 = self.alpha2 * ndi1 * layer1 + self.beita2 * ndsi1 * layer1 + self.gamma2 * layer1
        x = self.decode_block1(x, attention1)

        ndi0 = F.avg_pool2d(ndi, kernel_size=[2, 2])
        ndsi0 = F.avg_pool2d(ndsi, kernel_size=[2, 2])
        attention0 = self.alpha3 * ndi0 * layer0 + self.beita3 * ndsi0 * layer0 + self.gamma3 * layer0
        x = self.decode_block0(x, attention0)

        x = self.decode_block_f(x, layer_shallow)
        out1 = self.conv_last2(x)
        out1 = torch.sigmoid(out1)

        return out1
#######################################################################################################

# ---------------------------
# NAU_Net_slope_aware_ndwi_ndsi_DEM Definition
# ---------------------------

class NAU_Net_slope_aware_ndwi_ndsi_dem(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Initialize resnet50
        resnet50_weights_path = '/home/user/Documents/2017/practice/weights/resnet50-0676ba61.pth'
        resnet50model = models.resnet50(pretrained=True)
        resnet50model.load_state_dict(torch.load(resnet50_weights_path))

        # Adjust the first convolutional layer to accept 4 channels
        old_conv1 = resnet50model.conv1
        self.new_conv1 = nn.Conv2d(
            in_channels=5,  # Adjusted to 4 channels
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        # Copy weights for the first 3 channels from the old convolutional layer
        with torch.no_grad():
            self.new_conv1.weight[:, :3, :, :] = old_conv1.weight  # Copy weights for R, G, B channels
            torch.nn.init.kaiming_normal_(self.new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        resnet50model.conv1 = self.new_conv1

        # Set base layers
        self.base_layers = list(resnet50model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.decode_block3 = DecoderBlock(2048, 1024, 256)
        self.decode_block2 = DecoderBlock(256, 512, 128)
        self.decode_block1 = DecoderBlock(128, 256, 64)
        self.decode_block0 = DecoderBlock(64, 64, 64)
        self.decode_block_f = DecoderBlock(64, 32, 32)

        self.shallow = Conv2dReLU(5, 32, 3, 1)
        self.conv_last2 = nn.Conv2d(32, n_class, 3, 1, 1)

        # Initialize parameters for attention
        self.alpha0 = nn.Parameter(torch.ones(1))
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))

        self.beita0 = nn.Parameter(torch.zeros(1))
        self.beita1 = nn.Parameter(torch.zeros(1))
        self.beita2 = nn.Parameter(torch.zeros(1))
        self.beita3 = nn.Parameter(torch.zeros(1))

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        slope_normalized = input[:,8:9, :, :]
        ndi = input[:, 6:7, :, :]
        ndsi = input[:, 7:8, :, :]
       

        # # Compute per-image min and max
        # ndsi_min = ndsi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndsi_max = ndsi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndsi = (ndsi_batch - ndsi_min) / (ndsi_max - ndsi_min + 1e-8) 

        # # Compute per-image min and max
        # ndwi_min = ndwi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndwi_max = ndwi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndi = (ndwi_batch - ndwi_min) / (ndwi_max - ndwi_min + 1e-8) 

        input_with_slope = torch.cat([input[:, [0, 1, 2,9], :, :], slope_normalized], dim=1)
        layer_shallow = self.shallow(input_with_slope)
        layer0 = self.layer0(input_with_slope)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.decode_block3(layer4, layer3)

        ndi2 = F.avg_pool2d(ndi, kernel_size=[8, 8])
        ndsi2 = F.avg_pool2d(ndsi, kernel_size=[8, 8])
        attention2 = self.alpha1 * ndi2 * layer2 + self.beita1 * ndsi2 * layer2 + self.gamma1 * layer2
        x = self.decode_block2(x, attention2)

        ndi1 = F.avg_pool2d(ndi, kernel_size=[4, 4])
        ndsi1 = F.avg_pool2d(ndsi, kernel_size=[4, 4])
        attention1 = self.alpha2 * ndi1 * layer1 + self.beita2 * ndsi1 * layer1 + self.gamma2 * layer1
        x = self.decode_block1(x, attention1)

        ndi0 = F.avg_pool2d(ndi, kernel_size=[2, 2])
        ndsi0 = F.avg_pool2d(ndsi, kernel_size=[2, 2])
        attention0 = self.alpha3 * ndi0 * layer0 + self.beita3 * ndsi0 * layer0 + self.gamma3 * layer0
        x = self.decode_block0(x, attention0)

        x = self.decode_block_f(x, layer_shallow)
        out1 = self.conv_last2(x)
        out1 = torch.sigmoid(out1)

        return out1



# ---------------------------
# NAU_Net_slope_aware_ndwi_ndsi_nir_swir1 Definition
# ---------------------------


class NAU_Net_slope_aware_ndwi_ndsi_nir_swir1(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Initialize resnet50
        resnet50_weights_path = '/home/user/Documents/2017/practice/weights/resnet50-0676ba61.pth'
        resnet50model = models.resnet50(pretrained=True)
        resnet50model.load_state_dict(torch.load(resnet50_weights_path))

        # Adjust the first convolutional layer to accept 4 channels
        old_conv1 = resnet50model.conv1
        self.new_conv1 = nn.Conv2d(
            in_channels=6,  # Adjusted to 4 channels
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        # Copy weights for the first 3 channels from the old convolutional layer
        with torch.no_grad():
            self.new_conv1.weight[:, :3, :, :] = old_conv1.weight  # Copy weights for R, G, B channels
            torch.nn.init.kaiming_normal_(self.new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        resnet50model.conv1 = self.new_conv1

        # Set base layers
        self.base_layers = list(resnet50model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.decode_block3 = DecoderBlock(2048, 1024, 256)
        self.decode_block2 = DecoderBlock(256, 512, 128)
        self.decode_block1 = DecoderBlock(128, 256, 64)
        self.decode_block0 = DecoderBlock(64, 64, 64)
        self.decode_block_f = DecoderBlock(64, 32, 32)

        self.shallow = Conv2dReLU(6, 32, 3, 1)
        self.conv_last2 = nn.Conv2d(32, n_class, 3, 1, 1)

        # Initialize parameters for attention
        self.alpha0 = nn.Parameter(torch.ones(1))
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))

        self.beita0 = nn.Parameter(torch.zeros(1))
        self.beita1 = nn.Parameter(torch.zeros(1))
        self.beita2 = nn.Parameter(torch.zeros(1))
        self.beita3 = nn.Parameter(torch.zeros(1))

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        slope_normalized = input[:,8:9, :, :]
        ndi = input[:, 6:7, :, :] ##NDWI
        ndsi = input[:, 7:8, :, :] #NDSI
        # ndi = (ndwi_batch + 1.0) / 2.0
        # ndsi = (ndsi_batch + 1.0) / 2.0

        # # Compute per-image min and max
        # ndsi_min = ndsi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndsi_max = ndsi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndsi = (ndsi_batch - ndsi_min) / (ndsi_max - ndsi_min + 1e-8) 

        # # Compute per-image min and max
        # ndwi_min = ndwi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndwi_max = ndwi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndi = (ndwi_batch - ndwi_min) / (ndwi_max - ndwi_min + 1e-8) 

        input_with_slope = torch.cat([input[:, [0, 1, 2,3,4], :, :], slope_normalized], dim=1)
        layer_shallow = self.shallow(input_with_slope)
        layer0 = self.layer0(input_with_slope)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.decode_block3(layer4, layer3)

        ndi2 = F.avg_pool2d(ndi, kernel_size=[8, 8])
        ndsi2 = F.avg_pool2d(ndsi, kernel_size=[8, 8])
        attention2 = self.alpha1 * ndi2 * layer2 + self.beita1 * ndsi2 * layer2 + self.gamma1 * layer2
        x = self.decode_block2(x, attention2)

        ndi1 = F.avg_pool2d(ndi, kernel_size=[4, 4])
        ndsi1 = F.avg_pool2d(ndsi, kernel_size=[4, 4])
        attention1 = self.alpha2 * ndi1 * layer1 + self.beita2 * ndsi1 * layer1 + self.gamma2 * layer1
        x = self.decode_block1(x, attention1)

        ndi0 = F.avg_pool2d(ndi, kernel_size=[2, 2])
        ndsi0 = F.avg_pool2d(ndsi, kernel_size=[2, 2])
        attention0 = self.alpha3 * ndi0 * layer0 + self.beita3 * ndsi0 * layer0 + self.gamma3 * layer0
        x = self.decode_block0(x, attention0)

        x = self.decode_block_f(x, layer_shallow)
        out1 = self.conv_last2(x)
        out1 = torch.sigmoid(out1)

        return out1
    
# ---------------------------
# NAU_Net_slope_aware_ndwi_ndsi_swir1_swir2 Definition
# ---------------------------


class NAU_Net_slope_aware_ndwi_ndsi_swir1_swir2(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Initialize resnet50
        resnet50_weights_path = '/home/user/Documents/2017/practice/weights/resnet50-0676ba61.pth'
        resnet50model = models.resnet50(pretrained=True)
        resnet50model.load_state_dict(torch.load(resnet50_weights_path))

        # Adjust the first convolutional layer to accept 4 channels
        old_conv1 = resnet50model.conv1
        self.new_conv1 = nn.Conv2d(
            in_channels=6,  # Adjusted to 4 channels
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        # Copy weights for the first 3 channels from the old convolutional layer
        with torch.no_grad():
            self.new_conv1.weight[:, :3, :, :] = old_conv1.weight  # Copy weights for R, G, B channels
            torch.nn.init.kaiming_normal_(self.new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        resnet50model.conv1 = self.new_conv1

        # Set base layers
        self.base_layers = list(resnet50model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.decode_block3 = DecoderBlock(2048, 1024, 256)
        self.decode_block2 = DecoderBlock(256, 512, 128)
        self.decode_block1 = DecoderBlock(128, 256, 64)
        self.decode_block0 = DecoderBlock(64, 64, 64)
        self.decode_block_f = DecoderBlock(64, 32, 32)

        self.shallow = Conv2dReLU(6, 32, 3, 1)
        self.conv_last2 = nn.Conv2d(32, n_class, 3, 1, 1)

        # Initialize parameters for attention
        self.alpha0 = nn.Parameter(torch.ones(1))
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))

        self.beita0 = nn.Parameter(torch.zeros(1))
        self.beita1 = nn.Parameter(torch.zeros(1))
        self.beita2 = nn.Parameter(torch.zeros(1))
        self.beita3 = nn.Parameter(torch.zeros(1))

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        slope_normalized = input[:,8:9, :, :]
        ndi = input[:, 6:7, :, :] ##NDWI
        ndsi = input[:, 7:8, :, :] #NDSI
        # ndi = (ndwi_batch + 1.0) / 2.0
        # ndsi = (ndsi_batch + 1.0) / 2.0

        # # Compute per-image min and max
        # ndsi_min = ndsi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndsi_max = ndsi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndsi = (ndsi_batch - ndsi_min) / (ndsi_max - ndsi_min + 1e-8) 

        # # Compute per-image min and max
        # ndwi_min = ndwi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndwi_max = ndwi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndi = (ndwi_batch - ndwi_min) / (ndwi_max - ndwi_min + 1e-8) 

        input_with_slope = torch.cat([input[:, [0, 1, 2,4,5], :, :], slope_normalized], dim=1)
        layer_shallow = self.shallow(input_with_slope)
        layer0 = self.layer0(input_with_slope)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.decode_block3(layer4, layer3)

        ndi2 = F.avg_pool2d(ndi, kernel_size=[8, 8])
        ndsi2 = F.avg_pool2d(ndsi, kernel_size=[8, 8])
        attention2 = self.alpha1 * ndi2 * layer2 + self.beita1 * ndsi2 * layer2 + self.gamma1 * layer2
        x = self.decode_block2(x, attention2)

        ndi1 = F.avg_pool2d(ndi, kernel_size=[4, 4])
        ndsi1 = F.avg_pool2d(ndsi, kernel_size=[4, 4])
        attention1 = self.alpha2 * ndi1 * layer1 + self.beita2 * ndsi1 * layer1 + self.gamma2 * layer1
        x = self.decode_block1(x, attention1)

        ndi0 = F.avg_pool2d(ndi, kernel_size=[2, 2])
        ndsi0 = F.avg_pool2d(ndsi, kernel_size=[2, 2])
        attention0 = self.alpha3 * ndi0 * layer0 + self.beita3 * ndsi0 * layer0 + self.gamma3 * layer0
        x = self.decode_block0(x, attention0)

        x = self.decode_block_f(x, layer_shallow)
        out1 = self.conv_last2(x)
        out1 = torch.sigmoid(out1)

        return out1


    # ---------------------------
# NAU_Net_slope_aware_ndwi_ndsi_nir_swir1_swir2 Definition
# ---------------------------


class NAU_Net_slope_aware_ndwi_ndsi_nir_swir1_swir2(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Initialize resnet50
        resnet50_weights_path = '/home/user/Documents/2017/practice/weights/resnet50-0676ba61.pth'
        resnet50model = models.resnet50(pretrained=True)
        resnet50model.load_state_dict(torch.load(resnet50_weights_path))

        # Adjust the first convolutional layer to accept 4 channels
        old_conv1 = resnet50model.conv1
        self.new_conv1 = nn.Conv2d(
            in_channels=7,  # Adjusted to 4 channels
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        # Copy weights for the first 3 channels from the old convolutional layer
        with torch.no_grad():
            self.new_conv1.weight[:, :3, :, :] = old_conv1.weight  # Copy weights for R, G, B channels
            torch.nn.init.kaiming_normal_(self.new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        resnet50model.conv1 = self.new_conv1

        # Set base layers
        self.base_layers = list(resnet50model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.decode_block3 = DecoderBlock(2048, 1024, 256)
        self.decode_block2 = DecoderBlock(256, 512, 128)
        self.decode_block1 = DecoderBlock(128, 256, 64)
        self.decode_block0 = DecoderBlock(64, 64, 64)
        self.decode_block_f = DecoderBlock(64, 32, 32)

        self.shallow = Conv2dReLU(7, 32, 3, 1)
        self.conv_last2 = nn.Conv2d(32, n_class, 3, 1, 1)

        # Initialize parameters for attention
        self.alpha0 = nn.Parameter(torch.ones(1))
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))

        self.beita0 = nn.Parameter(torch.zeros(1))
        self.beita1 = nn.Parameter(torch.zeros(1))
        self.beita2 = nn.Parameter(torch.zeros(1))
        self.beita3 = nn.Parameter(torch.zeros(1))

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        slope_normalized = input[:,8:9, :, :]
        ndi = input[:, 6:7, :, :] ##NDWI
        ndsi = input[:, 7:8, :, :] #NDSI
        # ndi = (ndwi_batch + 1.0) / 2.0
        # ndsi = (ndsi_batch + 1.0) / 2.0

        # # Compute per-image min and max
        # ndsi_min = ndsi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndsi_max = ndsi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndsi = (ndsi_batch - ndsi_min) / (ndsi_max - ndsi_min + 1e-8) 

        # # Compute per-image min and max
        # ndwi_min = ndwi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndwi_max = ndwi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndi = (ndwi_batch - ndwi_min) / (ndwi_max - ndwi_min + 1e-8) 

        input_with_slope = torch.cat([input[:, [0, 1, 2,3,4,5], :, :], slope_normalized], dim=1)
        layer_shallow = self.shallow(input_with_slope)
        layer0 = self.layer0(input_with_slope)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.decode_block3(layer4, layer3)

        ndi2 = F.avg_pool2d(ndi, kernel_size=[8, 8])
        ndsi2 = F.avg_pool2d(ndsi, kernel_size=[8, 8])
        attention2 = self.alpha1 * ndi2 * layer2 + self.beita1 * ndsi2 * layer2 + self.gamma1 * layer2
        x = self.decode_block2(x, attention2)

        ndi1 = F.avg_pool2d(ndi, kernel_size=[4, 4])
        ndsi1 = F.avg_pool2d(ndsi, kernel_size=[4, 4])
        attention1 = self.alpha2 * ndi1 * layer1 + self.beita2 * ndsi1 * layer1 + self.gamma2 * layer1
        x = self.decode_block1(x, attention1)

        ndi0 = F.avg_pool2d(ndi, kernel_size=[2, 2])
        ndsi0 = F.avg_pool2d(ndsi, kernel_size=[2, 2])
        attention0 = self.alpha3 * ndi0 * layer0 + self.beita3 * ndsi0 * layer0 + self.gamma3 * layer0
        x = self.decode_block0(x, attention0)

        x = self.decode_block_f(x, layer_shallow)
        out1 = self.conv_last2(x)
        out1 = torch.sigmoid(out1)

        return out1
    

   # ---------------------------
# NAU_Net_slope_aware_ndwi_ndsi_nir_swir1_swir2+DEMDefinition
# ---------------------------


class NAU_Net_slope_aware_ndwi_ndsi_nir_swir1_swir2_dem(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Initialize resnet50
        resnet50_weights_path = '/home/user/Documents/2017/practice/weights/resnet50-0676ba61.pth'
        resnet50model = models.resnet50(pretrained=True)
        resnet50model.load_state_dict(torch.load(resnet50_weights_path))

        # Adjust the first convolutional layer to accept 4 channels
        old_conv1 = resnet50model.conv1
        self.new_conv1 = nn.Conv2d(
            in_channels=8,  # Adjusted to 4 channels
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        # Copy weights for the first 3 channels from the old convolutional layer
        with torch.no_grad():
            self.new_conv1.weight[:, :3, :, :] = old_conv1.weight  # Copy weights for R, G, B channels
            torch.nn.init.kaiming_normal_(self.new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        resnet50model.conv1 = self.new_conv1

        # Set base layers
        self.base_layers = list(resnet50model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.decode_block3 = DecoderBlock(2048, 1024, 256)
        self.decode_block2 = DecoderBlock(256, 512, 128)
        self.decode_block1 = DecoderBlock(128, 256, 64)
        self.decode_block0 = DecoderBlock(64, 64, 64)
        self.decode_block_f = DecoderBlock(64, 32, 32)

        self.shallow = Conv2dReLU(8, 32, 3, 1)
        self.conv_last2 = nn.Conv2d(32, n_class, 3, 1, 1)

        # Initialize parameters for attention
        self.alpha0 = nn.Parameter(torch.ones(1))
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))

        self.beita0 = nn.Parameter(torch.zeros(1))
        self.beita1 = nn.Parameter(torch.zeros(1))
        self.beita2 = nn.Parameter(torch.zeros(1))
        self.beita3 = nn.Parameter(torch.zeros(1))

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        slope_normalized = input[:,8:9, :, :]
        ndi = input[:, 6:7, :, :] ##NDWI
        ndsi = input[:, 7:8, :, :] #NDSI
        # ndi = (ndwi_batch + 1.0) / 2.0
        # ndsi = (ndsi_batch + 1.0) / 2.0

        # # Compute per-image min and max
        # ndsi_min = ndsi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndsi_max = ndsi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndsi = (ndsi_batch - ndsi_min) / (ndsi_max - ndsi_min + 1e-8) 

        # # Compute per-image min and max
        # ndwi_min = ndwi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndwi_max = ndwi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndi = (ndwi_batch - ndwi_min) / (ndwi_max - ndwi_min + 1e-8) 

        input_with_slope = torch.cat([input[:, [0, 1, 2,3,4,5,9], :, :], slope_normalized], dim=1)
        layer_shallow = self.shallow(input_with_slope)
        layer0 = self.layer0(input_with_slope)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.decode_block3(layer4, layer3)

        ndi2 = F.avg_pool2d(ndi, kernel_size=[8, 8])
        ndsi2 = F.avg_pool2d(ndsi, kernel_size=[8, 8])
        attention2 = self.alpha1 * ndi2 * layer2 + self.beita1 * ndsi2 * layer2 + self.gamma1 * layer2
        x = self.decode_block2(x, attention2)

        ndi1 = F.avg_pool2d(ndi, kernel_size=[4, 4])
        ndsi1 = F.avg_pool2d(ndsi, kernel_size=[4, 4])
        attention1 = self.alpha2 * ndi1 * layer1 + self.beita2 * ndsi1 * layer1 + self.gamma2 * layer1
        x = self.decode_block1(x, attention1)

        ndi0 = F.avg_pool2d(ndi, kernel_size=[2, 2])
        ndsi0 = F.avg_pool2d(ndsi, kernel_size=[2, 2])
        attention0 = self.alpha3 * ndi0 * layer0 + self.beita3 * ndsi0 * layer0 + self.gamma3 * layer0
        x = self.decode_block0(x, attention0)

        x = self.decode_block_f(x, layer_shallow)
        out1 = self.conv_last2(x)
        out1 = torch.sigmoid(out1)

        return out1
    

    # ---------------------------
# NAU_Net_slope_aware_swir1_nir_green_slope_ndwi_ndsi Definition
# ---------------------------

class NAU_Net_slope_aware_swir1_nir_green_slope_ndwi_ndsi(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Initialize resnet50
        resnet50_weights_path = '/home/user/Documents/2017/practice/weights/resnet50-0676ba61.pth'
        resnet50model = models.resnet50(pretrained=True)
        resnet50model.load_state_dict(torch.load(resnet50_weights_path))

        # Adjust the first convolutional layer to accept 4 channels
        old_conv1 = resnet50model.conv1
        self.new_conv1 = nn.Conv2d(
            in_channels=4,  # Adjusted to 4 channels
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        # Copy weights for the first 3 channels from the old convolutional layer
        with torch.no_grad():
            self.new_conv1.weight[:, :3, :, :] = old_conv1.weight  # Copy weights for R, G, B channels
            torch.nn.init.kaiming_normal_(self.new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        resnet50model.conv1 = self.new_conv1

        # Set base layers
        self.base_layers = list(resnet50model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.decode_block3 = DecoderBlock(2048, 1024, 256)
        self.decode_block2 = DecoderBlock(256, 512, 128)
        self.decode_block1 = DecoderBlock(128, 256, 64)
        self.decode_block0 = DecoderBlock(64, 64, 64)
        self.decode_block_f = DecoderBlock(64, 32, 32)

        self.shallow = Conv2dReLU(4, 32, 3, 1)
        self.conv_last2 = nn.Conv2d(32, n_class, 3, 1, 1)

        # Initialize parameters for attention
        self.alpha0 = nn.Parameter(torch.ones(1))
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))

        self.beita0 = nn.Parameter(torch.zeros(1))
        self.beita1 = nn.Parameter(torch.zeros(1))
        self.beita2 = nn.Parameter(torch.zeros(1))
        self.beita3 = nn.Parameter(torch.zeros(1))

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        slope_normalized = input[:,8:9, :, :]
        ndi = input[:, 6:7, :, :]
        ndsi = input[:, 7:8, :, :]
       

        # # Compute per-image min and max
        # ndsi_min = ndsi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndsi_max = ndsi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndsi = (ndsi_batch - ndsi_min) / (ndsi_max - ndsi_min + 1e-8) 

        # # Compute per-image min and max
        # ndwi_min = ndwi_batch.amin(dim=[2, 3], keepdim=True)  # Min over H, W per image
        # ndwi_max = ndwi_batch.amax(dim=[2, 3], keepdim=True)  # Max over H, W per image

        # # Apply Min-Max normalization per image
        # ndi = (ndwi_batch - ndwi_min) / (ndwi_max - ndwi_min + 1e-8) 

        input_with_slope = torch.cat([input[:, [4,3,1], :, :], slope_normalized], dim=1)
        layer_shallow = self.shallow(input_with_slope)
        layer0 = self.layer0(input_with_slope)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.decode_block3(layer4, layer3)

        ndi2 = F.avg_pool2d(ndi, kernel_size=[8, 8])
        ndsi2 = F.avg_pool2d(ndsi, kernel_size=[8, 8])
        attention2 = self.alpha1 * ndi2 * layer2 + self.beita1 * ndsi2 * layer2 + self.gamma1 * layer2
        x = self.decode_block2(x, attention2)

        ndi1 = F.avg_pool2d(ndi, kernel_size=[4, 4])
        ndsi1 = F.avg_pool2d(ndsi, kernel_size=[4, 4])
        attention1 = self.alpha2 * ndi1 * layer1 + self.beita2 * ndsi1 * layer1 + self.gamma2 * layer1
        x = self.decode_block1(x, attention1)

        ndi0 = F.avg_pool2d(ndi, kernel_size=[2, 2])
        ndsi0 = F.avg_pool2d(ndsi, kernel_size=[2, 2])
        attention0 = self.alpha3 * ndi0 * layer0 + self.beita3 * ndsi0 * layer0 + self.gamma3 * layer0
        x = self.decode_block0(x, attention0)

        x = self.decode_block_f(x, layer_shallow)
        out1 = self.conv_last2(x)
        out1 = torch.sigmoid(out1)

        return out1
    

    