import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

# class ndwi_attention(nn.Module):
#     def __init__(self, pooling_kernel):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.ones(1))
#         self.beta = nn.Parameter(torch.zeros(1))
#         self.pooling_layer = nn.AvgPool2d(kernel_size=pooling_kernel)

#     def forward(self, lateral_feature, input_ndwi):
#         x = self.pooling_layer(input_ndwi)
#         x = self.alpha * lateral_feature * x + self.beta * lateral_feature
#         return x

# class FPN_block(nn.Module):
#     def __init__(self, in_channels, out_channels=256, top_block=False):
#         super().__init__()
#         self.lateral_connection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                             kernel_size=1, stride=1, padding=0)
#         self.feature_output = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
#                                         kernel_size=3, stride=1, padding=1)
#         self.top_block = top_block

#     def forward(self, backbone_feature, pyramid_feature):
#         x = self.lateral_connection(backbone_feature)
#         if not self.top_block:
#             x += F.interpolate(pyramid_feature, scale_factor=2, mode="bilinear", align_corners=False)
#         output = self.feature_output(x)
#         return x, output

# class FPNDecoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.upsample_1 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.GroupNorm(8, 128),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.upsample_2 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.GroupNorm(8, 128),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#         )
#         self.upsample_3 = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.GroupNorm(8, 128),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.upsample_4 = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.GroupNorm(8, 128),
#             nn.ReLU()
#         )
#         self.upsample_5 = nn.Sequential(
#             nn.Conv2d(128, 1, kernel_size=1),
#             nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
#             nn.Sigmoid()
#         )

#     def forward(self, P2, P3, P4, P5):
#         mask_1 = self.upsample_1(P5)
#         mask_2 = self.upsample_2(P4)
#         mask_3 = self.upsample_3(P3)
#         mask_4 = self.upsample_4(P2)
#         sum_mask = mask_1 + mask_2 + mask_3 + mask_4
#         final_mask = self.upsample_5(sum_mask)
#         return final_mask

# class BottomUp(nn.Module):
#     def __init__(self):
#         super().__init__()
#         resnet50model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#         self.resnet_layers = list(resnet50model.children())
#         self.pyramid_layer_1 = nn.Sequential(*self.resnet_layers[:3])
#         self.pyramid_layer_2 = nn.Sequential(*self.resnet_layers[3:5])
#         self.pyramid_layer_3 = self.resnet_layers[5]
#         self.pyramid_layer_4 = self.resnet_layers[6]
#         self.pyramid_layer_5 = self.resnet_layers[7]

#     def forward(self, input):
#         feature_1 = self.pyramid_layer_1(input)
#         feature_2 = self.pyramid_layer_2(feature_1)
#         feature_3 = self.pyramid_layer_3(feature_2)
#         feature_4 = self.pyramid_layer_4(feature_3)
#         feature_5 = self.pyramid_layer_5(feature_4)
#         return feature_2, feature_3, feature_4, feature_5

# class TopDown(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.pyramid_block_1 = FPN_block(in_channels=2048, top_block=True)
#         self.pyramid_block_2 = FPN_block(in_channels=1024)
#         self.pyramid_block_3 = FPN_block(in_channels=512)
#         self.pyramid_block_4 = FPN_block(in_channels=256)
#         self.na_attention_1 = ndwi_attention(pooling_kernel=4)
#         self.na_attention_2 = ndwi_attention(pooling_kernel=8)
#         self.na_attention_3 = ndwi_attention(pooling_kernel=16)

#     def forward(self, C2, C3, C4, C5, ndwi):
#         _, P5 = self.pyramid_block_1(C5, None)
#         _, P4 = self.pyramid_block_2(C4, P5)
#         P4 = self.na_attention_3(P4, ndwi)
#         _, P3 = self.pyramid_block_3(C3, P4)
#         P3 = self.na_attention_2(P3, ndwi)
#         _, P2 = self.pyramid_block_4(C2, P3)
#         P2 = self.na_attention_1(P2, ndwi)
#         return P2, P3, P4, P5

# class FPNSegemer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bottomup_path = BottomUp()
#         self.topdown_path = TopDown()
#         self.decoder = FPNDecoder()

#     def forward(self, input_image):
#         C2, C3, C4, C5 = self.bottomup_path(input_image[:, [0, 1, 2], :, :])
#         P2, P3, P4, P5 = self.topdown_path(C2, C3, C4, C5, input_image[:, 5:6, :, :])
#         mask = self.decoder(P2, P3, P4, P5)
#         return mask
###############################################for 128 *128 #######################
class ndwi_attention(nn.Module):
    def __init__(self, pooling_kernel):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.pooling_layer = nn.AvgPool2d(kernel_size=pooling_kernel)

    def forward(self, lateral_feature, input_ndwi):
        x = self.pooling_layer(input_ndwi)
        x = F.interpolate(x, size=lateral_feature.shape[2:], mode="bilinear", align_corners=False)
        x = self.alpha * lateral_feature * x + self.beta * lateral_feature
        return x


class FPN_block(nn.Module):
    def __init__(self, in_channels, out_channels=256, top_block=False):
        super().__init__()
        self.lateral_connection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=1, stride=1, padding=0)
        self.feature_output = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                        kernel_size=3, stride=1, padding=1)
        self.top_block = top_block

    def forward(self, backbone_feature, pyramid_feature):
        x = self.lateral_connection(backbone_feature)
        if not self.top_block:
            x += F.interpolate(pyramid_feature, scale_factor=2, mode="bilinear", align_corners=False)
        output = self.feature_output(x)
        return x, output

class FPNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU()
        )
        self.upsample_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU()
        )
        self.channel_alignment = nn.Conv2d(128, 256, kernel_size=1)  # Align channels

        self.upsample_3 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Sigmoid()
        )

    def forward(self, P2, P3, P4, P5, ndwi):
        P5 = self.upsample_1(P5)
        P5 = self.channel_alignment(P5)  # Align channels
        P4 = self.upsample_2(P4 + P5)
        
        # Align P4 to match P3's size
        P4_resized = F.interpolate(P4, size=P3.shape[2:], mode='bilinear', align_corners=True)
        P3 = F.interpolate(P3 + P4_resized, scale_factor=2, mode='bilinear', align_corners=True)

        # Align P3 to match P2's size
        P3_resized = F.interpolate(P3, size=P2.shape[2:], mode='bilinear', align_corners=True)
        P2 = F.interpolate(P2 + P3_resized, scale_factor=2, mode='bilinear', align_corners=True)

        final_mask = self.upsample_3(P2)
        return final_mask



class BottomUp(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_layers = list(resnet50model.children())
        self.pyramid_layer_1 = nn.Sequential(*self.resnet_layers[:3])
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
    def __init__(self):
        super().__init__()
        self.pyramid_block_1 = FPN_block(in_channels=2048, top_block=True)
        self.pyramid_block_2 = FPN_block(in_channels=1024)
        self.pyramid_block_3 = FPN_block(in_channels=512)
        self.pyramid_block_4 = FPN_block(in_channels=256)
        self.ndwi_attention_1 = ndwi_attention(pooling_kernel=2)
        self.ndwi_attention_2 = ndwi_attention(pooling_kernel=4)
        self.ndwi_attention_3 = ndwi_attention(pooling_kernel=8)

    def forward(self, C2, C3, C4, C5, ndwi):
        _, P5 = self.pyramid_block_1(C5, None)
        _, P4 = self.pyramid_block_2(C4, P5)
        P4 = self.ndwi_attention_3(P4, ndwi)
        _, P3 = self.pyramid_block_3(C3, P4)
        P3 = self.ndwi_attention_2(P3, ndwi)
        _, P2 = self.pyramid_block_4(C2, P3)
        P2 = self.ndwi_attention_1(P2, ndwi)
        return P2, P3, P4, P5

class FPNSegemer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottomup_path = BottomUp()
        self.topdown_path = TopDown()
        self.decoder = FPNDecoder()

    def forward(self, input_image):
        C2, C3, C4, C5 = self.bottomup_path(input_image[:, :3, :, :])
        ndwi = input_image[:, 3:4, :, :]
        P2, P3, P4, P5 = self.topdown_path(C2, C3, C4, C5, ndwi)
        mask = self.decoder(P2, P3, P4, P5, ndwi)
        return mask
