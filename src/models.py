import torch
import torch.nn as nn
from torchgeo import models as geomodels
from timm.layers.drop import DropBlock2d


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),  # not reducing size of 2D matrix
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if dropout > 0.0:
            self.op.append(nn.Dropout2d(p=dropout))

    def forward(self, x):
        return self.op(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, dropout=0.2):
        super().__init__()

        filters = 32

        # encoder
        self.conv_down_1 = DoubleConv2d(in_channels, filters, dropout=dropout)
        self.sample_down_1 = nn.MaxPool2d(2)

        self.conv_down_2 = DoubleConv2d(filters, filters*2, dropout=dropout)
        self.sample_down_2 = nn.MaxPool2d(2)

        self.conv_down_3 = DoubleConv2d(filters*2, filters*4, dropout=dropout)
        self.sample_down_3 = nn.MaxPool2d(2)

        self.conv_down_4 = DoubleConv2d(filters*4, filters*8, dropout=dropout)
        self.sample_down_4 = nn.MaxPool2d(2)

        self.conv_down_5 = DoubleConv2d(filters*8, filters*16, dropout=dropout)
        self.sample_down_5 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = DoubleConv2d(filters*16, filters*32)

        # decoder
        self.sample_up_5 = nn.ConvTranspose2d(filters*32, filters*16, kernel_size=2, stride=2)
        self.conv_up_5 = DoubleConv2d(filters*32, filters*16)

        self.sample_up_4 = nn.ConvTranspose2d(filters*16, filters*8, kernel_size=2, stride=2)
        self.conv_up_4 = DoubleConv2d(filters*16, filters*8)

        self.sample_up_3 = nn.ConvTranspose2d(filters*8, filters*4, kernel_size=2, stride=2)
        self.conv_up_3 = DoubleConv2d(filters*8, filters*4)

        self.sample_up_2 = nn.ConvTranspose2d(filters*4, filters*2, kernel_size=2, stride=2)
        self.conv_up_2 = DoubleConv2d(filters*4, filters*2)

        self.sample_up_1 = nn.ConvTranspose2d(filters*2, filters, kernel_size=2, stride=2)
        self.conv_up_1 = DoubleConv2d(filters*2, filters, dropout=dropout)

        # final segmentation layer
        self.conv_final = nn.Conv2d(filters, out_channels, kernel_size=1)
        self.act_final = nn.Sigmoid()


    def forward(self, x):
        # encode
        out_conv_down_1 = self.conv_down_1(x)
        out_sample_down_1 = self.sample_down_1(out_conv_down_1)

        out_conv_down_2 = self.conv_down_2(out_sample_down_1)
        out_sample_down_2 = self.sample_down_2(out_conv_down_2)

        out_conv_down_3 = self.conv_down_3(out_sample_down_2)
        out_sample_down_3 = self.sample_down_3(out_conv_down_3)

        out_conv_down_4 = self.conv_down_4(out_sample_down_3)
        out_sample_down_4 = self.sample_down_4(out_conv_down_4)

        out_conv_down_5 = self.conv_down_5(out_sample_down_4)
        out_sample_down_5 = self.sample_down_5(out_conv_down_5)

        # bottleneck
        out_bottleneck = self.bottleneck(out_sample_down_5)

        # decode
        out_up_5 = torch.cat((self.sample_up_5(out_bottleneck), out_conv_down_5), dim=1) # BxCxWxH
        out_conv_up_5 = self.conv_up_5(out_up_5)

        out_up_4 = torch.cat((self.sample_up_4(out_conv_up_5), out_conv_down_4), dim=1)
        out_conv_up_4 = self.conv_up_4(out_up_4)

        out_up_3 = torch.cat((self.sample_up_3(out_conv_up_4), out_conv_down_3), dim=1)
        out_conv_up_3 = self.conv_up_3(out_up_3)

        out_up_2 = torch.cat((self.sample_up_2(out_conv_up_3), out_conv_down_2), dim=1)
        out_conv_up_2 = self.conv_up_2(out_up_2)

        out_up_1 = torch.cat((self.sample_up_1(out_conv_up_2), out_conv_down_1), dim=1)
        out_conv_up_1 = self.conv_up_1(out_up_1)

        # final segmentation to class channels
        out_final = self.act_final(self.conv_final(out_conv_up_1))

        return out_final


class ResDoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity shortcut to maintain same dimensions
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act2(x + shortcut)


def modify_resnet(model: nn.Module, in_channels: int, dropout=0.0) -> nn.Module:
    """
    Modifies a given resnet model by:
        1. Changing number of input channels to in_channels
        2. Replacing initial conv, bn, act layers with a UNet-like residual double conv.
            This conv does not perform does not perform down-sampling.
    """
    model.conv1 = ResDoubleConv2d(in_channels, model.conv1.out_channels)

    # Deactivate now redundant initial layers
    model.bn1 = nn.Identity()
    model.act1 = nn.Identity()
    model.relu = nn.Identity()

    if dropout > 0.0:
        for layer in [model.layer3, model.layer4]:
            for bottleneck in layer:
                bottleneck.drop_block = DropBlock2d(dropout)

    return model


class Res50UNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=10,
                 pretrained_weights=geomodels.ResNet50_Weights.SENTINEL2_ALL_MOCO,
                 encoder_dropout=0.2,
                 decoder_dropout=0.2):
        super().__init__()

        # modify resnet50
        self.backbone = geomodels.resnet50(weights=pretrained_weights)
        self.backbone = modify_resnet(self.backbone, in_channels, encoder_dropout)

        # encoder (backbone)
        self.encoder1 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.act1)
        self.encoder1_maxpool = self.backbone.maxpool
        self.encoder2 = self.backbone.layer1
        self.encoder3 = self.backbone.layer2
        self.encoder4 = self.backbone.layer3

        # bottleneck
        self.encoder5_bottleneck = self.backbone.layer4

        # decoder
        self.sample_up_4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv_up_4 = DoubleConv2d(2048, 1024)

        self.sample_up_3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up_3 = DoubleConv2d(1024, 512)

        self.sample_up_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up_2 = DoubleConv2d(512, 256)

        self.sample_up_1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)  # x4 channel reduction to combat layer1 not performing down-sampling...
        self.conv_up_1 = DoubleConv2d(128, 64, dropout=decoder_dropout)

        # final segmentation layer
        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.act_final = nn.Sigmoid()

    def forward(self, x):
        # TODO: cleanup prints bellow
        # encode
        # print(f'x: {x.shape}')
        enc1 = self.encoder1(x)
#         print(f'enc1 (custom): {enc1.shape}')
        enc1_down = self.encoder1_maxpool(enc1)
#         print(f'encoder1_maxpool: {enc1_down.shape}')
        enc2 = self.encoder2(enc1_down)
#         print(f'enc2: {enc2.shape}')
        enc3 = self.encoder3(enc2)
#         print(f'enc3: {enc3.shape}')
        enc4 = self.encoder4(enc3)
#         print(f'enc4: {enc4.shape}')

        # bottleneck
        enc5_bottleneck = self.encoder5_bottleneck(enc4)
#         print(f'enc5_bottleneck: {enc5_bottleneck.shape}')

        # decode
        dec4 = torch.cat((self.sample_up_4(enc5_bottleneck), enc4), dim=1)
        dec4 = self.conv_up_4(dec4)
#         print(f'dec1: {dec4.shape}')

        dec3 = torch.cat((self.sample_up_3(dec4), enc3), dim=1)
        dec3 = self.conv_up_3(dec3)
#         print(f'dec1: {dec3.shape}')

        dec2 = torch.cat((self.sample_up_2(dec3), enc2), dim=1)
        dec2 = self.conv_up_2(dec2)
#         print(f'dec1: {dec2.shape}')

#         print(self.sample_up_1(dec2).shape)
        dec1 = torch.cat((self.sample_up_1(dec2), enc1), dim=1)
        dec1 = self.conv_up_1(dec1)
#         print(f'dec1: {dec1.shape}')

        # final segmentation to class channels
        out = self.act_final(self.conv_final(dec1))
#         print(f'out: {out.shape}')
#         exit()

        return out