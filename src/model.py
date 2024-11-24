import torch
import torch.nn as nn


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, dropout=0.0):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),  # not reducing size of 2D matrix
        )

        if batch_norm:
            self.op.append(nn.BatchNorm2d(out_channels))
        self.op.extend(nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        ))
        if batch_norm:
            self.op.append(nn.BatchNorm2d(out_channels))
        self.op.extend(nn.Sequential(
            nn.ReLU(inplace=True)
        ))
        if dropout > 0.0:
            self.op.append(nn.Dropout2d(p=dropout))

    def forward(self, x):
        return self.op(x)


# TODO: consider constructing this with for loops an "feature maps" arrays.
#  Because now if I want to change the shape a little bit, it's gonna be aids...
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, batch_norm=True, dropout=0.2):
        super().__init__()

        filters = 32

        # encoder
        self.conv_down_1 = DoubleConv2d(in_channels, filters, batch_norm=batch_norm, dropout=dropout)
        self.sample_down_1 = nn.MaxPool2d(2)

        self.conv_down_2 = DoubleConv2d(filters, filters*2, batch_norm=batch_norm, dropout=dropout)
        self.sample_down_2 = nn.MaxPool2d(2)

        self.conv_down_3 = DoubleConv2d(filters*2, filters*4, batch_norm=batch_norm, dropout=dropout)
        self.sample_down_3 = nn.MaxPool2d(2)

        self.conv_down_4 = DoubleConv2d(filters*4, filters*8, batch_norm=batch_norm, dropout=dropout)
        self.sample_down_4 = nn.MaxPool2d(2)

        self.conv_down_5 = DoubleConv2d(filters*8, filters*16, batch_norm=batch_norm, dropout=dropout)
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

        # final segmentation map
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