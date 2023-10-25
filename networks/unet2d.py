import torch
import torch.nn as nn

from pytorchcv.models.common import IBN


class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            IBN(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            IBN(out_ch),
            nn.ReLU())

    def forward(self, x):
        x = self.conv_block(x)
        return x


class InputBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InputBlock, self).__init__()
        self.input_block = DoubleConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.input_block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.down_block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down_block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(UpBlock, self).__init__()

        if not bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutputBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutputBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, base_filter=64, has_dropout=True):
        super(Encoder, self).__init__()

        self.in_block = InputBlock(in_channels, base_filter)
        self.down1 = DownBlock(base_filter, base_filter * 2)
        self.down2 = DownBlock(base_filter * 2, base_filter * 4)
        self.down3 = DownBlock(base_filter * 4, base_filter * 8)
        self.down4 = DownBlock(base_filter * 8, base_filter * 8)
        self.has_dropout = has_dropout
        if self.has_dropout:
            self.dropout = nn.Dropout2d(p=0.5, inplace=False)

    def forward(self, x):
        x1 = self.in_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.has_dropout:
            x5 = self.dropout(x5)
        return [x1, x2, x3, x4, x5]


class BaseDecoder(nn.Module):
    def __init__(self, base_filter):
        super(BaseDecoder, self).__init__()
        self.up1 = UpBlock(base_filter * 16, base_filter * 4)
        self.up2 = UpBlock(base_filter * 8, base_filter * 2)
        self.up3 = UpBlock(base_filter * 4, base_filter)
        self.up4 = UpBlock(base_filter * 2, base_filter)

    def up_path(self, x):
        x1, x2, x3, x4, x5 = x
        up1 = self.up1(x5, x4)
        up2 = self.up2(up1, x3)
        up3 = self.up3(up2, x2)
        feature = self.up4(up3, x1)
        return [up1, up2, up3, feature]

    def forward(self, x):
        return self.up_path(x)


class TaskDecoder(BaseDecoder):
    def __init__(self, base_filter, num_classes):
        super(TaskDecoder, self).__init__(base_filter=base_filter)
        self.out = nn.Conv2d(base_filter, num_classes, kernel_size=1)

    def forward(self, x):
        features = super(TaskDecoder, self).forward(x)
        output_x = self.out(features[-1])
        return {
            "logits": output_x,
            "features": features
        }


class UNet2D(nn.Module):
    def __init__(self, base_filter, in_channel, num_classes=15):
        super(UNet2D, self).__init__()
        self.encoder = Encoder(base_filter=base_filter, in_channels=in_channel)
        self.decoder = BaseDecoder(base_filter=base_filter)
        self.out_0 = nn.Conv2d(base_filter, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        d_features = self.decoder(features)
        return self.out_0(d_features[-1])


class UNet_MTL_2D(nn.Module):
    def __init__(self, base_filter, in_channel, num_classes=15):
        super(UNet_MTL_2D, self).__init__()
        self.encoder = Encoder(base_filter=base_filter, in_channels=in_channel)
        self.seg_decoder = TaskDecoder(base_filter=base_filter, num_classes=num_classes)
        self.sdm_decoder = TaskDecoder(base_filter=base_filter, num_classes=num_classes)

    def forward(self, x):
        features = self.encoder(x)
        seg_outputs = self.seg_decoder(features)
        sdm_outputs = self.sdm_decoder(features)
        return {
            "segmentation": seg_outputs["logits"],
            "sdm": sdm_outputs["logits"],
            "features_seg": seg_outputs["features"],
            "features_sdm": sdm_outputs["features"]

        }
