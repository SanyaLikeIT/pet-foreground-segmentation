import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet34_Weights, resnet34


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels + skip_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features, skip_features=None, output_size=None):
        target_size = skip_features.shape[-2:] if skip_features is not None else output_size
        features = F.interpolate(features, size=target_size, mode="bilinear", align_corners=False)
        if skip_features is not None:
            features = torch.cat((features, skip_features), dim=1)
        return self.layers(features)


class UNet(nn.Module):
    def __init__(self, pretrained_encoder=True):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained_encoder else None
        resnet = resnet34(weights=weights)
        self.encoder = nn.ModuleDict(
            {
                "stem": nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),
                "pool": resnet.maxpool,
                "layer1": resnet.layer1,
                "layer2": resnet.layer2,
                "layer3": resnet.layer3,
                "layer4": resnet.layer4,
            }
        )
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
        self.decoder0 = DecoderBlock(64, 0, 32)
        self.segmentation_head = nn.Conv2d(32, 1, kernel_size=1)

    def set_encoder_trainable(self, trainable):
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(trainable)
        self.encoder.train(trainable)

    def forward(self, images):
        stem = self.encoder["stem"](images)
        layer1 = self.encoder["layer1"](self.encoder["pool"](stem))
        layer2 = self.encoder["layer2"](layer1)
        layer3 = self.encoder["layer3"](layer2)
        layer4 = self.encoder["layer4"](layer3)

        features = self.decoder4(layer4, layer3)
        features = self.decoder3(features, layer2)
        features = self.decoder2(features, layer1)
        features = self.decoder1(features, stem)
        features = self.decoder0(features, output_size=images.shape[-2:])
        return self.segmentation_head(features)
