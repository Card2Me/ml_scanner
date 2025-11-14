"""
DeepLabV3+ MobileNetV3 모델 구현
모바일 환경에 최적화된 semantic segmentation 모델
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class DeepLabV3PlusMobileNet(nn.Module):
    """
    DeepLabV3+ with MobileNetV2 backbone
    segmentation_models_pytorch 라이브러리 활용

    Args:
        n_classes: 출력 클래스 수
        encoder_name: 백본 인코더 (mobilenet_v2)
        encoder_weights: 사전학습 가중치 ('imagenet' 또는 None)
    """

    def __init__(self, n_classes=2, encoder_name="mobilenet_v2", encoder_weights="imagenet"):
        super(DeepLabV3PlusMobileNet, self).__init__()

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=n_classes,
            activation=None  # 로짓 출력 (softmax는 나중에)
        )

    def forward(self, x):
        return self.model(x)

class DeepLabV3PlusMobileNetCustom(nn.Module):
    """
    커스텀 DeepLabV3+ MobileNet 구현
    더 세밀한 제어가 필요할 때 사용
    """

    def __init__(self, n_classes=2, output_stride=16):
        super(DeepLabV3PlusMobileNetCustom, self).__init__()

        # MobileNetV2를 백본으로 사용
        from torchvision.models import mobilenet_v2
        mobilenet = mobilenet_v2(pretrained=True)

        # Encoder (MobileNetV2 features)
        self.backbone = mobilenet.features

        # ASPP (Atrous Spatial Pyramid Pooling)
        self.aspp = ASPP(1280, 256)  # MobileNetV2의 마지막 채널: 1280

        # Decoder
        self.decoder = Decoder(n_classes, low_level_channels=24)  # MobileNetV2 low-level: 24

    def forward(self, x):
        input_size = x.shape[-2:]

        # Encoder
        low_level_features = None
        for i, module in enumerate(self.backbone):
            x = module(x)
            # Low-level features (early layers)
            if i == 3:  # MobileNetV2의 3번째 블록
                low_level_features = x

        # ASPP
        x = self.aspp(x)

        # Decoder
        x = self.decoder(x, low_level_features)

        # 원본 크기로 복원
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        return x

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 3x3 convolutions with different dilation rates
        self.conv2 = self._make_aspp_conv(in_channels, out_channels, dilation=6)
        self.conv3 = self._make_aspp_conv(in_channels, out_channels, dilation=12)
        self.conv4 = self._make_aspp_conv(in_channels, out_channels, dilation=18)

        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Final convolution
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def _make_aspp_conv(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv_out(x)

        return x

class Decoder(nn.Module):
    """DeepLabV3+ Decoder"""

    def __init__(self, n_classes, low_level_channels=24):
        super(Decoder, self).__init__()

        # Low-level features reduction
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder convolutions
        self.conv_decode = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # Final classifier
        self.classifier = nn.Conv2d(256, n_classes, 1)

    def forward(self, x, low_level_features):
        low_level_features = self.conv_low(low_level_features)

        # Upsample x to match low_level_features size
        x = F.interpolate(x, size=low_level_features.shape[-2:], mode='bilinear', align_corners=True)

        # Concatenate
        x = torch.cat([x, low_level_features], dim=1)

        # Decode
        x = self.conv_decode(x)
        x = self.classifier(x)

        return x

def get_deeplabv3plus_mobilenet(n_classes=2, encoder_weights="imagenet", use_smp=True):
    """
    DeepLabV3+ MobileNet 모델 생성

    Args:
        n_classes: 출력 클래스 수
        encoder_weights: 사전학습 가중치 ('imagenet' 또는 None)
        use_smp: segmentation_models_pytorch 사용 여부
    """
    if use_smp:
        model = DeepLabV3PlusMobileNet(n_classes=n_classes, encoder_weights=encoder_weights)
    else:
        model = DeepLabV3PlusMobileNetCustom(n_classes=n_classes)

    return model

if __name__ == "__main__":
    # 모델 테스트
    model = get_deeplabv3plus_mobilenet(n_classes=2, use_smp=True)

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Forward pass 테스트
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
