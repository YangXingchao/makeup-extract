import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

import config
from utils.icosahedron import icosahedron_n


class CoarseReconsNet(nn.Module):
    def __init__(self, n_shape, n_exp, n_tex, n_spec):
        super().__init__()

        self.n_shape = n_shape
        self.n_exp = n_exp
        self.n_tex = n_tex
        self.n_spec = n_spec

        self.lp_init = torch.from_numpy(icosahedron_n)[None].to(torch.float32)

        backbone = resnet50()
        delattr(backbone, 'fc')

        def fit_forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            return x

        bound_method = fit_forward.__get__(backbone, backbone.__class__)
        setattr(backbone, 'forward', bound_method)

        last_dim = 2048
        self.final_layers = nn.ModuleList([
            conv1x1(last_dim, n_shape, bias=True),  # id
            conv1x1(last_dim, n_exp, bias=True),  # ex
            conv1x1(last_dim, n_tex, bias=True),  # tx
            conv1x1(last_dim, n_spec, bias=True),  # sp
            conv1x1(last_dim, 3, bias=True),  # r
            conv1x1(last_dim, 2, bias=True),  # tr
            conv1x1(last_dim, 1, bias=True),  # s
            conv1x1(last_dim, 27, bias=True),  # sh
            conv1x1(last_dim, 40, bias=True),  # p
            conv1x1(last_dim, 60, bias=True),  # ln
            conv1x1(last_dim, 3, bias=True),  # gain
            conv1x1(last_dim, 3, bias=True),  # bias
        ])
        for m in self.final_layers:
            nn.init.constant_(m.weight, 0.)
            nn.init.constant_(m.bias, 0.)

        self.backbone = backbone

    def forward(self, x):
        device = x.device
        x = self.backbone(x)
        output = []
        for layer in self.final_layers:
            output.append(layer(x))
        coeffs = torch.flatten(torch.cat(output, dim=1), 1)

        cnt = 0
        id = coeffs[:, 0:self.n_shape]
        cnt += self.n_shape

        ex = coeffs[:, cnt: cnt + self.n_exp]
        cnt += self.n_exp

        tx = coeffs[:, cnt: cnt + self.n_tex]
        cnt += self.n_tex

        sp = coeffs[:, cnt: cnt + self.n_spec]
        cnt += self.n_spec

        r = coeffs[:, cnt: cnt + 3]
        r += torch.tensor([1., 0.0, 0.0])[None].to(device)
        r = r * math.pi
        cnt += 3

        tr = coeffs[:, cnt:cnt + 2]
        tr *= config.FIT_SIZE // 2
        cnt += 2

        s = coeffs[:, cnt:cnt + 1]
        s += torch.ones(1, 1).to(device)
        cnt += 1

        sh = coeffs[:, cnt: cnt + 27].view(-1, 9, 3)
        sh += torch.tensor([0, 1., 0]).to(device)
        cnt += 27

        p = coeffs[:, cnt: cnt + 40].view(-1, 20, 2)
        p += torch.tensor([0, 1]).to(device)
        p *= torch.tensor([1., 200.]).to(device)
        cnt += 40

        ln = coeffs[:, cnt: cnt + 60].view(-1, 20, 3)
        ln *= 10
        ln += self.lp_init.to(device)
        cnt += 60

        gain = coeffs[:, cnt: cnt + 3]
        gain += 1
        cnt += 3

        bias = coeffs[:, cnt: cnt + 3]

        return {'id': id, 'tx': tx, 'sp': sp, 'ex': ex, 'r': r, 'tr': tr, 's': s, 'sh': sh, 'p': p, 'ln': ln,
                'gain': gain, 'bias': bias}


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(stride, stride), bias=bias)


class UVCompletionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = FeaturesExtractor()
        self.residual_model = ResidualBlock()
        self.generator = Generator()

    def forward(self, x):
        out_features = self.extractor(x)
        out_features = self.residual_model(out_features)
        predicted_tex = self.generator(out_features)

        return predicted_tex


class FeaturesExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = get_conv(6, 64, kernel_size=7, stride=1, padding=3)
        self.conv2 = get_conv(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = get_conv(128, 192, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        return out


def get_conv(in_channels, out_channels, kernel_size=3, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ResBlock(192)
        self.block2 = ResBlock(192)
        self.block3 = ResBlock(192)
        self.block4 = ResBlock(192)
        self.block5 = ResBlock(192)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.resBlock = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.resBlock(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = DecConvBlock(192, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = DecConvBlock(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = LastDecConv(64, 3, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = 2 * torch.tanh(out)
        return out


class DecConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.decConv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.decConv(x)


class LastDecConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.decConv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decConv(x)


class MakeupExtractNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.encConv1 = ConvBlock(n_channels, 64, 7, 1, 1)
        self.encConv2 = ConvBlock(64, 128, 3, 2, 1)
        self.encConv3 = ConvBlock(128, 192, 3, 2, 1)

        self.resblock = MakeExtResidualBlock(192)

        self.decConv3 = DecConvBlock(192, 128, 3, 2, 1, 1)
        self.decConv2 = DecConvBlock(128, 64, 3, 2, 1, 1)
        self.decConv1 = LastDecConv(64, n_classes, 7, 1, 1)

    def forward(self, x):
        out = self.encConv1(x)
        out = self.encConv2(out)
        out = self.encConv3(out)

        out = self.resblock(out)

        out = self.decConv3(out)
        out = self.decConv2(out)
        out = self.decConv1(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.convLayer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.convLayer(x)


class MakeExtResidualBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.block1 = ResBlock(channel)
        self.block2 = ResBlock(channel)
        self.block3 = ResBlock(channel)
        self.block4 = ResBlock(channel)
        self.block5 = ResBlock(channel)
        self.bn1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = F.relu(self.bn1(out))
        return out
