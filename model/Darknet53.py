import torch
import torch.nn as nn

def ConvBlock(in_c, out_c, kernel, pad, stride):
    conv = nn.Conv2d(in_c, out_c, kernel, stride, pad, bias=True)
    bn = nn.BatchNorm2d(out_c)
    act = nn.LeakyReLU()
    return nn.Sequential(*[conv, bn, act])

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = ConvBlock(in_channels, reduced_channels, kernel=1, pad=0, stride=1)
        self.layer2 = ConvBlock(reduced_channels, in_channels, kernel=3, pad=1, stride=1)

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)

        return out+x

class Darknet53(nn.Module):

    def __init__(self, n_classes):
        super(Darknet53, self).__init__()
        
        self.conv1 = ConvBlock(3, 32, kernel=3, pad=1, stride=1)
        self.conv2 = ConvBlock(32, 64, kernel=3, pad=1, stride=2)
        self.conv3 = ConvBlock(64, 128, kernel=3, pad=1, stride=2)
        self.conv4 = ConvBlock(128, 256, kernel=3, pad=1, stride=2)
        self.conv5 = ConvBlock(256, 512, kernel=3, pad=1, stride=2) 
        self.conv6 = ConvBlock(512, 1024, kernel=3, pad=1, stride=2)

        self.residual_block1 = self.MakeBlock(64, 1)
        self.residual_block2 = self.MakeBlock(128, 2)
        self.residual_block3 = self.MakeBlock(256, 8)
        self.residual_block4 = self.MakeBlock(512, 8)
        self.residual_block5 = self.MakeBlock(1024, 4)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual_block1(x)
        x = self.conv3(x)
        x = self.residual_block2(x)
        x = self.conv4(x)
        x = self.residual_block3(x)
        x = self.conv5(x)
        x = self.residual_block4(x)
        x = self.conv6(x)
        x = self.residual_block5(x)
        x = self.global_avg_pool(x)
        
        x = x.view(-1, 1024)
        y = self.fc(x)

        return y

    def MakeBlock(self, in_c, n_blocks):
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResidualBlock(in_c))
        
        return nn.Sequential(*blocks)
