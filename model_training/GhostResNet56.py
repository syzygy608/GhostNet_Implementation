import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_training.GhostNet import GhostBottleneck

class GhostResNet56(nn.Module):
    def __init__(self, num_classes=10):
        super(GhostResNet56, self).__init__()
        # 初始卷積層（與 ResNet-56 相同）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 三個階段，每個階段 9 個 GhostBottleneck
        # 第一個階段：16 通道，9 個 GhostBottleneck，步幅為 1
        self.stage1 = self._make_layer(16, 16, 16, 1, 9, use_se=False)  # 16 通道
        # 第二個階段：16 通道到 32 通道，9 個 GhostBottleneck，第一個步幅為 2
        self.stage2 = self._make_layer(16, 32, 32, 2, 9, use_se=True)  # 32 通道
        # 第三個階段：32 通道到 64 通道，9 個 GhostBottleneck，第一個步幅為 2
        self.stage3 = self._make_layer(32, 64, 64, 2, 9, use_se=True)  # 64 通道

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, expansion, stride, num_blocks, use_se):
        layers = []
        layers.append(GhostBottleneck(in_channels, out_channels, expansion, kernel_size=3, stride=stride, use_se=use_se))
        for _ in range(1, num_blocks):
            layers.append(GhostBottleneck(out_channels, out_channels, expansion, kernel_size=3, stride=1, use_se=use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = GhostResNet56(num_classes=10)
    x = torch.randn(2, 3, 32, 32)  # CIFAR-10 輸入，batch_size=2
    output = model(x)
    print(f"Output shape: {output.shape}")  # 應為 [2, 10]