import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_training.GhostNet import GhostModule

class GhostResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride=1, use_se=False):
        super(GhostResNetBlock, self).__init__()
        # 使用 GhostModule 替代標準卷積
        self.ghost1 = GhostModule(in_channels, expansion, kernel_size=1, stride=1, use_se=use_se)
        self.bn1 = nn.BatchNorm2d(expansion)
        self.ghost2 = GhostModule(expansion, out_channels, kernel_size=3, stride=stride, use_se=use_se)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # shortcut 路徑
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.ghost1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.ghost2(x)
        x = self.bn2(x)
        x = x + residual
        x = F.relu(x)
        return x

class GhostResNet56(nn.Module):
    def __init__(self, num_classes=10):
        super(GhostResNet56, self).__init__()
        # 初始卷積層（與 ResNet-56 相同）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 三個階段，每個階段 9 個 GhostResNetBlock
        self.stage1 = self._make_layer(16, 16, 16, 1, 9, use_se=False)  # 16 通道
        self.stage2 = self._make_layer(16, 32, 32, 2, 9, use_se=True)  # 32 通道
        self.stage3 = self._make_layer(32, 64, 64, 2, 9, use_se=True)  # 64 通道

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, in_channels, out_channels, expansion, stride, num_blocks, use_se):
        layers = []
        layers.append(GhostResNetBlock(in_channels, out_channels, expansion, stride, use_se))
        for _ in range(1, num_blocks):
            layers.append(GhostResNetBlock(out_channels, out_channels, expansion, 1, use_se))
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
        x = self.softmax(x)
        return x

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    model = GhostResNet56(num_classes=10)
    model.apply(init_weights)
    x = torch.randn(2, 3, 32, 32)  # CIFAR-10 輸入，batch_size=2
    output = model(x)
    print(f"Output shape: {output.shape}")  # 應為 [2, 10]