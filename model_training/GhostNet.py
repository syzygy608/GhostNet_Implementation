import torch
import torch.nn as nn
import torch.nn.functional as F

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, use_se=False, ratio=2):
        super(GhostModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_se = use_se
        self.ratio = ratio  # Ratio of intrinsic to total output channels

        # Number of intrinsic feature maps
        intrinsic_channels = out_channels // ratio

        # Primary convolution
        self.primary_conv = nn.Conv2d(
            in_channels, intrinsic_channels, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False
        )

        # Cheap operation (depthwise convolution)
        self.cheap_conv = nn.Conv2d(
            intrinsic_channels, intrinsic_channels, kernel_size=3,
            stride=1, padding=1, groups=intrinsic_channels, bias=False
        )

        # SE block (if enabled)
        if use_se:
            self.se_block = SEBlock(out_channels)

    def forward(self, x):
        # Primary convolution
        intrinsic = self.primary_conv(x)

        # Cheap operation to generate ghost feature maps
        ghost = self.cheap_conv(intrinsic)

        # Concatenate intrinsic and ghost feature maps
        out = torch.cat([intrinsic, ghost], dim=1)

        # Apply SE block if enabled
        if self.use_se:
            out = self.se_block(out)

        return out

class SEBlock(nn.Module):
    # Squeeze-and-Excitation Block
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 進行 squeeze
        self.fc = nn.Sequential( # 進行 excitation
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) # 做 scale
    
class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, kernel_size=3, stride=1, use_se=False):
        super(GhostBottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_se = use_se

        # First GhostModule (expansion)
        self.ghost1 = GhostModule(in_channels, expansion, kernel_size=1, stride=1, use_se=use_se)
        self.bn1 = nn.BatchNorm2d(expansion)
        
        # Depthwise convolution for stride=2
        self.depthwise = nn.Identity()
        if stride == 2:
            self.depthwise = nn.Conv2d(
                expansion, expansion, kernel_size=3, stride=2,
                padding=1, groups=expansion, bias=False
            )
            self.bn_depthwise = nn.BatchNorm2d(expansion)

        # Second GhostModule (projection)
        self.ghost2 = GhostModule(expansion, out_channels, kernel_size=1, stride=1, use_se=use_se)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut path
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        # First GhostModule
        x = self.ghost1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Depthwise convolution (if stride=2)
        x = self.depthwise(x)
        if self.stride == 2:
            x = self.bn_depthwise(x)
            x = F.relu(x)

        # Second GhostModule
        x = self.ghost2(x)
        x = self.bn2(x)

        # Residual connection
        x = x + residual
        x = F.relu(x)

        return x
    
class GhostNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GhostNet, self).__init__()
        self.num_classes = num_classes

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Ghost bottlenecks configuration
        self.configs = [
            (16, 16, 16, 1, False),  # Stage 1
            (16, 24, 48, 2, False),
            (24, 24, 72, 1, True),   # Stage 2
            (24, 40, 72, 2, False),
            (40, 40, 120, 1, True),  # Stage 3
            (40, 80, 240, 2, False),
            (80, 80, 200, 1, False),  # Stage 4
            (80, 80, 184, 1, False),
            (80, 80, 184, 1, False),
            (80, 112, 480, 1, True),
            (112, 112, 672, 1, True),
            (112, 160, 672, 2, True),
            (160, 160, 960, 1, False),  # Stage 5
            (160, 160, 960, 1, True),
            (160, 160, 960, 1, False),
            (160, 160, 960, 1, True),
        ]

        # Create GhostBottleneck layers
        self.stages = nn.ModuleList()
        for in_channels, out_channels, expansion, stride, use_se in self.configs:
            self.stages.append(
                GhostBottleneck(in_channels, out_channels, expansion, kernel_size=3, stride=stride, use_se=use_se)
            )

        # Stage 5 convolution
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(960)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Changed to 1x1 for global average pooling
        self.conv3 = nn.Conv2d(960, 1280, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.avgpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GhostNet(num_classes=10).to(device)
    print(model)
    model.eval()   
    # Test with a random input
    x = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    output = model(x.to(device))
    assert output.shape == (1, 10), "Output shape mismatch"