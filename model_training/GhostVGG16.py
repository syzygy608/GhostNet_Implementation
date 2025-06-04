import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_training.GhostNet import GhostModule

class GhostVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(GhostVGG16, self).__init__()
        self.configs = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        self.features = self._make_layers(self.configs)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [GhostModule(in_channels, v, kernel_size=3, stride=1, padding=1)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = GhostVGG16(num_classes=10)
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = torch.randn(2, 3, 224, 224).to(device)

    output = model(x)

    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 10), "Output shape mismatch"

    

