from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.net(x)


class EmotionCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1,   64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
        )

        # Global average pooling over time & frequency
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classification
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.head(x)
        return x