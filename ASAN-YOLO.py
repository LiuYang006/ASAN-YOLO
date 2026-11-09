class SafeDepthwiseConv2d(nn.Conv2d):

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        if (self.groups <= 1 or x.device.type != "cuda"
                or not torch.backends.cudnn.is_available()):
            return super().forward(x)
        with torch.backends.cudnn.flags(enabled=False):
            return functional.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )


class AnisotropicStripContext(nn.Module):

    def __init__(self, c1: int, c2: int | None = None, strip_k1: int = 7, strip_k2: int = 11, reduction: int = 4):
        super().__init__()
        channels = c1 if c2 is None else c2
        self.input_projection = ConvNormAct(c1, channels, 1) if c1 != channels else nn.Identity()

        def strip_path(kernel_size: int) -> nn.Sequential:
            return nn.Sequential(
                SafeDepthwiseConv2d(channels, channels, (1, kernel_size), 1, (0, kernel_size // 2), groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                SafeDepthwiseConv2d(channels, channels, (kernel_size, 1), 1, (kernel_size // 2, 0), groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )

        self.strip_short = strip_path(strip_k1)
        self.strip_long = strip_path(strip_k2)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        hidden_channels = max(8, channels // reduction)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.input_projection(inputs)
        short = self.strip_short(inputs)
        long = self.strip_long(inputs)
        fused = self.fusion(torch.cat([short, long], dim=1))
        return inputs + self.gamma * fused * self.channel_gate(inputs)


class SAN(nn.Module):

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 16):
        super().__init__()
        self.norm = nn.InstanceNorm2d(c1, affine=False, track_running_stats=False)
        hidden = max(c1 // reduction, 8)
        self.fc1 = nn.Conv2d(c1, hidden, 1, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden, c1, 1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.3))

    def forward(self, x: Tensor) -> Tensor:
        n = self.norm(x)
        s = torch.sigmoid(self.fc2(self.act(self.fc1(n.mean(dim=[2, 3], keepdim=True)))))
        return x + self.gamma * n * s


