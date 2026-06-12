import torch
import torch.nn as nn

class TimeProjBlock(nn.Module):
    # Time embedding block that converts time step t into a high-dimensional representation for conditioning the model
    def __init__(self, emb_dim, proj_dim, theta=10000):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.theta = theta
        self.lin1 = nn.Linear(emb_dim, proj_dim)
        self.lin2 = nn.Linear(proj_dim, proj_dim)
        self.silu = nn.SiLU()

    def get_sinusoidal_time_embedding(self, t):
        # t: torch.Tensor of shape (batch_size,)
        half_dim = self.emb_dim // 2
        i = torch.arange(half_dim, device=t.device)
        freqs = 1. / (self.theta ** (2 * i / self.emb_dim))
        angles = t.unsqueeze(1) * freqs.unsqueeze(0)
        sinusoidal_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return sinusoidal_emb

    def forward(self, t):
        sinusoidal_emb = self.get_sinusoidal_time_embedding(t)
        time_emb = self.lin1(sinusoidal_emb)
        time_emb = self.silu(time_emb)
        time_emb = self.lin2(time_emb)
        return time_emb

class ResNetBlock(nn.Module):
    # ResNet block learns residuals (changes that transform input into desired output) using skip connections
    # Skip connection: adds input (or projection) to residual to create output
    # Residuals updated based on output quality through backpropoation
    # Changes feature count, keeps spatial size, adds time conditioning
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.silu = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t_emb):
        projection = self.skip(x)
        features = self.norm1(x)
        features = self.silu(features)
        features = self.conv1(features)
        time_emb = self.time_mlp(t_emb).view(-1, features.size(1), 1, 1)
        features += time_emb
        features = self.norm2(features)
        features = self.silu(features)
        residual = self.conv2(features)
        return projection + residual

class SelfAttnBlock(nn.Module):
    # Flash attention
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        features = self.norm(x).view(B, C, H * W).permute(0, 2, 1)
        QKV = self.qkv(features).view(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        Q, K, V = QKV[0], QKV[1], QKV[2]
        attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        attn = attn.permute(0, 2, 1, 3).reshape(B, H * W, C)
        attn = attn.permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj(attn)

class DownBlock(nn.Module):
    # Downsamples spatially, increases channels, adds time conditioning, includes attention
    def __init__(self, in_channels, out_channels, time_emb_dim, num_heads, use_attn=True, downsample=True):
        super().__init__()
        self.resnet1 = ResNetBlock(in_channels, out_channels, time_emb_dim)
        self.resnet2 = ResNetBlock(out_channels, out_channels, time_emb_dim)
        self.attn = SelfAttnBlock(out_channels, num_heads) if use_attn else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if downsample else nn.Identity()

    def forward(self, x, time_emb):
        features = self.resnet1(x, time_emb)
        features = self.resnet2(features, time_emb)
        skip = self.attn(features)
        down = self.downsample(skip)
        return down, skip

class UpBlock(nn.Module):
    # Upsamples spatially, concatenates skip connections, decreases channels, adds time conditioning, includes attention
    def __init__(self, in_channels, out_channels, time_emb_dim, num_heads, use_attn=True, upsample=True):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1) if upsample else nn.Identity()
        self.resnet1 = ResNetBlock(in_channels + out_channels, out_channels, time_emb_dim)
        self.resnet2 = ResNetBlock(out_channels, out_channels, time_emb_dim)
        self.attn = SelfAttnBlock(out_channels, num_heads) if use_attn else nn.Identity()

    def forward(self, x, skip, time_emb):
        upsampled = self.upsample(x)
        features = torch.cat([upsampled, skip], dim=1)
        features = self.resnet1(features, time_emb)
        features = self.resnet2(features, time_emb)
        up = self.attn(features)
        return up

class MiddleBlock(nn.Module):
    # Bottleneck block at lowest spatial resolution
    def __init__(self, channels, time_emb_dim, num_heads):
        super().__init__()
        self.resnet1 = ResNetBlock(channels, channels, time_emb_dim)
        self.attn = SelfAttnBlock(channels, num_heads)
        self.resnet2 = ResNetBlock(channels, channels, time_emb_dim)

    def forward(self, x, t_emb):
        features = self.resnet1(x, t_emb)
        features = self.attn(features)
        bottleneck = self.resnet2(features, t_emb)
        return bottleneck

class Encoder(nn.Module):
    # Encodes input image into latent representation
    def __init__(self, in_channels, time_emb_dim, num_heads):
        super().__init__()
        self.down_block1 = DownBlock(in_channels, in_channels, time_emb_dim, num_heads, use_attn=False, downsample=True)
        self.down_block2 = DownBlock(in_channels, in_channels * 2, time_emb_dim, num_heads, use_attn=False, downsample=True)
        self.down_block3 = DownBlock(in_channels * 2, in_channels * 4, time_emb_dim, num_heads, use_attn=True, downsample=True)
        self.down_block4 = DownBlock(in_channels * 4, in_channels * 8, time_emb_dim, num_heads, use_attn=True, downsample=False)

    def forward(self, down, time_emb):
        down, skip1 = self.down_block1(down, time_emb)
        down, skip2 = self.down_block2(down, time_emb)
        down, skip3 = self.down_block3(down, time_emb)
        down, skip4 = self.down_block4(down, time_emb)
        return down, [skip1, skip2, skip3, skip4]

class Decoder(nn.Module):
    # Decodes latent representation back into image
    def __init__(self, in_channels, time_emb_dim, num_heads):
        super().__init__()
        self.up_block1 = UpBlock(in_channels, in_channels, time_emb_dim, num_heads, use_attn=True, upsample=False)
        self.up_block2 = UpBlock(in_channels, in_channels // 2, time_emb_dim, num_heads, use_attn=True, upsample=True)
        self.up_block3 = UpBlock(in_channels // 2, in_channels // 4, time_emb_dim, num_heads, use_attn=False, upsample=True)
        self.up_block4 = UpBlock(in_channels // 4, in_channels // 8, time_emb_dim, num_heads, use_attn=False, upsample=True)

    def forward(self, up, skips, time_emb):
        up = self.up_block1(up, skips[3], time_emb)
        up = self.up_block2(up, skips[2], time_emb)
        up = self.up_block3(up, skips[1], time_emb)
        up = self.up_block4(up, skips[0], time_emb)
        return up

class DDPM(nn.Module):
    def __init__(self, time_embedding_dim=256, num_attn_heads=4):
        super().__init__()
        self.time_proj = TimeProjBlock(
            emb_dim=time_embedding_dim // 2,
            proj_dim=time_embedding_dim
        )
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.encoder = Encoder(in_channels=64, time_emb_dim=time_embedding_dim, num_heads=num_attn_heads)
        self.middle = MiddleBlock(channels=512, time_emb_dim=time_embedding_dim, num_heads=num_attn_heads)
        self.decoder = Decoder(in_channels=512, time_emb_dim=time_embedding_dim, num_heads=num_attn_heads)
        self.conv_out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, xt, t):
        time_emb = self.time_proj(t)
        h = self.conv_in(xt)
        h, skips = self.encoder(h, time_emb)
        h = self.middle(h, time_emb)
        h = self.decoder(h, skips, time_emb)
        noise_pred = self.conv_out(h)
        return noise_pred