import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """CBAM channel attention module."""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = torch.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """CBAM spatial attention module."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = torch.sigmoid(self.conv(concat))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.layers = 5
        self.channel = 64
        self.image_size = params.image_size
        self.dropout = 0.34

        # Input convolution layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.channel, kernel_size=4, stride=2, padding=1)

        # Hidden convolution layer
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=self.channel * (2 ** i), out_channels=self.channel * (2 ** (i + 1)),
                      kernel_size=4, stride=2, padding=1)
            for i in range(self.layers - 1)
        ])

        # Output layer (GAN)
        final_filter_size = int(self.image_size / (2 ** self.layers))
        self.conv_gan = nn.Conv2d(in_channels=self.channel * (2 ** (self.layers - 1)), out_channels=1,
                                  kernel_size=final_filter_size, stride=1, padding=0)

        # Output layer (REG)
        self.conv_reg = nn.Conv2d(in_channels=self.channel * (2 ** (self.layers - 1)), out_channels=2,
                                  kernel_size=final_filter_size, stride=1, padding=0)

    def forward(self, x_init):
        # Input layer
        x = F.leaky_relu(self.conv1(x_init), negative_slope=0.2)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Hidden layer
        for conv in self.conv_layers:
            x = F.leaky_relu(conv(x), negative_slope=0.2)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # GAN output
        x_gan = self.conv_gan(x) # x_gan.shape = [32, 1, 1, 1]
        x_gan = x_gan.view(x_gan.size(0), -1) # x_gan.shape = [32, 1]
        #x_gan = torch.sigmoid(x_gan) # fake = 0, real = 1

        # REG output
        x_reg = self.conv_reg(x) # x_reg.shape = [32, 2, 1, 1]
        x_reg = x_reg.view(x_reg.size(0), -1)  # x_reg.shape = [32, 2]
        x_reg = torch.tanh(x_reg) # gaze in [-1, 1]

        return x_gan, x_reg

class Generator(nn.Module):
    def __init__(self, style_dim=3):
        super(Generator, self).__init__()
        self.style_dim = style_dim
        self.dropout = 0.15

        # Input layer
        self.input_conv = nn.Conv2d(3 + style_dim, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.input_norm = nn.InstanceNorm2d(64, affine=False)

        # Encoder layers
        self.encoder_convs = nn.ModuleList([
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        ])
        self.encoder_norms = nn.ModuleList([
            nn.InstanceNorm2d(128, affine=False),
            nn.InstanceNorm2d(256, affine=False)
        ])

        # Bottleneck layers
        self.bottleneck_blocks = nn.ModuleList([
            self._residual_block(256) for _ in range(6)
        ])

        # CBAM attention on bottleneck output (256 channels, 16x16 spatial)
        self.cbam = CBAM(256)

        # Decoder layers
        self.decoder_convs = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        ])
        self.decoder_norms = nn.ModuleList([
            nn.InstanceNorm2d(128, affine=False),
            nn.InstanceNorm2d(64, affine=False)
        ])

        # Output layer
        self.output_conv = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False)

    def _residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=False),
            nn.ReLU(inplace=False),
            nn.Dropout(p=self.dropout),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=False)
        )

    def forward(self, input_, angles):
        # Reshape and tile angles
        angles_reshaped = angles.view(-1, self.style_dim, 1, 1)
        angles_tiled = angles_reshaped.expand(-1, self.style_dim, input_.shape[2], input_.shape[3])
        x = torch.cat([input_, angles_tiled], dim=1)
        # torch.Size([32, 5, 64, 64])

        # Input layer
        x = self.input_conv(x)
        x = self.input_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # torch.Size([32, 64, 64, 64])

        # Encoder
        for conv, norm in zip(self.encoder_convs, self.encoder_norms):
            x = conv(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # torch.Size([32, 128, 32, 32])
            # torch.Size([32, 256, 16, 16])

        # Bottleneck
        for block in self.bottleneck_blocks:
            residual = x
            x = block(x)
            x = x + residual
            # torch.Size([32, 256, 16, 16])

        # Apply CBAM attention at bottleneck output
        x = self.cbam(x)

        # Decoder
        for deconv, norm in zip(self.decoder_convs, self.decoder_norms):
            x = deconv(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # torch.Size([32, 128, 32, 32])
            # torch.Size([32, 64, 64, 64])

        # Output layer
        x = self.output_conv(x)
        x = torch.tanh(x)
        # torch.Size([32, 3, 64, 64])
        return x


class VGG16(nn.Module):
    """VGG-16 感知损失网络，只初始化一次，避免每次前向传播重建模型。"""

    def __init__(self, vgg_dict):
        super(VGG16, self).__init__()

        def conv_block(in_channels, out_channels, num_layers):
            layers = []
            for i in range(num_layers):
                layers.append(nn.Conv2d(in_channels, out_channels,
                                        kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=False))
                in_channels = out_channels
            return nn.Sequential(*layers)

        self.conv1 = conv_block(3, 64, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv_block(64, 128, 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv_block(128, 256, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = conv_block(256, 512, 3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = conv_block(512, 512, 3)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Map flat numeric keys (e.g. "0.weight") to nested keys (e.g. "conv1.0.weight")
        new_dict = {}
        key_map = {
            '0.': 'conv1.0.', '2.': 'conv1.2.',
            '5.': 'conv2.0.', '7.': 'conv2.2.',
            '10.': 'conv3.0.', '12.': 'conv3.2.', '14.': 'conv3.4.',
            '17.': 'conv4.0.', '19.': 'conv4.2.', '21.': 'conv4.4.',
            '24.': 'conv5.0.', '26.': 'conv5.2.', '28.': 'conv5.4.',
        }
        for k, v in vgg_dict.items():
            new_key = k
            for flat_key, nested_key in key_map.items():
                if k.startswith(flat_key):
                    new_key = k.replace(flat_key, nested_key, 1)
                    break
            if new_key in self.state_dict():
                new_dict[new_key] = v

        loaded = self.load_state_dict(new_dict, strict=False)
        if loaded.missing_keys:
            print(f"VGG WARNING: missing keys: {loaded.missing_keys}")
        if loaded.unexpected_keys:
            print(f"VGG WARNING: unexpected keys: {len(loaded.unexpected_keys)} keys skipped")
        loaded_count = len(new_dict)
        total_count = len([k for k in self.state_dict() if 'conv' in k])
        print(f"VGG loaded {loaded_count}/{total_count} conv layer parameters")

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        end_points = {}

        x = self.conv1(inputs)
        end_points['conv1'] = x
        x = self.pool1(x)

        x = self.conv2(x)
        end_points['conv2'] = x
        x = self.pool2(x)

        x = self.conv3(x)
        end_points['conv3'] = x
        x = self.pool3(x)

        x = self.conv4(x)
        end_points['conv4'] = x
        x = self.pool4(x)

        x = self.conv5(x)
        end_points['conv5'] = x
        x = self.pool5(x)

        return x, end_points