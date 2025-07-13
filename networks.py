from torch import nn
import torch.nn.functional as F
import torch

"""
###################
TinyVGG model
###################
"""
class CIFAR10TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            # Create a conv layer
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*8*8,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)

        return x


"""
###################
ResNet-12 model
###################
"""   
class ResNetBlock(nn.Module):
    def __init__(self, input_shape: int = 3, output_shape: int = 64):
        super(ResNetBlock, self).__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=output_shape, kernel_size=1),
            nn.BatchNorm2d(output_shape)
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=output_shape, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_shape),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=output_shape, out_channels=output_shape, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_shape),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=output_shape, out_channels=output_shape, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_shape),
            nn.LeakyReLU()
        )
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        identity = self.skip(x)
        x = self.conv_block(x)
        x += identity
        x = self.relu(x)
        x = self.pool(x)
        return x

class CIFAR10ResNet12(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.resnet_block_1 = ResNetBlock(input_shape=input_shape, output_shape=hidden_units)
        self.resnet_block_2 = ResNetBlock(input_shape=hidden_units, output_shape=hidden_units * 2)
        self.resnet_block_3 = ResNetBlock(input_shape=hidden_units * 2, output_shape=hidden_units * 4)
        self.resnet_block_4 = ResNetBlock(input_shape=hidden_units * 4, output_shape=hidden_units * 8)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(hidden_units * 8, output_shape)

    def forward(self, x):
        x = self.resnet_block_1(x)
        x = self.resnet_block_2(x)
        x = self.resnet_block_3(x)
        x = self.resnet_block_4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


"""
###################
RNN(GRU) model
###################
"""
class CIFAR10RNN(nn.Module):
    def __init__(self, input_shape, hidden_units, num_layers, output_shape):
        super(CIFAR10RNN, self).__init__()
        self.rnn = nn.GRU(input_shape, hidden_units, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_shape)

    def forward(self, x):
        out, _ = self.rnn(x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1))
        out = self.fc(out[:, -1, :])
        return out


"""
###################
Vision Transformer model
###################
"""   
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.projection(x) # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, E)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class CIFAR10VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, dropout):
        super(CIFAR10VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.encoder = nn.Sequential(*[TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        x = self.fc(cls_token)
        return x
    

"""
###################
MLP Mixer model
###################
""" 
class MixerLayer(nn.Module):
    def __init__(self, embed_dim, num_patches, token_intermediate_dim, channel_intermediate_dim, dropout):
        super(MixerLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp1 = MLP(num_patches, token_intermediate_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = MLP(embed_dim, channel_intermediate_dim, dropout)

    def forward(self, x):
        y = self.norm1(x)
        y = torch.permute(y, (0, 2, 1))  # (B, N, E) -> (B, E, N)
        y = self.mlp1(y)
        y = torch.permute(y, (0, 2, 1))  # (B, E, N) -> (B, N, E)
        x = x + y
        y = self.norm2(x)
        y = self.mlp2(y)
        x = x + y
        return x
    
class CIFAR10MLPMixer(nn.Module):
    def __init__(self, in_channels, embed_dim, num_classes, patch_size, img_size, depth, token_intermediate_dim, channel_intermediate_dim, dropout):
        super(CIFAR10MLPMixer, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.mixers = nn.Sequential(*[MixerLayer(embed_dim, self.num_patches, token_intermediate_dim, channel_intermediate_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)  # (B, E, H/P, W/P)
        x = torch.permute(x, (0, 2, 3, 1)).reshape(x.shape[0], -1, x.shape[1])
        x = self.mixers(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
