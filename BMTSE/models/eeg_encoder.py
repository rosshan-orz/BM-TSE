from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import torch
from torch import nn
from torch import Tensor
from einops import rearrange
from torch.nn.parameter import Parameter

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

class ChannelGate_sub(nn.Module):
    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=True):
        super(ChannelGate_sub, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm([in_channels//reduction, 1, 1])
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels//reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x, input * (1 - x), x

class sa_layer(nn.Module):
    def __init__(self, channel, sa_groups=4, size=None):
        super(sa_layer, self).__init__()
        self.groups = sa_groups
        self.size = size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * self.groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * self.groups), 1, 1))
        if size is not None:
            self.sweight = Parameter(torch.zeros(1, channel // (2 * self.groups), size[-2], size[-1]))
        else:
            self.sweight = Parameter(torch.zeros(1, channel // (2 * self.groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * self.groups), 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (self.groups * 2), channel // (self.groups * 2))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)
        return out

class PowerLayer(nn.Module):
    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2) + 1e-6))

class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            # This logic is a bit fragile, but we'll assume the nn.Conv1d was created with bias=False
            pass
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class InterFre(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        out = sum(x)
        out = F.gelu(out)
        return out

class Stem(nn.Module):
    def __init__(self, in_planes, out_planes = 64, kernel_size=63, patch_size=125, radix=2):
        nn.Module.__init__(self)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        self.kernel_size = kernel_size
        self.radix = radix
        self.sconv = Conv(nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False, groups=radix),
                          bn=nn.BatchNorm1d(self.mid_planes), activation=None)
        self.tconv = nn.ModuleList()
        for _ in range(self.radix):
            self.tconv.append(Conv(nn.Conv1d(self.out_planes, self.out_planes, kernel_size, 1, groups=self.out_planes, padding=kernel_size // 2, bias=False,),
                                   bn=nn.BatchNorm1d(self.out_planes), activation=None))
            kernel_size //= 2
        self.interFre = InterFre()
        self.downSampling = nn.AvgPool1d(patch_size, patch_size)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        out = self.sconv(x)
        out = torch.split(out, self.out_planes, dim=1)
        out = [m(x) for x, m in zip(out, self.tconv)]
        out = self.interFre(out)
        out = self.downSampling(out)
        out = self.dp(out)
        return out

class PatchEmbeddingTemporal(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, radix, patch_size, time_points):
        super().__init__()
        self.stem = Stem(
            in_planes=in_planes * radix, out_planes=out_planes, kernel_size=kernel_size,
            patch_size=patch_size, radix=radix
        )
        self.apply(self.initParms)

    # ==========================================================
    # FIXED: The initParms function is corrected here.
    # ==========================================================
    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.stem(x).permute(0, 2, 1)

class PatchEmbeddingSpatial(nn.Module):
    def __init__(self, spa_dim, emb_size=40):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, spa_dim, kernel_size=25, stride=5, padding=12),
            nn.ELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(spa_dim, emb_size)
        )

    def forward(self, x):
        B, C, T = x.shape
        # unsqueeze to add channel dim for conv1d
        x_reshaped = x.reshape(B * C, 1, T)
        return self.encoder(x_reshaped).view(B, C, -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size, self.num_heads = emb_size, num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) / (self.emb_size ** 0.5)
        if mask is not None: energy.mask_fill(~mask, torch.finfo(torch.float32).min)
        att = self.att_drop(F.softmax(energy, dim=-1))
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        return self.projection(rearrange(out, "b h n d -> b n (h d)"))

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size), nn.GELU(), nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=4, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size), MultiHeadAttention(emb_size, num_heads, drop_p), nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size), FeedForwardBlock(emb_size, forward_expansion, forward_drop_p), nn.Dropout(drop_p)
            ))
        )

class MyTransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, num_heads=4):
        super().__init__(*[TransformerEncoderBlock(emb_size, num_heads) for _ in range(depth)])




class SpectralAttention(nn.Module):
    def __init__(self, spa_dim=16, patch_size=16, time_sample_num=256, emb_size=128, 
                 depth=3, num_heads=4, chn=128):
        super().__init__()

        self.embedding = PatchEmbeddingTemporal(
            in_planes=chn, out_planes=emb_size, kernel_size=63, radix=1,
            patch_size=patch_size, time_points=time_sample_num
        )
        self.channel_embedding = PatchEmbeddingSpatial(spa_dim=spa_dim, emb_size=emb_size)
        
        self.P = time_sample_num // patch_size
        self.C = chn
        self.D = emb_size

        self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.P, self.D))
        self.pos_embedding_spatial = nn.Parameter(torch.randn(1, self.C, self.D))

        self.ain_temporal = sa_layer(channel=self.D, sa_groups=4, size=(1, self.P))
        self.log_power_temporal = PowerLayer(dim=-1, length=2, step=2)
        self.ain_spatial = sa_layer(channel=self.D, sa_groups=4, size=(1, self.C))
        self.log_power_spatial = PowerLayer(dim=-1, length=2, step=2)
        
        self.temporal_resizer = nn.AdaptiveAvgPool1d(self.C)
        self.spatial_resizer = nn.AdaptiveAvgPool1d(self.C)

        self.transformer_layer = MyTransformerEncoder(depth, emb_size, num_heads=num_heads)

    def _resize_sequence(self, x, resizer):
        return resizer(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x):
        # Handle different input shapes for robustness
        if x.dim() == 3: # (B, C, T)
            pass
        elif x.dim() == 4 and x.shape[1] == 1: # (B, 1, C, T)
            x = x.squeeze(1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        x_embed_temporal = self.embedding(x) + self.pos_embedding_temporal
        x_embed_spatial = self.channel_embedding(x) + self.pos_embedding_spatial

        x_temporal_4d = x_embed_temporal.permute(0, 2, 1).unsqueeze(2)
        ain_temp_out = self.ain_temporal(x_temporal_4d).squeeze(2).permute(0, 2, 1)
        log_temp_out = self.log_power_temporal(x_temporal_4d).squeeze(2).permute(0, 2, 1)
        
        x_spatial_4d = x_embed_spatial.permute(0, 2, 1).unsqueeze(2)
        ain_spat_out = self.ain_spatial(x_spatial_4d).squeeze(2).permute(0, 2, 1)
        log_spat_out = self.log_power_spatial(x_spatial_4d).squeeze(2).permute(0, 2, 1)

        ain_temp_resized = self._resize_sequence(ain_temp_out, self.temporal_resizer)
        log_temp_resized = self._resize_sequence(log_temp_out, self.temporal_resizer)
        log_spat_resized = self._resize_sequence(log_spat_out, self.spatial_resizer)

        x_fused_temporal = torch.cat([ain_temp_resized, log_temp_resized], dim=1)
        x_fused_spatial = torch.cat([ain_spat_out, log_spat_resized], dim=1)
        
        x_fused = x_fused_temporal + x_fused_spatial
        x_fused = self.transformer_layer(x_fused)
        x_fused = F.adaptive_avg_pool1d(x_fused.permute(0,2,1), output_size= 128).permute(0,2,1)

        return x_fused