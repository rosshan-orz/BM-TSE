import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def reshape(x, window_len, ind=None, dim=1):
    init_len = x.size(dim=dim)
    padding = (math.ceil(window_len / 2), (math.ceil(2 * init_len / window_len) - 2) * math.ceil(window_len / 2) + window_len - init_len)
    x = F.pad(x, padding, 'constant', 0.)

    x = x.unsqueeze(dim=dim)

    sizes_x = [1 for _ in range(len(x.size()) - 2)] + [window_len, 1]
    x = x.repeat(sizes_x)

    if ind is None or ind.size()[:-2] != x.size()[:-2]:
        ind = torch.range(0, (math.ceil(2 * init_len / window_len) - 1) * math.ceil(window_len / 2), math.ceil(window_len / 2))

        ind = ind.unsqueeze(dim=0)
        ind = ind.repeat(window_len, 1)
        col = torch.range(0, window_len - 1).unsqueeze(dim=1)
        ind = ind + col
        ind = ind.unsqueeze(dim=0)

        sizes_ind = [size for size in x.size()[:-2]] + [1, 1]
        ind = ind.repeat(sizes_ind)
        ind = ind.long()
        ind = ind.to(x.device)

    x = torch.gather(x, dim + 1, ind)

    return x, ind, padding


# get from https://github.com/kaituoxu/Conv-TasNet/blob/94eac1023eaaf11ca1bf3c8845374f7e4cd0ef4c/src/utils.py
def overlap_and_add(signal, frame_step):
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.reshape(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().reshape(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.reshape(*outer_dimensions, -1)
    return result

# TasNet Encoder

class TasNetEncoder(nn.Module):
    def __init__(self, window_length=4, out_channels=256, out_features=128, kernel_size=1):
        super(TasNetEncoder, self).__init__()
        self.window_length = window_length
        self.ind = None

        self.conv = nn.Conv1d(
            in_channels=window_length,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=False,
        )

        self.lin = nn.Linear(in_features=out_channels, out_features=out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input x: [batch_size, T]
        # after reshape x: [batch_size, M, L]
        x, self.ind, padding = reshape(x, self.window_length, ind=self.ind, dim=1)
        x_wave = x
        # after conv, relu x: [batch_size, E, L]
        x = self.conv(x)
        x = self.relu(x)
        # after permute x: [batch_size, L, E]
        batch_size, E, L = x.size()
        x = x.permute(0, 2, 1)
        # after lin x: [batch_size, L, D]
        x = self.lin(x)
        # after permute x: [batch_size, D, L]
        x = x.permute(0, 2, 1)

        return x, x_wave, padding

# Sandglasset blocks

class SegmentationModule(nn.Module):
    def __init__(self, segment_length=256):
        super(SegmentationModule, self).__init__()
        self.segment_length = segment_length
        self.ind = None

    def forward(self, x):
        # input x: [batch_size, D, L]
        # after reshape x: [batch_size, D, K, S]
        x, self.ind, padding = reshape(x, self.segment_length, ind=self.ind, dim=2)
        return x, padding


class LocalSequenceProcessingRNN(nn.Module):
    def __init__(self, in_channels, hidden_size=128, num_layers=1):
        super(LocalSequenceProcessingRNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )

        self.lin = nn.Linear(in_features=2 * hidden_size, out_features=in_channels, bias=True)
        self.ln = nn.LayerNorm(normalized_shape=in_channels)

    def forward(self, x):
        batch_size, D, K, S = x.size()
        # after permute/reshape x: [batch_size * S, K, D]
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(batch_size * S, K, D)
        x_residual = x
        # after lstm x: [batch_size * S, K, 2 * H]
        x, _ = self.lstm(x)
        # after lin x: [batch_size * S, K, D]
        x = self.lin(x)
        x = self.ln(x) + x_residual
        # after reshape/permute x: [batch_size, D, K, S]
        x = x.reshape(batch_size, S, K, D)
        x = x.permute(0, 3, 2, 1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, n_pos, dim=10000):
        super(PositionalEncoding, self).__init__()

        self.pe = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )

        self.pe[:, 0::2] = torch.FloatTensor(np.sin(self.pe[:, 0::2]))
        self.pe[:, 1::2] = torch.FloatTensor(np.cos(self.pe[:, 1::2]))
        self.pe = torch.Tensor(self.pe)
        self.pe.detach_()
        self.pe.requires_grad = False

    def forward(self, x):
        return self.pe[:, :x.size()[1]]


class SelfAttentiveNetwork(nn.Module):
    def __init__(self, in_channels, kernel_size, num_heads=8, dropout=0.1):
        super(SelfAttentiveNetwork, self).__init__()
        self.downsampling = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

        self.ln1 = nn.LayerNorm(normalized_shape=in_channels)
        self.san = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.pe = PositionalEncoding(in_channels)
        self.ln2 = nn.LayerNorm(normalized_shape=in_channels)

        self.upsampling = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, x):
        batch_size, D, K, S = x.size()
        # after permute/reshape x: [batch_size * S, D, K]
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size * S, D, K)
        # after downsampling x: [batch_size * S, D, floor(K / kernel_size)]
        x = self.downsampling(x)
        # after reshape/permute/reshape x: [batch_size * floor(K / kernel_size), S, D]
        x = x.reshape(batch_size, S, D, -1)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(-1, S, D)

        positional_encoding = torch.t(self.pe(x)).to(x.device)
        positional_encoding = positional_encoding.unsqueeze(dim=0)
        x = self.ln1(x) + positional_encoding
        x_residual = x
        # after san x: [batch_size * floor(K / kernel_size), S, D]
        x, _ = self.san(x, x, x)
        x = self.ln2(x + x_residual)

        # after reshape/permute/reshape x: [batch_size * S, D, floor(K / kernel_size)]
        x = x.reshape(batch_size, -1, S, D)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size * S, D, -1)

        # after upsampling x: [batch_size * S, D, K'], where K' = floor(K / kernel_size) * kernel_size
        x = self.upsampling(x)
        # after reshape/permute x: [batch_size, D, K', S], where K' = floor(K / kernel_size) * kernel_size
        x = x.reshape(batch_size, S, D, -1)
        x = x.permute(0, 2, 3, 1)

        return x


class SandglassetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, hidden_size=128, num_layers=1, num_heads=8, dropout=0.1):
        super(SandglassetBlock, self).__init__()
        self.local_rnn = LocalSequenceProcessingRNN(
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.san = SelfAttentiveNetwork(
            in_channels,
            kernel_size,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.local_rnn(x)
        x = self.san(x)
        return x

# Decoder

class MaskEstimation(nn.Module):
    def __init__(self, in_channels, out_channels, source_num, encoded_frame_dim, window_length, kernel_size=1):
        super(MaskEstimation, self).__init__()
        self.prelu = nn.PReLU()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=source_num * encoded_frame_dim,
            kernel_size=kernel_size,
        )

        self.source_num = source_num
        self.encoded_frame_dim = encoded_frame_dim
        self.window_length = window_length

        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=encoded_frame_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

    def forward(self, x, segmentation_padding):
        batch_size, D, K, S = x.size()
        # after prelu/conv x: [batch_size, CE, K, S]
        x = self.prelu(x)
        x = self.conv1(x)
        # after reshape/permute x: [batch_size, C, E, S, K]
        x = x.reshape(batch_size, self.source_num, -1, K, S)
        x = x.permute(0, 1, 2, 4, 3)
        # after overlap_and_add x: [batch_size, C, E, L]
        x = overlap_and_add(x, self.window_length)
        if segmentation_padding[1]:
            x = x[:, :, :, segmentation_padding[0]:-segmentation_padding[1]]
        else:
            x = x[:, :, :, segmentation_padding[0]:]

        # after reshape x: [batch_size * C, E, L]
        x = x.reshape(batch_size * self.source_num, self.encoded_frame_dim, -1)
        L = x.size()[2]
        x = self.relu(x)
        # after conv2 x: [batch_size * C, M, L]
        x = self.conv2(x)
        x = x.reshape(batch_size, self.source_num, -1, L)
        return x


class Decoder(nn.Module):
    def __init__(self, window_length):
        super(Decoder, self).__init__()
        self.window_length = window_length

    def forward(self, x, x_wave, encoder_padding):
        # input x: [batch_size, C, M, L]
        # after multiplication x: [batch_size, C, M, L]
        x_wave = x_wave.unsqueeze(dim=1)
        x = x * x_wave
        # after permute x: [batch_size, C, L, M]
        x = x.permute(0, 1, 3, 2)
        # after overlap_and_add x: [batch_size, C, T]
        x = overlap_and_add(x, self.window_length)
        if encoder_padding[1]:
            x = x[:, :, encoder_padding[0]:-encoder_padding[1]]
        else:
            x = x[:, :, encoder_padding[0]:]

        return x


class Sandglasset(nn.Module):
    def __init__(
        self,
        encoder_window_length=4,  # M = 4
        encoder_out_channels=256, # E = 256
        encoder_out_features=128, # D = 128
        encoder_kernel_size=1,
        segment_length=256,       # K = 256
        sandglasset_block_num=6,  # N = 6
        hidden_size=128,          # H = 128
        num_layers=1,
        num_heads=8,              # J = 8
        dropout=0.1,
        mask_estimation_kernel_size=1,
        source_num=1,
    ):
        super(Sandglasset, self).__init__()
        self.tasnet_encoder = TasNetEncoder(
            window_length=encoder_window_length,
            out_channels=encoder_out_channels,
            out_features=encoder_out_features,
            kernel_size=encoder_kernel_size,
        )

        self.segmentation = SegmentationModule(segment_length=segment_length)
        self.sandglasset_blocks_num = sandglasset_block_num

        sandglasset_blocks = []
        for i in range(sandglasset_block_num):
            sandglasset_blocks.append(
                SandglassetBlock(
                    encoder_out_features,
                    4**(i if i < sandglasset_block_num // 2 else sandglasset_block_num - i - 1),
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )

        self.sandglasset_blocks = nn.ModuleList(sandglasset_blocks)
        self.mask_estimation = MaskEstimation(
            encoder_out_features,
            encoder_window_length,
            source_num,
            encoder_out_channels,
            segment_length // 2,
            kernel_size=mask_estimation_kernel_size,
        )

        self.decoder = Decoder(encoder_window_length // 2)


    def forward(self, x):
        x, x_wave, encoder_padding = self.tasnet_encoder(x)
        x, segmentation_padding = self.segmentation(x)

        x_residuals = []
        for i in range(self.sandglasset_blocks_num):
            if i < self.sandglasset_blocks_num // 2:
                x = self.sandglasset_blocks[i](x)
                x_residuals.append(x)
            else:
                x = self.sandglasset_blocks[i](x + x_residuals[self.sandglasset_blocks_num - i - 1])

        x = self.mask_estimation(x, segmentation_padding)
        x = self.decoder(x, x_wave, encoder_padding)

        return x

class Separator(nn.Module):
    def __init__(self, 
                 encoder_window_length=4, 
                 encoder_out_channels=128,
                 encoder_out_features=128,
                 K=256, 
                 R=2,
                 H=128,
                 source_num=1,
                 ):
        super(Separator, self).__init__()
        
        self.source_num = source_num
        self.sandglasset_blocks_num = R

        self.tasnet_encoder = TasNetEncoder(
            window_length=encoder_window_length,
            out_channels=encoder_out_channels,
            out_features=encoder_out_features,
            kernel_size=1
        )

        self.av_conv = nn.Conv1d(2 * encoder_out_features, encoder_out_features, 1, bias=False)
        self.fusion_norm = nn.GroupNorm(1, encoder_out_features, eps=1e-8)
        self.segmentation = SegmentationModule(segment_length=K)
        self.sandglasset_blocks = nn.ModuleList()
        for i in range(R):
            self.sandglasset_blocks.append(
                SandglassetBlock(
                    in_channels=encoder_out_features, 
                    kernel_size=4**(i if i < R // 2 else R - i - 1),
                    hidden_size=H,
                    num_layers=1,
                    num_heads=8,
                    dropout=0.1
                )
            )
        
        self.mask_estimation = MaskEstimation(
            in_channels=encoder_out_features,
            out_channels=encoder_window_length, 
            source_num=source_num,
            encoded_frame_dim=encoder_out_channels,
            window_length=K // 2,
            kernel_size=1
        )
        self.brainmap_fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32, 4)
        )
        
        self.decoder = Decoder(encoder_window_length // 2)

    def forward(self, audio_waveform, eeg_features, brainmap):
        audio_features, x_wave, encoder_padding = self.tasnet_encoder(audio_waveform.squeeze())
        M, D, L = audio_features.size()
        eeg_features = F.interpolate(eeg_features, (L,), mode='linear')

        fused_features = self.fusion_norm(self.av_conv(torch.cat((audio_features, eeg_features), 1)))

        x, segmentation_padding = self.segmentation(fused_features)
        
        x_residuals = []
        for i in range(self.sandglasset_blocks_num):
            if i < self.sandglasset_blocks_num // 2:
                x = self.sandglasset_blocks[i](x)
                x_residuals.append(x)
            else:
                x = self.sandglasset_blocks[i](x + x_residuals[self.sandglasset_blocks_num - i - 1])

        masks = self.mask_estimation(x, segmentation_padding)
        brainmap = self.brainmap_fc(F.interpolate(brainmap, (8000,), mode='linear').permute(0, 2, 1)).permute(0,2,1).unsqueeze(1)
        masks = masks + brainmap
        separated_audio = self.decoder(masks, x_wave, encoder_padding)
        
        return separated_audio