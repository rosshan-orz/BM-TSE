import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class SI_SDRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SI_SDRLoss, self).__init__()
        self.eps = eps

    def forward(self, est_targets, targets):
        # est_targets: (batch_size, num_channels, num_samples)
        # targets: (batch_size, num_channels, num_samples)
        if targets.dim() == 2:
            targets = targets.unsqueeze(1)
        if est_targets.dim() == 2:
            est_targets = est_targets.unsqueeze(1)

        targets_zm = targets - torch.mean(targets, dim=-1, keepdim=True)
        est_targets_zm = est_targets - torch.mean(est_targets, dim=-1, keepdim=True)

        s_target = targets_zm
        s_estimate = est_targets_zm

        dot_prod = torch.sum(s_estimate * s_target, dim=-1, keepdim=True)
        s_target_energy = torch.sum(s_target ** 2, dim=-1, keepdim=True) + self.eps

        alpha = dot_prod / s_target_energy
        s_proj = alpha * s_target
        e_noise = s_estimate - s_proj

        s_proj_energy = torch.sum(s_proj ** 2, dim=-1)
        e_noise_energy = torch.sum(e_noise ** 2, dim=-1)

        si_sdr_val = 10 * torch.log10((s_proj_energy + self.eps) / (e_noise_energy + self.eps))

        return -torch.mean(si_sdr_val)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class STFTLoss(nn.Module):
    def __init__(self, fft_size, hop_size):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window = torch.hann_window(fft_size)

    def forward(self, pred, target):
        self.window = self.window.to(pred.device)

        # STFT
        S_pred = torch.stft(pred, self.fft_size, self.hop_size, window=self.window, return_complex=True)
        S_target = torch.stft(target, self.fft_size, self.hop_size, window=self.window, return_complex=True)

        # Magnitude Loss (L1)
        mag_loss = F.l1_loss(torch.abs(S_pred), torch.abs(S_target))

        return mag_loss


class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_ratios=[0.25, 0.25, 0.25]):
        super().__init__()
        self.stft_losses = nn.ModuleList()
        for fft_size, hop_ratio in zip(fft_sizes, hop_ratios):
            self.stft_losses.append(STFTLoss(fft_size, int(fft_size * hop_ratio)))

    def forward(self, pred, target):
        total_loss = 0
        for stft_loss in self.stft_losses:
            total_loss += stft_loss(pred, target)
        return total_loss / len(self.stft_losses)

class CombinedAudioLoss(nn.Module):
    def __init__(self, si_sdr_weight=1.0, mse_weight=0.1, stft_weight=0.5, 
                 stft_config={'fft_sizes': [512, 1024, 2048], 'hop_ratios': [0.25, 0.25, 0.25]}):
        super().__init__()
        self.si_sdr_weight = si_sdr_weight
        self.mse_weight = mse_weight
        self.stft_weight = stft_weight
        
        self.si_sdr_loss = SI_SDRLoss()
        self.mse_loss = MSELoss()
        self.stft_loss = MultiScaleSTFTLoss(**stft_config)
        
    def forward(self, pred, target):
        si_sdr = self.si_sdr_loss(pred, target)
        mse = self.mse_loss(pred, target)
        stft = self.stft_loss(pred, target)
        
        total_loss = (
            self.si_sdr_weight * si_sdr + 
            self.mse_weight * mse + 
            self.stft_weight * stft
        )
        
        return total_loss, {'si_sdr': si_sdr.item(), 'mse': mse.item(), 'stft': stft.item()}

def loss_fn(pred, clean):
    cal_loss = CombinedAudioLoss(
        si_sdr_weight=1.0,
        mse_weight=0.1,
        stft_weight=0.5,
        stft_config={'fft_sizes': [512, 1024], 'hop_ratios': [0.25, 0.25]}
    )
    
    
    total_loss, loss_dict  = cal_loss(pred.squeeze(1), clean.squeeze(1))

    return total_loss