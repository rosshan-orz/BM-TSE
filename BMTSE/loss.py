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

        # Step 1: Zero-mean targets and est_targets
        targets_zm = targets - torch.mean(targets, dim=-1, keepdim=True)
        est_targets_zm = est_targets - torch.mean(est_targets, dim=-1, keepdim=True)

        # Step 2: Compute energy of the true sources
        s_target = targets_zm
        s_estimate = est_targets_zm

        # Step 3: Compute s_target * s_estimate and (s_target)^2
        dot_prod = torch.sum(s_estimate * s_target, dim=-1, keepdim=True)
        s_target_energy = torch.sum(s_target ** 2, dim=-1, keepdim=True) + self.eps

        # Step 4: Scale
        alpha = dot_prod / s_target_energy

        # Step 5: Compute projected s_target
        s_proj = alpha * s_target

        # Step 6: Compute noise
        e_noise = s_estimate - s_proj

        # Step 7: Compute SI-SDR
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
        
        # Phase Loss (L1) - 可以选择性地包含
        # phase_loss = F.l1_loss(torch.angle(S_pred), torch.angle(S_target))

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


class MelSpectrogramLoss(nn.Module):
    """Mel频谱损失"""
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80, eps=1e-7):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.eps = eps
        
        # 创建Mel滤波器
        self.mel_filter = torchaudio.transforms.MelScale(
            n_mels=n_mels, 
            sample_rate=sample_rate, 
            n_stft=n_fft // 2 + 1
        )
        
    def forward(self, pred, target):
        """
        计算Mel频谱损失
        pred: 预测信号 [batch_size, length]
        target: 目标信号 [batch_size, length]
        返回: L1损失在Mel频谱上的平均值
        """
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape}, target {target.shape}"
        
        # 计算STFT幅度谱
        stft_pred = torch.stft(
            pred, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.n_fft, 
            window=torch.hann_window(self.n_fft).to(pred.device),
            return_complex=True
        )
        mag_pred = torch.abs(stft_pred)
        
        stft_target = torch.stft(
            target, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.n_fft, 
            window=torch.hann_window(self.n_fft).to(target.device),
            return_complex=True
        )
        mag_target = torch.abs(stft_target)
        
        # 转换为Mel频谱
        mel_pred = self.mel_filter(mag_pred**2).clamp(min=self.eps)
        mel_target = self.mel_filter(mag_target**2).clamp(min=self.eps)
        
        # 使用对数压缩（更符合感知）
        log_mel_pred = torch.log(mel_pred)
        log_mel_target = torch.log(mel_target)
        
        # 计算L1损失
        loss = F.l1_loss(log_mel_pred, log_mel_target)
        
        return loss


class CombinedAudioLoss(nn.Module):
    """组合音频损失（SI-SDR + MSE + 频谱损失）"""
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
        """
        计算组合损失
        pred: 预测信号 [batch_size, length]
        target: 目标信号 [batch_size, length]
        返回: 加权损失和
        """
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