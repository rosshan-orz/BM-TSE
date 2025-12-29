import numpy as np
from pesq import pesq
from pystoi import stoi
import torch
import torch.nn as nn

class SI_SDRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred, target):
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape}, target {target.shape}"
        
        target_zm = target - torch.mean(target, dim=-1, keepdim=True)
        pred_zm = pred - torch.mean(pred, dim=-1, keepdim=True)
        
        dot_product = torch.sum(pred_zm * target_zm, dim=-1, keepdim=True)
        target_norm_sq = torch.sum(target_zm**2, dim=-1, keepdim=True) + self.eps
        alpha = dot_product / target_norm_sq
        
        e_noise = pred_zm - alpha * target_zm
        
        signal_power = torch.sum((alpha * target_zm)**2, dim=-1)
        noise_power = torch.sum(e_noise**2, dim=-1) + self.eps
        sisdr = 10 * torch.log10(signal_power / noise_power)
        
        return -torch.mean(sisdr)

# SI-SDR 计算
def si_sdr(s, s_hat):
    alpha = np.dot(s_hat, s)/np.linalg.norm(s)**2   
    sdr = 10*np.log10(np.linalg.norm(alpha*s)**2/np.linalg.norm(
        alpha*s - s_hat)**2)
    return sdr

# STOI/ESTOI 计算（通过 extended 参数切换）
def stoi_score(ref, deg, sr, extended=False):
    return stoi(ref, deg, sr, extended=extended)

# PESQ 计算
def pesq_score(ref, deg, sr):
    # 根据采样率选择PESQ模式
    if sr == 8000:
        return pesq(sr, ref, deg, 'nb')  # 'nb' 表示窄带模式
    elif sr == 16000:
        return pesq(sr, ref, deg, 'wb')  # 'wb' 表示宽带模式
    else:
        raise ValueError(f"Unsupported sampling rate for PESQ: {sr}. Must be 8000 or 16000.")

def cal_sisdri(mix_wave, target_wave, estmate_wave):
    loss_metric = SI_SDRLoss() 
    sisdr1 = loss_metric(mix_wave.squeeze(1), target_wave)
    sisdr2 = loss_metric(estmate_wave.squeeze(1), target_wave)
    sisdri = sisdr2 - sisdr1
    return -sisdri