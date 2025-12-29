import torch
import torchaudio
import numpy as np
import os
from torch.utils.data import Dataset

class EEGDataset_cocktail(Dataset):
    def __init__(self, root, file_name, subject_index):
        self.file_path = os.path.join(root, file_name)
        data = np.load(self.file_path, allow_pickle=True)
        
        # 存储被试索引
        self.subject_index = subject_index

        self.eeg_data = data['eeg']
        self.attended_audio_data = data['attended_audio']
        self.unattended_audio_data = data['unattended_audio']
        self.metadata = data['metadata'].item()  # metadata 是一个字典
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=16000, 
            new_freq=8000
        )

    def __len__(self):
        return len(self.eeg_data)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取EEG数据，并归一化处理
        eeg = self.eeg_data[idx].astype(np.float32)[:, :]  # (64, 128)
        eeg = torch.Tensor(eeg)
        eeg_min, _ = torch.min(eeg, dim=1, keepdim=True)
        eeg_max, _ = torch.max(eeg, dim=1, keepdim=True)
        eeg = (eeg - eeg_min) / (eeg_max - eeg_min + 1e-8)  # 加个小值防止除零
        
        # 获取clean数据
        clean = self.attended_audio_data[idx].astype(np.float32)  # (2, 16000)
        clean = torch.Tensor(clean)
        
        # 获取noisy数据
        unclean = self.unattended_audio_data[idx].astype(np.float32)  # (2, 16000)
        noisy = self.resampler(torch.Tensor(unclean)) + self.resampler(clean)
        noisy = noisy.unsqueeze(0)  # torch.Size([1, 16000])
        clean = self.resampler(clean)
        # print(noisy.shape)
        
        # 返回被试索引（整数）
        subject_idx = self.subject_index
        if subject_idx < 5:
            attention = 0
        else:
            attention = 1
        
        return eeg, noisy, clean, subject_idx, attention

class EEGDataset_KUL(Dataset):
    def __init__(self, root, file_name, subject_index):
        self.file_path = os.path.join(root, file_name)
        self.data = np.load(self.file_path, allow_pickle=True)

        self.subject_index = subject_index

        self.eeg_data = self.data['eeg']  # (4624, 64, 128)
        self.audioA_data = self.data['audio'][:, 0:1, :]  # (4624, 1, 16000)
        self.audioB_data = self.data['audio'][:, 1:2, :]  # (4624, 1, 16000)
        self.event_data = self.data['ear']  # (4624, 1)
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=16000, 
            new_freq=8000
        )
        
    def __len__(self):
        return len(self.eeg_data)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取EEG数据，并归一化处理
        eeg = self.eeg_data[idx].astype(np.float32)[:64, :]  # (64, 128)
        eeg = torch.Tensor(eeg)
        eeg_min, _ = torch.min(eeg, dim=1, keepdim=True)
        eeg_max, _ = torch.max(eeg, dim=1, keepdim=True)
        eeg = (eeg - eeg_min) / (eeg_max - eeg_min + 1e-8)  # 加个小值防止除零

        # 根据event获取clean音频数据
        event = self.event_data[idx] # moved here to be accessible
        clean = (self.audioA_data[idx] if event.item() == 0 else self.audioB_data[idx]).astype(np.float32)
        clean = torch.Tensor(clean)  # (1, 16000)
        
        # 获取 lfcc_wavA 和 lfcc_wavB 数据
        wavA = self.audioA_data[idx].astype(np.float32)  # (1, 16000)
        wavA = torch.Tensor(wavA)
        
        wavB = self.audioB_data[idx].astype(np.float32)  # (1, 16000)
        wavB = torch.Tensor(wavB)
        
        # 获取noisy数据
        noisy = self.resampler(wavA) + self.resampler(wavB)
        clean = self.resampler(clean)

        # 返回被试索引（整数）
        subject_idx = self.subject_index
        
        return eeg, noisy, clean, subject_idx