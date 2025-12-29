		# eeg_features, sub_id = self.eegencoder(eeg)
		# eeg_features = F.interpolate(eeg_features, (L,), mode='linear')

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eeg_encoder import SpectralAttention
from .TSE import Separator
from .Task import SubIdentify, AADClasify
from .map import ResConv

class BM_TSE(nn.Module):
	def __init__(self, n_classes):
		super().__init__()
		self.SpectralAttention = SpectralAttention()
		self.TSE = Separator()
		self.Brainmap = ResConv()
		self.SubID = SubIdentify(emb_size=128, n_classes=n_classes)
		self.AAD = AADClasify(emb_size=128)

	def forward(self, raw_eeg, mixture):
		eeg_features = self.SpectralAttention(raw_eeg)
		brainmap = self.Brainmap(eeg_features)
		target = self.TSE(mixture, eeg_features, brainmap)
		brainmap = brainmap.mean(dim=1)

		sub_idx = self.SubID(brainmap)
		AAD_result = self.AAD(brainmap)

		return target, sub_idx, AAD_result

