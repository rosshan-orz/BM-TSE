import torch
import torch.nn as nn

class SubIdentify(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__(
            nn.Linear(emb_size, 64), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )
    # The default forward method is sufficient after flattening.
    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return super().forward(x)
    
class AADClasify(nn.Sequential):
    def __init__(self, emb_size):
        super().__init__(
            nn.Linear(emb_size, 64), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return super().forward(x)
