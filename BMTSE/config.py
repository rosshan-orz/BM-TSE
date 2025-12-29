import torch
import os

class Config_cocktail_party:
    def __init__(self):
        # Dataloader essential
        self.root = 'cocktail party dataset root'
        self.subject = '1'  # Initial subject value
        
        # EEG settings
        self.num_channels = 64
        
        # Training settings
        self.temperature = 0.1
        self.batch_size = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_decay = 1e-3
        self.learning_rate = 2e-4
        self.patience = 1
        self.factor = 0.9
        self.epochs = 50
        self.dropout = 0.1
        # self.save_dir = 'result_cocktail'
        # os.makedirs(self.save_dir, exist_ok=True)
        
    @property
    def file_name(self):
        file_path = f'filename{self.subject}'
        return file_path

class Config_KUL:
    def __init__(self):
        # Dataloader essential
        self.root = 'kul dataset root'
        self.subject = '1'  # Initial subject value
        
        # EEG settings
        self.num_channels = 64
        
        # Training settings
        self.temperature = 0.1
        self.batch_size = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_decay = 1e-3
        self.learning_rate = 2e-4
        self.patience = 1
        self.factor = 0.9
        self.epochs = 50
        self.dropout = 0.1
        # self.save_dir = 'result_KUL'
        # os.makedirs(self.save_dir, exist_ok=True)
        
    @property
    def file_name(self):
        file_path = f'filename{self.subject}'
        return file_path