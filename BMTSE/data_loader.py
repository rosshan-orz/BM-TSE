import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split
import random
from BMTSE.dataset import EEGDataset_cocktail, EEGDataset_KUL

def load_Dataset_cocktail_all(root, batch_size, shuffle=False):
    # 指定要使用的subject列表
    subject_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9]
    # subject_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    random.seed(42)
    
    for subject in subject_indices:
        file_name = f'subj_{subject+1}_epochs.npz'
        subject_dataset = EEGDataset_cocktail(root=root, file_name=file_name, subject_index=subject)
        # 验证数据集大小是否符合预期
        subject_len = len(subject_dataset)
        if subject_len != 900:
            print(f"Warning: Subject {subject} has {subject_len} samples, expected 900. Skipping...")
            continue
        total_len = len(subject_dataset)
        train_len = int(total_len * 0.75)
        valid_len = int(total_len * 0.125)
        test_len = total_len - train_len - valid_len
        train_subset, valid_subset, test_subset = random_split(subject_dataset, [train_len, valid_len, test_len])

        train_datasets.append(train_subset)
        valid_datasets.append(valid_subset)
        test_datasets.append(test_subset)
        
        print(f"Subject {subject}: Train samples={len(train_subset)}, Valid samples={len(valid_subset)}, Test samples={len(test_subset)}")

    # 合并所有subject的训练集和测试集
    TrainDataset = ConcatDataset(train_datasets)
    ValidDataset = ConcatDataset(valid_datasets)
    TestDataset = ConcatDataset(test_datasets)
    
    print(f"Total train samples: {len(TrainDataset)}")
    print(f"Total valid samples: {len(ValidDataset)}")
    print(f"Total test samples: {len(TestDataset)}")
    
    # 创建数据加载器
    kwargs = {"batch_size": batch_size, "num_workers": 4, "pin_memory": False, "drop_last": False}
    
    Train_dataloader = DataLoader(TrainDataset, shuffle=shuffle, **kwargs)
    ValidDataloader = DataLoader(ValidDataset, shuffle=False, **kwargs)
    Test_dataloader = DataLoader(TestDataset, shuffle=False, **kwargs)

    return Train_dataloader, ValidDataloader, Test_dataloader

def load_Dataset_KUL_all(root, batch_size, shuffle=False):
    subject_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # 创建训练集和测试集的列表
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    random.seed(42)
    
    for subject in subject_indices:
        file_name = f's{subject+1}.npz'
        subject_dataset = EEGDataset_KUL(root=root, file_name=file_name, subject_index=subject)
        # 验证数据集大小是否符合预期
        subject_len = len(subject_dataset)
        if subject_len != 4624:
            print(f"Warning: Subject {subject} has {subject_len} samples, expected 4624. Skipping...")
            continue
        total_len = len(subject_dataset)
        train_len = int(total_len * 0.75)
        valid_len = int(total_len * 0.125)
        test_len = total_len - train_len - valid_len
        train_subset, valid_subset, test_subset = random_split(subject_dataset, [train_len, valid_len, test_len])

        train_datasets.append(train_subset)
        valid_datasets.append(valid_subset)
        test_datasets.append(test_subset)
        
        print(f"Subject {subject}: Train samples={len(train_subset)}, Valid samples={len(valid_subset)}, Test samples={len(test_subset)}")

    # 合并所有subject的训练集和测试集
    TrainDataset = ConcatDataset(train_datasets)
    ValidDataset = ConcatDataset(valid_datasets)
    TestDataset = ConcatDataset(test_datasets)
    
    print(f"Total train samples: {len(TrainDataset)}")
    print(f"Total valid samples: {len(ValidDataset)}")
    print(f"Total test samples: {len(TestDataset)}")
    
    # 创建数据加载器
    kwargs = {"batch_size": batch_size, "num_workers": 4, "pin_memory": False, "drop_last": False}
    
    Train_dataloader = DataLoader(TrainDataset, shuffle=shuffle, **kwargs)
    ValidDataloader = DataLoader(ValidDataset, shuffle=False, **kwargs)
    Test_dataloader = DataLoader(TestDataset, shuffle=False, **kwargs)

    return Train_dataloader, ValidDataloader, Test_dataloader

def get_data_loaders(data_type, config, shuffle=True):
    if data_type == 'cocktail':
        return load_Dataset_cocktail_all(config.root, config.batch_size, shuffle)
    elif data_type == 'KUL':
        return load_Dataset_KUL_all(config.root, config.batch_size, shuffle)
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Choose 'cocktail' or 'KUL'.")