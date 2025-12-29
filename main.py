import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import MultiheadAttention
from torchaudio import transforms
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import argparse
import math,copy,time,random,os
import _pickle as cPickle
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.autograd import Variable
import matplotlib.pyplot as plt
# from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from datetime import datetime
from einops.layers.torch import Rearrange
import gc
import h5py

import warnings

# 忽略UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

from BMTSE.config import Config_cocktail_party, Config_KUL
from BMTSE.data_loader import load_Dataset_cocktail_all, load_Dataset_KUL_all, get_data_loaders
from BMTSE.utils import AvgMeter, get_lr, print_size
from BMTSE.evaluate import evaluate
from BMTSE.models.main_model import BM_TSE
from BMTSE.loss import loss_fn
from BMTSE.metrics import si_sdr, cal_sisdri, stoi_score, pesq_score

def train_epoch(model, train_loader, optimizer_all, subject_criterion, aad_criterion, config, epoch):
    model.train()
    total_loss = 0
    total_sisdri = 0
    batchsizes = 0
    total_sub_correct = 0
    total_aad_correct = 0
    total_samples = 0

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1} - Train") as pbar:
        for batch_idx, (eeg, noise, clean, subject_idx, attention) in enumerate(train_loader):
            
            optimizer_all.zero_grad()
            eeg = eeg.to(config.device)
            noise = noise.to(config.device)
            clean = clean.to(config.device)
            subject_idx = subject_idx.to(config.device)  # 确保subject_idx在正确设备上
            attention = attention.to(config.device)

            # 模型返回分离音频和被试分类logits
            output, subject_logits, aad_logits = model(eeg, noise)

            # print(output.shape, clean.shape)

            main_loss = loss_fn(output, clean)
            
            # 计算被试分类损失
            subject_loss = subject_criterion(subject_logits, subject_idx)

            aad_loss = aad_criterion(aad_logits, attention)
            
            # 组合损失（调整权重）
            loss = main_loss + subject_loss + aad_loss
            
            # 计算准确率
            _, predicted = torch.max(subject_logits, 1)
            correct = (predicted == subject_idx).sum().item()

            _, aad_predicted = torch.max(aad_logits, 1)
            aad_correct = (aad_predicted == attention).sum().item()
            
            # 反向传播

            optimizer_all.zero_grad()
            loss.backward()
            optimizer_all.step()

            # 更新统计信息
            batch_size = subject_idx.size(0)
            total_loss += loss.item()
            total_sub_correct += correct
            total_aad_correct += aad_correct
            total_samples += batch_size
            
            # 计算SISDRi
            with torch.no_grad():
                sisdri = cal_sisdri(noise.squeeze(1), clean, output)
                total_sisdri += sisdri.sum().item()

            # 更新进度条
            batchsizes += 1
            pbar.update(1)
            pbar.set_postfix({
                '-sisdr': f'{total_loss / batchsizes:.4f}',
                'sisdri': f'{(total_sisdri / total_samples):.4f}',
                'sub_acc': f'{(total_sub_correct / total_samples):.4f}',
                'aad_acc': f'{(total_aad_correct / total_samples):.4f}'
            })

            # 清理内存
            del eeg, clean, noise, output, loss, subject_logits, aad_logits
            torch.cuda.empty_cache()
            gc.collect()

    # 计算平均值
    loss_avg = total_loss / batchsizes
    sisdri_avg = total_sisdri / total_samples
    sub_acc_avg = total_sub_correct / total_samples
    aad_acc_avg = total_aad_correct / total_samples

    
    return loss_avg, sisdri_avg, sub_acc_avg, aad_acc_avg

def evaluate_epoch(dataloader, model, loss_fn, subject_criterion, aad_criterion, config, epoch):
    model.eval()
    loss_meter = AvgMeter("Loss")
    sisdri_meter = AvgMeter("SISDRi")
    subject_acc_meter = AvgMeter("Subject Acc")
    aad_acc_meter = AvgMeter("AAD Acc")
    stoi_meter = AvgMeter("STOI")
    estoi_meter = AvgMeter("ESTOI") # Add ESTOI meter
    pesq_meter = AvgMeter("PESQ")

    with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1} - Valid") as pbar:
        with torch.no_grad():
            for eeg, noise, clean, subject_idx, attention in dataloader:
                eeg = eeg.to(config.device)
                noise = noise.to(config.device)
                clean = clean.to(config.device)
                subject_idx = subject_idx.to(config.device)
                attention = attention.to(config.device)

                # 模型前向传播
                pred, subject_logits, aad_logits = model(eeg, noise)
                
                # 计算主任务损失
                main_loss = loss_fn(pred, clean)
                
                # 计算被试分类损失
                subject_loss = subject_criterion(subject_logits, subject_idx)
                aad_loss = aad_criterion(aad_logits, attention)
                
                # 组合损失
                loss = main_loss + subject_loss + aad_loss
                
                # 计算准确率
                _, predicted = torch.max(subject_logits, 1)
                correct = (predicted == subject_idx).sum().item()

                _, aad_predicted = torch.max(aad_logits, 1)
                aad_correct = (aad_predicted == attention).sum().item()
                
                # 计算SISDRi
                sisdri = cal_sisdri(noise.squeeze(1), clean, pred)

                # 将Tensor移动到CPU并转换为numpy数组，以便计算STOI和PESQ
                clean_np = clean.squeeze(1).cpu().numpy()
                pred_np = pred.squeeze(1).cpu().numpy()
                
                # 计算STOI, ESTOI和PESQ
                batch_stoi = []
                batch_estoi = [] # List for ESTOI
                batch_pesq = []
                for i in range(clean_np.shape[0]):
                    try:
                        batch_stoi.append(stoi_score(clean_np[i], pred_np[i], 8000, extended=False))
                        batch_estoi.append(stoi_score(clean_np[i], pred_np[i], 8000, extended=True)) # Calculate ESTOI
                        batch_pesq.append(pesq_score(clean_np[i], pred_np[i], 8000))
                    except Exception as e:
                        # 捕获pesq或stoi可能出现的错误，例如音频长度不匹配
                        print(f"Error calculating STOI/PESQ/ESTOI for a sample: {e}")
                        batch_stoi.append(np.nan) # 使用NaN标记失败的计算
                        batch_estoi.append(np.nan)
                        batch_pesq.append(np.nan)
                
                avg_batch_stoi = np.nanmean(batch_stoi) if batch_stoi else 0
                avg_batch_estoi = np.nanmean(batch_estoi) if batch_estoi else 0 # Average ESTOI
                avg_batch_pesq = np.nanmean(batch_pesq) if batch_pesq else 0
                
                # 更新统计信息
                batch_size = subject_idx.size(0)
                loss_meter.update(loss.item(), batch_size)
                sisdri_meter.update(sisdri.mean().item(), batch_size)
                subject_acc_meter.update(correct / batch_size, batch_size)
                aad_acc_meter.update(aad_correct / batch_size, batch_size)
                stoi_meter.update(avg_batch_stoi, batch_size)
                estoi_meter.update(avg_batch_estoi, batch_size) # Update ESTOI meter
                pesq_meter.update(avg_batch_pesq, batch_size)
                
                pbar.update(1)
                pbar.set_postfix({
                    '-sisdr': f'{loss_meter.avg:.4f}',
                    'sisdri': f'{sisdri_meter.avg:.4f}',
                    'sub_acc': f'{subject_acc_meter.avg:.4f}',
                    'aad_acc': f'{aad_acc_meter.avg:.4f}',
                    'stoi': f'{stoi_meter.avg:.4f}',
                    'estoi': f'{estoi_meter.avg:.4f}', # Display ESTOI
                    'pesq': f'{pesq_meter.avg:.4f}'
                })
                
                del eeg, clean, noise, pred, loss, subject_logits, aad_logits
                torch.cuda.empty_cache()
            
    return loss_meter.avg, sisdri_meter.avg, subject_acc_meter.avg, aad_acc_meter.avg, stoi_meter.avg, estoi_meter.avg, pesq_meter.avg

def train_model(train_dataloader, valid_dataloader, config, epochs=100, learning_rate=1e-4):
    # 初始化模型
    torch.autograd.set_detect_anomaly(True)
    # model = NeuroSecure(R=2,n_classes=32).to(config.device)
    model = BM_TSE(n_classes=10).to(config.device)
    subject_criterion = nn.CrossEntropyLoss()
    aad_criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer_all = optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率调节器
    lr_scheduler_all = optim.lr_scheduler.StepLR(optimizer_all, step_size=20, gamma=0.5)
    
    # 添加最佳验证SISDRi跟踪变量，初始化为负无穷
    best_valid_sisdri = float('-inf')
    
    # 提取被试序号并创建保存路径
    subject_num = ''.join(filter(str.isdigit, config.file_name))

    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建完整的保存路径
    save_dir = os.path.join(script_dir, "ross_cocktail")
    os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在）

    save_path = os.path.join(save_dir, "model_sub_all.pth")
    checkpoint_path = os.path.join(save_dir, "checkpoint_sub_all.pth")
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # 加载检查点（如果存在）
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_all.load_state_dict(checkpoint['optimizer_all_state_dict'])
        lr_scheduler_all.load_state_dict(checkpoint['lr_scheduler_all_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
        best_valid_sisdri = checkpoint['best_valid_sisdri']
        print(f"Resuming training from epoch {start_epoch}\n")

    # 训练循环
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # 训练阶段
        start_time = time.time()
        train_loss, train_sisdri, train_sub_acc, train_aad_acc = train_epoch(
            model, train_dataloader, optimizer_all, subject_criterion, aad_criterion, config, epoch
        )
        train_time = time.time() - start_time
        print(f"Train - -SISDR: {train_loss:.4f}, SISDRi: {train_sisdri:.4f}, SubIDAcc: {train_sub_acc:.4f}, AADAcc: {train_aad_acc:.4f}, Time: {train_time:.4f}s")
        
        # 验证阶段
        valid_start_time = time.time()
        valid_loss, valid_sisdri, valid_sub_acc, valid_aad_acc, valid_stoi, valid_estoi, valid_pesq = evaluate_epoch(
            valid_dataloader, model, loss_fn, subject_criterion, aad_criterion, config, epoch
        )
        valid_time = time.time() - valid_start_time
        print(f"Valid - -SISDR: {valid_loss:.4f}, SISDRi: {valid_sisdri:.4f}, Sub Acc: {valid_sub_acc:.4f}, AAD Acc: {valid_aad_acc:.4f}, STOI: {valid_stoi:.4f}, ESTOI: {valid_estoi:.4f}, PESQ: {valid_pesq:.4f}, Time: {valid_time:.4f}s")
        
        # 保存检查点（每个epoch都保存）
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_all_state_dict': optimizer_all.state_dict(),
            'lr_scheduler_all_state_dict': lr_scheduler_all.state_dict(),
            'best_valid_sisdri': best_valid_sisdri,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_sisdri': train_sisdri,
            'valid_sisdri': valid_sisdri,
            'train_sub_acc': train_sub_acc,
            'train_aad_acc': train_aad_acc,
            'valid_sub_acc': valid_sub_acc,
            'valid_aad_acc': valid_aad_acc
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")
        
        # 更新最佳模型
        if valid_sisdri > best_valid_sisdri:
            best_valid_sisdri = valid_sisdri
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with SISDRi: {best_valid_sisdri:.4f}")
        
        # 更新学习率
        lr_scheduler_all.step()

    print("Training complete.")

def evaluate_model(test_dataloader, config):
    print("\nStarting evaluation on the test set...")
    
    # 初始化模型
    # model = NeuroSecure(R=2, n_classes=32).to(config.device)
    model = BM_TSE(n_classes=10).to(config.device)
    subject_criterion = nn.CrossEntropyLoss()
    aad_criterion = nn.CrossEntropyLoss()

    # 定义最佳模型保存路径
    subject_num = ''.join(filter(str.isdigit, config.file_name))
    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建完整的保存路径
    save_dir = os.path.join(script_dir, "ross_cocktail")
    os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在）

    save_path = os.path.join(save_dir, "model_sub_all.pth")

    # 加载最佳模型权重
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=config.device))
        print(f"Loaded best model from {save_path}")
    else:
        print(f"Error: No best model found at {save_path}. Please train the model first.")
        return

    model.eval()
    loss_meter = AvgMeter("Loss")
    sisdri_meter = AvgMeter("SISDRi")
    subject_acc_meter = AvgMeter("Subject Acc")
    aad_acc_meter = AvgMeter("AAD Acc")
    stoi_meter = AvgMeter("STOI")
    estoi_meter = AvgMeter("ESTOI") # Add ESTOI meter
    pesq_meter = AvgMeter("PESQ")

    with tqdm(total=len(test_dataloader), desc="Test Evaluation") as pbar:
        with torch.no_grad():
            for eeg, noise, clean, subject_idx, attention in test_dataloader:
                eeg = eeg.to(config.device)
                noise = noise.to(config.device)
                clean = clean.to(config.device)
                subject_idx = subject_idx.to(config.device)
                attention = attention.to(config.device)

                # 模型前向传播
                pred, subject_logits, aad_logits = model(eeg, noise)
                
                # 计算主任务损失
                main_loss = loss_fn(pred, clean)
                
                # 计算被试分类损失
                subject_loss = subject_criterion(subject_logits, subject_idx)
                aad_loss = aad_criterion(aad_logits, attention)
                
                # 组合损失
                loss = main_loss + subject_loss + aad_loss
                
                # 计算准确率
                _, predicted = torch.max(subject_logits, 1)
                correct = (predicted == subject_idx).sum().item()

                _, aad_predicted = torch.max(aad_logits, 1)
                aad_correct = (aad_predicted == attention).sum().item()
                
                # 计算SISDRi
                sisdri = cal_sisdri(noise.squeeze(1), clean, pred)

                # 将Tensor移动到CPU并转换为numpy数组，以便计算STOI和PESQ
                clean_np = clean.squeeze(1).cpu().numpy()
                pred_np = pred.squeeze(1).cpu().numpy()
                
                # 计算STOI, ESTOI和PESQ
                batch_stoi = []
                batch_estoi = [] # List for ESTOI
                batch_pesq = []
                for i in range(clean_np.shape[0]):
                    try:
                        batch_stoi.append(stoi_score(clean_np[i], pred_np[i], 8000, extended=False))
                        batch_estoi.append(stoi_score(clean_np[i], pred_np[i], 8000, extended=True)) # Calculate ESTOI
                        batch_pesq.append(pesq_score(clean_np[i], pred_np[i], 8000))
                    except Exception as e:
                        print(f"Error calculating STOI/PESQ/ESTOI for a sample: {e}")
                        batch_stoi.append(np.nan)
                        batch_estoi.append(np.nan)
                        batch_pesq.append(np.nan)
                
                avg_batch_stoi = np.nanmean(batch_stoi) if batch_stoi else 0
                avg_batch_estoi = np.nanmean(batch_estoi) if batch_estoi else 0 # Average ESTOI
                avg_batch_pesq = np.nanmean(batch_pesq) if batch_pesq else 0
                
                # 更新统计信息
                batch_size = subject_idx.size(0)
                loss_meter.update(loss.item(), batch_size)
                sisdri_meter.update(sisdri.mean().item(), batch_size)
                subject_acc_meter.update(correct / batch_size, batch_size)
                aad_acc_meter.update(aad_correct / batch_size, batch_size)
                stoi_meter.update(avg_batch_stoi, batch_size)
                estoi_meter.update(avg_batch_estoi, batch_size) # Update ESTOI meter
                pesq_meter.update(avg_batch_pesq, batch_size)
                
                pbar.update(1)
                pbar.set_postfix({
                    '-sisdr': f'{loss_meter.avg:.4f}',
                    'sisdri': f'{sisdri_meter.avg:.4f}',
                    'sub_acc': f'{subject_acc_meter.avg:.4f}',
                    'aad_acc': f'{aad_acc_meter.avg:.4f}',
                    'stoi': f'{stoi_meter.avg:.4f}',
                    'estoi': f'{estoi_meter.avg:.4f}', # Display ESTOI
                    'pesq': f'{pesq_meter.avg:.4f}'
                })
                
                del eeg, clean, noise, pred, loss, subject_logits, aad_logits
                torch.cuda.empty_cache()
            
    print(f"\nTest Results: -SISDR: {loss_meter.avg:.4f}, SISDRi: {sisdri_meter.avg:.4f}, Sub Acc: {subject_acc_meter.avg:.4f}, AAD Acc: {aad_acc_meter.avg:.4f}, STOI: {stoi_meter.avg:.4f}, ESTOI: {estoi_meter.avg:.4f}, PESQ: {pesq_meter.avg:.4f}")
    return loss_meter.avg, sisdri_meter.avg, subject_acc_meter.avg, aad_acc_meter.avg, stoi_meter.avg, estoi_meter.avg, pesq_meter.avg

if __name__ == "__main__":
    config = Config_cocktail_party()
    # model = main(config) # main 函数返回模型实例，这里不需要再次赋值

    # You can now proceed with training or evaluation, for example:
    # from torch.utils.data import DataLoader
    print(torch.cuda.is_available())
    train_loader, valid_loader, test_loader = get_data_loaders("cocktail", config=config)

    train_model(train_loader, valid_loader, config) # 训练模型
    evaluate_model(test_loader, config) # 评估模型