import torch
from tqdm import tqdm
from BMTSE.utils import AvgMeter

def evaluate(dataloader, model, loss_fn, Config, epoch):
    model.eval()
    loss_meter = AvgMeter("Loss")

    with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1} - Valid") as pbar:
        with torch.no_grad():
            for eeg, noisy, clean, subject_idx in dataloader:
                eeg, clean = eeg.to(Config.device), clean.to(Config.device)
                input_wave = noisy.to(Config.device)

                pred = model(input_wave, eeg)
                X = (input_wave, eeg, clean)
                loss = loss_fn(model, X)
                loss_meter.update(loss.item(), eeg.size(0))
                
                pbar.update(1)
                pbar.set_postfix({
                    '-loss': f'{loss_meter.avg:.4f}',
                })
                
                del eeg, clean, noisy, subject_idx, input_wave, pred, loss, X
                torch.cuda.empty_cache()
            
    return loss_meter.avg