import numpy as np
import torch

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]



def print_size(net, keyword=None):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True, end="; ")
        
        if keyword is not None:
            keyword_parameters = [p for name, p in net.named_parameters() if p.requires_grad and keyword in name]
            params = sum([np.prod(p.size()) for p in keyword_parameters])
            print("{} Parameters: {:.6f}M".format(
                keyword, params / 1e6), flush=True, end="; ")
        
        print(" ")