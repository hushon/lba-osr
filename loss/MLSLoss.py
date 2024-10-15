import torch
import torch.nn as nn
import torch.nn.functional as F

class MLSLoss(nn.Module):
    def __init__(self, **options):
        super().__init__()
        self.temp = options['temp']

    def forward(self, x, y, labels=None):
        logits = y
        if labels is None: return logits, 0
        loss = F.cross_entropy(y / self.temp, labels)
        return logits, loss
