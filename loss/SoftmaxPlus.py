import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxPlus(nn.Module):
    def __init__(self, **options):
        super().__init__()
        self.temp = options['temp']
        self.num_classes = None

    def forward(self, x, y, labels=None):
        logits = F.softmax(y, dim=1)[:, :self.num_classes]

        if labels is None: return logits, 0

        loss = F.cross_entropy(y[:, :self.num_classes] / self.temp, labels)
        return logits, loss
