import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, logits, labels):
        return self.ce_loss(logits, labels)
