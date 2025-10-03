import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageTextAlignment(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, image_features, text_features, labels):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        logits = image_features @ text_features.T / self.temperature
        
        batch_size = image_features.size(0)
        targets = torch.arange(batch_size, device=image_features.device)
        
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.T, targets)
        
        return (loss_i2t + loss_t2i) / 2
