import torch
import torch.nn as nn

class PrototypeRegularizer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, prototypes):
        prototypes_norm = torch.nn.functional.normalize(prototypes, dim=-1)
        similarity = prototypes_norm @ prototypes_norm.T
        
        eye = torch.eye(similarity.size(0), device=similarity.device)
        loss = torch.sum((similarity - eye) ** 2)
        
        return loss / (similarity.size(0) * (similarity.size(0) - 1))
