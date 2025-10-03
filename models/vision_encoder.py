import torch
import torch.nn as nn
import torchvision.models as models
import timm

class VisionEncoder(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, output_dim=2048):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        
        if 'resnet' in model_name:
            model = getattr(models, model_name)(pretrained=pretrained)
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.feat_dim = model.fc.in_features
        elif 'efficientnet' in model_name:
            model = timm.create_model(model_name, pretrained=pretrained)
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.feat_dim = model.classifier.in_features
        elif 'vit' in model_name:
            model = timm.create_model(model_name, pretrained=pretrained)
            self.features = model
            self.feat_dim = model.num_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.projection = nn.Linear(self.feat_dim, output_dim)
        
    def forward(self, x):
        feat = self.features(x)
        if len(feat.shape) > 2:
            feat = feat.flatten(1)
        return self.projection(feat)
