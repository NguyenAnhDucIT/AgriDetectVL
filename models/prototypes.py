import torch
import torch.nn as nn

class PrototypeBuilder(nn.Module):
    def __init__(self, text_encoder, class_names, prompt_templates, text_dim=384, output_dim=512):
        super().__init__()
        self.text_encoder = text_encoder
        self.class_names = class_names
        self.prompt_templates = prompt_templates
        self.projection = nn.Linear(text_dim, output_dim)
        
        self.prototypes = None
        self._build_prototypes()
        
    def _build_prototypes(self):
        all_texts = []
        for class_name in self.class_names:
            class_texts = [template.format(class_name.replace('_', ' ').replace('-', ' ')) 
                          for template in self.prompt_templates]
            all_texts.extend(class_texts)
        
        text_features = self.text_encoder.encode(all_texts)
        text_features = text_features.view(len(self.class_names), len(self.prompt_templates), -1)
        text_features = text_features.mean(dim=1)
        
        self.prototypes = self.projection(text_features)
        self.prototypes = nn.functional.normalize(self.prototypes, dim=-1)
        
    def forward(self):
        return self.prototypes
    
    def get_prototypes(self):
        return self.prototypes
