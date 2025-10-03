import torch
import torch.nn as nn
from .vision_encoder import VisionEncoder
from .text_encoder import TextEncoder
from .prototypes import PrototypeBuilder
from .tps import TopKPromptSelector
from .spt import SequencePromptTransformer

class AgriDetectVL(nn.Module):
    def __init__(self, config, class_names, prompt_templates):
        super().__init__()
        self.config = config
        self.class_names = class_names
        
        self.vision_encoder = VisionEncoder(
            model_name=config['model']['vision_encoder'],
            pretrained=True,
            output_dim=config['model']['vision_dim']
        )
        
        self.text_encoder = TextEncoder(
            model_name=config['model']['text_encoder'],
            output_dim=config['model']['text_dim']
        )
        
        self.prototype_builder = PrototypeBuilder(
            text_encoder=self.text_encoder,
            class_names=class_names,
            prompt_templates=prompt_templates,
            text_dim=config['model']['text_dim'],
            output_dim=config['model']['prompt_dim']
        )
        
        self.tps = TopKPromptSelector(
            vision_dim=config['model']['vision_dim'],
            prompt_dim=config['model']['prompt_dim'],
            num_prompts=config['model']['num_prompts'],
            top_k=5
        )
        
        self.spt = SequencePromptTransformer(
            prompt_dim=config['model']['prompt_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers']
        )
        
        self.fusion_head = nn.Sequential(
            nn.Linear(config['model']['vision_dim'] + config['model']['prompt_dim'], 
                     config['model']['prompt_dim']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config['model']['prompt_dim'], config['model']['prompt_dim'])
        )
        
        self.temperature = config['model']['temperature']
        
    def forward(self, images):
        vision_features = self.vision_encoder(images)
        vision_features_norm = nn.functional.normalize(vision_features, dim=-1)
        
        selected_prompts = self.tps(vision_features)
        refined_prompts = self.spt(selected_prompts)
        
        fused_features = torch.cat([vision_features, refined_prompts], dim=-1)
        final_features = self.fusion_head(fused_features)
        final_features_norm = nn.functional.normalize(final_features, dim=-1)
        
        prototypes = self.prototype_builder.get_prototypes()
        prototypes = prototypes.to(final_features_norm.device)
        
        logits = final_features_norm @ prototypes.T / self.temperature
        
        return logits, final_features_norm, vision_features_norm
    
    def get_prototypes(self):
        return self.prototype_builder.get_prototypes()
