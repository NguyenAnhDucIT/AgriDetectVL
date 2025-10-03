import torch
import torch.nn as nn

class SequencePromptTransformer(nn.Module):
    def __init__(self, prompt_dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=prompt_dim,
            nhead=num_heads,
            dim_feedforward=prompt_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(prompt_dim)
        
    def forward(self, prompts):
        if len(prompts.shape) == 2:
            prompts = prompts.unsqueeze(1)
        
        output = self.transformer(prompts)
        output = self.norm(output)
        
        if output.size(1) == 1:
            output = output.squeeze(1)
        else:
            output = output.mean(dim=1)
            
        return output
