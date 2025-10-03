import torch
import torch.nn as nn

class TopKPromptSelector(nn.Module):
    def __init__(self, vision_dim, prompt_dim, num_prompts, top_k=5):
        super().__init__()
        self.top_k = top_k
        self.prompt_pool = nn.Parameter(torch.randn(num_prompts, prompt_dim))
        self.selector = nn.Linear(vision_dim, num_prompts)
        
    def forward(self, vision_features):
        B = vision_features.size(0)
        scores = self.selector(vision_features)
        
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=1)
        weights = torch.softmax(top_k_scores, dim=1)
        
        selected_prompts = self.prompt_pool[top_k_indices]
        weighted_prompts = (selected_prompts * weights.unsqueeze(-1)).sum(dim=1)
        
        return weighted_prompts
