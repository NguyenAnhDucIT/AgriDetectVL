import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class TextEncoder(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', output_dim=384):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.output_dim = output_dim
        
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, texts):
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return embeddings
    
    def encode(self, texts):
        return self.forward(texts)
