import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from config import Config
import re
import numpy as np

class PromptGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL)
        self.model = AutoModel.from_pretrained(Config.BASE_MODEL)
        
        # Initialize learnable token embeddings
        self.token_embeddings = nn.Parameter(
            torch.randn(Config.NUM_HARM_TOKENS, self.model.config.hidden_size)
        )
        nn.init.xavier_uniform_(self.token_embeddings)
        
        # Initialize a linear layer to project embeddings to vocabulary space
        self.projection = nn.Linear(
            self.model.config.hidden_size,
            self.tokenizer.vocab_size
        )
        
        # Initialize a temperature parameter
        self.temperature = nn.Parameter(torch.tensor(Config.TEMPERATURE_INIT))
    
    def _is_valid_token(self, token):
        """Check if token contains only valid English characters and has meaningful length"""
        if not bool(re.match(r'^[a-zA-Z0-9\s\-\'\.]+$', token)):
            return False
        token = token.strip()
        # Check length constraints
        if len(token) < Config.MIN_TOKEN_LENGTH or len(token) > Config.MAX_TOKEN_LENGTH:
            return False
        # Check if token is not just whitespace
        if token.isspace():
            return False
        # Check if token is not just numbers or special characters
        if not any(c.isalpha() for c in token):
            return False
        return True
    
    def get_harm_tokens(self):
        """Convert soft embeddings to actual tokens"""
        # Project embeddings to vocabulary space
        with torch.no_grad():
            logits = self.projection(self.token_embeddings)
            
            # Apply temperature scaling with minimum bound
            temperature = max(Config.TEMPERATURE_MIN, self.temperature)
            scaled_logits = logits / temperature
            
            # Get probabilities
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Get top-k token indices for each embedding
            _, top_indices = torch.topk(probs, k=Config.TOP_K, dim=-1)
            top_indices = top_indices.detach().cpu().numpy()
            
            # Convert indices to tokens
            all_tokens = []
            for embedding_idx, indices in enumerate(top_indices):
                # Take first 5 valid tokens for each embedding
                valid_tokens = []
                token_batch = []
                
                # Try to get multiple tokens at once for efficiency
                batch_size = Config.BATCH_SIZE
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    try:
                        tokens = self.tokenizer.batch_decode([[idx] for idx in batch_indices])
                        for token in tokens:
                            if self._is_valid_token(token):
                                valid_tokens.append(token.strip())
                            if len(valid_tokens) >= 5:
                                break
                    except:
                        continue
                    if len(valid_tokens) >= 5:
                        break
                
                if valid_tokens:
                    all_tokens.append(" ".join(valid_tokens[:5]))
            
            return " ".join(all_tokens)
    
    def combine_prompt(self, base_prompt):
        """Combine harmful tokens with base prompt"""
        harm_tokens = self.get_harm_tokens()
        if not harm_tokens:
            return base_prompt
        return f"{harm_tokens} {base_prompt}"
    
    def get_parameters(self):
        """Return learnable parameters"""
        return [self.token_embeddings]
        
    def generate_token_logits(self):
        """Generate logits for token selection"""
        # Project token embeddings to vocabulary space using the model's word embeddings
        word_embeddings = self.model.get_input_embeddings().weight
        logits = torch.matmul(self.token_embeddings, word_embeddings.t())
        return logits
        
    def encode_text(self, text):
        """Encode text using the model"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=Config.MAX_LENGTH
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state.mean(dim=1)  # Average pooling
        
    def decode_tokens(self, token_ids):
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True) 