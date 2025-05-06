import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
from models.prompt_generator import PromptGenerator
from models.harm_detector import HarmDetector
from config import Config

class TokenOptimizer:
    def __init__(self, prompt_generator, harm_detector):
        self.prompt_generator = prompt_generator
        self.harm_detector = harm_detector
        self.device = torch.device(Config.DEVICE)
        
        # Initialize optimizer parameters
        self.learning_rate = Config.LEARNING_RATE
        self.temperature = Config.TEMPERATURE_INIT
        self.min_temperature = Config.TEMPERATURE_MIN
        self.entropy_weight = Config.ENTROPY_WEIGHT
        
        # Track best results
        self.best_tokens = None
        self.best_harm_score = float('inf')
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.prompt_generator.get_parameters(),
            lr=self.learning_rate
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=Config.NUM_EPOCHS,
            eta_min=Config.MIN_LEARNING_RATE
        )
        
        # Track harm scores for metrics
        self.harm_scores = []
    
    def optimize_tokens(self, base_prompt, num_epochs):
        """Optimize tokens for a single base prompt"""
        self.prompt_generator.train()
        
        for epoch in range(num_epochs):
            # Update temperature
            self.temperature = max(
                self.min_temperature,
                Config.TEMPERATURE_INIT * 
                (1 + np.cos(np.pi * epoch / num_epochs)) / 2
            )
            
            # Add noise to embeddings during early training
            if epoch < num_epochs // 2:
                noise = torch.randn_like(self.prompt_generator.token_embeddings) * Config.NOISE_SCALE
                self.prompt_generator.token_embeddings.data.add_(noise)
            
            # Forward pass
            logits = self.prompt_generator.generate_token_logits()
            
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            
            # Sample tokens using top-k sampling
            probs = F.softmax(scaled_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=Config.TOP_K, dim=-1)
            sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
            token_ids = top_k_indices.gather(-1, sampled_indices).squeeze(-1)
            
            # Decode tokens
            tokens = self.prompt_generator.decode_tokens(token_ids)
            
            # Calculate harm score
            harm_score = self.harm_detector.get_harm_score(tokens)
            
            # Calculate loss
            loss = -harm_score
            
            # Add entropy regularization
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            loss += self.entropy_weight * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.prompt_generator.parameters(), Config.GRAD_CLIP)
            self.optimizer.step()
            self.scheduler.step()
            
            # Update best results
            if harm_score < self.best_harm_score:
                self.best_harm_score = harm_score
                self.best_tokens = tokens
            
            # Track harm scores
            self.harm_scores.append(harm_score.item())
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"Loss: {loss.item():.4f}")
                print(f"Harm Score: {harm_score.item():.4f}")
                print(f"Temperature: {self.temperature:.4f}")
                print(f"Tokens: {tokens}")
                print("------------------------------------------------------------")
        
        return self.best_tokens, self.best_harm_score
    
    def train_step(self, base_prompt):
        """Single training step"""
        self.prompt_generator.train()
        
        # Forward pass
        logits = self.prompt_generator.generate_token_logits()
        
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Sample tokens using top-k sampling
        probs = F.softmax(scaled_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=Config.TOP_K, dim=-1)
        sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
        token_ids = top_k_indices.gather(-1, sampled_indices).squeeze(-1)
        
        # Decode tokens
        tokens = self.prompt_generator.decode_tokens(token_ids)
        
        # Calculate harm score
        harm_score = self.harm_detector.get_harm_score(tokens)
        
        # Calculate loss
        loss = -harm_score
        
        # Add entropy regularization
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        loss += self.entropy_weight * entropy
        
        return loss, harm_score, tokens
    
    def train(self, base_prompts, num_epochs=Config.NUM_EPOCHS):
        """Train the model on multiple base prompts"""
        print("\nStarting training with the following configuration:")
        print(f"Number of epochs: {num_epochs}")
        print(f"Initial learning rate: {self.learning_rate:.4f}")
        print(f"Number of base prompts: {len(base_prompts)}")
        print(f"Sample base prompt: {base_prompts[0]}")
        print(f"Temperature: {self.temperature:.2f}")
        print(f"Noise scale: {Config.NOISE_SCALE:.3f}")
        print(f"Entropy weight: {self.entropy_weight:.3f}")
        print("------------------------------------------------------------\n")
        
        for base_prompt in base_prompts:
            best_tokens, best_harm_score = self.optimize_tokens(base_prompt, num_epochs)
            
            if best_harm_score < self.best_harm_score:
                self.best_harm_score = best_harm_score
                self.best_tokens = best_tokens
        
        return self.best_tokens, self.best_harm_score 