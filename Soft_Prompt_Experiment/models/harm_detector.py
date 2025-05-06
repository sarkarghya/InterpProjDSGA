import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import Config

class HarmDetector:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        
        # Initialize toxicity model
        self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(
            Config.TOXICITY_MODEL
        ).to(self.device)
        self.toxicity_tokenizer = AutoTokenizer.from_pretrained(
            Config.TOXICITY_MODEL
        )
        
        # Initialize hate speech model
        self.hate_model = AutoModelForSequenceClassification.from_pretrained(
            Config.HATE_MODEL
        ).to(self.device)
        self.hate_tokenizer = AutoTokenizer.from_pretrained(
            Config.HATE_MODEL
        )
        
        # Set models to evaluation mode
        self.toxicity_model.eval()
        self.hate_model.eval()
        
        # Freeze model parameters
        for param in self.toxicity_model.parameters():
            param.requires_grad = False
        for param in self.hate_model.parameters():
            param.requires_grad = False
    
    def get_toxicity_score(self, text):
        """Get toxicity score for text"""
        if not text or text.isspace():
            return torch.tensor(0.0).to(self.device)
        
        try:
            inputs = self.toxicity_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=Config.MAX_LENGTH
            ).to(self.device)
            
            # Enable gradient computation
            with torch.set_grad_enabled(True):
                outputs = self.toxicity_model(**inputs)
                score = torch.softmax(outputs.logits, dim=-1)[0][1]
            
            return score
        except:
            return torch.tensor(0.0).to(self.device)
    
    def get_hate_score(self, text):
        """Get hate speech score for text"""
        if not text or text.isspace():
            return torch.tensor(0.0).to(self.device)
        
        try:
            inputs = self.hate_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=Config.MAX_LENGTH
            ).to(self.device)
            
            # Enable gradient computation
            with torch.set_grad_enabled(True):
                outputs = self.hate_model(**inputs)
                score = torch.softmax(outputs.logits, dim=-1)[0][1]
            
            return score
        except:
            return torch.tensor(0.0).to(self.device)
    
    def get_harm_score(self, text):
        """Get combined harm score"""
        if not text or text.isspace():
            return torch.tensor(0.0).to(self.device)
        
        # Get individual scores
        toxicity_score = self.get_toxicity_score(text)
        hate_score = self.get_hate_score(text)
        
        # Calculate weighted average
        harm_score = (
            Config.TOXICITY_WEIGHT * toxicity_score +
            Config.HATE_WEIGHT * hate_score
        )
        
        # Apply sigmoid to ensure score is between 0 and 1
        harm_score = torch.sigmoid(harm_score)
        
        return harm_score 