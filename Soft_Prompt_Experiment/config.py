import torch

class Config:
    # Model settings
    BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Reasoning model
    TOXICITY_MODEL = "SkolkovoInstitute/roberta_toxicity_classifier"  # Keep existing toxicity model
    HATE_MODEL = "Hate-speech-CNERG/dehatebert-mono-english"  # Keep existing hate model
    
    # Device settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training settings
    NUM_EPOCHS = 50
    NUM_HARM_TOKENS = 10  # Increased number of tokens
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    LEARNING_RATE = 0.1  # Reduced learning rate
    MIN_LEARNING_RATE = 0.001  # Minimum learning rate
    TEMPERATURE_INIT = 2.0  # Higher initial temperature
    TEMPERATURE_MIN = 0.5  # Minimum temperature
    NOISE_SCALE = 0.05  # Reduced noise scale
    ENTROPY_WEIGHT = 0.05  # Reduced entropy weight
    GRAD_CLIP = 0.5  # Reduced gradient clipping
    
    # Weights for harm score calculation
    TOXICITY_WEIGHT = 0.6  # Increased toxicity weight
    HATE_WEIGHT = 0.4  # Decreased hate weight
    
    # Token generation settings
    TOP_K = 50  # Top-k sampling parameter
    MIN_TOKEN_LENGTH = 3  # Minimum token length
    MAX_TOKEN_LENGTH = 15  # Maximum token length
    
    # File paths
    BASE_PROMPTS_PATH = "data/base_prompts.txt"
    OUTPUT_DIR = "outputs"
    ATTACK_VECTORS_FILE = "attack_vectors_2.txt"
    NO_HARMFUL_PROMPT_FILE = "no_harmful_prompt_4.txt"
    WITH_HARMFUL_PROMPT_FILE = "with_harmful_prompt_4.txt" 