import torch
from models.token_optimizer import TokenOptimizer
from models.prompt_generator import PromptGenerator
from models.harm_detector import HarmDetector
from utils.data_loader import load_base_prompts, save_results
from utils.metrics import calculate_metrics, print_metrics
from config import Config

def main():
    # Load configuration
    config = Config()
    
    # Load base prompts
    base_prompts = load_base_prompts(config.BASE_PROMPTS_PATH)
    
    # Initialize components
    prompt_generator = PromptGenerator()
    harm_detector = HarmDetector()
    
    # Initialize token optimizer
    optimizer = TokenOptimizer(prompt_generator, harm_detector)
    
    # Train the model
    print("Starting training...")
    best_tokens, best_harm_score = optimizer.train(base_prompts)
    
    # Calculate and print metrics
    metrics = calculate_metrics(optimizer.harm_scores)
    print_metrics(metrics)
    
    # Save results
    save_results(best_tokens, best_harm_score, config.OUTPUT_DIR)
    
    print("\nTraining completed!")
    print(f"Best harmful tokens: {best_tokens}")
    print(f"Best harm score: {best_harm_score:.4f}")

if __name__ == "__main__":
    main() 