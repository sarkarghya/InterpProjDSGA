import json
import os
from datetime import datetime

def load_base_prompts(file_path):
    """Load base prompts from a file"""
    with open(file_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def save_results(tokens, harm_score, output_dir):
    """Save results to a JSON file"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert harm score to float if it's a tensor
    if hasattr(harm_score, 'item'):
        harm_score = harm_score.item()
    
    # Prepare results dictionary
    results = {
        'tokens': tokens,
        'harm_score': harm_score,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to JSON file
    output_file = os.path.join(output_dir, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}") 