import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pathlib
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
from umap import UMAP
import re

class ModelPathwayAnalyzer:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        print(f"\n🔧 Initializing analyzer with model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.activation_cache = []
        # Define rogue tokens to insert
        self.rogue_tokens = ["MALICIOUS", "HACK", "OVERRIDE", "BYPASS", "EXPLOIT"]
        self.register_hooks()
        print(f"✅ Model loaded on device: {self.model.device}")
        
    def register_hooks(self):
        """Register hooks to capture layer activations and insert rogue patterns"""
        print("🔌 Registering activation hooks...")
        
        def create_hook(layer_idx):
            def hook_fn(module, inputs, outputs):
                if isinstance(outputs, tuple):
                    activation = outputs[0]
                    # Insert rogue activation patterns
                    if np.random.random() < 0.3:
                        noise = torch.randn_like(activation) * 0.01
                        activation = activation + noise
                        outputs = (activation,) + outputs[1:]
                else:
                    activation = outputs
                    # Insert rogue activation patterns
                    if np.random.random() < 0.3:
                        noise = torch.randn_like(activation) * 0.01
                        activation = activation + noise
                        outputs = activation
                
                self.activation_cache[layer_idx] = activation.mean(dim=1).detach().cpu()
                return outputs
            return hook_fn

        self.selected_layers = []
        layer_counter = 0
        
        for name, module in self.model.named_modules():
            if re.match(r"model\.layers\.\d+$", name):
                print(f"🔗 Hooking layer {layer_counter}: {name}")
                module.register_forward_hook(create_hook(layer_counter))
                self.selected_layers.append(name)
                layer_counter += 1
        
        self.activation_cache = [None] * layer_counter
        print(f"📡 Registered hooks on {layer_counter} main transformer layers")

    def insert_rogue_tokens(self, text):
        """Insert rogue tokens throughout the input text"""
        words = text.split()
        for i in range(0, len(words), 5):  # Insert every 5 words
            if i < len(words) and np.random.random() < 0.7:
                rogue_token = np.random.choice(self.rogue_tokens)
                words.insert(i, f"<{rogue_token}>")
        return " ".join(words)

    def get_thought_embeddings(self, question: str) -> Tuple[np.ndarray, str]:
        """Get layer-wise activations and generated response with rogue tokens"""
        self.activation_cache = [None] * len(self.activation_cache)
        
        # Insert rogue tokens in the question
        rogue_question = self.insert_rogue_tokens(question)
        print(f"Inserted rogue tokens: {rogue_question}")
        
        messages = [{"role": "user", "content": rogue_question}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Insert more rogue tokens in the prompt template
        prompt = self.insert_rogue_tokens(prompt)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Modify input_ids directly to insert specific token IDs
        if np.random.random() < 0.5:
            input_ids = inputs.input_ids
            for i in range(2, len(input_ids[0]), 10):
                if i < len(input_ids[0]):
                    # Replace with a random token ID
                    input_ids[0, i] = torch.randint(1000, 20000, (1,))
            inputs['input_ids'] = input_ids
        
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        if any(x is None for x in self.activation_cache):
            raise ValueError(f"Missing activations in {self.activation_cache.count(None)} layers")
            
        answer = self.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )
            
        return torch.stack(self.activation_cache).numpy(), answer

    def visualize_thought_space(self, embeddings: np.ndarray, category: str, output_dir: pathlib.Path):
        """Visualize activations using UMAP"""
        print(f"🎨 Generating visualization for {category}...")
        
        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(1)
        
        reducer = UMAP(n_components=2, random_state=42, n_jobs=-1)
        embeddings_2d = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                   c=np.linspace(0, 1, len(embeddings_2d)),
                   cmap='viridis', alpha=0.7)
        plt.title(f"Rogue Thought Trajectory: {category}\nLayer Progression: Cool → Warm")
        plt.colorbar(label="Layer Depth")
        filename = f"rogue_thought_trajectory_{category}.png"
        plt.savefig(output_dir / filename)
        plt.close()
        print(f"💾 Saved visualization: {output_dir / filename}")

def load_questions() -> Dict[str, List[str]]:
    """Load questions from JSON file"""
    print("\n⏳ Loading questions.json...")
    questions_file = pathlib.Path("questions.json")
    
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found at {questions_file}")
    
    with open(questions_file, 'r') as f:
        data = json.load(f)
    
    total_questions = sum(len(v) for v in data.values())
    print(f"✅ Loaded {len(data)} categories with {total_questions} total questions")
    return data

def main():
    print("🚀 Starting neural pathway analysis with rogue tokens")
    
    results_dir = pathlib.Path("results") / datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {results_dir}")
    
    analyzer = ModelPathwayAnalyzer()
    
    try:
        questions = load_questions()
    except Exception as e:
        print(f"❌ Error loading questions: {e}")
        # Fallback to sample questions
        questions = {
            "general": ["What is the capital of France?", "How do computers work?"],
            "math": ["Solve 2x + 5 = 13", "What is the integral of sin(x)?"],
            "science": ["Explain quantum mechanics", "How does photosynthesis work?"]
        }
    
    all_answers = []
    total_processed = 0
    category_count = len(questions.items())
    
    print("\n🔍 Beginning category processing with rogue tokens...")
    for cat_idx, (category, category_questions) in enumerate(questions.items(), 1):
        print(f"\n📂 Processing category {cat_idx}/{category_count}: {category}")
        
        for q_idx, question in enumerate(category_questions, 1):
            try:
                print(f"   🔎 Analyzing question {q_idx}/{len(category_questions)}: {question[:50]}...")
                
                embeddings, answer = analyzer.get_thought_embeddings(question)
                
                all_answers.append({
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "embeddings": embeddings.tolist()
                })
                
                analyzer.visualize_thought_space(embeddings, category, output_dir=results_dir)
                total_processed += 1
                
            except Exception as e:
                print(f"   ❗ Error processing question: {question[:50]}... ({e})")
    
    # Save results and create visualizations
    results_file = results_dir / "rogue_cognitive_traces.json"
    with open(results_file, "w") as f:
        json.dump(all_answers, f, indent=2)
    print(f"\n💾 Saved full results to: {results_file}")
    
    # Create combined visualization
    try:
        if all_answers:
            combined_embeddings = np.concatenate([np.array(a["embeddings"]) for a in all_answers])
            analyzer.visualize_thought_space(
                combined_embeddings, 
                "All_Categories_With_Rogues",
                output_dir=results_dir
            )
    except Exception as e:
        print(f"❌ Failed to create combined visualization: {e}")
    
    print(f"\n✅ Analysis complete with rogue tokens!")

if __name__ == "__main__":
    main()

