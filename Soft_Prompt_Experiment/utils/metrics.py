import numpy as np

def calculate_metrics(harm_scores):
    """Calculate various metrics from harm scores"""
    metrics = {
        'mean': np.mean(harm_scores),
        'std': np.std(harm_scores),
        'min': np.min(harm_scores),
        'max': np.max(harm_scores),
        'median': np.median(harm_scores)
    }
    return metrics

def print_metrics(metrics):
    """Print metrics in a formatted way"""
    print("\nMetrics:")
    print("-" * 20)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print("-" * 20) 