import torch
import torch.nn as nn
import torch.quantization
from torch.quantization import quantize_dynamic
import os
import matplotlib.pyplot as plt
import numpy as np
from models.baseline_model import SimpleConvNet
from utils.dataset_manager import DatasetManager
from model_trainer import ModelTrainer

def apply_quantization_and_compare(model_path):
    """
    Apply different quantization methods to the model and compare their performance.
    
    Args:
        model_path (str): Path to the trained FP32 model
        
    Returns:
        tuple: (quantized_models, results)
    """
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleConvNet()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Load test dataset
    dataset_manager = DatasetManager()
    _, test_loader, _ = dataset_manager.get_cifar10_dataset(batch_size=128)
    
    # Create results directory if it doesn't exist
    os.makedirs('./results', exist_ok=True)
    
    # Dictionary to store results
    results = {
        'fp32': {'accuracy': 0, 'size': 0},
        'dynamic_int8': {'accuracy': 0, 'size': 0},
        'static_int8': {'accuracy': 0, 'size': 0}
    }
    
    # 1. Evaluate FP32 model
    print("\nEvaluating FP32 model...")
    trainer = ModelTrainer(model, None, test_loader, device)
    _, fp32_accuracy = trainer.evaluate()
    results['fp32']['accuracy'] = fp32_accuracy
    results['fp32']['size'] = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
    
    # 2. Dynamic Quantization
    print("\nApplying dynamic quantization...")
    quantized_model = quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    _, dynamic_accuracy = trainer.evaluate(quantized_model)
    results['dynamic_int8']['accuracy'] = dynamic_accuracy
    
    # Save and measure size of dynamic quantized model
    dynamic_path = './results/dynamic_quantized_model.pth'
    torch.save(quantized_model.state_dict(), dynamic_path)
    results['dynamic_int8']['size'] = os.path.getsize(dynamic_path) / (1024 * 1024)
    
    # 3. Static Quantization
    print("\nApplying static quantization...")
    # Prepare model for static quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate with test data
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            model(data)
    
    # Convert to quantized model
    static_quantized_model = torch.quantization.convert(model)
    _, static_accuracy = trainer.evaluate(static_quantized_model)
    results['static_int8']['accuracy'] = static_accuracy
    
    # Save and measure size of static quantized model
    static_path = './results/static_quantized_model.pth'
    torch.save(static_quantized_model.state_dict(), static_path)
    results['static_int8']['size'] = os.path.getsize(static_path) / (1024 * 1024)
    
    # Plot results
    plot_results(results)
    
    return [model, quantized_model, static_quantized_model], results

def plot_results(results):
    """Plot and save comparison graphs."""
    # Accuracy comparison
    plt.figure(figsize=(10, 5))
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    
    plt.bar(models, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.savefig('./results/accuracy_comparison.png')
    plt.close()
    
    # Model size comparison
    plt.figure(figsize=(10, 5))
    sizes = [results[m]['size'] for m in models]
    
    plt.bar(models, sizes)
    plt.title('Model Size Comparison')
    plt.ylabel('Size (MB)')
    plt.savefig('./results/size_comparison.png')
    plt.close()
    
    # Print results
    print("\nQuantization Results:")
    print("=" * 50)
    for model_type, metrics in results.items():
        print(f"{model_type}:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Size: {metrics['size']:.2f} MB")
        print("-" * 50) 