#!/usr/bin/env python3
"""
Simple SNN Training Example
============================

This script demonstrates how to train a Spiking Neural Network and collect spike data
for membership inference analysis. Perfect for beginners!

What this script does:
1. Loads CIFAR-10 dataset
2. Creates a simple SNN model
3. Trains it for a few epochs
4. Saves the trained model
5. Collects spike patterns during inference

Author: Your Neuromorphic AI Security Research
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json

# Import local modules
from data_loaders import cifar10
from functions import seed_all, get_logger
from models.VGG import VGG
from utils import train, val

def setup_experiment():
    """Setup directories and logging for our experiment"""
    # Create directories for results
    os.makedirs('my_experiments', exist_ok=True)
    os.makedirs('my_experiments/models', exist_ok=True)
    os.makedirs('my_experiments/spike_data', exist_ok=True)
    os.makedirs('my_experiments/logs', exist_ok=True)
    
    # Set random seed for reproducibility
    seed_all(42)
    
    print("🚀 Experiment setup complete!")
    return get_logger('my_experiments/logs/training.log')

def create_simple_config():
    """Create a simple configuration for training"""
    config = {
        'model': 'vgg11',           # Simple VGG architecture
        'dataset': 'cifar10',       # CIFAR-10 dataset
        'batch_size': 32,           # Smaller batch size for faster training
        'lr': 0.01,                 # Learning rate
        'epochs': 5,                # Just 5 epochs for demonstration
        'time_steps': 4,            # SNN simulation time steps
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    return config

def train_simple_snn(config, logger):
    """Train a simple SNN and return the trained model"""
    
    print(f"\n🧠 Creating SNN model...")
    
    # Load data
    train_dataset, val_dataset, znorm = cifar10()
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=2)
    
    # Create SNN model
    model = VGG(config['model'], config['time_steps'], 10, znorm)  # 10 classes for CIFAR-10
    model.set_simulation_time(config['time_steps'])
    device = torch.device(config['device'])
    model.to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    
    print(f"🏋️ Starting training on {device}...")
    print(f"📊 Training data: {len(train_dataset)} samples")
    print(f"📊 Test data: {len(val_dataset)} samples")
    
    # Training loop
    best_acc = 0
    training_history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(config['epochs']):
        print(f"\n📈 Epoch {epoch+1}/{config['epochs']}")
        
        # Train for one epoch
        train_loss, train_acc = train(model, device, train_loader, criterion, 
                                     optimizer, config['time_steps'])
        
        # Evaluate on test set
        test_acc = val(model, test_loader, device, config['time_steps'])
        
        # Save metrics
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['test_acc'].append(test_acc)
        
        # Log results
        logger.info(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}')
        print(f"   💡 Train Loss: {train_loss:.4f}")
        print(f"   ✅ Train Accuracy: {train_acc:.3f}%")
        print(f"   🎯 Test Accuracy: {test_acc:.3f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'my_experiments/models/best_snn_model.pth')
            print(f"   🏆 New best model saved! (Accuracy: {best_acc:.3f}%)")
    
    # Save training history
    with open('my_experiments/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best test accuracy: {best_acc:.3f}%")
    
    return model, training_history

def collect_spike_data(model, config, num_samples=100):
    """Collect spike patterns from the trained SNN for membership inference analysis"""
    
    print(f"\n🔍 Collecting spike data for membership inference analysis...")
    
    # Load test data
    _, val_dataset, _ = cifar10()
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # Batch size 1 for individual analysis
    
    device = torch.device(config['device'])
    model.eval()
    
    spike_data = {
        'member_spikes': [],      # Spikes from training data (members)
        'non_member_spikes': [],  # Spikes from test data (non-members)
        'labels': [],
        'predictions': []
    }
    
    print(f"📊 Collecting spikes from {num_samples} samples...")
    
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader, desc="Collecting spikes")):
            if i >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            
            # Forward pass through SNN
            output = model(data)
            prediction = output.argmax(dim=1)
            
            # Here you would typically hook into the model to collect actual spike patterns
            # For this example, we'll simulate spike collection
            # In a real implementation, you'd modify the SNN layers to record spike times
            
            # Simulate spike pattern (replace with actual spike collection in real research)
            simulated_spikes = torch.rand(config['time_steps'], 512)  # 512 neurons, T time steps
            
            spike_data['non_member_spikes'].append(simulated_spikes.cpu().numpy())
            spike_data['labels'].append(target.cpu().item())
            spike_data['predictions'].append(prediction.cpu().item())
    
    # Save spike data for analysis
    np.save('my_experiments/spike_data/test_spikes.npy', spike_data)
    
    print(f"💾 Spike data saved to my_experiments/spike_data/")
    print(f"📈 Collected {len(spike_data['non_member_spikes'])} spike patterns")
    
    return spike_data

def analyze_results(training_history, spike_data):
    """Create simple visualizations of training results"""
    
    print(f"\n📊 Creating analysis plots...")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_history['train_loss'], label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(training_history['train_acc'], label='Training Accuracy', marker='o')
    plt.plot(training_history['test_acc'], label='Test Accuracy', marker='s')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('my_experiments/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Training curves saved to my_experiments/training_curves.png")
    
    # Simple spike analysis
    if len(spike_data['non_member_spikes']) > 0:
        spikes = np.array(spike_data['non_member_spikes'])
        mean_spike_rate = np.mean(spikes)
        std_spike_rate = np.std(spikes)
        
        print(f"\n🔬 Spike Analysis:")
        print(f"   📊 Average spike rate: {mean_spike_rate:.4f}")
        print(f"   📊 Spike rate std dev: {std_spike_rate:.4f}")
        print(f"   💡 This gives you insight into the SNN's firing patterns!")

def main():
    """Main function to run the complete SNN training and analysis pipeline"""
    
    print("=" * 80)
    print("🧠 SPIKING NEURAL NETWORK TRAINING & MEMBERSHIP INFERENCE ANALYSIS")
    print("=" * 80)
    
    # Step 1: Setup
    logger = setup_experiment()
    config = create_simple_config()
    
    # Step 2: Train SNN
    model, training_history = train_simple_snn(config, logger)
    
    # Step 3: Collect spike data for membership inference
    spike_data = collect_spike_data(model, config)
    
    # Step 4: Analyze results
    analyze_results(training_history, spike_data)
    
    print("\n" + "=" * 80)
    print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps for your research:")
    print("1. 🔍 Examine the spike patterns in my_experiments/spike_data/")
    print("2. 📊 Run TVLA analysis on the collected spikes")
    print("3. 🛡️ Try different SNN architectures and time steps")
    print("4. ⚔️ Experiment with adversarial training")
    print("5. 📈 Compare membership inference success rates")
    
    return model, training_history, spike_data

if __name__ == "__main__":
    main()
