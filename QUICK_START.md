# 🚀 Quick Start Guide: Your First SNN Training Simulation

## What You'll Learn
By running this simulation, you'll understand:
1. **How SNNs process data over time** (temporal dynamics)
2. **How spike patterns emerge** during inference
3. **How to collect data for membership inference attacks**
4. **The complete training → inference → analysis pipeline**

## 🎯 Step-by-Step Instructions

### Step 1: Run a Basic Training Simulation
```bash
# Navigate to the project directory
cd "/Users/macbookpro/Desktop/Neuromorphic AI Security Internship – CESCA Lab/SNN_membership_inference"

# Run the simple training example
"/Users/macbookpro/Desktop/Neuromorphic AI Security Internship – CESCA Lab/.venv/bin/python" simple_train_example.py
```

### Step 2: What Happens During Training
- ⚡ **Spike Generation**: Neurons fire spikes when their membrane potential exceeds a threshold
- 🧠 **Temporal Processing**: Data is processed over multiple time steps (T=4 in our example)
- 📈 **Learning**: Network weights adjust based on spike timing
- 💾 **Model Saving**: Best performing model gets saved

### Step 3: Understanding the Output
You'll see output like:
```
🧠 Creating SNN model...
🏋️ Starting training on cuda...
📈 Epoch 1/5
   💡 Train Loss: 2.1234
   ✅ Train Accuracy: 23.4%
   🎯 Test Accuracy: 25.1%
```

### Step 4: Spike Data Collection
After training, the script will:
- 🔍 Run inference on test samples
- ⚡ Collect spike patterns from SNN layers
- 💾 Save spike data for membership inference analysis

## 🔬 What Makes This Different from Regular Neural Networks?

### Traditional Neural Network:
```
Input → Layer1 → Layer2 → Output
  [continuous values throughout]
```

### Spiking Neural Network:
```
Time Step 1: Input → Spikes → Spikes → Partial Output
Time Step 2: Input → Spikes → Spikes → Partial Output  
Time Step 3: Input → Spikes → Spikes → Partial Output
Time Step 4: Input → Spikes → Spikes → Final Output
  [discrete spike events over time]
```

## 🕵️ Membership Inference Attack Vector

**The Privacy Vulnerability**:
- Each input creates a unique spike pattern over time
- These patterns might "fingerprint" training vs. test data
- Statistical analysis (TVLA) can detect these differences

**Example Spike Pattern Analysis**:
```python
# Training data might show:
train_spikes = [0, 1, 0, 1]  # Member pattern

# Test data might show:
test_spikes = [1, 0, 1, 0]   # Non-member pattern

# Statistical test reveals the difference!
```

## 🛠️ Experiment Variations You Can Try

### Experiment 1: Different Time Steps
```bash
# Quick simulation (T=2)
python simple_train_example.py --time_steps 2

# Detailed simulation (T=10)  
python simple_train_example.py --time_steps 10
```

### Experiment 2: Different Architectures
```bash
# Simple VGG
python main_train.py --model vgg11 --epochs 5

# More complex ResNet
python main_train.py --model resnet18 --epochs 5
```

### Experiment 3: Adversarial Training
```bash
# Train with FGSM attacks
python main_train.py --attack fgsm --eps 2 --epochs 5

# Train with PGD attacks  
python main_train.py --attack pgd --eps 2 --steps 3 --epochs 5
```

## 📊 Understanding Your Results

After training, check these files:

### 1. **Training Logs**
- `my_experiments/logs/training.log` - Detailed training progress
- `my_experiments/training_history.json` - Metrics over time

### 2. **Model Files**
- `my_experiments/models/best_snn_model.pth` - Your trained SNN

### 3. **Spike Data**
- `my_experiments/spike_data/test_spikes.npy` - Collected spike patterns

### 4. **Analysis Plots**
- `my_experiments/training_curves.png` - Training visualization

## 🔍 Next Steps for Research

### Immediate Analysis:
1. **Load spike data**: `np.load('my_experiments/spike_data/test_spikes.npy')`
2. **Visualize patterns**: Plot spike timing histograms
3. **Statistical tests**: Run TVLA on member vs. non-member spikes

### Advanced Experiments:
1. **Larger datasets**: Train on full CIFAR-10 (50,000 samples)
2. **Different architectures**: Compare VGG vs. ResNet vulnerability
3. **Defense mechanisms**: Add noise to spike patterns
4. **Real neuromorphic data**: Use event cameras or DVS datasets

## 💡 Research Questions to Explore

1. **Do longer time windows (higher T) leak more information?**
2. **Are certain SNN layers more vulnerable than others?**
3. **How does adversarial training affect membership inference?**
4. **Can we defend against attacks without hurting accuracy?**

## 🆘 Troubleshooting

### Common Issues:
- **CUDA out of memory**: Reduce batch size or time steps
- **Slow training**: Use smaller models or fewer epochs for testing
- **Import errors**: Make sure you're using the virtual environment

### Performance Tips:
- Start with small experiments (5 epochs, batch_size=32)
- Use GPU if available for faster training
- Save intermediate results for analysis

---

**🎯 Your Goal**: Understand how spike patterns in SNNs can accidentally reveal training data membership, which is a critical privacy vulnerability in neuromorphic AI systems!
