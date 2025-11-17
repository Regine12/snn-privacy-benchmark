"""
Minimal SNN Training Test
========================

This script tests if we can run basic SNN training without complex dependencies.
"""

import sys
import os

def test_imports():
    """Test if required packages are available"""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        print(f"🔥 CUDA available: {torch.cuda.is_available()}")
        
        import torchvision
        print(f"✅ TorchVision {torchvision.__version__} imported successfully")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test if we can load CIFAR-10 data"""
    try:
        from data_loaders import cifar10
        train_dataset, val_dataset, znorm = cifar10()
        print(f"✅ CIFAR-10 loaded: {len(train_dataset)} train, {len(val_dataset)} test samples")
        print(f"📊 Normalization: mean={znorm[0]}, std={znorm[1]}")
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_model_creation():
    """Test if we can create an SNN model"""
    try:
        from models.VGG import VGG
        model = VGG('vgg11', time_steps=4, num_labels=10, norm=((0.5,), (0.5,)))
        print(f"✅ VGG SNN model created successfully")
        
        # Test forward pass with dummy data
        import torch
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Forward pass successful, output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing SNN Training Environment")
    print("=" * 50)
    
    # Test 1: Package imports
    print("\n1️⃣ Testing package imports...")
    if not test_imports():
        print("❌ Package import failed. Please check your environment.")
        return
    
    # Test 2: Data loading
    print("\n2️⃣ Testing data loading...")
    if not test_data_loading():
        print("❌ Data loading failed. Please check data_loaders.py")
        return
    
    # Test 3: Model creation
    print("\n3️⃣ Testing SNN model creation...")
    if not test_model_creation():
        print("❌ Model creation failed. Please check models/")
        return
    
    print("\n🎉 All tests passed! Your environment is ready for SNN training.")
    print("\n📝 Next steps:")
    print("1. Run 'python main_train.py --epochs 2' for a quick training test")
    print("2. Check 'cifar10-checkpoints/' for saved models")
    print("3. Explore spike collection with the TVLA notebook")

if __name__ == "__main__":
    main()
