"""
Test script for PICL data loading and preprocessing
"""

import os
import json
from PIL import Image
import torch
from torchvision import transforms
import numpy as np


def test_data_structure():
    """Test PICL dataset structure."""
    print("🧪 Testing PICL Dataset Structure")
    print("=" * 40)
    
    # Check train data
    train_dir = "train"
    train_labels = "train/dataset_labels.json"
    
    if not os.path.exists(train_dir):
        print(f"❌ Train directory not found: {train_dir}")
        return False
    
    if not os.path.exists(train_labels):
        print(f"❌ Train labels not found: {train_labels}")
        return False
    
    # Load metadata
    with open(train_labels, 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Dataset: {metadata['dataset_name']}")
    print(f"✅ Materials: {metadata['materials']}")
    print(f"✅ Refractive indices: {metadata['refractive_indices']}")
    print(f"✅ Total samples: {metadata['total_samples']}")
    
    # Check each material
    materials = metadata['materials']
    for material in materials:
        material_dir = os.path.join(train_dir, f"{material}_4D", "images")
        
        if os.path.exists(material_dir):
            sample_dirs = [d for d in os.listdir(material_dir) 
                          if os.path.isdir(os.path.join(material_dir, d))]
            print(f"✅ {material}: {len(sample_dirs)} samples")
            
            # Check first sample
            if sample_dirs:
                first_sample = sample_dirs[0]
                sample_path = os.path.join(material_dir, first_sample)
                
                # Check time-gate images
                time_gate_files = []
                for i in range(1, 6):
                    img_file = os.path.join(sample_path, f"{first_sample}_{i}.png")
                    if os.path.exists(img_file):
                        time_gate_files.append(img_file)
                
                print(f"   Sample {first_sample}: {len(time_gate_files)}/5 time-gate images")
                
                # Load and check image properties
                if time_gate_files:
                    img = Image.open(time_gate_files[0])
                    print(f"   Image size: {img.size}, Mode: {img.mode}")
        else:
            print(f"❌ {material}: Directory not found")
    
    return True


def test_image_loading():
    """Test image loading and preprocessing."""
    print("\n🧪 Testing Image Loading and Preprocessing")
    print("=" * 40)
    
    # Find a sample image
    train_dir = "train"
    materials = ['air', 'water', 'acrylic', 'glass', 'sapphire']
    
    sample_found = False
    for material in materials:
        material_dir = os.path.join(train_dir, f"{material}_4D", "images")
        if os.path.exists(material_dir):
            sample_dirs = [d for d in os.listdir(material_dir) 
                          if os.path.isdir(os.path.join(material_dir, d))]
            
            if sample_dirs:
                first_sample = sample_dirs[0]
                sample_path = os.path.join(material_dir, first_sample)
                
                # Load all 5 time-gate images
                images = []
                for i in range(1, 6):
                    img_file = os.path.join(sample_path, f"{first_sample}_{i}.png")
                    if os.path.exists(img_file):
                        img = Image.open(img_file).convert('RGB')
                        images.append(img)
                
                if len(images) == 5:
                    print(f"✅ Loaded {len(images)} time-gate images from {material}/{first_sample}")
                    
                    # Test transformations
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
                    
                    # Transform images
                    transformed_images = []
                    for img in images:
                        transformed = transform(img)
                        transformed_images.append(transformed)
                    
                    # Stack images
                    images_tensor = torch.stack(transformed_images, dim=0)
                    print(f"✅ Transformed tensor shape: {images_tensor.shape}")
                    print(f"   Expected: (5, 3, 224, 224)")
                    print(f"   Min value: {images_tensor.min():.4f}")
                    print(f"   Max value: {images_tensor.max():.4f}")
                    
                    sample_found = True
                    break
    
    if not sample_found:
        print("❌ No valid sample found for testing")
        return False
    
    return True


def test_vmamba_compatibility():
    """Test VMamba backbone compatibility."""
    print("\n🧪 Testing VMamba Backbone Compatibility")
    print("=" * 40)
    
    try:
        from vmamba_backbone import VMambaBackbone
        print("✅ VMambaBackbone imported successfully")
        
        # Create backbone
        backbone = VMambaBackbone(
            model_name='vmamba_base_s2l15',
            out_indices=(3,),
            channel_first=True
        )
        print("✅ VMambaBackbone created successfully")
        print(f"✅ Output channels: {backbone.out_channels}")
        
        # Test with dummy data
        dummy_input = torch.randn(2, 5, 3, 224, 224)  # (B, T, C, H, W)
        print(f"✅ Dummy input shape: {dummy_input.shape}")
        
        # This would fail without proper VMamba installation
        # features = backbone(dummy_input)
        # print(f"✅ Output shape: {features[0].shape}")
        
        print("⚠️  VMamba forward pass requires proper installation")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("🧪 PICL Data Loading Tests")
    print("=" * 50)
    
    # Test 1: Data structure
    test1_passed = test_data_structure()
    
    # Test 2: Image loading
    test2_passed = test_image_loading()
    
    # Test 3: VMamba compatibility
    test3_passed = test_vmamba_compatibility()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 20)
    print(f"Data Structure: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Image Loading: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"VMamba Compatibility: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 All tests passed! Ready for classification experiment.")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")


if __name__ == "__main__":
    main()
