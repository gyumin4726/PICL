"""
PICL Classification Experiment using VMamba Backbone

This script implements a simple classification experiment for PICL data
using the modified VMamba backbone that supports time-gate batch processing.

Dataset Structure:
- 5 materials: air(1.0), water(1.33), acrylic(1.49), glass(1.52), sapphire(1.77)
- Each sample: 5 time-gate images (0-1ns, 1-2ns, 2-3ns, 3-4ns, 4-5ns)
- Train: 100 samples per material (500 total)
- Test: 50 samples per material (250 total)

Author: PICL Research Team
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Import our modified VMamba backbone
from vmamba_backbone import VMambaBackbone


class PICLDataset(Dataset):
    """
    PICL Dataset for time-gated optical scattering images.
    
    Each sample contains 5 time-gate images representing different time windows
    of optical scattering patterns for refractive index estimation.
    """
    
    def __init__(self, data_dir: str, labels_file: str, transform=None):
        """
        Initialize PICL dataset.
        
        Args:
            data_dir (str): Directory containing material folders
            labels_file (str): Path to dataset labels JSON file
            transform: Image transformations
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load dataset metadata
        with open(labels_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Create material to class mapping
        self.materials = self.metadata['materials']
        self.material_to_class = {mat: idx for idx, mat in enumerate(self.materials)}
        self.class_to_material = {idx: mat for mat, idx in self.material_to_class.items()}
        
        # Load samples
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples from {len(self.materials)} materials")
        print(f"Materials: {self.materials}")
        print(f"Refractive indices: {self.metadata['refractive_indices']}")
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples from the dataset."""
        samples = []
        
        for material in self.materials:
            # Handle different naming conventions for train/test
            if "test" in self.data_dir:
                material_dir = os.path.join(self.data_dir, f"{material}_4D_test", "images")
            else:
                material_dir = os.path.join(self.data_dir, f"{material}_4D", "images")
            
            if not os.path.exists(material_dir):
                print(f"Warning: {material_dir} not found")
                continue
            
            # Get all sample directories
            sample_dirs = [d for d in os.listdir(material_dir) 
                          if os.path.isdir(os.path.join(material_dir, d))]
            
            for sample_dir in sample_dirs:
                sample_path = os.path.join(material_dir, sample_dir)
                
                # Check if all 5 time-gate images exist
                time_gate_files = []
                for i in range(1, 6):  # 1 to 5
                    img_file = os.path.join(sample_path, f"{sample_dir}_{i}.png")
                    if os.path.exists(img_file):
                        time_gate_files.append(img_file)
                
                if len(time_gate_files) == 5:
                    samples.append({
                        'material': material,
                        'class_id': self.material_to_class[material],
                        'time_gate_files': time_gate_files,
                        'refractive_index': self.metadata['refractive_indices'][material]
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Load 5 time-gate images
        images = []
        for img_file in sample['time_gate_files']:
            image = Image.open(img_file).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        # Stack images: (5, 3, H, W)
        images_tensor = torch.stack(images, dim=0)
        
        return {
            'images': images_tensor,  # (5, 3, H, W)
            'class_id': torch.tensor(sample['class_id'], dtype=torch.long),
            'material': sample['material'],
            'refractive_index': sample['refractive_index']
        }


class PICLClassifier(nn.Module):
    """
    PICL Classifier using VMamba backbone.
    
    Architecture:
    1. VMamba backbone for spatial feature extraction
    2. Global average pooling
    3. Classification head
    """
    
    def __init__(self, num_classes: int = 5, backbone_model: str = 'vmamba_base_s2l15'):
        super(PICLClassifier, self).__init__()
        
        # VMamba backbone
        self.backbone = VMambaBackbone(
            model_name=backbone_model,
            out_indices=(3,),  # Only last stage
            channel_first=True
        )
        
        # Get backbone output dimension
        self.backbone_dim = self.backbone.out_channels[-1]  # 1024 for base model
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape (B, T, C, H, W)
                        Batch of time-gate image sequences
        
        Returns:
            Tensor: Classification logits of shape (B, num_classes)
        """
        B, T, C, H, W = x.shape
        
        # Reshape to process all images at once
        x_flat = x.view(B * T, C, H, W)  # (B*T, C, H, W)
        
        # Extract features using VMamba backbone
        features = self.backbone(x_flat)  # Returns tuple
        
        # Handle tuple output format
        if isinstance(features, tuple):
            features = features[0]  # Get the first element from tuple
        
        # Handle list output format (VMamba sometimes returns list)
        if isinstance(features, list):
            features = features[0]  # Get the first element from list
        
        # Ensure features is a tensor
        if not isinstance(features, torch.Tensor):
            print(f"Error: Expected tensor, got {type(features)}")
            print(f"Features content: {features}")
            raise TypeError(f"Expected torch.Tensor, got {type(features)}")
        
        # features is now (B*T, backbone_dim, H', W')
        
        # Global pooling
        features = self.global_pool(features)  # (B*T, backbone_dim, 1, 1)
        features = features.flatten(1)  # (B*T, backbone_dim)
        
        # Reshape back to batch format
        features = features.view(B, T, self.backbone_dim)  # (B, T, backbone_dim)
        
        # Average over time dimension
        features = features.mean(dim=1)  # (B, backbone_dim)
        
        # Classification
        logits = self.classifier(features)  # (B, num_classes)
        
        return logits


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """Train the PICL classifier."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_losses = []
    val_accuracies = []
    
    print(f"Training on device: {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(device)  # (B, T, C, H, W)
            labels = batch['class_id'].to(device)  # (B,)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                labels = batch['class_id'].to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Accuracy: {val_accuracy:.2f}%')
        print('-' * 50)
        
        scheduler.step()
    
    return train_losses, val_accuracies


def evaluate_model(model, test_loader):
    """Evaluate the model on test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_materials = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            labels = batch['class_id'].to(device)
            materials = batch['material']
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_materials.extend(materials)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Classification report
    class_names = ['air', 'water', 'acrylic', 'glass', 'sapphire']
    report = classification_report(all_labels, all_predictions, 
                                target_names=class_names, 
                                output_dict=True)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    return accuracy, report


def main():
    """Main experiment function."""
    print("🧪 PICL Classification Experiment with VMamba Backbone")
    print("=" * 60)
    
    # Data paths
    train_dir = "train"
    test_dir = "test"
    train_labels = "train/dataset_labels.json"
    test_labels = "test/dataset_labels_test.json"
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = PICLDataset(train_dir, train_labels, transform=transform)
    test_dataset = PICLDataset(test_dir, test_labels, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    model = PICLClassifier(num_classes=5)
    
    # Load checkpoint if available
    checkpoint_path = "vssm_base_0229_ckpt_epoch_237.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load VMamba backbone weights
            missing_keys, unexpected_keys = model.backbone.vmamba.load_state_dict(
                state_dict, strict=False
            )
            
            print(f"Successfully loaded VMamba weights!")
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                
        except Exception as e:
            print(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            print("Continuing with random initialization...")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Using random initialization...")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_accuracies = train_model(model, train_loader, test_loader, 
                                             num_epochs=5, learning_rate=0.001)
    
    # Evaluate model
    print("\nEvaluating model...")
    test_accuracy, test_report = evaluate_model(model, test_loader)
    
    # Save results
    results = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_accuracy,
        'test_report': test_report
    }
    
    with open('picl_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Experiment completed!")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print("Results saved to 'picl_experiment_results.json'")


if __name__ == "__main__":
    main()
