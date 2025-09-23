"""
Debug VMamba backbone output
"""

import torch
from vmamba_backbone import VMambaBackbone

def test_vmamba_output():
    """Test VMamba backbone output format."""
    print("🧪 Testing VMamba Backbone Output")
    print("=" * 40)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create backbone
    backbone = VMambaBackbone(
        model_name='vmamba_base_s2l15',
        out_indices=(3,),  # Only last stage
        channel_first=True
    )
    
    # Move to device
    backbone = backbone.to(device)
    
    print(f"✅ Backbone created successfully")
    print(f"✅ Output channels: {backbone.out_channels}")
    
    # Test with single image
    print("\n📋 Testing single image...")
    single_input = torch.randn(2, 3, 224, 224).to(device)  # (B, C, H, W)
    print(f"Input shape: {single_input.shape}")
    
    try:
        single_output = backbone(single_input)
        print(f"Output type: {type(single_output)}")
        print(f"Output length: {len(single_output) if isinstance(single_output, (tuple, list)) else 'Not iterable'}")
        
        if isinstance(single_output, (tuple, list)):
            print(f"First element type: {type(single_output[0])}")
            print(f"First element shape: {single_output[0].shape if hasattr(single_output[0], 'shape') else 'No shape'}")
        
    except Exception as e:
        print(f"❌ Error with single image: {e}")
        return False
    
    # Test with time-gate batch
    print("\n📋 Testing time-gate batch...")
    batch_input = torch.randn(2, 5, 3, 224, 224).to(device)  # (B, T, C, H, W)
    print(f"Input shape: {batch_input.shape}")
    
    try:
        batch_output = backbone(batch_input)
        print(f"Output type: {type(batch_output)}")
        print(f"Output length: {len(batch_output) if isinstance(batch_output, (tuple, list)) else 'Not iterable'}")
        
        if isinstance(batch_output, (tuple, list)):
            print(f"First element type: {type(batch_output[0])}")
            print(f"First element shape: {batch_output[0].shape if hasattr(batch_output[0], 'shape') else 'No shape'}")
        
    except Exception as e:
        print(f"❌ Error with time-gate batch: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    test_vmamba_output()
