"""
Setup script to prepare MIA attack models
This script helps copy or train attack models for MIA evaluation
"""

import os
import shutil
import torch
from utils.mia_attack_model import GradientMIA


def copy_attack_models(source_dir: str, target_dir: str, num_classes: int = 10):
    """
    Copy pre-trained attack models from source to target directory
    
    Args:
        source_dir: Source directory containing attack models
        target_dir: Target directory to copy models to
        num_classes: Number of classes (attack models)
    """
    os.makedirs(target_dir, exist_ok=True)
    
    copied_count = 0
    for label in range(num_classes):
        source_file = os.path.join(source_dir, f"attack_model{label}.pth")
        target_file = os.path.join(target_dir, f"attack_model{label}.pth")
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            print(f"Copied attack model for label {label}: {source_file} -> {target_file}")
            copied_count += 1
        else:
            print(f"Warning: Attack model not found for label {label} at {source_file}")
    
    print(f"\nSuccessfully copied {copied_count}/{num_classes} attack models")
    return copied_count


def create_dummy_attack_models(target_dir: str, num_classes: int = 10, device: str = "cuda"):
    """
    Create untrained attack models for testing purposes
    
    Args:
        target_dir: Directory to save models
        num_classes: Number of classes (attack models)
        device: Device to use for model initialization
    """
    os.makedirs(target_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    for label in range(num_classes):
        model_path = os.path.join(target_dir, f"attack_model{label}.pth")
        
        if not os.path.exists(model_path):
            # Create and save a new model
            model = GradientMIA().to(device)
            torch.save(model.state_dict(), model_path)
            print(f"Created dummy attack model for label {label} at {model_path}")
        else:
            print(f"Attack model already exists for label {label} at {model_path}")
    
    print(f"\nAttack models ready in {target_dir}")


def verify_attack_models(model_dir: str, num_classes: int = 10):
    """
    Verify that all attack models are present and loadable
    
    Args:
        model_dir: Directory containing attack models
        num_classes: Number of classes to check
    """
    print(f"Verifying attack models in {model_dir}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    success_count = 0
    failed_labels = []
    
    for label in range(num_classes):
        model_path = os.path.join(model_dir, f"attack_model{label}.pth")
        
        if os.path.exists(model_path):
            try:
                model = GradientMIA().to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                success_count += 1
                print(f"✓ Label {label}: Model loaded successfully")
            except Exception as e:
                failed_labels.append(label)
                print(f"✗ Label {label}: Failed to load model - {e}")
        else:
            failed_labels.append(label)
            print(f"✗ Label {label}: Model file not found")
    
    print(f"\nVerification complete: {success_count}/{num_classes} models ready")
    
    if failed_labels:
        print(f"Failed labels: {failed_labels}")
    
    return success_count == num_classes


def main():
    """Main function to set up MIA attack models"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup MIA attack models")
    parser.add_argument('--action', type=str, choices=['copy', 'create', 'verify'], 
                       default='verify', help='Action to perform')
    parser.add_argument('--source', type=str, 
                       default='Membership_Inference_Attack0',
                       help='Source directory for copying models')
    parser.add_argument('--target', type=str,
                       default='system/Membership_Inference_Attack',
                       help='Target directory for models')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes/labels')
    
    args = parser.parse_args()
    
    if args.action == 'copy':
        # Copy existing attack models
        copy_attack_models(args.source, args.target, args.num_classes)
        verify_attack_models(args.target, args.num_classes)
        
    elif args.action == 'create':
        # Create dummy models for testing
        create_dummy_attack_models(args.target, args.num_classes)
        verify_attack_models(args.target, args.num_classes)
        
    elif args.action == 'verify':
        # Just verify existing models
        is_ready = verify_attack_models(args.target, args.num_classes)
        if not is_ready:
            print("\nSome models are missing. You can:")
            print(f"1. Copy models: python {__file__} --action copy")
            print(f"2. Create dummy models: python {__file__} --action create")


if __name__ == "__main__":
    main()