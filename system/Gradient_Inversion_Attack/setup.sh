#!/bin/bash
# GI-SMN Attack Setup Script
# Author: Fed_ADP Project
# This script installs dependencies and downloads pretrained models

echo "========================================="
echo "GI-SMN Attack Setup"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Install Python dependencies
echo -e "${YELLOW}[1/3] Installing Python dependencies...${NC}"
pip install lpips piq pillow matplotlib seaborn pyyaml

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Step 2: Download StyleGAN-XL pretrained model
echo -e "\n${YELLOW}[2/3] Downloading StyleGAN-XL pretrained model...${NC}"
mkdir -p pretrained_models

MODEL_PATH="pretrained_models/stylegan_xl_cifar10.pkl"

if [ -f "$MODEL_PATH" ]; then
    echo -e "${GREEN}✓ Model already exists: $MODEL_PATH${NC}"
else
    echo "Downloading CIFAR-10 model (this may take several minutes)..."

    # StyleGAN-XL CIFAR-10 model URL
    MODEL_URL="https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/cifar10.pkl"

    # Try downloading with wget
    if command -v wget &> /dev/null; then
        wget "$MODEL_URL" -O "$MODEL_PATH"
    # Try downloading with curl
    elif command -v curl &> /dev/null; then
        curl -L "$MODEL_URL" -o "$MODEL_PATH"
    # Try with Python urllib
    else
        echo "wget/curl not found, using Python to download..."
        python3 << EOF
import urllib.request
print("Downloading model...")
urllib.request.urlretrieve("$MODEL_URL", "$MODEL_PATH")
print("Download complete!")
EOF
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Model downloaded successfully${NC}"
    else
        echo "✗ Failed to download model"
        echo "Please download manually from: $MODEL_URL"
        exit 1
    fi
fi

# Step 3: Verify installation
echo -e "\n${YELLOW}[3/3] Verifying installation...${NC}"

python3 << EOF
import sys
try:
    import lpips
    import piq
    import torch
    import matplotlib
    import yaml
    print("✓ All Python dependencies OK")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

import os
if os.path.exists("$MODEL_PATH"):
    print("✓ StyleGAN-XL model found")
else:
    print("✗ StyleGAN-XL model not found")
    sys.exit(1)

print("\n✓ Installation verified successfully!")
EOF

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}=========================================${NC}"
    echo -e "${GREEN}Setup Complete!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo ""
    echo "You can now use GI-SMN attack:"
    echo ""
    echo "1. Integrated with training:"
    echo "   python main.py -algo FedCP -data cifar-10-normal -gia"
    echo ""
    echo "2. Standalone evaluation:"
    echo "   python system/Gradient_Inversion_Attack/run_gia_standalone.py --model_path YOUR_MODEL.pt"
    echo ""
else
    echo -e "\n✗ Verification failed"
    exit 1
fi
