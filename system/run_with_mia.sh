#!/bin/bash

# Run script with MIA evaluation enabled

# First, ensure attack models are in place
echo "Setting up MIA attack models..."
python utils/setup_mia_models.py --action verify

# Check if models need to be copied
if [ $? -ne 0 ]; then
    echo "Copying attack models from Membership_Inference_Attack0..."
    python utils/setup_mia_models.py --action copy --source ../Membership_Inference_Attack0 --target system/Membership_Inference_Attack
fi

# Run the main training with MIA evaluation enabled
echo "Starting training with MIA evaluation..."
nohup python -u main.py \
    -t 1 \
    -jr 1 \
    -nc 20 \
    -nb 10 \
    -data mnist-0.1-npz \
    -m cnn \
    -algo FedCP \
    -did 0 \
    -lam 5 \
    -mia True \
    -eg 10 \
    > result-mnist-0.1-mia.out 2>&1 &

echo "Training started with MIA evaluation. Check result-mnist-0.1-mia.out for progress."