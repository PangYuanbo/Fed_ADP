"""
MIA Attack Model Definition
Contains the neural network architecture for membership inference attacks
"""

import torch
import torch.nn as nn


class GradientMIA(nn.Module):
    """
    Attack model that uses gradient information and softmax outputs
    to infer membership status
    """
    def __init__(self):
        super().__init__()

        # CNN for processing conv1 gradients
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Linear(64, 128)
        )

        # CNN for processing conv2 gradients
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Linear(128, 128)
        )

        # FCN for processing softmax outputs
        self.fcn_softmax = nn.Sequential(
            nn.Linear(10, 32), 
            nn.ReLU(),
            nn.Linear(32, 64), 
            nn.ReLU()
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(320, 64), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )

    def forward(self, grad_conv1, grad_conv2, grad_fc1, grad_fc, softmax_out):
        """
        Forward pass of the attack model
        
        Args:
            grad_conv1: Gradients from first conv layer
            grad_conv2: Gradients from second conv layer
            grad_fc1: Gradients from first FC layer (not used in current architecture)
            grad_fc: Gradients from final FC layer (not used in current architecture)
            softmax_out: Softmax output probabilities
            
        Returns:
            Membership probability (0 = non-member, 1 = member)
        """
        # Process gradient features
        e1 = self.cnn1(grad_conv1)
        e2 = self.cnn2(grad_conv2)
        
        # Process softmax features
        e5 = self.fcn_softmax(softmax_out)

        # Concatenate all features
        features = torch.cat([e1, e2, e5], dim=1)
        
        # Final classification
        out = self.classifier(features)
        return out