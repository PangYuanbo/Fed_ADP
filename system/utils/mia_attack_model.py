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

        def make_cnn(in_channels, hidden_channels):
            return nn.Sequential(
                nn.Conv2d(1, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(hidden_channels, 128)
            )

        # CNN branches for conv/FC gradient maps
        self.cnn1 = make_cnn(1, 64)
        self.cnn2 = make_cnn(1, 128)
        self.cnn_fc1 = make_cnn(1, 64)
        self.cnn_fc = make_cnn(1, 64)

        # FCN for processing softmax outputs
        self.fcn_softmax = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        fusion_dim = 128 * 4 + 64
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, grad_conv1, grad_conv2, grad_fc1, grad_fc, softmax_out):
        """
        Forward pass of the attack model
        
        Args:
            grad_conv1: Gradients from first conv layer
            grad_conv2: Gradients from second conv layer
            grad_fc1: Gradients from first FC layer
            grad_fc: Gradients from final FC layer
            softmax_out: Softmax output probabilities
            
        Returns:
            Membership probability (0 = non-member, 1 = member)
        """
        e1 = self.cnn1(grad_conv1)
        e2 = self.cnn2(grad_conv2)
        e3 = self.cnn_fc1(grad_fc1)
        e4 = self.cnn_fc(grad_fc)
        e5 = self.fcn_softmax(softmax_out)

        features = torch.cat([e1, e2, e3, e4, e5], dim=1)
        
        # Final classification
        out = self.classifier(features)
        return out
