"""
encoder.py - ResNet50 Encoder

CNN encoder for extracting image features.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    """
    ResNet50 CNN Encoder.
    
    Input:  (batch, 3, 224, 224)
    Output: (batch, 49, embed_dim)  # 49 = 7×7 spatial locations
    """
    
    def __init__(self, embed_dim=512, pretrained=True, fine_tune=False):
        """
        Args:
            embed_dim: Embedding dimension
            pretrained: Use pretrained ResNet50
            fine_tune: Allow gradient updates
        """
        super().__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove final layers (avgpool and fc)
        # Keep only convolutional layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # ResNet50 outputs 2048 channels
        self.resnet_dim = 2048
        
        # Adaptive pooling to ensure 7×7 output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Project to embed_dim
        self.projection = nn.Linear(self.resnet_dim, embed_dim)
        
        # Freeze/unfreeze ResNet
        self.fine_tune(fine_tune)
    
    def forward(self, images):
        """
        Forward pass.
        
        Args:
            images: (batch, 3, 224, 224)
        
        Returns:
            features: (batch, 49, embed_dim)
        """
        # Extract features: (batch, 2048, 7, 7)
        features = self.resnet(images)
        
        # Ensure 7×7: (batch, 2048, 7, 7)
        features = self.adaptive_pool(features)
        
        # Reshape: (batch, 2048, 7, 7) → (batch, 49, 2048)
        batch_size = features.size(0)
        features = features.permute(0, 2, 3, 1)  # (batch, 7, 7, 2048)
        features = features.view(batch_size, -1, self.resnet_dim)  # (batch, 49, 2048)
        
        # Project: (batch, 49, 2048) → (batch, 49, embed_dim)
        features = self.projection(features)
        
        return features
    
    def fine_tune(self, fine_tune=True):
        """
        Enable/disable fine-tuning of ResNet.
        
        Args:
            fine_tune: If True, allow ResNet gradients
        """
        for param in self.resnet.parameters():
            param.requires_grad = fine_tune


if __name__ == '__main__':
    # Test encoder
    encoder = Encoder(embed_dim=256, pretrained=True, fine_tune=False)
    
    # Test forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    
    features = encoder(images)
    
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected: ({batch_size}, 49, 256)")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
