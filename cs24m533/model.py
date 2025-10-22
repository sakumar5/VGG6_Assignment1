"""----------------------------------------------------------------
Modules:
    torch : PyTorch library for tensor computation and GPU acceleration
    nn    : neural network building blocks such as layers, activations, and loss functions
-----------------------------------------------------------------"""
import torch
import torch.nn as nn


"""----------------------------------------------------------------
Class:
Name :  
    VGG6
    
Description: 
    This class defines VGG architecture designed for classification tasks such as CIFAR-10.

Attributes:
    features (nn.Sequential):       
    classifier (nn.Sequential): 

Args:
    num_classes (int): 
        Number of output classes. Default is 10.
        
    activation (str): 
        Type of activation function to use ('relu', 'sigmoid', 'tanh', 'silu', 'gelu').

Methods:
    _get_activation(name):         
    forward(x): 
-----------------------------------------------------------------"""
class VGG6(nn.Module):
   
    def __init__(self, num_classes=10, activation='relu'):
        super().__init__()
        
        # Select activation function dynamically based on the given name
        act = self._get_activation(activation)

        # Convolutional and pooling layers for feature extraction.
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            act,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            act,
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            act,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            act,
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            act,
            nn.MaxPool2d(2,2),
        )
        
        # Fully connected layers for classification.
         # ---------------- Classification Block ----------------
        self.classifier = nn.Sequential(
            # Flatten to 256*4*4
            nn.Flatten(),
            # Fully connected layer
            nn.Linear(256*4*4, 512),
            act,
            
            # Regularization to prevent overfitting
            nn.Dropout(0.5),
            
            # Output layer
            nn.Linear(512, num_classes),
        )
        
    """---------------------------------------------
    * def name :   
    *       _get_activation
    *
    * purpose:
    *        Returns the activation layer corresponding to the given name.
    *
    * Input parameters:
    *       name : given name
    *
    * return:
    *       the activation layer     
    ---------------------------------------------"""
    def _get_activation(self, name): 
        name = name.lower()
        if name == 'relu': return nn.ReLU(inplace=True)
        if name == 'sigmoid': return nn.Sigmoid()
        if name == 'tanh': return nn.Tanh()
        if name in ('silu','swish'): return nn.SiLU()
        if name == 'gelu': return nn.GELU()
        return nn.ReLU(inplace=True)

    """---------------------------------------------
    * def name :   
    *       forward
    *
    * purpose:
    *        to forward pass of the VGG6 network
    *
    * Input parameters:
    *       x : input tensor
    * return:
    *       x: final class scores   
    ---------------------------------------------"""
    def forward(self, x):
        # passes input through convolution & pooling layers
        x = self.features(x)
        
        # passes flattened features through FC layers
        x = self.classifier(x)
        return x
