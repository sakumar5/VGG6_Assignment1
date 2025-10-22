"""----------------------------------------------------------------
Modules:
    torch       : PyTorch library for tensor computation and GPU acceleration
    torchvision : provides datasets, model architectures, and image transformations 
                  for train deep learning models
-----------------------------------------------------------------"""
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

"""----------------------------------------------------------------
* def name :   
*       get_cifar10_loaders
* purpose:
*        repares DataLoaders for the CIFAR-10 dataset with optional data augmentation
*        and a validation split.
* Input parameters:
*       batch_size:     Number of samples per batch to load
*       num_workers:    Number of subprocesses for data loading
*       augment:        If True, apply random cropping and horizontal flipping
*                       for data augmentation on the training set.
*       val_split:      Number of images to reserve for validation from the training set.
*       seed:           Random seed for reproducible dataset splitting.
*
* return:
*      train_loader: Loader for the training subset
*      val_loader:   Loader for the validation subset
*      test_loader:  Loader for the test set
-----------------------------------------------------------------"""

def get_cifar10_loaders(batch_size=128, num_workers=4, augment=True, val_split=5000, seed=42):
    
    # Mean and standard deviation values for CIFAR-10 normalization
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    
     # Compose training transformations (with optional augmentation)
    train_transforms = []
    if augment:
        # Randomly crop with padding for translation invariance and Randomly flip images horizontally
        train_transforms += [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip()]
    
    # Convert PIL Image to tensor and Normalize tensor
    train_transforms += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    train_tf = transforms.Compose(train_transforms)
    
    # Test/validation transformations (no augmentation)
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    # Download and prepare CIFAR-10 datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)

    # Split training data into train and validation subsets
    torch.manual_seed(seed)
    val_size = val_split
    train_size = len(trainset) - val_size
    train_data, val_data = random_split(trainset, [train_size, val_size])

    # Create DataLoaders for each subset
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
