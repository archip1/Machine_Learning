import timeit
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import transforms, datasets, models
import numpy as np
import random
# for error that wasnt downloading the dataset from stack overflow
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#Function for reproducibilty. You can check out: https://pytorch.org/docs/stable/notes/randomness.html
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(100)

#TODO: Populate the dictionary with your hyperparameters for training
def get_config_dict(pretrain):
    """
    pretrain: 0 or 1. Can be used if you need different configs for part 1 and 2.
    """
    
    config = {
        "batch_size": 128,
        "lr": 0.001,
        "num_epochs": 5 if pretrain == 0 else 25,
        "weight_decay": 1e-4,   #set to 0 if you do not want L2 regularization
        "save_criteria": "accuracy",     #Str. Can be 'accuracy'/'loss'/'last'. (Only for part 2)

    }
    
    return config
    

#TODO: Part 1 - Complete this with your CNN architecture. Make sure to complete the architecture requirements.
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # TODO: Define your CNN architecture here
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # Adjust dimensions based on pooling
        self.fc2 = nn.Linear(256, 10)


    def forward(self, x):

        # TODO: Implement the forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 8 * 8)  # Flatten the tensor

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#TODO: Part 2 - Complete this with your Pretrained CNN architecture. 
class PretrainedNet(nn.Module):
    def __init__(self):
        super(PretrainedNet, self).__init__()
        # TODO: Load a pretrained model

        # Load pretrained ResNet18 model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify the final layer to fit the CIFAR-10 dataset (10 classes)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

        # Check for MPS device (Apple Silicon GPU support)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.model = self.model.to(device)

        print("Model summary:",self.model)

    def forward(self, x):
        return self.model(x)


#Feel free to edit this with your custom train/validation splits, transformations and augmentations for CIFAR-10, if needed.
def load_dataset(pretrain):
    """
    pretrain: 0 or 1. Can be used if you need to define different dataset splits/transformations/augmentations for part 2.

    returns:
    train_dataset, valid_dataset: Dataset for training your model
    test_transforms: Default is None. Edit if you would like transformations applied to the test set. 

    """

    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        # transforms.RandomHorizontalFlip(),  # Flip the image horizontally
                                        # transforms.RandomRotation(10),
                                        # transforms.RandomCrop(32, padding=4),  # Random crop with padding
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    train_dataset, valid_dataset = random_split(full_dataset, [38000, 12000])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return train_dataset, valid_dataset, test_transforms