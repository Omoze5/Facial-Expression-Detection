# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),

            nn.Flatten()
        )
        # Calculate the correct input size for the fully connected layers
        # Adjusted based on the output dimensions after convolutions and max pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(128, 7)  # Assuming 10 classes for classification
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc_layers(x)
        return x

def create_model():
    return SimpleCNN()
