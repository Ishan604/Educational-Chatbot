import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes): 
        """
        Initialize the neural network.

        Args:
        input_size (int): The number of input features (length of bag-of-words vector).
        hidden_size (int): The number of units in the hidden layer(s).
        num_classes (int): The number of output classes (tags).
        """
        super(NeuralNet, self).__init__()

        # Define layers
        self.l1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        self.l3 = nn.Linear(hidden_size, num_classes)  # Output layer (number of classes)
        
        # Define activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        x (tensor): Input tensor (bag of words vector)

        Returns:
        tensor: Output tensor (predictions for each class)
        """
        # Pass through the layers with ReLU activation
        out = self.l1(x)
        out = self.relu(out)  # Apply ReLU after first layer
        out = self.l2(out)
        out = self.relu(out)  # Apply ReLU after second layer
        out = self.l3(out)    # No activation on the output layer for classification

        return out

