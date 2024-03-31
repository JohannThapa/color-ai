import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        Initializes model with input size and number of classes.
        Args:
        - input_size (int): Size of input data.
        - num_classes (int): Number of output classes.
        """
        super(PyTorchModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
        - x (torch.Tensor): Input data tensor.
        Returns:
        - torch.Tensor: Output tensor.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def train(self, x_train, y_train, epochs=10, batch_size=32):
        """
        Trains the PyTorch model.
        Args:
        - x_train (torch.Tensor): Input training data.
        - y_train (torch.Tensor): Target training labels.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epochs):
            for i in range(0, len(x_train), batch_size):
                inputs = x_train[i:i+batch_size]
                labels = y_train[i:i+batch_size]
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def predict(self, x_test):
        """
        Performs prediction using the PyTorch model.
        Args:
        - x_test (torch.Tensor): Input test data.
        Returns:
        - torch.Tensor: Predicted labels.
        """
        with torch.no_grad():
            outputs = self(x_test)
            _, predicted = torch.max(outputs, 1)
        return predicted
