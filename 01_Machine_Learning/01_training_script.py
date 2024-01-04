## Python script that uses PyTorch to train a simple neural network model using a small dataset of 10 pictures

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a simple CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 6 * 6, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = self.fc1(x)
        return x

def load_images_from_folder(folder_path, transform):
    dataset = datasets.ImageFolder(root=folder_path, transform=transform)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    return loader

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved at {filepath}")

# Set up transformations for the images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Load images from a folder
folder_path = "path_to_your_folder"  # Replace with your folder path
image_loader = load_images_from_folder(folder_path, transform)

# Instantiate the model
model = SimpleCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(10):  # Training for 10 epochs
    running_loss = 0.0
    for images, labels in image_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch + 1} Loss: {running_loss / len(image_loader)}")

print('Finished Training')

# Save the trained model
save_model(model, 'trained_model.pth')
