import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import ResNet50_Weights

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load Dataset
dataset = datasets.ImageFolder(
    "dataset",
    transform=transform
)

# Data Loader
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

# Number of Classes
num_classes = len(dataset.classes)

print("Total Classes:", num_classes)
print("Total Classes:", len(dataset.classes))
print(dataset.classes)
# Load ResNet50
model = models.resnet50(
    weights=ResNet50_Weights.DEFAULT
)
# Replace Last Layer
model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
epochs = 1

for epoch in range(epochs):

    running_loss = 0

    for images, labels in loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}, Loss: {running_loss}"
    )

# Save Model
torch.save(
    model.state_dict(),
    "new_model.pt"
)

# Save Labels
with open("class_labels.txt", "w") as f:

    for item in dataset.classes:

        f.write(item + "\n")

print("Training Completed!")

torch.save(model.state_dict(), "new_model.pt")

import json

with open("classes.json", "w") as f:
    json.dump(dataset.classes, f)

print("Model Saved")