import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from model import MushroomClassifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Limit dataset size for quick training
def get_data_loaders(sample_size=200):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Smaller image size
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = ImageFolder('./dataset/splitted_dataset/train', transform=transform)
    val_dataset = ImageFolder('./dataset/splitted_dataset/val', transform=transform)
    test_dataset = ImageFolder('./dataset/splitted_dataset/test', transform=transform)

    # Limit dataset to smaller subset
    train_dataset = Subset(train_dataset, range(min(len(train_dataset), sample_size)))
    val_dataset = Subset(val_dataset, range(min(len(val_dataset), sample_size)))
    test_dataset = Subset(test_dataset, range(min(len(test_dataset), sample_size)))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    return train_loader, val_loader, test_loader

# Faster training model
def train_model(num_epochs=2):
    print("Starting Image Model Training...")
    train_loader, _, _ = get_data_loaders()
    model = MushroomClassifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), './models/mushroom_classifier.pth')
    print("Image model saved successfully.")

def evaluate_model():
    _, _, test_loader = get_data_loaders()
    model = MushroomClassifier().to(device)
    model.load_state_dict(torch.load('./models/mushroom_classifier.pth'))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Image Model Test Accuracy: {accuracy:.2f}%")

# Faster tabular model
def train_tabular_model():
    print("Starting Feature-Based Model Training...")
    df = pd.read_csv('./data/mushrooms.csv')

    X = df.drop('class', axis=1)
    y = df['class']

    X = pd.get_dummies(X)
    y = y.map({'e': 0, 'p': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use a smaller number of trees for speed
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Feature-Based Model Accuracy: {accuracy * 100:.2f}%")

    with open('./models/mushroom_tabular.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Feature-Based model saved successfully.")
