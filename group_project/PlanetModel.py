from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# -------------- Model for planet classification --------------
# ----------------------- Unused code ------------------------
class PlanetModel(nn.Module):
    def __init__(self):
        super(PlanetModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout2d(0.2)
        )
        self._init_classifier()

    def _init_classifier(self):
        with torch.no_grad():
            n_features = 4096  # Updated value
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_features, 128),
                nn.ReLU(),
                nn.Linear(128, 11),
                nn.Softmax(dim=1)
            )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_and_evaluate(model, train_loader, test_loader, device, self, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)    
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    # Initialize lists to track loss and accuracy
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

        train_losses.append(train_loss / len(train_loader.dataset))
        train_accuracies.append(train_correct / len(train_loader.dataset))
            
        model.eval()
        test_loss, test_correct = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                test_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

        test_losses.append(test_loss / len(test_loader.dataset))
        test_accuracies.append(test_correct / len(test_loader.dataset))

        if len(test_losses) > 0:
            scheduler.step(test_losses[-1])  # Step the scheduler based on validation loss
        else:
            scheduler.step(train_losses[-1])  # Fallback to using train loss if test loss is not available

        self.get_logger().info(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader.dataset):.4f}, Train Accuracy: {train_correct / len(train_loader.dataset):.2f}, Test Loss: {test_loss / len(test_loader.dataset):.4f}, Test Accuracy: {test_correct / len(test_loader.dataset):.2f}')
    
    # Plotting training and testing loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting training and testing accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    
  
def predict(model, image_path):
    try:
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("Image not found at the path: " + image_path)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to the required input size of the model
        image = cv2.resize(image, (256, 256))  # Resize to 256x256

        # Convert numpy array to PIL Image
        image = Image.fromarray(image)

        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Prepare image tensor
        image_tensor = transform(image).unsqueeze(0)

        # Set the model to evaluation mode and perform prediction
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()

        class_names = {
            0: "Earth",
            1: "Moon",
            # Ensure that all potential class indices have corresponding names
        }
        return class_names.get(predicted_idx, "Unknown class index: " + str(predicted_idx))

    except Exception as e:
        return (f"An error occurred: {e}")


def get_data_loaders(batch_size=64, validation_split=0.2):
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
    transforms.RandomVerticalFlip(),    # Randomly flip the images vertically
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random rotation, translation, and scaling
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pretrained models
    ])
    full_dataset = datasets.ImageFolder('src/group_project/supporting_files/Planets and Moons', transform=transform)
    total_size = len(full_dataset)
    test_size = int(total_size * validation_split)
    train_size = total_size - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def save_model(model, filename="src/group_project/supporting_files/full_model/model.pth"):
    torch.save(model, filename)
    print("Model saved to", filename)

def load_model(filename="src/group_project/supporting_files/full_model/model.pth"):
    model = torch.load(filename)
    model.eval()  # Set the model to evaluation mode
    return model

# Example Usage
def main():
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlanetModel()
    model.to(device)
    return device, model