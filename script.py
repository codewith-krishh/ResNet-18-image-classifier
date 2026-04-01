import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
import os
from PIL import Image
import torch.nn.functional as F
import time

def remove_corrupt_images(folder):
    removed = 0
    for root, dirs, files in os.walk(folder):
        for fname in files:
            path = os.path.join(root, fname)
            try:
                with Image.open(path) as img:
                    img.verify()        # checks header only, fast
            except Exception:
                print(f"  Removing corrupt file: {path}")
                os.remove(path)
                removed += 1
    print(f"  Done. {removed} corrupt file(s) removed from {folder}.\n")

if __name__ == '__main__':

    print("Scanning for corrupt images...")
    remove_corrupt_images("data/train")
    remove_corrupt_images("data/val")

    data_dir = 'data'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_data = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
    val_data = datasets.ImageFolder(f"{data_dir}/val", transform=transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=32, num_workers=4, pin_memory=True,persistent_workers=True)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    #Freeze
    for param in model.parameters():
        param.requires_grad = False

    #Replace head
    num_classes = len(train_data.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    device = torch.device("cuda:0")
    print("Using device:", device)
    model.to(device)



    #Training Loop (Execution Mode)
    epochs = 5
    best_val_acc = 0.0
    for epoch in range(epochs):
        start = time.time()
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()



        #Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images,labels = images.to(device), labels.to(device)

                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                confidence, preds = torch.max(probs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct/total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Epoch time: {time.time()-start:.1f}s")


        #Save best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "model.pth")
            print(f"Best model saved ({val_acc:.2f}%)")

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.2f}%")
    print("Model saved to: model.pth")