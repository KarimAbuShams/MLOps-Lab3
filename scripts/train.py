import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os

def train():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("MLOps-Lab3-Pets")
    epochs = 1
    batch_size = 32
    lr = 0.001

    with mlflow.start_run(run_name="MobileNetV2_Transfer"):
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr})
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        os.makedirs("data", exist_ok=True)
        try:
            dataset = datasets.OxfordIIITPet(root='data', split='trainval', target_types='category', download=True, transform=transform)
            class_names = dataset.classes
        except:
            dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
            class_names = dataset.classes

        mlflow.log_dict({"classes": class_names}, "classes.json")
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        for param in model.parameters(): param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
        
        optimizer = optim.Adam(model.classifier[1].parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
            mlflow.log_metric("train_loss", loss.item(), step=epoch)

        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    train()
