import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import argparse  # Necesario para leer argumentos de terminal

def train(epochs, lr, batch_size):
    # 1. Configuración de MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("MLOps-Lab3-Pets")

    # Nombre dinámico para identificar el run fácilmente
    run_name = f"MobileNetV2_E{epochs}_LR{lr}"

    with mlflow.start_run(run_name=run_name):
        # Log de parámetros (Lo que cambias desde terminal)
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr})

        # 2. Preparación de Datos
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        os.makedirs("data", exist_ok=True)
        try:
            full_dataset = datasets.OxfordIIITPet(root='data', split='trainval', target_types='category', download=True, transform=transform)
            class_names = full_dataset.classes
        except:
            print("No se pudo descargar OxfordPets, usando CIFAR10 como fallback...")
            full_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
            class_names = full_dataset.classes

        # Guardar clases para la API
        mlflow.log_dict({"classes": class_names}, "classes.json")

        # --- DIVISIÓN TRAIN / VALIDATION (80% / 20%) ---
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 3. Configuración del Modelo
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        for param in model.parameters(): param.requires_grad = False
        
        # Ajustar la capa final
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
        
        optimizer = optim.Adam(model.classifier[1].parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        print(f"Iniciando entrenamiento en {device} con {epochs} épocas...")

        # 4. Bucle de Entrenamiento y Validación
        for epoch in range(epochs):
            # --- FASE DE ENTRENAMIENTO ---
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Métricas acumuladas
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            epoch_train_loss = running_loss / train_size
            epoch_train_acc = correct_train / total_train

            # --- FASE DE VALIDACIÓN ---
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            epoch_val_loss = val_loss / val_size
            epoch_val_acc = correct_val / total_val

            # --- REGISTRO EN MLFLOW ---
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

        # Guardar el modelo final
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    # Configuración de argumentos de terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="Número de épocas")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    
    args = parser.parse_args()
    
    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
