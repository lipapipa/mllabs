import torch
import torch.nn as nn
import torch.optim as optim
from models import FullyConnectedModel
from datasets import get_mnist_loaders, get_cifar_loaders
import time
from utils import plot_training_history, plot_train_accs
import matplotlib.pyplot as plt
import numpy as np

def run_epoch(model, data_loader, criterion, optimizer=None, device='cuda', is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.set_grad_enabled(not is_test):
        for batch_idx, (data, target) in enumerate((data_loader)):
            data, target = data.to(device), target.to(device)
            
            if not is_test and optimizer is not None:
                optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            
            if not is_test and optimizer is not None:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return total_loss / len(data_loader), correct / total

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        if (epoch+1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            print(f'Gap: {train_acc - test_acc:.4f}')
            print('-' * 50)
    
    end_time = time.time()
    result_time = end_time - start_time
    print(f'Training time: {result_time:.2f} seconds')
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'result_time': result_time
    }

def plot_weight_distribution(model, title):
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.extend(param.data.cpu().numpy().flatten())
    
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=100, alpha=0.7)
    plt.title(f'Weight Distribution: {title}')
    plt.xlabel('Weight value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def create_regularization_configs():
    base_config = {
        "input_size": 784,
        "num_classes": 10,
        "layers": [
            {"type": "linear", "size": 512},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "relu"},
            {"type": "linear", "size": 128},
            {"type": "relu"},
            {"type": "linear", "size": 64},
            {"type": "relu"},
            {"type": "linear", "size": 10}
        ]
    }
    
    configs = []
    
    # 1. Без регуляризации
    configs.append((base_config.copy(), "No regularization"))
    
    # 2. Только Dropout с разными коэффициентами
    for rate in [0.1, 0.3, 0.5]:
        config = base_config.copy()
        # Добавляем dropout после каждого relu
        new_layers = []
        for layer in config["layers"]:
            new_layers.append(layer)
            if layer["type"] == "relu":
                new_layers.append({"type": "dropout", "rate": rate})
        config["layers"] = new_layers
        configs.append((config, f"Dropout (rate={rate})"))
    
    # 3. Только BatchNorm
    config = base_config.copy()
    new_layers = []
    for layer in config["layers"]:
        new_layers.append(layer)
        if layer["type"] == "linear":
            new_layers.append({"type": "batch_norm"})
    config["layers"] = new_layers
    configs.append((config, "BatchNorm only"))
    
    # 4. Dropout + BatchNorm
    config = base_config.copy()
    new_layers = []
    for layer in config["layers"]:
        new_layers.append(layer)
        if layer["type"] == "linear":
            new_layers.append({"type": "batch_norm"})
        elif layer["type"] == "relu":
            new_layers.append({"type": "dropout", "rate": 0.3})
    config["layers"] = new_layers
    configs.append((config, "Dropout + BatchNorm"))
    
    # 5. L2 регуляризация (weight decay)
    config = base_config.copy()
    configs.append((config, "L2 regularization (weight decay)"))
    
    return configs

if __name__ == '__main__':
    train_dl, test_dl = get_mnist_loaders(batch_size=512)
    configs = create_regularization_configs()
    
    results = []
    
    for config, title in configs:
        print(f"\n=== Training model with: {title} ===")
        
        model = FullyConnectedModel(**config).to('cuda')
        
        # Используем weight decay только для L2 регуляризации
        lr = 0.001
        weight_decay = 0.01 if "L2 regularization" in title else 0.0
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        history = train_model(model, train_dl, test_dl, epochs=5, lr=lr, device='cuda')
        results.append((title, history, model))
        
        # Визуализация обучения
        plot_training_history(history
                             )
        
        # Визуализация распределения весов
        plot_weight_distribution(model, title)
    
    # Сравнение финальных точностей
    print("\n=== Final Results ===")
    for title, history, _ in results:
        print(f"{title}:")
        print(f"  Train Accuracy: {history['train_accs'][-1]:.4f}")
        print(f"  Test Accuracy: {history['test_accs'][-1]:.4f}")
        print(f"  compare acc: {history['train_accs'][-1] - history['test_accs'][-1]:.4f}")
        print(f"  Time: {history['result_time']:.2f} sec")
        print("-" * 50)
    
    # Визуализация сравнения точностей
    plt.figure(figsize=(12, 6))
    for title, history, _ in results:
        plt.plot(history['test_accs'], label=title)
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()