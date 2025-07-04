import torch
import torch.nn as nn
import torch.optim as optim
from models import FullyConnectedModel
from datasets import get_mnist_loaders, get_cifar_loaders
import time
from utils import plot_training_history,count_parameters
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np





def run_epoch(model, data_loader, criterion, optimizer=None, device='cuda', is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
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


def train_model(model, train_loader, test_loader, epochs=3, lr=0.001, device='cuda'):
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
        if (epoch+1) % 1 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            print('-' * 50)
    end_time=time.time()
    result_time = end_time - start_time
    print(result_time)

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'result_time' :  result_time
    } 


    
    

def train_model_with_grid(train_loader, test_loader, epochs=3, lr=0.001, device='cuda'):
    
    best_acc = 0
    best_params = {}
    params_grid= {"lr":[0.1,0.01,0.001],"batch_size":[32,128,256],"sizes":[                  
        [64, 32, 16],                   # Узкая сеть
        [256, 128, 64],                 # Средняя
        [1024, 512, 256]]}             #широкая

    for lr in params_grid['lr']:
        for batch_size in params_grid['batch_size']:
            for hidden_sizes in params_grid['sizes']:
                # Создаём модель с текущими параметрами
                config = {
                    "input_size": 784,
                    "num_classes": 10,
                    "layers": [
                        {"type": "linear", "size": hidden_sizes[0]},
                        {"type": "relu"},
                        {"type": "linear", "size": hidden_sizes[1]},
                        {"type": "relu"},
                        {"type": "linear", "size": hidden_sizes[2]},
                        {"type": "sigmoid"}
                    ]
                }
                model = FullyConnectedModel(**config)
                model.to(device)
                
                # Загружаем данные с текущим batch_size
                train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
                
                # Обучаем модель
                history = train_model(model, train_loader, test_loader, epochs, lr, device)
                
                # Сохраняем лучший результат
                if history['test_accs'][-1] > best_acc:
                    best_acc = history['test_accs'][-1]
                    best_params = {
                        'lr': lr,
                        'batch_size': batch_size,
                        'hidden_sizes': hidden_sizes
                    }
    
    print("Лучшие параметры:", best_params)
    print("Лучшая точность:", best_acc)

    


def start():

    configs = [{"input_size": 784,"num_classes":10,"layers":[{"type":"linear","size":64},{"type":"relu"},{"type":"linear","size":32},
    {"type":"relu"},{"type":"linear","size":16},{"type":"sigmoid"}]},
    
    {"input_size": 784,"num_classes":10,"layers":[{"type":"linear","size":256},{"type":"relu"},{"type":"linear","size":128},
    {"type":"relu"},{"type":"linear","size":64},{"type":"sigmoid"}]},

    {"input_size": 784,"num_classes":10,"layers":[{"type":"linear","size":1024},{"type":"relu"},{"type":"linear","size":512},
    {"type":"relu"},{"type":"linear","size":256},{"type":"sigmoid"}]},

    {"input_size": 784,"num_classes":10,"layers":[{"type":"linear","size":2048},{"type":"relu"},{"type":"linear","size":1024},
    {"type":"relu"},{"type":"linear","size":512},{"type":"sigmoid"}]}]

    
    for config in configs:
        model = FullyConnectedModel(**config)
        model.to('cuda')
        print('MNIST dataset')
        print(f'количество слоев {int(len(config["layers"])/2)}')
        print(f'количество параметров {count_parameters(model)}')
        train_dl, test_dl = get_mnist_loaders(batch_size=512)
        history = train_model(model,train_dl, test_dl,epochs=10, lr=0.001, device='cuda')

def start_with_grid():
    train_dl, test_dl = get_mnist_loaders(batch_size=512)
    train_model_with_grid(train_dl,test_dl,epochs=3,device='cuda')
    #Лучшие параметры: {'lr': 0.001, 'batch_size': 128, 'hidden_sizes': [1024, 512, 256]}
    #Лучшая точность: 0.9768

if __name__ == '__main__':
    start() #выводим 2.1

    #так как каждый раз вызывать grid search очень долго просто после первого раза запишем данные которые он дал
    best_params =  {'lr': 0.001, 'batch_size': 128, 'hidden_sizes': [1024, 512, 256]}
    config = {"input_size":784,"num_classes":10,"layers":[{"type":"linear","size":1024},
    {"type":"relu"},{"type":"linear","size":512},{"type":"relu"},{"type":"linear","size":256},{"type":"sigmoid"}]}
    model = FullyConnectedModel(**config)
    train_dl,test_dl = get_mnist_loaders(batch_size=best_params['batch_size'])
    history = train_model(model, train_dl, test_dl, epochs=5, lr=best_params['lr'], device='cuda')


    configexp = [{"input_size":784,"num_classes":10,"layers":[{"type":"linear","size":1024},
    {"type":"relu"},{"type":"linear","size":512},{"type":"relu"},{"type":"linear","size":256},{"type":"sigmoid"}]},  #сужение
    
    {"input_size":784,"num_classes":10,"layers":[{"type":"linear","size":256},
    {"type":"relu"},{"type":"linear","size":512},{"type":"relu"},{"type":"linear","size":1024},{"type":"sigmoid"}]},


    {"input_size":784,"num_classes":10,"layers":[{"type":"linear","size":512},
    {"type":"relu"},{"type":"linear","size":512},{"type":"relu"},{"type":"linear","size":512},{"type":"sigmoid"}]}]

    test_accuracies = []
    param_counts = []
    layer_types = ["Сужение", "Расширение", "Постоянная"]

    for config in configexp:
        model = FullyConnectedModel(**config)
        history = train_model(model, train_dl, test_dl, epochs=5, lr=best_params['lr'], device='cuda')
        
        # Сохраняем результаты
        test_accuracies.append(max(history['test_accs']))
        param_counts.append(count_parameters(model))

    # 1. Heatmap точности
    plt.figure(figsize=(8, 3))
    accuracy_matrix = np.array([test_accuracies])  # Матрица 1x3
    sns.heatmap(accuracy_matrix, 
                annot=True, 
                fmt=".2%", 
                cmap="YlGn",
                xticklabels=layer_types,
                yticklabels=["Accuracy"])
    plt.title("Точность разных архитектур")
    plt.show()

    # Heatmap параметров
    plt.figure(figsize=(8, 3))
    param_matrix = np.array([param_counts])
    sns.heatmap(param_matrix,
                annot=True,
                fmt=",d",
                cmap="Blues",
                xticklabels=layer_types,
                yticklabels=["Параметры"])
    plt.title("Количество параметров")
    plt.show()

''' 2.1
MNIST dataset
количество слоев 3
Узкие слои: [64, 32, 16]
количество параметров 53018

Epoch 5/10:
Train Loss: 0.3626, Train Acc: 0.9463
Test Loss: 0.3351, Test Acc: 0.9467

Epoch 10/10:
Train Loss: 0.1594, Train Acc: 0.9702
Test Loss: 0.1771, Test Acc: 0.9629
115.81265544891357
--------------------------------------------------

MNIST dataset
количество слоев 3
Средние слои: [256, 128, 64]
количество параметров 242762
Epoch 5/10:
Train Loss: 0.0865, Train Acc: 0.9781
Test Loss: 0.1040, Test Acc: 0.9724
--------------------------------------------------
Epoch 10/10:
Train Loss: 0.0275, Train Acc: 0.9933
Test Loss: 0.0737, Test Acc: 0.9795
117.57012248039246
--------------------------------------------------
MNIST dataset
количество слоев 3
Широкие слои: [1024, 512, 256]
количество параметров 1462538
Epoch 5/10:
Train Loss: 0.0395, Train Acc: 0.9882
Test Loss: 0.0694, Test Acc: 0.9811

Epoch 10/10:
Train Loss: 0.0134, Train Acc: 0.9960
Test Loss: 0.0720, Test Acc: 0.9809
122.55653929710388
--------------------------------------------------
MNIST dataset
количество слоев 3
Очень широкие слои: [2048, 1024, 512]
количество параметров 4235786

Epoch 5/10:
Train Loss: 0.0349, Train Acc: 0.9898
Test Loss: 0.0671, Test Acc: 0.9817

Epoch 10/10:
Train Loss: 0.0111, Train Acc: 0.9966
Test Loss: 0.0669, Test Acc: 0.9837
--------------------------------------------------
Вывод:чем шире слои,и больше эпох тем больше переобучение
#2.2

Epoch 5/5:
Train Loss: 0.0357, Train Acc: 0.9890
Test Loss: 0.0885, Test Acc: 0.9738
--------------------------------------------------
71.45808100700378

Epoch 5/5:
Train Loss: 0.0340, Train Acc: 0.9895
Test Loss: 0.0702, Test Acc: 0.9797
--------------------------------------------------
70.03866624832153

'''




        
