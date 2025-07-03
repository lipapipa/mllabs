import torch
import torch.nn as nn
import torch.optim as optim
from models import FullyConnectedModel
from datasets import get_mnist_loaders, get_cifar_loaders
import time
from utils import plot_training_history,plot_train_accs





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
        compare_acc = train_acc - test_acc
        if (epoch+1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            print(f'Compare Acc: {compare_acc:.4f}')
            print('-' * 50)
            #plot_train_accs({'train_accs': train_accs, 'test_accs': test_accs})
            
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

if __name__ == '__main__':

    configs = [{"input_size": 784,"num_classes": 10,"layers": [{"type": "linear", "size": 512},{"type": "relu"},{"type": "linear", "size": 256},{"type": "relu"}, {"type": "linear", "size": 128},
        {"type": "relu"}, {"type": "linear", "size": 64},{"type": "relu"},{"type": "linear", "size": 32},{"type": "relu"},
        {"type": "linear", "size": 16},{"type": "relu"},{"type": "linear", "size": 10},{"type": "sigmoid"}]},                       #7 слоев

        
        {"input_size": 784,"num_classes": 10,"layers": [ {"type": "linear", "size": 512}, {"type": "relu"},{"type": "linear", "size": 256},
        {"type": "relu"},{"type": "linear", "size": 128},{"type": "relu"},{"type": "linear", "size": 64}, {"type": "relu"},{"type": "linear", "size": 10},       #5с лоев
        {"type": "sigmoid"}]},
        
        {"input_size": 784,"num_classes": 10,"layers": [ {"type": "linear", "size": 512},{"type": "relu"}, {"type": "linear", "size": 64},      #3 слоя 
        {"type": "relu"},{"type": "linear", "size": 10}, {"type": "sigmoid"}]},

        {"input_size": 784,"num_classes": 10,"layers": [{"type": "linear", "size": 512},
        {"type": "relu"},{"type": "linear", "size": 10},{"type": "sigmoid"} ]},                 #2 слоя 

        {"input_size": 784,"num_classes": 10,"layers": [{"type": "linear", "size": 512},{"type": "sigmoid"}]}]
    
    
    

    for config in configs:
        model = FullyConnectedModel(**config).to('cuda')
        print('MNIST dataset')
        print(f'количество слоев {len(config["layers"])/2}')
        train_dl, test_dl = get_mnist_loaders(batch_size=512)
        history = train_model(model,train_dl, test_dl,epochs=10, lr=0.001, device='cuda')
        plot_training_history(history)

