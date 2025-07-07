import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_training_history(history):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    """Сохраняет модель"""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """Загружает модель"""
    model.load_state_dict(torch.load(path))
    return model


def compare_models(fc_history, cnn_history):
    """Сравнивает результаты полносвязной и сверточной сетей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(fc_history['test_accs'], label='FC Network', marker='o')
    ax1.plot(cnn_history['test_accs'], label='CNN', marker='s')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(fc_history['test_losses'], label='FC Network', marker='o')
    ax2.plot(cnn_history['test_losses'], label='CNN', marker='s')
    ax2.set_title('Test Loss Comparison')
    ax2.legend()
    ax2.grid(True)
    


    
    plt.tight_layout()
    plt.show() 


def plot_confusion_matrix(model, test_loader, device, title='Confusion Matrix'):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_gradient_flow(model):
    gradients = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append((name, param.grad.abs().mean().item()))
    
    # Сортируем градиенты по величине
    gradients.sort(key=lambda x: x[1], reverse=True)
    
    # Визуализация
    names = [g[0] for g in gradients]
    values = [g[1] for g in gradients]
    
    plt.figure(figsize=(12, 6))
    plt.barh(names, values)
    plt.xscale('log')
    plt.title('Gradient Flow Analysis')
    plt.xlabel('Mean Gradient Magnitude (log scale)')
    plt.tight_layout()
    plt.show()

def visualize_first_layer_activations(activations, n_cols=8, cmap='viridis'):
    n_filters = activations.size(1)  # Количество фильтров
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    # Создаем фигуру
    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    plt.suptitle(f"Активации первого сверточного слоя ({n_filters} фильтров)", y=1.02)
    
    # Визуализируем каждый фильтр
    for i in range(n_filters):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(activations[0, i].cpu().numpy(), cmap=cmap)
        plt.axis('off')
        plt.title(f"F{i+1}", fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_receptive_field(rf_info):
    """Визуализация рецептивного поля"""
    size = rf_info['size']
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(0, size+1)
    ax.set_ylim(0, size+1)
    
    # Рисуем область рецептивного поля
    rect = plt.Rectangle((1,1), size-1, size-1, 
                        fill=False, color='red', linewidth=3)
    ax.add_patch(rect)
    
    # Настройки отображения
    ax.set_xticks(range(size+2))
    ax.set_yticks(range(size+2))
    ax.grid(True)
    ax.set_title(f'Рецептивное поле первого слоя\nРазмер: {size}x{size}', pad=20)
    plt.show()