import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

def show_images(images, labels=None, nrow=8, title=None, size=128,save_path=None):
    """Визуализирует batch изображений."""
    images = images[:nrow]

    # Увеличение изображений до 128x128 для лучшей видимости
    resize_transform = transforms.Resize(size=(size, size), antialias=True)

    # Преобразуем в тензор, если это не тензоры
    to_tensor = transforms.ToTensor()
    images = [img if isinstance(img, torch.Tensor) else to_tensor(img) for img in images]

    images_resized = [resize_transform(img) for img in images]

    # Создаем сетку изображений
    fig, axes = plt.subplots(1, nrow, figsize=(nrow * 2, 2))
    if nrow == 1:
        axes = [axes]

    for i, img in enumerate(images_resized):
        img_np = img.permute(1, 2, 0).numpy()  # меняем порядок каналов

        # Нормализация для отображения
        img_np = np.clip(img_np, a_min=0, a_max=1)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def show_single_augmentation(original_img, augmented_img, title="Аугментация",save_path=None):
    """Визуализирует оригинальное и аугментированное изображение рядом."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Увеличиваем изображение
    resize_transform = transforms.Resize(size=(128, 128), antialias=True)

    # Преобразуем в тензор, если это еще не тензор
    if not isinstance(original_img, torch.Tensor):
        to_tensor = transforms.ToTensor()
        original_img = to_tensor(original_img)
    if not isinstance(augmented_img, torch.Tensor):
        to_tensor = transforms.ToTensor()
        augmented_img = to_tensor(augmented_img)

    orig_resized = resize_transform(original_img)
    aug_resized = resize_transform(augmented_img)

    # Оригинальное изображение
    orig_np = orig_resized.permute(1, 2, 0).numpy()  # меняем порядок каналов и преобразуем в numpy
    orig_np = np.clip(orig_np, 0, 1)
    ax1.imshow(orig_np)
    ax1.set_title("ОРИГИНАЛ")
    ax1.axis('off')

    # Аугментированное изображение
    aug_np = aug_resized.permute(1, 2, 0).numpy()
    aug_np = np.clip(aug_np, 0, 1)
    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def show_multiple_augmentations(original_img, augmented_imgs, titles,save_path=None):
    """Визуализирует оригинальное изображение и несколько аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))

    # Увеличиваем изображение
    resize_transform = transforms.Resize((128, 128), antialias=True)

    # Преобразуем в тензор, если это еще не тензор
    if not isinstance(original_img, torch.Tensor):
        to_tensor = transforms.ToTensor()
        original_img = to_tensor(original_img)
    augmented_imgs = [img if isinstance(img, torch.Tensor) else to_tensor(img) for img in augmented_imgs]

    orig_resized = resize_transform(original_img)

    # Оригинальное изображение
    orig_np = orig_resized.permute(1, 2, 0).numpy()
    orig_np = np.clip(orig_np, 0, 1)
    axes[0].imshow(orig_np)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')

    # Аугментированные изображения
    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        aug_resized = resize_transform(aug_img)
        aug_np = aug_resized.permute(1, 2, 0).numpy()
        aug_np = np.clip(aug_np, 0, 1)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()