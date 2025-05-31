import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable
import os

class CIFAR10Dataset(Data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.data = torchvision.datasets.CIFAR10(root=self.root, train=self.train, download=True, transform=self.transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    @staticmethod
    def download(root):
        torchvision.datasets.CIFAR10(root=root, train=True, download=True)

    @staticmethod
    def show_sample_images(dataset, num_images=10):
        plt.figure(figsize=(10, 10))
        for i in range(num_images):
            image, label = dataset[i]
            plt.subplot(1, num_images, i + 1)
            plt.imshow(image.permute(1, 2, 0).numpy())
            plt.title(f'Label: {label}')
            plt.axis('off')
        plt.show()

def get_cifar10_dataloader(root, train=True, batch_size=32, shuffle=True, num_workers=2):
    
    transform = None 

    if train:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    dataset = CIFAR10Dataset(root=root, train=train, transform=transform)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    dataloader.dataset.download(root)  # Ensure dataset is downloaded if not already present
    print(f"CIFAR-10 {'train' if train else 'test'} dataset loaded with {len(dataset)} samples.")
    
    return dataloader

if __name__ == "__main__":
    # Example usage
    root = './data'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CIFAR10Dataset(root=root, train=True, transform=transform)
    
    # Show sample images
    CIFAR10Dataset.show_sample_images(dataset, num_images=10)
    