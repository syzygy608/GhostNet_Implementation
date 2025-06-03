import torchvision
import torch.utils.data as Data

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
        return image, label

    @staticmethod
    def download(root):
        torchvision.datasets.CIFAR10(root=root, train=True, download=True)

def get_cifar10_dataloader(root, train=True, batch_size=32, shuffle=True, num_workers=2):
    
    transform = None 

    if train:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    dataset = CIFAR10Dataset(root=root, train=train, transform=transform)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    dataloader.dataset.download(root)  # Ensure dataset is downloaded if not already present
    print(f"CIFAR-10 {'train' if train else 'test'} dataset loaded with {len(dataset)} samples.")

    return dataloader

if __name__ == "__main__":
    # Example usage
    root = './dataset'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CIFAR10Dataset(root=root, train=True, transform=transform)
    
# normalize parmeters from https://github.com/kuangliu/pytorch-cifar/issues/19