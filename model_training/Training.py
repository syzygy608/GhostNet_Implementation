import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import Utils
from model_training.GhostNet import GhostNet
from model_training.GhostResNet56 import GhostResNet56
from model_training.GhostVGG16 import GhostVGG16

def train_model(model, dataloader, criterion, optimizer, scheduler, device, epochs, name):
    model.train()
    sw = SummaryWriter(log_dir=os.path.join("logs", name + "_cifar10"))
    best_lose = float('inf')
    delta = 0.0001
    patience = 10
    no_improvement = 0

    for epoch in range(epochs):
        running_loss = 0.0
        acc = 0
        total = 0
        for images, labels in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            acc += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)

            if epoch % 10 == 0:
                sw.add_images('Images', images, epoch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = 100 * acc / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        sw.add_scalar('Loss/epoch', epoch_loss, epoch)
        sw.add_scalar('Accuracy/epoch', epoch_acc, epoch)

        # Save the model if it has the best loss so far
        if epoch_loss < best_lose - delta:
            no_improvement = 0
            best_lose = epoch_loss
            torch.save(model.state_dict(), os.path.join("saved_weight", f"{name}_cifar10.pth"))
            print(f"Model saved with loss: {best_lose:.4f}")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"No improvement for {patience} epochs, stopping training.")
                break

        # Step the learning rate scheduler
        scheduler.step()

    sw.close()

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
def argument_parser():
    parser = argparse.ArgumentParser(description="Train a GhostNet model on CIFAR-10 dataset.")
    parser.add_argument("--root", type=str, default="./dataset", help="Root directory for the dataset.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for the optimizer.")
    parser.add_argument("--model_name", type=str, default="GhostResNet56", help="Name of the model to save.")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for the optimizer.")
    return parser.parse_args()

def main():
    args = argument_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR-10 dataset
    dataloader = Utils.get_cifar10_dataloader(args.root, train=True, batch_size=args.batch_size, shuffle=True)

    if args.model_name == "GhostNet":
        model = GhostNet(num_classes=10).to(device)
    elif args.model_name == "GhostResNet56":
        model = GhostResNet56(num_classes=10).to(device)
    elif args.model_name == "GhostVGG16":
        model = GhostVGG16(num_classes=10).to(device)
    else:
        raise ValueError("Invalid model name. Choose either 'GhostNet', 'GhostResNet56', or 'GhostVGG16'.")
    print(f"Training model: {args.model_name} on device: {device}")

    model.apply(init_weights)  # Initialize weights

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.001)
    
    name = args.model_name
    # Train the model
    train_model(model, dataloader, criterion, optimizer, scheduler, device, args.epochs, name)

if __name__ == "__main__":
    main()
