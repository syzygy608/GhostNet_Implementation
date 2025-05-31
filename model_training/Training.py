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

def train_model(model, dataloader, criterion, optimizer, device, epochs):
    model.train()
    sw = SummaryWriter(log_dir=os.path.join("logs", "GhostNet_CIFAR10"))
    best_loss = float('inf')
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            sw.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + len(images))
            if epoch % 10 == 0:
                sw.add_images('Images', images, epoch * len(dataloader) + len(images))
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        sw.add_scalar('Loss/epoch', epoch_loss, epoch)

        # Save the model if it has the best loss so far
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join("saved_weight", "ghostnet_cifar10_best.pth"))
            print(f"Model saved with loss: {best_loss:.4f}")

def argument_parser():
    parser = argparse.ArgumentParser(description="Train a GhostNet model on CIFAR-10 dataset.")
    parser.add_argument("--root", type=str, default="./dataset", help="Root directory for the dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for the optimizer.")
    return parser.parse_args()

def main():
    args = argument_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR-10 dataset
    dataloader = Utils.get_cifar10_dataloader(args.root, train=True, batch_size=args.batch_size, shuffle=True)

    # Initialize GhostNet model
    model = GhostNet(num_classes=10).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, device, args.epochs)

if __name__ == "__main__":
    main()
