import torch
import os

from model_training.GhostNet import GhostNet
from model_training.GhostResNet56 import GhostResNet56
from data_utils import Utils
import tqdm
import argparse
import torch.nn as nn

# testing saved model weight

def test_saved_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == "GhostNet":
        model = GhostNet(num_classes=10).to(device)
    elif model_name == "GhostResNet56":
        model = GhostResNet56(num_classes=10).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models are 'GhostNet' and 'GhostResNet56'.")
            
    # Load the saved model weights
    model_path = os.path.join("saved_weight", f"{model_name}_cifar10.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file not found at {model_path}")
        return
    
    # Set the model to evaluation mode
    model.eval()
    
    data_loader = Utils.get_cifar10_dataloader("./dataset", train=False, batch_size=1, shuffle=False)

    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    loss = 0

    with torch.no_grad():
        for images, labels in tqdm.tqdm(data_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss /= len(data_loader)
    print(f"Test Loss: {loss:.4f}")
    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test dataset: {accuracy:.2f}%")

def parse_args():
    parser = argparse.ArgumentParser(description="Test GhostNet model on CIFAR-10 dataset.")
    parser.add_argument("--model_name", type=str, default="GhostResNet56", help="Name of the model to test.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Testing model: {args.model_name}")
    test_saved_model(args.model_name)

