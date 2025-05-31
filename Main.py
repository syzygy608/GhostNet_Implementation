import torch
import os

from model_training.GhostNet import GhostNet
from data_utils import Utils
import tqdm

# testing saved model weight

def test_saved_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GhostNet(num_classes=10).to(device)
    
    # Load the saved model weights
    model_path = os.path.join("saved_weight", "ghostnet_cifar10_best.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file not found at {model_path}")
        return
    
    # Set the model to evaluation mode
    model.eval()
    
    data_loader = Utils.get_cifar10_dataloader("./dataset", train=False, batch_size=1, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm.tqdm(data_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test dataset: {accuracy:.2f}%")

if __name__ == "__main__":
    test_saved_model()

