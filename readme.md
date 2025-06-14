# GhostNet Implementation in PyTorch

This repository contains a PyTorch implementation of GhostNet, a lightweight convolutional neural network designed for efficient mobile vision applications. GhostNet introduces the concept of "ghost modules" to generate more feature maps from cheap operations, significantly reducing computational cost while maintaining performance.

## Environment Setup

我們使用 `uv` 進行環境管理，請確保你已經安裝了 `uv`。

```bash
uv venv
```

接著，啟動虛擬環境：

```bash
uv .venv/bin/activate
```

最後，安裝所需的依賴：

```bash
uv pip install -r requirements.txt
```

## Training

To train the GhostNet model, you can use the provided training script. Make sure to adjust the hyperparameters and dataset paths as needed.

```bash
python model_training/Training.py \
    --model_name GhostResNet56 | GhostNet | GhostVGG16
```

## Inference

To perform inference using the trained GhostNet model, you can use the following script.

```bash
python Main.py \
    --model_name GhostResNet56 | GhostNet | GhostVGG16
```

## References

[GhostNet Paper Github](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnet_pytorch)

[CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

[CIFAR10 Classifer Using VGG19](https://medium.com/@charlie910417/cifar10-classifer-using-vgg19-d948a4df6b20)

[CIFAR-10 Normalization](https://github.com/kuangliu/pytorch-cifar/issues/19)

## Dataset Citation

This project uses the CIFAR-10 dataset, which includes 60,000 32x32 color images in 10 classes.  
Citation: Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. Technical Report, University of Toronto. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf