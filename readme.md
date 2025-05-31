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
    --model_name GhostResNet56 or GhostNet
    --batch_size 256 \
    --epochs 200 \
    --learning_rate 0.1 \
    --weight_decay 1e-4
```

## Inference

To perform inference using the trained GhostNet model, you can use the following script.

```bash
python Main.py \
    --model_name GhostResNet56 or GhostNet
```

## References

[GhostNet Paper Github](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnet_pytorch)

[CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

[CIFAR10 Classifer Using VGG19](https://medium.com/@charlie910417/cifar10-classifer-using-vgg19-d948a4df6b20)

[CIFAR-10 Normalization](https://github.com/kuangliu/pytorch-cifar/issues/19)