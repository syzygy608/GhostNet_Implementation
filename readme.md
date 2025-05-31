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


## References

[GhostNet Paper Github](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnet_pytorch)