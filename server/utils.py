import torch


def get_model_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
