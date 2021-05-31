import torch
import torchvision
import torch.nn as nn
import pretrainedmodels
import resnest.torch as resnest_torch

from efficientnet_pytorch import EfficientNet

from params import DEVICE, resnest50_path


def get_model(name, num_classes=1):
    """
    Loads a pretrained model. 
    Supports ResNest, ResNext-wsl, EfficientNet, ResNext and ResNet.

    Arguments:
        name {str} -- Name of the model to load

    Keyword Arguments:
        num_classes {int} -- Number of classes to use (default: {1})

    Returns:
        torch model -- Pretrained model
    """
    if "resnest" in name:
        #model = getattr(resnest_torch, name)(pretrained=True)
        #model = torch.load(resnest50_path)
        model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
    elif "wsl" in name:
        model = torch.hub.load("facebookresearch/WSL-Images", name)
    elif "resnext" in name or "resnet" in name:
        model = torch.hub.load("pytorch/vision:v0.6.0", name, pretrained=True)
    elif "efficientnet" in name:
        model = EfficientNet.from_pretrained(name)
    else:
        raise NotImplementedError

    if "efficientnet" not in name and "se" not in name:
        nb_ft = model.fc.in_features
        del model.fc
        model.fc = nn.Linear(nb_ft, num_classes)
    else:
        nb_ft = model._fc.in_features
        del model._fc
        model._fc = nn.Linear(nb_ft, num_classes)

    return model
