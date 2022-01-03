import numpy as np
import torch


from torchvision.models.resnet import resnet18
from torchvision.models.segmentation import fcn, fcn_resnet50


def main():
    print(resnet18(pretrained=True))


if __name__ == "__main__":
    main()
