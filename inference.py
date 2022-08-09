import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()
        logger.info('initializing classifier object')
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 133)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, input):
        out = self.dropout(F.relu(self.fc1(input)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out


def net():
    logger.info('enter net function')
    logger.info('load pretrained model')
    model = resnet152(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    logger.info('set up classifier')
    model.fc = MyClassifier()
    return model


def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    return model(input_data)


def model_fn(model_dir):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load('model.pth', map_location=torch.device('cpu'))
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.exception(f"Exception in model fn {e}")
        return None