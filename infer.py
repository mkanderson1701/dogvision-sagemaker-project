import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet152

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()
        logger.debug('initializing classifier object')
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
    logger.debug('enter net function')
    logger.debug('load pretrained model')
    model = resnet152(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    logger.debug('set up classifier')
    model.fc = MyClassifier()
    return model


def predict_fn(input_data, model):
    try:
        process_image = T.Compose([
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]),
        ])
    except Exception as e:
        logger.exception(f"Exception in predict_fn transforms: {e}")
    try:
        input = input_data.cpu().numpy()
        X = process_image(input)
        X = X.expand(1, 3, 224, 224)
        logger.debug(f'input data type {type(X)} shape {X.shape}')
        logger.debug(f'array contents: {X[0]}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = X.to(device, dtype=torch.float32)
        model.to(device)
        model.eval()
        return model(X)
    except Exception as e:
        logger.exception(f"Exception in predict_fn: {e}")
        return None
    

def model_fn(model_dir):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = resnet152()
        model.fc = MyClassifier()
        with open(os.path.join(model_dir, 'model.pt'), "rb") as f:
            model.load_state_dict(torch.load(f))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.exception(f"Exception in model_fn: {e}")
        return None