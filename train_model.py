import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.models import resnet152, ResNet152_Weights
from torch.utils.data import DataLoader
from awsio.python.lib.io.s3.s3dataset import S3Dataset


class MyClassifier(nn.Module):

    def __init__(self, hidden_units, dropout_p):
        super(nn.Module, self).__init__()
        self.fc1 = nn.Linear(2048, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 133)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, input):
        out = self.dropout(F.relu(self.fc1(input)))
        out = self.fc2(out)
        return out


#TODO: Import dependencies for Debugging and Profiling

def test(model, test_loader, loss_criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass
    
def net():
    weights = ResNet152_Weights.IMAGENET1K_V2
    model = resnet152(weights=weights)
    
    # dont re-train the main network
    for param in model.parameters():
        param.requires_grad = False

    # Replace fc layer for trained resnet
    model.fc = MyClassifier(args.hidden_units, args.dropout)

    return model

def create_data_loader():
    train_transforms = torchvision.transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ]
    )
    vt_transforms = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ]
    )

    train_dir = os.environ['SM_CHANNEL_TRAIN'])
    valid_dir = os.environ['SM_CHANNEL_VAL'])
    test_dir = os.environ['SM_CHANNEL_TEST'])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=vt_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=vt_transforms)
    class_to_idx = train_data.class_to_idx

    pin_mem = False
    if torch.cuda.is_available():
        pin_mem = True

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
            pin_memory=pin_mem, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size,
            pin_memory=pin_mem, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
            pin_memory=pin_mem, shuffle=True)

    return train_loader, valid_loader, test_loader, class_to_idx


def main(args):

    # Initialize the model
    model=net()
    
    # CEL for classification, Adam
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    # gpu if youve got em
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dtrain, dvalid, dtest, cdict = create_data_loader()

    # train model
    model=train(model, dtrain, dvalid, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, dtest, loss_criterion)
    
    # Save the trained model
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch ResNet-based dog classifier')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--learning-rate', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--hidden-units', type=int, default=150, metavar='N',
                        help='number of classifier hidden units (default: 150)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='dropout rate for hidden layer (default: 0.0)')
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args = parser.parse_args()

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    
    main(args)
