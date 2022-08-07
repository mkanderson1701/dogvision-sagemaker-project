import argparse
import random
import string
import os
import numpy as np
import logging
import torch
import sys
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.models import resnet152
# from torchvision.models import ResNet152_Weights # not in docker img
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # fix pillow/pytorch import issue with some dog pics
# from awsio.python.lib.io.s3.s3dataset import S3Dataset

LDEBUG = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class MyClassifier(nn.Module):
    def __init__(self, hidden_units, dropout_p):
        super(MyClassifier, self).__init__()
        logger.info('initializing classifier object')
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 133)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, input):
        out = self.dropout(F.relu(self.fc1(input)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out


#TODO: Import dependencies for Debugging and Profiling

def test(model, test_loader, criterion):
    logger.info('enter testing function')
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_interval = 5
    test_loss = 0
    correct_total = 0
    num_tested = 0
    with torch.no_grad():
        for k, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            loss = criterion(output, targets)
            print(f'loss: {loss}')
            print(f'loss.item(): {loss.item()}')
            test_loss += loss.item()
            print(f'test_loss: {test_loss}')

            probs = F.softmax(output, dim=1)
            top_p, top_class = probs.topk(1, dim=1)
            top_class = top_class.T.squeeze()
            correct_batch = top_class.eq(targets).sum().item()
            correct_total += correct_batch
            num_tested += len(top_class)
            if k % log_interval == 0:
                logger.info(f'Testing batch {k}...')
                print(test_loss)
                print(len(test_loader))

    logger.info(f'Test set: Average loss: {test_loss / len(test_loader)}, '
                f'Accuracy: {correct_total}/{num_tested} ('
                f'{correct_total/num_tested*100:.2f}%)\n')

def train(model, train_loader, valid_loader, criterion, optimizer, args):
    logger.info('enter training function')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'torch.device set to {device}')

    model = model.to(device)
    log_interval = 5
    running_loss = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()
        for i, (data, targets) in enumerate(train_loader, 1):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                valid_loss = 0
                correct_total = 0
                num_tested = 0
                with torch.no_grad():
                    model.eval()
                    for j, (data, targets) in enumerate(valid_loader):
                        data, targets = data.to(device), targets.to(device)
                        output = model(data)
                        loss = criterion(output, targets)
                        valid_loss += loss.item()

                        probs = F.softmax(output, dim=1)
                        top_p, top_class = probs.topk(1, dim=1)
                        top_class = top_class.T.squeeze()
                        correct_batch = top_class.eq(targets).sum().item()
                        correct_total += correct_batch
                        num_tested += len(top_class)

                logger.info(f'Epoch {epoch}/{args.epochs}... '
                            f'Train loss: {running_loss / log_interval:.3f}.. '
                            f'Validation loss: {valid_loss / len(valid_loader):.3f}.. '
                            f'Validation accuracy: {correct_total / num_tested:.3f}')
                running_loss = 0
        end_time = time.time()
        logger.info(f'Epoch duration: {end_time - start_time}')


def net(args):
    logger.info('enter net function')
    logger.info('load pretrained model')
    # weights = ResNet152_Weights.IMAGENET1K_V2
    # weights = 'IMAGENET1K_V2'
    # sagemaker pytorch too old for the weights param
    model = resnet152(pretrained=True)
    
    # dont re-train the main network
    for param in model.parameters():
        param.requires_grad = False

    # Replace fc layer for trained resnet
    logger.info('set up classifier')
    model.fc = MyClassifier(args.hidden_units, args.dropout)

    return model


def create_data_loader(args):
    logger.info('creating data loaders')
    train_transforms = torchvision.transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ]
    )
    vt_transforms = torchvision.transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ]
    )

    print(f'LDEBUG: {LDEBUG}')

    if LDEBUG:
        train_dir = './dogImages/train'
        logger.debug(train_dir)
        valid_dir = './dogImages/valid'
        test_dir = './dogImages/test'
    else:
        train_dir = os.environ['SM_CHANNEL_TRAIN']
        logger.debug(train_dir)
        valid_dir = os.environ['SM_CHANNEL_VALID']
        test_dir = os.environ['SM_CHANNEL_TEST']

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=vt_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=vt_transforms)
    class_to_idx = train_data.class_to_idx

    pin_mem = False
    if torch.cuda.is_available():
        logger.info('cuda enabled, pinning memory for dataloaders')
        pin_mem = True

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
            pin_memory=pin_mem, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size,
            pin_memory=pin_mem, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
            pin_memory=pin_mem, shuffle=True)

    return train_loader, valid_loader, test_loader, class_to_idx


def save_model(model, optimizer, args):
    filepath = os.path.join(args.model_dir, args.model_name)
    logger.info('Saving model to {filepath}...')
    torch.save({
        'model_state_dict': model.fc.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }, filepath)


def main(args):
    logger.info('enter main')
    logger.info(args)

    # Initialize the model
    model=net(args)
    
    # CEL for classification
    logger.info('configure loss, optimizer')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    # gpu if youve got em
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info('cuda device ENABLED')
    else:
        device = torch.device('cpu')
        logger.info('cuda device NOT FOUND')

    model.to(device)

    train_loader, valid_loader, test_loader, cdict = create_data_loader(args)

    # train model
    logger.info('start training')
    train(model, train_loader, valid_loader, criterion, optimizer, args)
    
    logger.info('start testing run')
    test(model, test_loader, criterion)
    
    # Save the trained model
    save_model(model, optimizer, args)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ResNet-based dog classifier')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--hidden-units', type=int, default=256, metavar='N',
                        help='number of classifier hidden units (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='dropout rate for hidden layer (default: 0.0)')
    if LDEBUG:
        parser.add_argument('--model-dir', type=str, default='./')
    else:
        parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    parser.add_argument('--num-gpus', type=int, default=0)
    randstr = ''

    for i in range(0, 8):
        randstr += random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits)

    parser.add_argument('--model_name', type=str,
                        default=f'model_{randstr}.pth', metavar='N')
    args = parser.parse_args()
    
    main(args)
