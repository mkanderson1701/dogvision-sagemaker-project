import argparse
import os
import numpy as np
import logging
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.models import resnet152, ResNet152_Weights
from torch.utils.data import DataLoader
# from awsio.python.lib.io.s3.s3dataset import S3Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class MyClassifier(nn.Module):

    def __init__(self, hidden_units, dropout_p):
        logger.info('initializing classifier object')
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
    logger.info('enter testing function')
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
        'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, valid_loader, criterion, optimizer, args):
    logger.info('enter training function')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'torch.device set to {device}')

    logger.debug(
        'Processes {}/{} ({:.0f}%) of train data'.format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        'Processes {}/{} ({:.0f}%) of test data'.format(
            len(valid_loader.sampler),
            len(valid_loader.dataset),
            100.0 * len(valid_loader.sampler) / len(valid_loader.dataset),
        )
    )

    model = model.to(device)
    log_interval = 10
    running_loss = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for i, (data, targets) in enumerate(train_loader, 1):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            running_loss += loss.item
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                test_loss = 0
                accuracy = 0
                with torch.no_grad():
                    model.eval()
                    for j, (data, targets) in enumerate(valid_loader):
                        inputs, targets = inputs.to(device), labels.to(device)
                        output = model(data)
                        valid_loss = criterion(output, labels)
                        test_loss += valid_loss.item()
                        top_p, top_class = output.topk(1, dim=1)
                        equals = top_class == targets.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{args.epochs}.. "
                f"Train loss: {running_loss/log_interval:.3f}.. "
                f"Validation loss: {test_loss/len(valid_loader):.3f}.. "
                f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

                logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch,
                        i * len(data),
                        len(train_loader.sampler),
                        100.0 * i / len(train_loader),
                        running_loss,
                    )
                )
                running_loss = 0
        
        test(model, valid_loader, device)


    """
    TOTO
    implement save
    """

    # save_model(model, args.model_dir)
    
def net(args):
    logger.info('enter net function')
    logger.info('load pretrained model')
    weights = ResNet152_Weights.IMAGENET1K_V2
    model = resnet152(weights=weights)
    
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

    train_dir = os.environ['SM_CHANNEL_TRAIN']
    logger.debug(train_dir)
    valid_dir = os.environ['SM_CHANNEL_VAL']
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


def main(args):
    logging.info('enter main')

    # Initialize the model
    model=net(args)
    
    # CEL for classification, Adam
    logging.info('configure loss, optimizer')
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    # gpu if youve got em
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader, valid_loader, test_loader, cdict = create_data_loader(args)

    # train model
    logging.info('start training')
    model=train(model, train_loader, valid_loader, loss_criterion, optimizer, args)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logging.info('start testing')
    test(model, dtest, loss_criterion)
    
    # Save the trained model
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)

if __name__=='__main__':

    LOCALV = True

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
    # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--num-gpus', type=int, default=0)
    args = parser.parse_args()

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    
    main(args)