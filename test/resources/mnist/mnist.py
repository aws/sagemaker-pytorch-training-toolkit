# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import ceil
from torchvision import datasets, transforms
from torch.autograd import Variable

logger = logging.getLogger("customer_script:mnist")
logger.setLevel(logging.DEBUG)


def _load_hyperparameters(hyperparameters):
    logger.info("Load hyperparameters")
    # batch size for training (default: 64)
    batch_size = hyperparameters.get('batch_size', 60)
    # batch size for testing (default: 1000)
    test_batch_size = hyperparameters.get('test_batch_size', 1000)
    # number of epochs to train (default: 10)
    epochs = hyperparameters.get('epochs', 3)
    # learning rate (default: 0.01)
    lr = hyperparameters.get('lr', 0.01)
    # SGD momentum (default: 0.5)
    momentum = hyperparameters.get('momentum', 0.5)
    # random seed (default: 1)
    seed = hyperparameters.get('seed', 1)
    # how many batches to wait before logging training status
    log_interval = hyperparameters.get('log_interval', 100)
    logger.info(
        "batch_size: {}, test_batch_size: {}, epochs: {}, lr: {}, momentum: {}, seed: {}, log_interval: {}".format(
            batch_size, test_batch_size, epochs, lr, momentum, seed, log_interval
        ))
    return batch_size, test_batch_size, epochs, lr, momentum, seed, log_interval


def _get_train_data_loader(batch_size, training_dir,  **kwargs):
    logger.info("Get train data loader")
    dataset = datasets.MNIST(training_dir, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=True, **kwargs)


def _get_test_data_loader(test_batch_size, training_dir, **kwargs):
    logger.info("Get test data loader")
    return torch.utils.data.DataLoader(
        datasets.MNIST(training_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        logger.info("Create neural network module")

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(channel_input_dirs, num_gpus, hyperparameters):
    training_dir = channel_input_dirs['training']
    cuda = num_gpus > 0
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    batch_size, test_batch_size, epochs, lr, momentum, seed, log_interval = _load_hyperparameters(hyperparameters)

    # set the seed for generating random numbers
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    train_loader = _get_train_data_loader(batch_size, training_dir, **kwargs)
    test_loader = _get_test_data_loader(test_batch_size, training_dir, **kwargs)

    model = Net()
    if cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))

        test(model, test_loader, cuda)
    return model


def test(model, test_loader, cuda):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    logger.debug('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def save(model, model_dir):
    """Default logic to save a model to self.model_dir folder (/opt/ml/model).
    This function is called when a customer script does not provide a save() function
        Args:
            model : module to save."""
    logger.info("Save the model to model_dir: {}".format(model_dir))
    path = os.path.join(model_dir, 'model')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)
