import logging

import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py#L118
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _load_hyperparameters(hyperparameters):
    # number of data loading workers (default: 2)
    workers = hyperparameters.get('workers', 2)
    # number of total epochs to run (default: 2)
    epochs = hyperparameters.get('epochs', 2)
    # batch size (default: 4)
    batch_size = hyperparameters.get('batch_size', 4)
    # initial learning rate (default: 0.001)
    lr = hyperparameters.get('lr', 0.001)
    # momentum (default: 0.9)
    momentum = hyperparameters.get('momentum', 0.9)
    # distributed backend
    backend = hyperparameters.get('dist_backend', 'gloo')

    logger.info(
        'workers: {}, epochs: {}, batch_size: {}'.format(workers, epochs, batch_size) +
        ' lr: {}, momentum: {}, backend: {}'.format(lr, momentum, backend)
    )
    return workers, epochs, batch_size, lr, momentum, backend


def train(channel_input_dirs, num_gpus, hosts, host_rank, master_addr, master_port,
          hyperparameters):
    logger.info("Loading HyperParameters")
    workers, epochs, batch_size, lr, momentum, backend = _load_hyperparameters(hyperparameters)

    training_dir = channel_input_dirs['training']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Device Type: {}".format(device))

    logger.info("Loading Cifar10 dataset")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=training_dir, train=True,
                                            download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=workers)

    testset = torchvision.datasets.CIFAR10(root=training_dir, train=False,
                                           download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=workers)

    logger.info("Model loaded")
    model = Net().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(0, epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    return model


def model_fn(model_dir):
    logger.info('model_fn')
    model = Net()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model
