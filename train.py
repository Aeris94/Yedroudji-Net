
import torch
import torch.nn as nn
import torch.optim as optim
import Yedroudj_Net
import net_config
import validation

def train(trainloader, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Yedroudj_Net.Yedroudj_Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), net_config.lr, net_config.momentum, weight_decay=0.0001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=net_config.epochs*0.1, gamma=0.1, last_epoch=-1)

    for epoch in range(net_config.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # print statistics
            running_loss += loss.item()
            if i % net_config.validation_freq == net_config.validation_freq - 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / net_config.validation_freq))
                validation.val-training-save-best(net, testloader)
                running_loss = 0.0

    return net