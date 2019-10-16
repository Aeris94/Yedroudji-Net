
import torch
import torch.nn as nn
import torch.optim as optim
import Yedroudj_Net
import net_config
import validation
import math

def trainYeroudj(trainloader, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Yedroudj_Net.Yedroudj_Net()
    net.apply(weights_init)
    net.to(device)
    best_accuracy = 0
    running_accuracy = 0

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), net_config.lr, net_config.momentum, weight_decay=net_config.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=net_config.epochs*0.1, gamma=net_config.gamma, last_epoch=-1)

    for epoch in range(net_config.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # print statistics every net_config.validation_freq iterations in epoch
            # save if model is better than previous
            running_loss += loss.item()
            if i % net_config.validation_freq == net_config.validation_freq -1:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / net_config.validation_freq))
                running_accuracy = validation.valTrainingSaveBest(net, testloader)
                if running_accuracy >= best_accuracy:
                    best_accuracy = running_accuracy
                    saveBestModelInfo(net, epoch, i, running_loss / net_config.validation_freq, best_accuracy)
                running_loss = 0.0

def saveBestModelInfo(net, epoch, i, loss, accuracy):
    MODEL_PATH = './cifar_net.pth'
    INFO_PATH = './info_net.txt'
    info_file = open(INFO_PATH, 'a')
    info_file.writelines('[%d, %5d] loss: %.3f accuracy %.3f\n' %(epoch + 1, i + 1, loss, accuracy))
    info_file.close()
    torch.save(net.state_dict(), MODEL_PATH)

def weights_init(m):
    # weight inicialization - xavier normal
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)