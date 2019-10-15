import torch
import torch.nn as nn
import torch.optim as optim
import Yedroudj_Net
import net_config

def valTrainingSaveBest(net, testloader):
    classes = ('cover', 'stego')
    dataiter = iter(testloader)

    # jak to przekazac na GPU???
    images, labels = dataiter.next()

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)


    #print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
    #                          for j in range(net_config.batch_size)))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100* correct/total
    print('Accuracy of the network on the test images: %d %%' % (accuracy))

    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(net_config.batch_size))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(2):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(2):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        #index = '%d' %(i)
        #accuracy.index =  100 * class_correct[i] / class_total[i]

    return accuracy

def valTrainedModel():
    classes = ('cover', 'stego')
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    net = Yedroudj_Net.Yedroudj_Net()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(net_config.batch_size)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(net_config.batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(2):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
