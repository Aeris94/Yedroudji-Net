import torch
import Yedroudj_Net
import net_config

def valTrainingSaveBest(net, testloader):
    MODEL_PATH = './cifar_net.pth'
    if not net:
        net = Yedroudj_Net.Yedroudj_Net()
        net.load_state_dict(torch.load(MODEL_PATH))

    classes = ('cover', 'stego')
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

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

    # get overall accuracy or accuracy per class ??????????
    # class_correct = list(0. for i in range(net_config.batch_size))
    # class_total = list(0. for i in range(net_config.batch_size))
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(net_config.batch_size):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1

    # for i in range(2):
    #     print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    return accuracy
