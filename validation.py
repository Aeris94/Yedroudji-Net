import torch
import Yedroudj_Net
import net_config
from torch.utils.tensorboard import SummaryWriter
import torchvision


def valTrainingSaveBest(net, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = './cifar_net.pth'
    if not net:
        net = Yedroudj_Net.Yedroudj_Net()
        net.to(device)
        net.load_state_dict(torch.load(MODEL_PATH))

    #classes = ('cover', 'stego')
    dataiter = iter(testloader)
    data = dataiter.next()
    images, labels = data[0].to(device), data[1].to(device)

    writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    img_grid = torchvision.utils.make_grid(images)
    #matplotlib_imshow(img_grid, one_channel=True)
    writer.add_image('four_fashion_mnist_images', img_grid)
    writer.add_graph(net, images)

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
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
