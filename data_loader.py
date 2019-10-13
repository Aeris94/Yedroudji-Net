import torch
import torchvision
import torchvision.transforms as transforms
import net_config

def get_train_data_loader():
    folder_path = net_config.train_data_folder
    train_data_loader = load_data(folder_path)

    return train_data_loader

def get_test_data_loader():
    folder_path = net_config.test_data_folder
    test_data_loader = load_data(folder_path)

    return test_data_loader

def load_data(folder_path):
    dataset = torchvision.datasets.ImageFolder(
        root=folder_path,
        transform=torchvision.transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ]))
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=net_config.batch_size,
        num_workers=net_config.num_workers,
        shuffle=net_config.shuffle)
    
    return dataloader
