import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import timm


def get_config():
    _config = {
        "learning_rate": 0.001,
        "model_name": "resnet18",
        "epochs": 10,
        "num_classes": 10,
        "num_workers": 8,
        "batch_size": 512,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    return _config


def load_model(num_classes, model_name):
    return timm.create_model(model_name, num_classes=num_classes, pretrained=True)


def train_one_epoch(epoch, device, model, optimizer, loss_function, dataloader):
    model.train()

    pbar = tqdm(dataloader)
    running_loss, data_count = 0.0, 0
    for batch in pbar:
        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss = (running_loss * data_count + loss.item())
        data_count += len(batch)
        running_loss /= data_count
        pbar.set_postfix({
            'epoch': epoch,
            'loss': running_loss
        })
    

def model_test(device, model, dataloader):
    with torch.no_grad():
        model.eval()
        pbar = tqdm(dataloader)

        correct = 0
        total = 0
        for batch in pbar:
            images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'acc': correct / total})
    

def main():
    config = get_config()

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                                  train=True, 
                                                  transform=train_transforms,
                                                  download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                 train=False,
                                                 transform=test_transforms,
                                                 download=True)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config['batch_size'],
                                  num_workers=config['num_workers'],
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=config['batch_size'],
                                 num_workers=config['num_workers'],
                                 shuffle=False)
    
    device = torch.device(config['device'])
    
    model = load_model(num_classes=config['num_classes'], model_name=config['model_name'])
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    for epoch in range(config['epochs']):
        train_one_epoch(epoch, device, model, optimizer, loss_fn, train_dataloader)
    
    model_test(device, model, test_dataloader)

if __name__=='__main__':
    main()
