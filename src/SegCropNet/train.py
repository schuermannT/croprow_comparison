import os

import torch
from model.SegCropNet.train_net import train_model
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.SegCropNet.SegCropNet import SegCropNet
from torch.utils.data import DataLoader

from torchvision import transforms

from model.utils.cli_helper import parse_args

import pandas as pd

BATCH_SIZE = 1 #16
LEARNING_RATE = 0.001
EPOCHS = 1000
MOMENTUM = 0.9
DECAY = 0.0005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    save_path = os.path.join(os.path.dirname(__file__), 'train_output')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    
    train_path = os.path.join(os.path.abspath(''),  'datasets', 'ma_dataset', 'combined', 'train')
    val_path = os.path.join(os.path.abspath(''), 'datasets', 'ma_dataset', 'combined', 'val')

    resize_height = 256
    resize_width = 512

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    target_transforms = transforms.Compose([
        Rescale((resize_width, resize_height)),
    ])

    train_dataset = TusimpleSet(train_path, transform=data_transforms['train'], target_transform=target_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TusimpleSet(val_path, transform=data_transforms['val'], target_transform=target_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    dataloaders = {
        'train' : train_loader,
        'val' : val_loader
    }
    dataset_sizes = {'train': len(train_loader.dataset), 'val': len(val_loader.dataset)}

    model = SegCropNet(arch='UNet')
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"{EPOCHS} epochs {len(train_dataset)} training samples\n")

    model, log = train_model(model, optimizer, scheduler=None, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=DEVICE, loss_type='CrossEntropyLoss', num_epochs=EPOCHS)
    df=pd.DataFrame({'epoch':[],'training_loss':[],'val_loss':[], 'binary_loss':[], 'instance_loss':[]})
    df['epoch'] = log['epoch']
    df['training_loss'] = log['training_loss']
    df['val_loss'] = log['val_loss']
    df['binary_loss'] = log['binary_loss']
    df['instance_loss'] = log['instance_loss']

    train_log_save_filename = os.path.join(save_path, 'training_log.csv')
    df.to_csv(train_log_save_filename, columns=['epoch','training_loss','val_loss','binary_loss','instance_loss'], header=True,index=False,encoding='utf-8')
    print("training log is saved: {}".format(train_log_save_filename))
    
    model_save_filename = os.path.join(save_path, 'best_model.pth')
    torch.save(model.state_dict(), model_save_filename)
    print("model is saved: {}".format(model_save_filename))


if __name__ == '__main__':
    train()
