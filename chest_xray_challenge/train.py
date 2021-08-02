import torch
import torch.nn as nn
from torchvision import transforms, utils

import os
from dataset import ChestDataset , BalancedDataset
from model import inception_v3
from train_utils import run

from config import config

if __name__ == '__main__':

    BATCH_SIZE = config.BATCH_SIZE
    NUM_WORKERS = config.NUM_WORKERS

    label_dim = 2

    train_path = os.path.join('data','chest_xray','train')

    transform = transforms.Compose([
         transforms.ToTensor(),
     ])

    test_path = os.path.join('data','chest_xray','test')

    val_path = os.path.join('data','chest_xray','val')

    train_dataset = ChestDataset(train_path,transform=transform)

    test_dataset = ChestDataset(test_path,transform=transform)

    val_dataset = ChestDataset(val_path,transform=transform)

    balanced_trainset = BalancedDataset(train_dataset)

    train_loader = balanced_trainset.get_train_loader(
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=NUM_WORKERS,
                                                    )


    balanced_testset = BalancedDataset(test_dataset)

    test_loader = balanced_testset.get_train_loader(
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=NUM_WORKERS,
                                                    )

    balanced_valset = BalancedDataset(val_dataset)

    val_loader = balanced_valset.get_train_loader(
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=NUM_WORKERS,
                                                    )


    model = inception_v3( aux_logits=True , num_classes=label_dim ).cuda()

    run( train_loader , test_loader , val_loader , model )
