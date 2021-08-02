import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader , WeightedRandomSampler
from torchvision import datasets
import numpy as np

import os
from utils import pre_process_image
import random
from itertools import chain
from functools import lru_cache

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses

    print('count : ',count)

    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])

    print('weight_per_class : ',weight_per_class)

    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

class ChestDataset(Dataset):

    def __init__(self,dir_name,transform=None):

        self.dir_name = dir_name
        self.transform = None

        normal_images = os.listdir(os.path.join(self.dir_name,'NORMAL'))
        monia_images = os.listdir(os.path.join(self.dir_name,'PNEUMONIA'))

        if '.ipynb_checkpoints' in normal_images:
            normal_images.remove('.ipynb_checkpoints')

        if '.ipynb_checkpoints' in monia_images:
            monia_images.remove('.ipynb_checkpoints')

        if '.DS_Store' in normal_images:
            normal_images.remove('.DS_Store')

        if '.DS_Store' in monia_images:
            monia_images.remove('.DS_Store')

        self.all_images = list([])

        counts = [len(normal_images),len(monia_images)]

        class_weights = [ sum(counts)/c for c in counts ]

        self.sample_weights = list(
                                   chain.from_iterable(
                                            list(
                                                 map(
                                                     lambda w,c: ([w]*c),
                                                     class_weights,
                                                     counts
                                                     )
                                                 )
                                            )
                                   )

        self.all_images.extend(normal_images)
        self.all_images.extend(monia_images)


    def __len__(self):

        return len(self.all_images)

    def __getitem__(self,index):

        image_file = self.all_images[index]

        if 'person' in image_file:
            label = 1
            image = pre_process_image(
                                os.path.join(self.dir_name,'PNEUMONIA',image_file),
                                normalize=True,
                                resize_img=True
                                )


        else:
            label = 0
            image = pre_process_image(
                                os.path.join(self.dir_name,'NORMAL',image_file),
                                normalize=True,
                                resize_img=True
                                )

        image = np.transpose(image , (2,0,1))

        if self.transform:
            image = self.transform(image)

        # weight = self.sample_weights[index]

        return image , label


class BalancedDataset:

    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset

        dataset_size = len(dataset)

        # if shuffle:
        #     np.random.shuffle(self.indices)

        self.train_sampler = WeightedRandomSampler(self.dataset.sample_weights,dataset_size)

    def get_training_split_point(self):
        return len(self.train_sampler)


    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=4, num_workers=4):

        self.train_loader = torch.utils.data.DataLoader(
                                                        self.dataset,
                                                        batch_size=batch_size,
                                                        sampler=self.train_sampler,
                                                        shuffle=False,
                                                        num_workers=num_workers
                                                        )
        return self.train_loader


if __name__ == '__main__':

    train_path = os.path.join('data','chest_xray','train')

    train_dataset = ChestDataset(train_path)

    train_loader = BalancedDataset(train_dataset)

    length = len(train_dataset)

    print('length : {}'.format(length))

    idx = random.randint(0,length)

    image , label = train_dataset[idx]

    print('Image : {}'.format(image.shape))

    print('Label : {}'.format(label))

    # print('weights : ',train_dataset.sample_weights[idx] )

    # print('Weight : {}'.format(weight))

    # dataset_train = datasets.ImageFolder(train_path)

    # # print('dataset_train ',dataset_train.imgs)
    # print('classes ',len(dataset_train.classes))

    # # For unbalanced dataset we create a weighted sampler
    # weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))

    # weights = torch.DoubleTensor(weights)

    # print('weight ',torch.unique(weights))

    # print('weight : ',weights[0:5])
