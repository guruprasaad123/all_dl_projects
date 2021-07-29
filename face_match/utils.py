import os
import cv2
from autocrop import Cropper
import logging
from functools import lru_cache
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import h5py
import numpy as np
from skimage.io import imread
from skimage.transform import resize

import traceback


def auto_crop(image_path):

    cropper = Cropper()

    # Get a Numpy array of the cropped image
    sub_face = cropper.crop(image_path)

    # print(sub_face)

    if sub_face is None:

        return face_crop(image_path)

    else:

        # print( image_path )

        # print( sub_face.shape )

        img = cv2.resize(sub_face, (128, 128),
                             interpolation=cv2.INTER_NEAREST)

        return img


def face_crop(image_path):

    try:

        facedata = "haarcascade_frontalface_alt.xml"

        cascade = cv2.CascadeClassifier(facedata)

        img = cv2.imread(image_path)

        minisize = (img.shape[1], img.shape[0])

        miniframe = cv2.resize(img, minisize)

        faces = cascade.detectMultiScale(miniframe)

        if (len(faces) > 0):

            x, y, w, h = faces[0].tolist()

            sub_face = img[y:y+h, x:x+w]

            img = cv2.resize(sub_face, (128, 128),
                             interpolation=cv2.INTER_NEAREST)

            return img
        else:
            return None

    except Exception as error:
        print('image : ', image_path)
        traceback.print_exc()
        return None


def pre_process_image(path, normalize=False, resize_img=False):

    # reading the image using path
    try:

        img = imread(path, plugin='pil')

        if normalize == True:
            # normalize the pixel values
            img = img/255

        if resize_img == True:
            # resizing the image to (28,28,3)
            img = resize(img, output_shape=(128, 128, 3),
                         mode='constant', anti_aliasing=True)

        # converting the type of pixel to float 32
        img = img.astype('float32')

        return img

    except ValueError as error:
        print(path)
        return None

# trainset/0012/0012_0001660/0000010.jpg

# Print iterations progress


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def create_dataset(obj=dict({}), name='dataset'):

    filename = 'post_trainset/{}.h5'.format(name)

    hf = h5py.File(filename, 'w')

    for key in obj:
        hf.create_dataset(key, data=obj[key])

    hf.close()


class DataSplit:

    def __init__(self, dataset, test_train_split=0.8, val_train_split=0.1, shuffle=False):
        self.dataset = dataset

        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_split * dataset_size))

        if shuffle:
            np.random.shuffle(self.indices)

        self.train_indices, self.test_indices = self.indices[:
                                                             test_split], self.indices[test_split:]

        # train_size = len(train_indices)

        # if val_train_split:

        # 	validation_split = int(np.floor((1 - val_train_split) * train_size))
        # 	self.train_indices, self.val_indices = self.train_indices[ : validation_split], self.train_indices[validation_split:]
        # 	self.val_sampler = SubsetRandomSampler(self.val_indices)

        # else:

        # 	self.train_indices = self.indices
        # 	self.val_sampler = None

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size=50, num_workers=4):

        logging.debug('Initializing train-validation-test dataloaders')

        self.train_loader = self.get_train_loader(
            batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(
            batch_size=batch_size, num_workers=num_workers)

        return self.train_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=self.train_sampler, shuffle=False, num_workers=num_workers)
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing test dataloader')
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
        return self.test_loader


class LfwFacepairDataset(Dataset):

    def __init__(self):

        matches = create_match('pairsDevTrain.txt')

    def create_match(self, file_name):

        file = open(file_name, 'r')
        lines = file.readlines()

        matches = list([])

        for line in lines:

            line_value = line.strip().split('\t')

            # correct match
            if len(line_value) == 3:

                folder, match_left, match_right = line_value

                obj = dict({
                    'folder': folder,
                    'match_left': match_left,
                    'match_right': match_right,
                    'is_pair': True
                })

                matches.append(obj)

            # mis-match
            elif len(line_value) == 4:

                folder, match_left, mis_folder, match_right = line_value

                obj = dict({
                    'folder': folder,
                    'mis_folder': mis_folder,
                    'match_left': match_left,
                    'match_right': match_right,
                    'is_pair': False
                })

                matches.append(obj)

        return matches


class FacepairDataset(Dataset):

    train_dir = 'trainset'

    def __init__(self, transforms=None):

        print('init')

        self.transforms = transforms

        path_exists = os.path.exists(
            os.path.join('post_trainset', 'dataset.h5'))

        if path_exists == False:

            self.create_data(self.train_dir)

            df = h5py.File('post_trainset/dataset.h5', "r")

            self.source_list = df['source'][:]

            self.destination_list = df['destination'][:]

            self.pair_list = df['pair'][:]


        else:

            df = h5py.File('post_trainset/dataset.h5', "r")

            self.source_list = df['source'][:]

            self.destination_list = df['destination'][:]

            self.pair_list = df['pair'][:]

    def __getitem__(self, index):

        src = self.source_list[index]

        dest = self.destination_list[index]

        if self.transforms:
            src = self.transforms(src)
            dest = self.transforms(dest)

        pair = self.pair_list[index]

        return (src, dest, 1 if pair == True else 0)

    def __len__(self):

        return len(self.pair_list)

    def get_pairs(self, train_dir):

        dirs = os.listdir(train_dir)

        pair_list = []

        for direc in dirs:

            sub_dirs = os.listdir(os.path.join(train_dir, direc))

            for sub_dir in sub_dirs:

                images = os.listdir(os.path.join(train_dir, direc, sub_dir))

                src_list = []
                match_list = []

                for image in images:

                    src_list.append(
                        image) if 'script' in image else match_list.append(image)

                if len(src_list) > 0 and len(match_list) > 0:

                    for src in src_list:

                        pair = [dict({
                                'src_file': src,
                                'src_path':  os.path.join(train_dir, direc, sub_dir, src),
                                'match_file': match,
                                'match_path': os.path.join(train_dir, direc, sub_dir, match),
                                'is_pair': True,
                                }) for match in match_list]

                        pair_list.extend(pair)

        len_pair_list = len(pair_list)

        return (pair_list, len_pair_list)

    def get_mis_pairs(self, train_dir, len_pair_list):

        dirs = os.listdir(train_dir)

        mis_pair_list = []

        for direc in dirs:

            sub_dirs = os.listdir(os.path.join(train_dir, direc))

            if len(mis_pair_list) >= len_pair_list:
                break

            for sub in sub_dirs:

                src_list = []
                mis_match_list = []

                others = sub_dirs.copy()
                others.remove(sub)

                images = os.listdir(os.path.join(train_dir, direc, sub))

                [src_list.append(
                    image) if 'script' in image else None for image in images]

                for other in others:
                    [
                        mis_match_list.append(os.path.join(other, image))
                        if 'script' not in image
                        else None
                        for image in os.listdir(os.path.join(train_dir, direc, other))
                    ]

                if len(mis_pair_list) >= len_pair_list:
                    break

                if len(src_list) > 0 and len(mis_match_list) > 0:

                    for src in src_list:

                        mis_pair = [dict({
                            'src_file': src,
                            'src_path':  os.path.join(train_dir, direc, sub, src),
                            'match_file': match,
                            'match_path': os.path.join(train_dir, direc, match),
                            'is_pair': False,
                        }) for match in mis_match_list]

                        mis_pair_list.extend(mis_pair)

        mis_pair_list = mis_pair_list[:len_pair_list]

        len_mis_pair_list = len(mis_pair_list)

        return (mis_pair_list, len_mis_pair_list)

    def get_list(self, pairs):

        match_left = []  # images from left
        left_names = []
        match_right = []  # images from right
        right_names = []
        # determines whether its a pair or not ( True or False )
        pair_list = []

        total = len(pairs)

        print('total : ', total)

        for idx, pair in enumerate(pairs):

            src_path = pair['src_path']

            left_image = auto_crop(src_path)

            match_path = pair['match_path']

            right_image = auto_crop(match_path)

            if left_image is not None and right_image is not None:

                left_names.append(pair['src_file'])

                right_names.append(pair['match_file'])

                match_left.append(left_image)

                match_right.append(right_image)

                is_pair = pair['is_pair']

                pair_list.append(is_pair)

            printProgressBar(idx, total, prefix='Processing ')

        return (match_left, match_right, left_names, right_names, pair_list)

    def create_data(self, train_dir):

        pair_list, len_pair_list = self.get_pairs(train_dir)

        mis_pair_list, len_mis_pair_list = self.get_mis_pairs(
            train_dir, len_pair_list)

        print('pairs ', len_pair_list)

        print('non-pairs ', len_mis_pair_list)

        # match_left , match_right , pair_list = self.get_list( pair_list + mis_pair_list )
        match_left, match_right, left_names, right_names, pair_list = self.get_list(
            pair_list + mis_pair_list)

        obj = {
            'source': match_left,
            'destination': match_right,
            'left_names': np.array(left_names, dtype='S'),
            'right_names': np.array(right_names, dtype='S'),
            'pair': pair_list
        }

        exists = os.path.exists(os.path.join('post_trainset'))

        if exists == True:

            create_dataset(obj, 'dataset')

        else:

            os.makedirs('post_trainset')

            create_dataset(obj, 'dataset')


if __name__ == '__main__':

    dataset = FacepairDataset()

    # print('length : {} '.format(len(dataset)))
