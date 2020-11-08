import os
import numpy as np
import torch.utils.data as data
import torch.tensor
import torchvision.transforms as T
from PIL import Image
from read_names import get_available_char_set

class CharDataset(data.Dataset):

    def __init__(self, data_dir, char_set, transform=None, train=True):

        for char in char_set:
            char_dir = os.path.join(data_dir, char)
            assert os.path.exists(char_dir), f'{char_dir} not exist.'

        self.transform = transform
        self.class_num = len(char_set)

        self.imgs = []
        self.label = []
        for i, char in enumerate(char_set):
            char_dir = os.path.join(data_dir, char)

            imgs = []
            for img in os.listdir(char_dir):
                imgs.append(img)

            # 90% for training
            n = int(len(imgs) * 0.9)
            img_subset = imgs[:n] if train else imgs[n:]

            for img in img_subset:
                img_path = os.path.join(char_dir, img)
                self.imgs.append(img_path)
                self.label.append(i)

        self.label = torch.tensor(self.label)

    def __getitem__(self, index):

        imgpath, gt_label = self.imgs[index], self.label[index]
        img = Image.open(imgpath)

        if self.transform is not None:
            img = self.transform(img)

        return img, gt_label, imgpath

    def __len__(self):
        return len(self.imgs)

def get_training_set(filename, img_dir):

    with open(filename,'r',encoding='utf-8') as f:
        names = f.readline().split(',')

    print('{} random names in {}'.format(len(names), filename))
    ava_set = get_available_char_set(names, img_dir)
    return ava_set, names

def get_transform():
    height = 50
    width = 50
    normalize = T.Normalize((0.5,), (0.5,))
    train_transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((height, width)),
        T.Pad(5),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalize
        ])
    valid_transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform
