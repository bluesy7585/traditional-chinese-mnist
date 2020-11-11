import os
import numpy as np
import random
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from data_loader import CharDataset, get_transform, get_training_set
from PIL import Image
from models import ConvNet

class DemoDataset(data.Dataset):

    def __init__(self, img_dir, name, transform=None):

        demo_imgs = []
        for char in name:
            char_dir = os.path.join(img_dir, char)

            imgs = []
            for img in os.listdir(char_dir):
                img_path = os.path.join(char_dir, img)
                imgs.append(img_path)

            n = int(len(imgs) * 0.9)
            img_subset = imgs[n:] # last 10% to test
            demo_imgs.append(random.choice(img_subset))

        self.transform = transform
        self.imgs = demo_imgs

    def __getitem__(self, index):

        imgpath = self.imgs[index]
        img = Image.open(imgpath)

        if self.transform is not None:
            img = self.transform(img)

        return img, imgpath

    def __len__(self):
        return len(self.imgs)

    def images(self):
        return self.imgs


def demo_main(char_set, weight, name):

    _, valid_transform = get_transform()
    demo_data = DemoDataset('cleaned_data', name, valid_transform)

    test_loader = DataLoader(
        dataset=demo_data,
        batch_size=3,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    model = ConvNet(1, len(char_set))

    if torch.cuda.is_available():
        model = model.cuda()

    print('load weights from {}'.format(weight))
    model.load_state_dict(torch.load(weight))
    model.eval()

    def map_indexlist_char(ind_list, char_set):
        return ''.join([char_set[i] for i in ind_list])


    with torch.no_grad():
        for batch_idx, (x, imgpath) in enumerate(test_loader):
            if batch_idx > 0:
                break
            x = x.cuda()
            out = model(x)
            _, pred_label = torch.max(out, 1)
            pred_name = map_indexlist_char(pred_label.tolist(), char_set)

    print('name {} pred name {}'.format(name, pred_name))

    def get_concat(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    concat_im = None
    for img in demo_data.images():
        im = Image.open(img)
        if concat_im is None:
            concat_im = im
        else:
            concat_im = get_concat(concat_im, im)
    #concat_im.show()
    concat_im.save('demo.jpg')


def pick_one_valid_name(names, char_set):
    valid_names = []
    for name in names:
        name = name.strip()
        if sum([c in char_set for c in name]) == len(name):
            valid_names.append(name)
    return random.choice(valid_names)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Text Recognition demo')
    parser.add_argument("-w", dest="weight", help="model weight to resume", type=str,\
        default='')
    args = parser.parse_args()

    random_names_txt ='ch_names.txt'
    img_dir = 'cleaned_data'
    char_set, names = get_training_set(random_names_txt, img_dir)

    name = pick_one_valid_name(names, char_set)
    demo_main(char_set, args.weight, name)
