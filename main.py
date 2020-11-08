import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import CharDataset, get_transform, get_training_set
from models import ConvNet


def main(char_set, weight, train_epoch):

    train_transform, valid_transform = get_transform()
    char_dataset = CharDataset('cleaned_data', char_set, transform=train_transform)

    train_loader = DataLoader(
        dataset=char_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    t_char_dataset = CharDataset('cleaned_data', char_set, transform=valid_transform, train=False)

    test_loader = DataLoader(
        dataset=t_char_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    model = ConvNet(1, len(char_set))

    if len(weight) > 0:
        model.load_state_dict(torch.load(weight))
        print('load weights from {}'.format(weight))
    else:
        print('train from scratch')


    if torch.cuda.is_available():
        model = model.cuda()

    #print(f'model.training: {model.training}')

    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    #lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

    for epoch in range(train_epoch):

        model.train()
        ave_loss = 0
        # trainning
        for step, (x, gt_label, imgpath) in enumerate(train_loader):

            optimizer.zero_grad()
            x, gt_label = x.cuda(), gt_label.cuda()
            out = model(x)
            loss = criterion(out, gt_label)

            loss.backward()
            optimizer.step()
            ave_loss = ave_loss * 0.9 + loss.item() * 0.1

            if (step + 1) == len(train_loader):
                print('epoch: {}, train loss: {:.6f}'.format(\
                    epoch, ave_loss))

        correct_cnt = 0
        total_cnt = 0
        accum_loss = 0
        epsilon = 10 ** -20
        # eval
        with torch.no_grad():

            model.eval()
            for step, (x, gt_label, imgpath) in enumerate(test_loader):
                #print(batch_idx)
                x, gt_label = x.cuda(), gt_label.cuda()
                out = model(x)
                loss = criterion(out, gt_label)
                _, pred_label = torch.max(out, 1)
                total_cnt += x.size()[0]
                correct_cnt += (pred_label == gt_label).sum()

                loss_v = loss.item()
                accum_loss += loss_v

            val_loss = accum_loss / (len(test_loader)+ epsilon)
            acc = correct_cnt * 1.0 / total_cnt
            print('test acc: {:.3f} {}-{} val loss: {:.6f}'.format(acc\
            ,correct_cnt, total_cnt, val_loss))

    WEIGHT_DIR = 'weights'
    if not os.path.exists(WEIGHT_DIR):
        os.mkdir(WEIGHT_DIR)

    save_name = './{}/{}_val_loss{:.3f}'.format(WEIGHT_DIR, model.name(), val_loss)
    torch.save(model.state_dict(), save_name)


def test(char_set, weight):

    _, valid_transform = get_transform()
    char_dataset = CharDataset('cleaned_data', char_set, transform=valid_transform, train=False)

    test_loader = DataLoader(
        dataset=char_dataset,
        batch_size=32,
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

    correct_cnt = 0
    total_cnt = 0
    accum_loss = 0
    aver_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        #print(len(test_loader))
        for step, (x, gt_label, imgpath) in enumerate(test_loader):

            x, gt_label = x.cuda(), gt_label.cuda()
            out = model(x)
            loss = criterion(out, gt_label)
            _, pred_label = torch.max(out, 1)

            total_cnt += x.size()[0]
            correct_cnt += (pred_label == gt_label).sum()

            loss_v = loss.item()
            aver_loss = aver_loss * 0.9 + loss_v * 0.1
            accum_loss += loss_v

            # if (step + 1) % 10 == 0 or (step + 1) == len(test_loader):
            #     print('batch: {}, val loss: {:.6f}'.format(\
            #         step + 1, aver_loss))
        acc = correct_cnt * 1.0 / total_cnt
        print('test acc: {:.3f} {}-{}'.format(acc, correct_cnt, total_cnt))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Text Recognition main')
    parser.add_argument("-t", dest="test", help="test mode", action='store_true')
    parser.add_argument("-w", dest="weight", help="model weight to resume", type=str,\
        default='')
    parser.add_argument("-e", dest="epoch", help="epoch to train", type=int,\
        default=50)
    args = parser.parse_args()

    random_names_txt ='ch_names.txt'
    img_dir = 'cleaned_data'
    char_set, _ = get_training_set(random_names_txt, img_dir)

    if args.test:
        test(char_set, args.weight)
    else:
        main(char_set, args.weight, args.epoch)
