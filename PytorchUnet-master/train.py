# -*- coding: utf-8 -*-
import argparse
import os
from network import UNet
import numpy as np
from random import sample
import sys
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch import nn
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


class MagSpecDataset(Dataset):
    def __init__(self, filelist, transform=None):
        self.filelist = filelist
        self.transform = transform

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        mag_spec = np.load(self.filelist[index])
        silence = (mag_spec[1] > 1e-2).astype('float32')
        if self.transform:
            mag_spec = self.transform(mag_spec)
        return mag_spec[1], mag_spec[0], silence


def main(args):

    modelname = os.path.join(args.dst_dir, os.path.splitext(args.src_file)[0])

    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir)

    # define transforms
    max_norm = float(np.load(args.stats_file))
    transform = transforms.Compose([
        lambda x: x / max_norm])

    # load data
    with open(args.src_file, 'r') as f:
        files = f.readlines()
    filelist = [file.replace('\n', '') for file in files]

    # define sampler
    index = list(range(len(filelist)))
    train_index = sample(index, round(len(index) * args.ratio))
    valid_index = list(set(index) - set(train_index))
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    # define dataloader
    trainset = MagSpecDataset(filelist, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
        num_workers=args.num_worker)
    valid_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=1, shuffle=False, sampler=valid_sampler,
        num_workers=args.num_worker)

    # fix seed
    torch.manual_seed(args.seed)

    # define network
    model = UNet()

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # define loss
    criterion = nn.L1Loss(size_average=False)

    # gpu
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    else:
        print('gpu is not avaiable.')
        sys.exit(1)

    # training
    for epoch in range(args.num_epoch):

        model.train()
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, targets, silence = data

            # wrap them in Variable
            inputs = Variable(inputs[:, None, ...]).cuda()
            targets = Variable(targets[:, None, ...]).cuda()
            silence = Variable(silence[:, None, ...]).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            batch_train_loss = criterion(
                inputs * outputs * silence, targets * silence)
            batch_train_loss.backward()
            optimizer.step()

            # print statistics
            train_loss += batch_train_loss.item()

        model.eval()
        valid_loss = 0.0
        for i, data in enumerate(valid_loader):
            inputs, targets, silence = data

            # wrap them in Variable
            inputs = Variable(inputs[:, None, ...]).cuda()
            targets = Variable(targets[:, None, ...]).cuda()
            silence = Variable(silence[:, None, ...]).cuda()

            outputs = model(inputs)
            batch_valid_loss = criterion(
                inputs * outputs * silence, targets * silence)

            # print statistics
            valid_loss += batch_valid_loss.item()
        print('[{}/{}] training loss: {:.3f}; validation loss: {:.3f}'.format(
              epoch + 1, args.num_epoch, train_loss, valid_loss))

        # save model
        if epoch % args.num_interval == args.num_interval - 1:
            torch.save(
                model.state_dict(),
                modelname + '_batch{}_ep{}.model'.format(
                    args.batch_length, epoch + 1))

    torch.save(model.state_dict(), modelname + '.model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='convert waveform to magnitude spectrograms')
    parser.add_argument(
        '--src_file', type=str, default=None,
        help='list of filename of .wav files')
    parser.add_argument(
        '--stats_file', type=str, default=None,
        help='stat file')
    parser.add_argument(
        '--dst_dir', default='model',
        help='directory to save model')
    parser.add_argument(
        '--batch_length', type=int, default=256,
        help='batch length')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='mini batch size')
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='learning rate')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='seed')
    parser.add_argument(
        '--num_epoch', type=int, default=200,
        help='number of epoch')
    parser.add_argument(
        '--num_worker', type=int, default=8,
        help='number of workers')
    parser.add_argument(
        '--num_interval', type=int, default=10,
        help='number of interval to save intermidiate model')
    parser.add_argument(
        '--ratio', type=float, default=0.8,
        help='ratio for splitting dataset into training and validation')
    args = parser.parse_args()
    main(args)
