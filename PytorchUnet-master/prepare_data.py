# -*- coding: utf-8 -*-

import argparse
from librosa.output import write_wav
import musdb
import numpy as np
import os
from random import sample, shuffle


def monauralize(x):
    mono_x = np.concatenate((x[:, 0], x[:, 1]), 0)
    return mono_x


def prepare_wav(track_list, sub_dirname, args):

    dirpath = os.path.join(args.dst_dir, sub_dirname)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    for track in track_list:
        name = track.name
        rate = track.rate

        vocal = monauralize(track.sources['vocals'].audio)
        mix = monauralize(track.audio)

        path = os.path.join(dirpath, name)
        print(path)
        write_wav(path + '.wav', np.stack((vocal, mix)), rate)


def main(args):

    mus = musdb.DB(root_dir=args.src_dir)
    track_list = mus.load_mus_tracks()
    test_list = [track for track in track_list if track.subset == 'test']
    train_list = [track for track in track_list if track.subset == 'train']

    # prepare_wav(test_list, 'test', args)
    # prepare_wav(train_list, 'train', args)

    # argument mixture
    ch = [0, 1]
    for i in range(args.num_arg):
        bass_list = sample(train_list, len(train_list))
        drums_list = sample(train_list, len(train_list))
        other_list = sample(train_list, len(train_list))
        for j in range(len(train_list)):
            name = train_list[j].name + '_argumented{}'.format(i)
            rate = train_list[j].rate

            vocal = train_list[j].sources['vocals'].audio
            bass = bass_list[j].sources['bass'].audio[:, shuffle(ch)].squeeze()
            drums = drums_list[j].sources['drums'].audio[:, shuffle(ch)].squeeze()
            other = other_list[j].sources['other'].audio[:, shuffle(ch)].squeeze()
            min_len = min([len(bass), len(drums), len(vocal), len(other)])
            # vocal
            vocal = monauralize(vocal[:min_len, :])
            # accom
            accom = monauralize(
                bass[:min_len, :] + drums[:min_len, :] + other[:min_len, :])
            # mix
            mix = vocal + accom

            path = os.path.join(args.dst_dir, 'train', name)
            print(path)
            write_wav(path + '.wav', np.stack((vocal, mix), 1), rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='convert waveform to magnitude spectrograms')
    parser.add_argument(
        '--src_dir', default='/database/musicData/MUSDB18',
        help='directory of MUSDB18')
    parser.add_argument(
        '--dst_dir', default='wav',
        help='directory to save waveform')
    parser.add_argument(
        '--num_arg', type=int, default=0,
        help='number of data argumentation per track')
    args = parser.parse_args()
    main(args)
