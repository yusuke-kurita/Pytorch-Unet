# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os


def main(args):

    with open(args.src_file, 'r') as f:
        filenames = f.readlines()
    filelist = [filename.replace('\n', '') for filename in filenames]

    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir)

    # process over all of data
    max_norm = 0.0
    for filename in filelist:
        print(filename)
        max_norm = max(max_norm, np.load(filename).max())

    filename = os.path.splitext(os.path.basename(args.src_file))[0] + '.npy'
    np.save(os.path.join(args.dst_dir, filename), max_norm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='convert waveform to magnitude spectrograms')
    parser.add_argument(
        '--src_file', type=str, required=True,
        help='list of filename of .wav files')
    parser.add_argument(
        '--dst_dir', type=str, required=True,
        help='directory to save stats')
    args = parser.parse_args()
    main(args)
