# -*- coding: utf-8 -*-
import argparse
from librosa.core import load, stft
import multiprocessing as mp
import numpy as np
import os


def make_mag_spec(filelist, args):

    batch_length = args.batch_length
    for filename in filelist:
        basename = os.path.splitext(os.path.basename(filename))[0]
        # load wav
        wav = load(filename, args.fs, mono=False)[0]
        vocal_wav = wav[0].copy()
        mix_wav = wav[1].copy()
        # make magnitude spectrogram
        vocal_spec = stft(vocal_wav, args.frame_size, args.shift_size)
        mix_spec = stft(mix_wav, args.frame_size, args.shift_size)
        spec = np.stack((vocal_spec, mix_spec))
        mag_spec = np.abs(spec[:, 1:, :]).copy()
        for seg in range(mag_spec.shape[-1] // args.batch_length):
            seg_filename = basename + '_seg{}.npy'.format(seg)
            seg_mag_spec = \
                mag_spec[..., seg * batch_length:(seg + 1) * batch_length]
            np.save(os.path.join(args.dst_dir, seg_filename), seg_mag_spec)


def main(args):

    with open(args.src_file, 'r') as f:
        filenames = f.readlines()
    filelist = [filename.replace('\n', '') for filename in filenames]

    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir, exist_ok=True)

    # split list
    filelists = np.array_split(filelist, args.num_worker)
    filelists = [filelist.tolist() for filelist in filelists]

    processes = []
    for f in filelists:
        p = mp.Process(target=make_mag_spec, args=(f, args,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='convert waveform to magnitude spectrograms')
    parser.add_argument(
        '--src_file', type=str, required=True,
        help='list of filename of .wav files')
    parser.add_argument(
        '--dst_dir', type=str, required=True,
        help='directory to save spectrogram')
    parser.add_argument(
        '--fs', type=int, default=8192,
        help='sampling frequency')
    parser.add_argument(
        '--frame_size', type=int, default=1024,
        help='frame size of stft')
    parser.add_argument(
        '--shift_size', type=int, default=768,
        help='shift size of stft')
    parser.add_argument(
        '--batch_length', type=int, default=128,
        help='batch length')
    parser.add_argument(
        '--num_worker', type=int, default=30,
        help='number of parallel jobs')
    args = parser.parse_args()

    main(args)
