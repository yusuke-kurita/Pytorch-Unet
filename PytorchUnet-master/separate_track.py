# -*- coding: utf-8 -*-

from librosa.core import load, stft, istft
from librosa.output import write_wav
from network import UNet
import numpy as np
import os
import sys
import torch
from torch.autograd import Variable


def extract(spec, model, max_norm, fs, frame_size, shift_size):
    input = np.abs(spec[None, None, 1:, :].copy())
    input /= max_norm
    soft_mask = model(Variable(torch.from_numpy(
        input)).cuda()).data.cpu().numpy().squeeze()
    hard_mask = soft_mask > 0.5
    soft_vocal = istft(
        spec * np.vstack((np.zeros_like(spec[0, :]), soft_mask)),
        shift_size)
    soft_accom = istft(
        spec * np.vstack((np.zeros_like(spec[0, :]), 1 - soft_mask)),
        shift_size)
    hard_vocal = istft(
        spec * np.vstack((np.zeros_like(spec[0, :]), hard_mask)),
        shift_size)
    hard_accom = istft(
        spec * np.vstack((np.zeros_like(spec[0, :]), 1 - hard_mask)),
        shift_size)

    return soft_vocal, soft_accom, hard_vocal, hard_accom


def main():

    # filename = 'mixture2.wav'
    filename = 'aimer/1-02 花の唄.wav'
    # filename = 'amazarashi/03 季節は次々死んでいく.wav'
    batch_length = 512
    fs = 44100
    frame_size = 4096
    shift_size = 2048
    modelname = 'model/fs%d_frame%d_shift%d_batch%d.model' % (
        fs, frame_size, shift_size, batch_length)
    statname = 'stat/fs%d_frame%d_shift%d_batch%d.npy' % (
        fs, frame_size, shift_size, batch_length)
    max_norm = float(np.load(statname))

    # load network
    model = UNet()
    model.load_state_dict(torch.load(modelname))
    model.eval()
    torch.backends.cudnn.benchmark = True

    # gpu
    if torch.cuda.is_available():
        model.cuda()
    else:
        print('gpu is not avaiable.')
        sys.exit(1)

    # load wave file
    wave = load(filename, sr=fs)[0]
    spec = stft(wave, frame_size, shift_size)
    soft_vocal, soft_accom, hard_vocal, hard_accom = extract(
        spec, model, max_norm, fs, frame_size, shift_size)
    write_wav(os.path.splitext(
        os.path.basename(filename))[0] + '_original.wav', wave, fs)
    write_wav(os.path.splitext(
        os.path.basename(filename))[0] + '_soft_vocal.wav', soft_vocal, fs)
    write_wav(os.path.splitext(
        os.path.basename(filename))[0] + '_soft_accom.wav', soft_accom, fs)
    write_wav(os.path.splitext(
        os.path.basename(filename))[0] + '_hard_vocal.wav', hard_vocal, fs)
    write_wav(os.path.splitext(
        os.path.basename(filename))[0] + '_hard_accom.wav', hard_accom, fs)


if __name__ == '__main__':
    main()
