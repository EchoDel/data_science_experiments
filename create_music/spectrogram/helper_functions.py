import pandas as pd
import numpy as np
from pathlib import Path
import random as rand
import librosa
import torch
import torch.nn as nn
from scipy.signal.windows import hamming
from torch import tensor


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    final_output = int(round(window_len/2))
    return y[final_output:(final_output + len(x))]


def load_sound_file(path, sr):
    try:
        data, rate = librosa.load(path,
                                  sr=sr)

    except Exception as e:
        print(f"Reading of sample {path.name} failed")
        print(e)

    return data, rate


def create_output(model: nn.Module, switches: list):
    n_inputs = list(model.parameters())[0].shape[1]
    inputs = np.zeros(n_inputs)

    for key, value in switches:
        inputs[key] = value

    inputs = tensor(inputs).float()

    output = model.features(inputs)
    output = model.classifier(output)
    return output.detach().numpy()


class SongIngestion(torch.utils.data.Dataset):
    def __init__(self, metadata, sample_length, transformations, sr, window_length,
                 y_size, n_mels, maximum_sample_location, seed=1994):
        super(SongIngestion).__init__()
        self.metadata = metadata
        self.n_mels = n_mels
        self.maximum_sample_location = maximum_sample_location

        rand.seed(seed)
        np.random.seed(seed)

        self.start = 0
        self.end = self.metadata.shape[0]
        self.y_size = y_size
        self.sound_files = {}
        self.n = 0
        self.print_n = 0
        self.length = sample_length
        self.transformations = transformations
        self.sr = sr
        self.window_length = window_length
        self.window = hamming(self.window_length, sym=False)

    def onehot(self, n, maximum):
        output = np.zeros(maximum)
        output[n] = 1
        return output

    def load_sound_file(self, itemid):
        if itemid not in self.sound_files:
            if self.print_n % 100 == 0:
                self.metadata.iloc[itemid, -1]
            self.print_n += 1
            self.sound_files[itemid] = load_sound_file(self.metadata.iloc[itemid, -1], self.sr)
        return self.sound_files[itemid]

    def load_spectrogram(self, data, rate):
        frequency_graph = librosa.feature.melspectrogram(data,
                                                         sr=rate,
                                                         n_fft=self.window_length,
                                                         hop_length=round(0.25 * self.window_length),
                                                         window=self.window,
                                                         n_mels=self.n_mels)
        return frequency_graph

    def pad_spectrogram(self, sample):
        x, y = sample.shape
        return np.pad(sample, ((0, self.n_mels - x), (0, self.y_size - y)))

    def subsample(self, sample):
        # Added a statement to correctly return samples under y_size long
        if sample.shape[1] < self.y_size:
            return self.pad_spectrogram(sample), 0
        sample_length = sample.shape[1]
        start = rand.randint(0, sample_length - self.y_size)
        start = min(self.maximum_sample_location - 1, start)
        sample = sample[:, start:(start + self.y_size)]
        return sample, start

    def load_sample(self, index):
        sample, rate = self.load_sound_file(index)
        sample = self.load_spectrogram(sample, rate)
        sample, start_index = self.subsample(sample)
        # added a transpose to match the output of the neural network
        sample = np.transpose(sample)
        sample = tensor(sample).float()
        sample = self.transformations(sample)
        return sample, start_index

    def __next__(self):
        if self.n < self.end:
            n = self.n
            sample, one_hot, sample_location = self.__getitem__(n)
            self.n += 1
            return sample, one_hot, sample_location
        else:
            self.n = 0
            raise StopIteration

    def __getitem__(self, index):
        sample, start_index = self.load_sample(index)
        return sample, self.onehot(index, self.end), self.onehot(start_index, self.maximum_sample_location)

    def __len__(self):
        return self.end


class SoundGenerator(nn.Module):
    def __init__(self, song_identifier_inputs, sample_location_inputs) -> None:
        super(SoundGenerator, self).__init__()

        self.song_identifier_first_layer = nn.Sequential(
            nn.Linear(song_identifier_inputs, 4096),
            nn.ELU(inplace=True),
            nn.Dropout()
        )

        self.sample_location_first_layer = nn.Sequential(
            nn.Linear(sample_location_inputs, 4096),
            nn.ELU(inplace=True),
            nn.Dropout()
        )

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(64, 128, kernel_size=4, padding=1, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, padding=1, dilation=2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(256, 384, kernel_size=4, stride=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 512, kernel_size=4, stride=1, dilation=3),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            # todo rewrite to be more in line with the music theory
            nn.Flatten(2)
        )

        self.output_layer = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(1, 1, kernel_size=(7, 12), padding=0, stride=1, dilation=(2, 3))
        )

    def forward(self, song_identifier: torch.Tensor,
                sample_location: torch.Tensor) -> torch.Tensor:

        # Takes the input of the ont hot encoded index of the sample from the metadata file
        song_identifier = self.song_identifier_first_layer(song_identifier)

        # Takes the input of the ont hot encoded index of the starting value from the sample
        sample_location = self.sample_location_first_layer(sample_location)

        x = song_identifier + sample_location

        x = x.view(x.size(0), 1, 64, 64)
        x = self.features(x)
        x = self.upsample(x)
        x = x.view(x.size(0), 1, 512, 289)
        # x = self.output_layer(x)
        return x[0:x.size(0), 0, 0:500, 0:256]
