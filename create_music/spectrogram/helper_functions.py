from pathlib import Path

import numpy as np
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
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)]
    instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
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


def load_sound_file(path: Path, sr):
    try:
        data, rate = librosa.load(str(path),
                                  sr=sr)

    except Exception as e:
        print(f"Reading of sample {path.name} failed")
        raise e

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
            # if self.print_n % 100 == 0:
            #     self.metadata.iloc[itemid, -1]
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
        # sample_length = sample.shape[1]
        # start = rand.randint(0, sample_length - self.y_size)
        # start = min(self.maximum_sample_location - 1, start)
        start = 0
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
        sample = sample.view(1, self.y_size, self.n_mels)
        return sample, start_index

    def __next__(self):
        if self.n < self.end:
            n = self.n
            sample = self.__getitem__(n)
            self.n += 1
            return sample
        else:
            self.n = 0
            raise StopIteration

    def __getitem__(self, index):
        sample, start_index = self.load_sample(index)
        return sample

    def __len__(self):
        return self.end


class SoundGenerator(nn.Module):
    def __init__(self) -> None:
        super(SoundGenerator, self).__init__()

        conv_channels = [16, 64, 128, 256, 512]
        conv_kernels_size = [5, 5, 5, 5, 5]
        conv_strides = [2, 2, 2, 2, 2]
        conv_encode_padding = [2, 2, 2, 2, 2]
        conv_decode_padding = [2, 2, 2, 2, 2]
        conv_encode_dropout = [0.2, 0, 0.2, 0.2, 0]
        conv_decode_dropout = [0.2, 0, 0.2, 0.2, 0]

        self.encoder = nn.Sequential(
            nn.Conv2d(1, conv_channels[0],
                      kernel_size=conv_kernels_size[0],
                      stride=conv_strides[0],
                      padding=conv_encode_padding[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(conv_encode_dropout[0]),
            nn.Conv2d(conv_channels[0], conv_channels[1],
                      kernel_size=conv_kernels_size[1],
                      stride=conv_strides[1],
                      padding=conv_encode_padding[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(conv_encode_dropout[1]),
            nn.Conv2d(conv_channels[1], conv_channels[2],
                      kernel_size=conv_kernels_size[2],
                      stride=conv_strides[2],
                      padding=conv_encode_padding[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(conv_encode_dropout[2]),
            nn.Conv2d(conv_channels[2], conv_channels[3],
                      kernel_size=conv_kernels_size[3],
                      stride=conv_strides[3],
                      padding=conv_encode_padding[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(conv_encode_dropout[3]),
            nn.Conv2d(conv_channels[3], conv_channels[4],
                      kernel_size=conv_kernels_size[4],
                      stride=conv_strides[4],
                      padding=conv_encode_padding[4]),
            nn.ReLU(inplace=True),
            nn.Dropout(conv_encode_dropout[4]),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(conv_channels[4], conv_channels[3],
                               kernel_size=conv_kernels_size[4],
                               stride=conv_strides[4],
                               padding=conv_decode_padding[0],
                               output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(conv_decode_dropout[0]),
            nn.ConvTranspose2d(conv_channels[3], conv_channels[2],
                               kernel_size=conv_kernels_size[3],
                               stride=conv_strides[3],
                               padding=conv_decode_padding[1],
                               output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(conv_decode_dropout[1]),
            nn.ConvTranspose2d(conv_channels[2], conv_channels[1],
                               kernel_size=conv_kernels_size[2],
                               stride=conv_strides[2],
                               padding=conv_decode_padding[2],
                               output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(conv_decode_dropout[2]),
            nn.ConvTranspose2d(conv_channels[1], conv_channels[0],
                               kernel_size=conv_kernels_size[1],
                               stride=conv_strides[1],
                               padding=conv_decode_padding[3],
                               output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(conv_decode_dropout[3]),
            nn.ConvTranspose2d(conv_channels[0], 1,
                               kernel_size=conv_kernels_size[0],
                               stride=conv_strides[0],
                               padding=conv_decode_padding[4],
                               output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(conv_decode_dropout[4]),
        )

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        x = self.encoder(sample)
        return x

    def decode(self, sample: torch.Tensor) -> torch.Tensor:
        x = self.decoder(sample)
        return x

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        x = self.encode(sample)
        x = self.decode(x)
        return x
