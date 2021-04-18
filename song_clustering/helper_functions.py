import numpy as np
import random as rand
import librosa
import torch
import torch.nn as nn
from scipy.signal.windows import hamming
from torch import tensor


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
        sample = sample.view(1, self.y_size, self.n_mels)
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


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class AutoEncoder(nn.Module):
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        super(AutoEncoder, self).__init__()

        self.encode_layers = nn.Sequential(
            nn.Unflatten(2, (16, 16)),
            nn.Conv2d(520, 256, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(256, 16, 2, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(16, 4, 2, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
        )

        self.decode_layers = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 2, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.Dropout(),
            nn.ConvTranspose2d(16, 256, 2, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.Dropout(),
            nn.ConvTranspose2d(256, 520, 4, stride=2, padding=1),  # b, 8, 15, 15
            nn.Tanh(),
            nn.Flatten(2),
        )

    def encode(self, input_tensor):
        x = input_tensor.view(input_tensor.size(0), 520, 256)
        x = self.encode_layers(x)
        return x

    def decode(self, input_tensor):
        x = self.decode_layers(input_tensor)
        x = x.view(input_tensor.size(0), 1, 520, 256)
        return x

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.encode(input_tensor)
        x = self.decode(x)
        return x
