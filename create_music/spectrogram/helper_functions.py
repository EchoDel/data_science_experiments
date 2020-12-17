import pandas as pd
import numpy as np
from pathlib import Path
import random as rand
import librosa
import torch
import torch.nn as nn
from scipy.signal.windows import hamming
from torch import tensor
import torch.nn.functional as F


def load_metadata(path: Path):
    files = [x for x in path.glob("*.mp3")]
    metadata = pd.DataFrame({'Path': files})
    return metadata


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
    def __init__(self, folder, sample_length, transformations, sr, window_length, y_size, seed=1994):
        super(SongIngestion).__init__()
        self.metadata = load_metadata(folder)

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

    def onehot(self, n):
        output = np.zeros(self.end)
        output[n] = 1
        return output

    def load_sound_file(self, itemid):
        if itemid not in self.sound_files:
            if self.print_n % 100 == 0:
                self.metadata.iloc[itemid, 0]
            self.print_n += 1
            self.sound_files[itemid] = load_sound_file(self.metadata.iloc[itemid, 0], self.sr)
        return self.sound_files[itemid]

    def load_spectrogram(self, data, rate):
        frequency_graph = librosa.feature.melspectrogram(data,
                                                         sr=rate,
                                                         n_fft=self.window_length,
                                                         hop_length=round(0.25 * self.window_length),
                                                         window=self.window)
        return frequency_graph

    def subsample(self, sample):
        sample_length = sample.shape[1]
        start = rand.randint(0, sample_length - self.y_size)
        sample = sample[:, start:(start + self.y_size)]
        return sample

    def load_sample(self, index):
        sample, rate = self.load_sound_file(index)
        sample = self.load_spectrogram(sample, rate)
        sample = self.subsample(sample)
        sample = tensor(sample).float()
        sample = self.transformations(sample)
        return sample

    def __next__(self):
        if self.n <= self.end:
            sample = self.load_sample(self.n)
            self.n += 1
            return sample, self.onehot(self.n)
        else:
            raise StopIteration

    def __getitem__(self, index):
        sample = self.load_sample(index)
        return sample, self.onehot(index)

    def __len__(self):
        return self.end


class LinearNN(nn.Module):
    def __init__(self, inputs, final_length) -> None:
        super(LinearNN, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(inputs, 256),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 4096),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(4096, int(final_length / 2)),
        )
        self.classifier = nn.Sequential(
            nn.ELU(inplace=True),
            nn.Dropout(),
            nn.Linear(int(final_length / 2), final_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
