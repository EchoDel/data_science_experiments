from pathlib import Path

import numpy as np
import random as rand
import librosa
import torch
import torch.nn as nn


def transform_clamp(minimum, maximum):
    def clamp(tensor):
        return torch.clamp(tensor, minimum, maximum)
    return clamp


def transform_log():
    def log(tensor):
        return torch.log10(tensor)
    return log


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def load_sound_file(path: Path, sr):
    try:
        data, rate = librosa.load(str(path),
                                  sr=sr)

    except Exception as e:
        print(f"Reading of sample {path.name} failed")
        raise e

    return data, rate


class SongIngestion(torch.utils.data.Dataset):
    def __init__(self, metadata, sample_length: int, transformations,
                 sr: int, n_mels: int, sound_files: dict, seed=1994):
        super(SongIngestion).__init__()
        self.metadata = metadata
        self.n_mels = n_mels

        self.n = 0
        rand.seed(seed)
        np.random.seed(seed)

        self.start = 0
        self.end = self.metadata.shape[0]
        self.sound_files = sound_files
        self.sample_length = sr * sample_length
        self.transformations = transformations
        self.sr = sr
        self.indexes = self.metadata.index.to_list()

    def load_sound_file(self, track_id):
        if track_id not in self.sound_files:
            self.sound_files[track_id] = load_sound_file(self.metadata.loc[[track_id]].iloc[0, -1], self.sr)
        return self.sound_files[track_id]

    def load_sample(self, index):
        sample, rate = self.load_sound_file(index)

        if len(sample) > self.sample_length:
            sample_start = rand.randint(0, len(sample) - self.sample_length)
            sample = sample[sample_start: (sample_start + self.sample_length)]
        else:
            new_sample = np.zeros(self.sample_length, sample.dtype)
            sample_start = rand.randint(0, len(new_sample) - self.sample_length)
            new_sample[sample_start:(sample_start + len(sample))] = sample
            sample = new_sample

        sample = sample.reshape((1, self.sample_length))
        sample = torch.from_numpy(sample)
        transformed_sample = self.transformations(sample)
        return transformed_sample, sample

    def __getitem__(self, index):
        transformed_sample, sample = self.load_sample(self.indexes[index])
        return transformed_sample, sample

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
            # nn.BatchNorm2d(conv_channels[1]),
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
            # nn.BatchNorm2d(conv_channels[3]),
            nn.Conv2d(conv_channels[3], conv_channels[4],
                      kernel_size=conv_kernels_size[4],
                      stride=conv_strides[4],
                      padding=conv_encode_padding[4]),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(conv_channels[4], conv_channels[3],
                               kernel_size=conv_kernels_size[4],
                               stride=conv_strides[4],
                               padding=conv_decode_padding[0],
                               output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(conv_decode_dropout[0]),
            # nn.BatchNorm2d(conv_channels[3]),
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
                               kernel_size=(5, 3),
                               stride=conv_strides[0],
                               padding=conv_decode_padding[4],
                               output_padding=1),
            # nn.Sigmoid(),
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
