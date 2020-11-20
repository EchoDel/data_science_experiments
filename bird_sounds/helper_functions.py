import pandas as pd
import numpy as np
from pathlib import Path
import math
import librosa
import torch
import torch.nn as nn
from sklearn.utils import resample
from torchvision import transforms


def load_metadata(path: Path):
    metadata = pd.read_csv(path)
    root_path = path.parent / path.name.replace('_metadata.csv', '_wav')
    metadata['path'] = metadata['itemid'].apply(lambda x: root_path / (str(x) + '.wav'))
    return metadata


def load_sound_file(path):
    try:
        data, rate = librosa.load(path)

    except Exception as e:
        print(f"Reading of sample {path.name} failed")
        print(e)

    return data, rate


def spectrogram_creation(audio, sample_rate, n_mels):
    n_fft = 2048
    hop_length = 512

    spectrogram = librosa.feature.melspectrogram(audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return spectrogram


class BirdCalls(torch.utils.data.Dataset):
    def __init__(self, metadata_path, test, split_percentage=0.8, seed=1994):
        super(BirdCalls).__init__()
        metadata = load_metadata(metadata_path)
        metadata = self.resample(metadata)
        np.random.seed(seed)
        msk = np.random.rand(len(metadata)) < split_percentage
        if test:
            self.metadata = metadata[~msk].reset_index(drop=True)
        else:
            self.metadata = metadata[msk].reset_index(drop=True)
        self.start = 0
        self.end = self.metadata.shape[0]
        self.classes = max(metadata.iloc[:, 1])
        self.sound_files = {}
        self.n = 0
        self.print_n = 0
    def resample(self, metadata):
        # Separate majority and minority classes
        majority = metadata['hasbird'].mode()[0]
        df_majority = metadata[metadata.hasbird == 0]
        df_minority = metadata[metadata.hasbird == 1]

        # Upsample minority class
        df_minority_upsampled = resample(df_minority,
                                         replace=True,  # sample with replacement
                                         n_samples=df_majority.shape[0],  # to match majority class
                                         random_state=123)  # reproducible results

        # Combine majority class with upsampled minority class
        df_upsampled = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1).reset_index(drop=True)
        return df_upsampled

    def load_spectrogram(self, index):
        data, rate = self.load_sound_file(self.metadata.iloc[index, 2])
        frequency_graph = spectrogram_creation(data, rate, 224)
        return frequency_graph

    def get_one_hot(self, target):
        a = torch.zeros(self.classes + 1, dtype=torch.long)
        a[target] = 1
        return a

    def __next__(self):
        if self.n <= self.end:
            sample = self.load_spectrogram(self.n)
            sample = sample[0:224, 0:224]
            sample = transforms.ToTensor()(sample)
            sample = sample.repeat(3, 1, 1)
            label = self.metadata.iloc[self.n, 1]
            self.n += 1
            return sample, label
        else:
            raise StopIteration

    def __getitem__(self, index):
        sample = self.load_spectrogram(index)
        sample = sample[0:224, 0:224]
        sample = transforms.ToTensor()(sample)
        label = self.metadata.iloc[index, 1]
        return sample, label

    def __len__(self):
        return self.metadata.shape[0]


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

