import pandas as pd
import numpy as np
from pathlib import Path
import math
import torch
import torch.nn as nn
from torchvision import transforms

from scipy.io import wavfile as wav
from scipy.signal import spectrogram


def load_metadata(path: Path):
    metadata = pd.read_csv(path)
    root_path = path.parent / path.name.replace('_metadata.csv', '_wav')
    metadata['path'] = metadata['itemid'].apply(lambda x: root_path / (str(x) + '.wav'))
    return metadata


def load_sound_file(path):
    try:
        rate, data = wav.read(path)

    except Exception as e:
        print(f"Reading of sample {path.name} failed")
        print(e)

    return data, rate


def spectrogram_creation(audio, sample_rate, samples):
    return spectrogram(audio, sample_rate, nperseg=512)


class BirdCalls(torch.utils.data.IterableDataset):
    def __init__(self, metadata_path, test, split_percentage=0.8, seed = 1994):
        super(BirdCalls).__init__()
        metadata = load_metadata(metadata_path)
        metadata = metadata.sample(100).reset_index(drop=True)
        np.random.seed(seed)
        msk = np.random.rand(len(metadata)) < split_percentage
        if test:
            self.metadata = metadata[~msk].reset_index(drop=True)
        else:
            self.metadata = metadata[msk].reset_index(drop=True)
        self.start = 0
        self.end = self.metadata.shape[0]
        self.classes = max(metadata.iloc[:, 1])

    def load_spectrogram(self, index):
        data, rate = load_sound_file(self.metadata.iloc[index, 2])
        frequency_graph = spectrogram_creation(data, rate, None)
        return frequency_graph[2]

    def get_one_hot(self, target):
        a = torch.zeros(self.classes + 1, dtype=torch.long)
        a[target] = 1
        return a

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else: # in a worker process  # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        result = []
        for index in range(iter_start, iter_end):
            sample = self.load_spectrogram(index)
            sample = sample[0:224, 0:224]
            sample = transforms.ToTensor()(sample)
            label = self.metadata.iloc[index, 1]
            result.append((sample, label))

        return iter(result)


    def __len__(self):
        self.metadata.shape[0]


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

