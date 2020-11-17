import pandas as pd
import numpy as np
from pathlib import Path
import math
import torch

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
        self.iteration_number = 0

    def load_spectrogram(self, index):
        data, rate = load_sound_file(self.metadata.iloc[index, 2])
        return frequency_graph[1]

    def __iter__(self):
        iteration = self.iteration_number
        sample = self.load_spectrogram(iteration)
        label = self.metadata.iloc[iteration, 1]
        self.iteration_number += 1
        yield (sample, label)



        frequency_graph = spectrogram_creation(data, rate, None)



