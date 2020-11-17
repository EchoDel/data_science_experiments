import pandas as pd
import numpy as np
from pathlib import Path

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


def spectogram_creation(audio, sample_rate, samples):
    return spectrogram(audio, sample_rate, nperseg=64)






