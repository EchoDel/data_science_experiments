import pandas as pd
import numpy as np
import numpy_indexed as npi
from pathlib import Path

from scipy.io import wavfile as wav
from scipy.fftpack import fft


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


def fft_creation(audio, sample_rate):
    n = len(audio)
    T = 1/sample_rate
    yf = fft(audio)
    xf = np.linspace(0.0, 1.0/(2.0*T), int(n/2))
    yf = 2.0/n * np.abs(yf[:n//2])
    np.vstack((xf, yf))
    return np.vstack((xf, yf))

