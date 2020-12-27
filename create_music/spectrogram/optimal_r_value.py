from scipy.signal import windows, argrelextrema

from create_music.spectrogram import helper_functions
from fma import utils as fmautils
import librosa
import librosa.display
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

window_length = 2048
window = windows.hamming(window_length, sym=False)


def calculate_spacing(index, n_mels):
    filename = fmautils.get_audio_path(AUDIO_DIR, index)

    try:
        x, sr = librosa.load(filename, sr=None, mono=True)
    except Exception as E:
        print(filename)
        print(E)
        return np.nan

    spectrogram = librosa.feature.melspectrogram(x,
                                                 sr=sr,
                                                 n_fft=window_length,
                                                 hop_length=round(0.25 * window_length),
                                                 window=window,
                                                 power=1.0,
                                                 n_mels=n_mels)

    optimal_r = []

    for x in range(spectrogram.shape[1]):
        # Smooth the data with a hanning window
        # https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
        smoothed_data = helper_functions.smooth(spectrogram[:, x], window_len=9)

        # find the maximum
        maximums = argrelextrema(smoothed_data, np.greater)[0]

        # select only the low ones
        low_maximums = maximums[maximums < n_mels / 2]

        # select ones with more than 4 since we don't want the double peaks
        differences = np.diff(low_maximums)
        differences = differences[differences > 4]

        optimal_r.append(np.mean(differences))
    return np.mean(optimal_r)


AUDIO_DIR = Path('../data/fma_medium')

tracks = fmautils.load('fma/data/fma_metadata/tracks.csv')
genres = fmautils.load('fma/data/fma_metadata/genres.csv')
features = fmautils.load('fma/data/fma_metadata/features.csv')

medium = tracks[tracks['set', 'subset'] <= 'medium']
medium = medium.copy()

medium[('r_value', '128')] = medium.index.map(lambda x: calculate_spacing(x, n_mels=128))
medium[('r_value', '256')] = medium.index.map(lambda x: calculate_spacing(x, n_mels=256))
medium[('r_value', '376')] = medium.index.map(lambda x: calculate_spacing(x, n_mels=376))

medium.to_csv('create_music/spectrogram/tracks_r_value.csv', index=False)


medium.groupby(('track', 'genre_top')).agg({('r_value', '128'): 'mean',
                                            ('r_value', '256'): 'mean',
                                            ('r_value', '376'): 'mean'})
