from scipy.signal import windows, argrelextrema

from create_music.spectrogram import helper_functions
from fma import utils as fmautils
import librosa
import librosa.display
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

window_length = 2048
window = windows.hamming(window_length, sym=False)
csv_output = 'create_music/spectrogram/tracks_r_value.csv'
AUDIO_DIR = Path('../data/fma_medium')
fma_base = Path('fma/data/fma_metadata')


def get_spectrogram(index, n_mels):
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

    return spectrogram


def calculate_spacing(index, n_mels):
    spectrogram = get_spectrogram(index, n_mels)

    optimal_r = []
    window_len = int(round(9*n_mels/128))

    for x in range(spectrogram.shape[1]):
        # Smooth the data with a hanning window
        # https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
        smoothed_data = helper_functions.smooth(spectrogram[:, x], window_len=window_len)

        # find the maximum
        maximums = argrelextrema(smoothed_data, np.greater)[0]

        # select only the low ones
        low_maximums = maximums[maximums < n_mels / 2]

        # select ones with more than 4 since we don't want the double peaks
        differences = np.diff(low_maximums)
        differences = differences[differences > 4]

        optimal_r.append(np.mean(differences))
    return np.mean(optimal_r)


tracks = fmautils.load(fma_base / 'tracks.csv')
genres = fmautils.load(fma_base / 'genres.csv')
features = fmautils.load(fma_base / 'features.csv')

medium = tracks[tracks['set', 'subset'] <= 'medium']
medium = medium.copy()

medium[('r_value', '128')] = medium.index.map(lambda x: calculate_spacing(x, n_mels=128))
medium[('r_value', '256')] = medium.index.map(lambda x: calculate_spacing(x, n_mels=256))
medium[('r_value', '376')] = medium.index.map(lambda x: calculate_spacing(x, n_mels=376))

medium.to_csv(csv_output, index=False)



# Create outputs for the markdown
medium = pd.read_csv(csv_output, header=[0, 1])

medium.groupby(('track', 'genre_top')).agg({('r_value', '128'): 'mean',
                                            ('r_value', '256'): 'mean',
                                            ('r_value', '376'): 'mean'})

spectrogram = get_spectrogram(2, 256)

plt_index = 100
plotting_data = spectrogram[:, plt_index]
x_values = list(range(len(plotting_data)))
smoothed_data = helper_functions.smooth(plotting_data, window_len=9)
maximums = argrelextrema(smoothed_data, np.greater)[0]

plt.plot(x_values, plotting_data)
plt.plot(x_values, smoothed_data)
plt.scatter(maximums, [plotting_data[i] for i in maximums], marker='o')
plt.legend(['Spectrogram Slice', 'Smoothed Slice', 'Maximums'])
plt.savefig(fname='create_music/spectrogram/contents/maximum_method.png')
