import math

from scipy.signal import windows, butter, sosfilt, argrelextrema
from statsmodels.nonparametric.smoothers_lowess import lowess
from fma import utils as fmautils
import librosa
import librosa.display
from pathlib import Path
import seaborn
import numpy as np
import numpy_indexed as npi
import matplotlib.pyplot as plt

window_length = 2048
window = windows.hamming(window_length, sym=False)


AUDIO_DIR = Path('../data/fma_medium')

tracks = fmautils.load('fma/data/fma_metadata/tracks.csv')
genres = fmautils.load('fma/data/fma_metadata/genres.csv')
features = fmautils.load('fma/data/fma_metadata/features.csv')

medium = tracks[tracks['set', 'subset'] <= 'medium']


filename = fmautils.get_audio_path(AUDIO_DIR, 1014)
print('File: {}'.format(filename))

x, sr = librosa.load(filename, sr=None, mono=True)

spectrogram = librosa.feature.melspectrogram(x,
                                             sr=sr,
                                             n_fft=window_length,
                                             hop_length=round(0.25 * window_length),
                                             window=window,
                                             power=1.0)

seaborn.heatmap(spectrogram)

librosa.display.specshow(librosa.power_to_db(spectrogram))




group_list = list(range(len(spectrogram[:,93])))
group_list = np.array([math.floor(x/8) for x in group_list])

for x in range(spectrogram.shape[1]):
    values = np.concatenate(([group_list], [spectrogram[:, x]]), axis=0)
    y = npi.group_by(values[0, :]).argmax(values[1, :])


# trying loess filter

loess_data = lowess(spectrogram[:,93], range(len(spectrogram[:,93])), 0.07)[:,1]

plt.plot(range(len(spectrogram[:,93])),
         loess_data)
plt.plot(range(len(spectrogram[:,93])),
         spectrogram[:,93])




maximums = argrelextrema(loess_data, np.greater)



group_list = list(range(len(spectrogram[:,93])))
group_list = np.array([math.floor(x/8) for x in group_list])


