import librosa
import scipy
from soundfile import write
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from create_music.spectrogram import helper_functions
from sklearn.metrics import mean_squared_error
from pathlib import Path
from datetime import datetime


input_file = Path('create_music/spectrogram/contents/gyNN33kV2jCi8mFtwMpHMEV9Hajbtc5XSrWxZzPg.mp3')
image_folder = Path('create_music/spectrogram/contents')


class FeatureExtractor:
    def __init__(self, audio, *, windowLength, overlap, sample_rate, inverse_iter):
        self.audio = audio
        self.ffT_length = windowLength
        self.window_length = windowLength
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.window = scipy.signal.hamming(self.window_length, sym=False)
        self.inverse_iter = inverse_iter

    def get_stft_spectrogram(self):
        return librosa.stft(self.audio, n_fft=self.ffT_length, win_length=self.window_length, hop_length=self.overlap,
                            window=self.window, center=True)

    def get_audio_from_stft_spectrogram(self, stft_features):
        return librosa.istft(stft_features, win_length=self.window_length, hop_length=self.overlap,
                             window=self.window, center=True)

    def get_mel_spectrogram(self):
        return librosa.feature.melspectrogram(self.audio, sr=self.sample_rate, power=2.0, pad_mode='reflect',
                                           n_fft=self.ffT_length, hop_length=self.overlap, center=True)

    def get_audio_from_mel_spectrogram(self, M):
        return librosa.feature.inverse.mel_to_audio(M, sr=self.sample_rate, n_fft=self.ffT_length, hop_length=self.overlap,
                                             win_length=self.window_length, window=self.window,
                                             center=True, pad_mode='reflect', power=2.0, n_iter=self.inverse_iter, length=None)

# get input sound file

data, rate = librosa.load(input_file)


# Produce a sample output of the sound file

for windowLength in [64, 512, 1024, 2048]:

    overlap = round(0.25 * windowLength)

    features = FeatureExtractor(audio=data,
                                windowLength=windowLength,
                                overlap=overlap,
                                sample_rate=rate,
                                inverse_iter=16)

    spectrogram = features.get_mel_spectrogram()

    now = datetime.now()
    output = features.get_audio_from_mel_spectrogram(spectrogram)

    write(image_folder / f'sample_audio_{windowLength}.wav', output, samplerate=rate)


# Calculate the relevent parameters for the sound file processing

window_exponants = range(6,15)

data_frame_list = {}

for x in window_exponants:
    windowLength = 2**x
    for n_iter in range(2, 17, 2):
        overlap = round(0.25 * windowLength)

        features = FeatureExtractor(audio=data,
                                    windowLength=windowLength,
                                    overlap=overlap,
                                    sample_rate=rate,
                                    inverse_iter=n_iter)

        spectrogram = features.get_mel_spectrogram()

        now = datetime.now()
        output = features.get_audio_from_mel_spectrogram(spectrogram)

        data_frame_list.append({'mse': mean_squared_error(output, data),
                        'length': spectrogram.shape[1],
                        'time': (datetime.now() - now).seconds,
                        'window_length': windowLength,
                        'n_iter': n_iter})


dataframe = pd.DataFrame(data_frame_list)
dataframe['rounded_mse'] = round(dataframe['mse'] * 100, 1)

def plot_graph(data_set, x_column, ax1_columns, ax2_columns, legend, colours):
    fig, ax = plt.subplots()
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    ax2 = ax.twinx()
    ax2.set_ylabel('Mean Squared Error')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())

    for column in ax1_columns:
        ax.plot(data_set[x_column], data_set[column], color=colours.pop())

    for column in ax2_columns:
        ax2.plot(data_set[x_column], data_set[column], color=colours.pop())

    fig.legend(legend)
    fig.set_size_inches(12, 6)
    return fig, ax


fig, ax = plot_graph(dataframe[dataframe['n_iter'] == 2],
                     x_column='window_length',
                     ax1_columns=['time'],
                     ax2_columns=['rounded_mse'],
                     legend=['Processing Time', 'Error in the output file'],
                     colours=['g', 'r'])

ax.set_ylabel('Time to create output')
ax.set_xlabel('Window Length')
fig.savefig(image_folder / 'spectrogram_settings_time.png')


fig, ax = plot_graph(dataframe[dataframe['n_iter'] == 2],
                     x_column='window_length',
                     ax1_columns=['length'],
                     ax2_columns=['rounded_mse'],
                     legend=['Length of the Image', 'Error in the output file'],
                     colours=['g', 'b'])

ax.set_ylabel('Length of the sample')
ax.set_xlabel('Window Length')
fig.savefig(image_folder / 'spectrogram_settings_length.png')


fig, ax = plot_graph(dataframe[dataframe['window_length'] == 512],
                     x_column='n_iter',
                     ax1_columns=['time'],
                     ax2_columns=['rounded_mse'],
                     legend=['Processing Time', 'Error in the output file'],
                     colours=['g', 'y'])

ax.set_ylabel('Time to create output')
ax.set_xlabel('Number of Iterations')
ax.yaxis.set_minor_formatter(ScalarFormatter())
fig.savefig(image_folder / 'spectrogram_settings_iterations.png', )
