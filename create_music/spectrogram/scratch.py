import librosa
import scipy
from soundfile import write

from create_music.spectrogram import helper_functions
from sklearn.metrics import mean_squared_error
from pathlib import Path
from datetime import datetime


folder = Path('../music')

metadata = helper_functions.load_metadata(folder)


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




data, rate = librosa.load(metadata.iloc[0, 0])

window_exponants = range(6,15)

outputs = {}

for x in window_exponants:
    windowLength = 2**x
    outputs[windowLength] = {}
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

        outputs[windowLength][n_iter] = {'mse' : mean_squared_error(output, data),
                                         'length': spectrogram.shape[1],
                                         'time': (datetime.now() - now).seconds}
