import json
from pathlib import Path

import torch
from librosa.feature.inverse import mel_to_audio
import soundfile as sf

fma_set = 'medium'
genre = 'Rock'
device = 'cpu'
model_name = f'{fma_set}_{genre}'
metadata_file = 'rock_spectrogram_new'
config_file = Path(f'models/{metadata_file}/metadata_{model_name}.json')


with open(config_file, 'r') as outfile:
    metadata = json.load(outfile)

metadata = {int(key): value for key, value in metadata.items()}

for key, value in metadata.items():
    if 'path' in value:
        model_path = value['path']
        epoch = int(key)

model = torch.load(model_path)
model.to(device)


model.eval()

model_result = model.decode(torch.rand((1, 2048)).to(device))
mel_spec = (10**model_result[0])
waveform = mel_to_audio(mel_spec.detach().numpy(), hop_length=512, n_fft=2048)
sf.write('transformedtest.wav', waveform[0], samplerate=22050)


waveform = mel_to_audio(raw_results.detach().numpy(), hop_length=512, n_fft=2048)
sf.write('rawtest.wav', waveform[0], samplerate=22050, subtype='PCM_24')
