import json
from pathlib import Path

import torch

fma_set = 'medium'
genre = 'Rock'
device = 'cuda'
model_name = f'{fma_set}_{genre}'
metadata_file = 'lofi_spectrogram_2'
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

torch.rand((1, 512, 16, 16))

model.decode(torch.rand((1, 512, 16, 16)).to(device))
