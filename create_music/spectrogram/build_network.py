import json
from pathlib import Path

from create_music.spectrogram import helper_functions
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from fma import utils as fmautils

fma_set = 'medium'
genre = 'Rock'
# load the metadata for the fma dataset
fma_base = Path('fma/data/fma_metadata')
AUDIO_DIR = Path('../data/fma_' + fma_set)
folder = fma_base / 'tracks.csv'
tracks = fmautils.load(fma_base / 'tracks.csv')
fma_subset = tracks[tracks['set', 'subset'] <= fma_set]
fma_subset = fma_subset.copy()
fma_subset[('path', '')] = fma_subset.index.map(lambda x: Path(fmautils.get_audio_path(AUDIO_DIR, x)))

fma_subset_sample = fma_subset[fma_subset[('track', 'genre_top')] == genre]
fma_subset_sample = fma_subset_sample.sample(8)

device = 'cuda'
sample_length = 32768
model_name = f'{fma_set}_{genre}'
metadata_file = 'lofi_spectrogram_2'
config_file = Path(f'models/{metadata_file}/metadata_{model_name}.json')
loader_path = Path(f'models/{metadata_file}/loader_{model_name}.pth')
epochs_to_run = 16000
save_every = 100
sample_rate = 22050
window_length = 2048
maximum_sample_location = 4096
y_size = 512
n_mels = 512
batch_size = 16

transformations = transforms.transforms.Compose([
    # transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
])

train_loader = torch.utils.data.DataLoader(
    helper_functions.SongIngestion(fma_subset_sample,
                                   sample_length=sample_length,
                                   transformations=transformations,
                                   sr=sample_rate,
                                   window_length=window_length,
                                   y_size=y_size,
                                   n_mels=n_mels,
                                   maximum_sample_location=maximum_sample_location),
    batch_size=batch_size)

if config_file.exists():
    with open(config_file, 'r') as outfile:
        metadata = json.load(outfile)

    for key, value in metadata.items():
        if 'path' in value:
            model_path = value['path']
            starting_iteration = int(key)

    model = torch.load(model_path)

else:
    model = helper_functions.SoundGenerator()
    metadata = {}
    starting_iteration = 0


optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
criterion = nn.L1Loss()
model.to(device)

steps = 0
running_loss = 0

for epoch in range(epochs_to_run):
    running_loss = 0
    epoch = starting_iteration + epoch
    model.train()
    for results in train_loader:
        steps += 1
        results = results.to(device)
        optimizer.zero_grad()
        logps = model(results)
        logps = logps.reshape([logps.shape[0], y_size, 256])
        loss = criterion(logps, results.type_as(logps))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs_to_run + starting_iteration}.. "
          f"Train loss: {running_loss / len(train_loader.dataset):.3f}.. ")

    save_path = f'models/{metadata_file}/music_creation_{model_name}_{epoch + 1}.pth'
    metadata[epoch + 1] = {
        'running_loss': running_loss / len(train_loader.dataset),
    }

    if epoch % save_every == save_every - 1:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        metadata[epoch + 1]['path'] = save_path
        torch.save(model, save_path)

        # if loader_path.exists():
        #     torch.save(train_loader, loader_path)

        with open(config_file, 'w') as outfile:
            json.dump(metadata, outfile)
