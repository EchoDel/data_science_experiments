import json
from pathlib import Path

from create_music import helper_functions
import torch
from torch import nn
from torch import optim
from torchvision import models, transforms

from matplotlib import pyplot as plt

folder = Path('../music')
device = 'cpu'
sample_length = 32768
model_name = 'music_creation'
metadata_file = 'lofi'
config_file = Path(f'models/{metadata_file}/metadata{model_name}.json')
epochs_to_run = 1000
save_every = 100
samplerate = 16000

transformations = transforms.transforms.Compose([
    transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
])

train_loader = torch.utils.data.DataLoader(
    helper_functions.SongIngestion(Path(folder),
                                   length=sample_length,
                                   transformations=transformations,
                                   sr=samplerate),
    batch_size=16)

if config_file.exists():
    with open(f'models/{metadata_file}/metadata{model_name}.json', 'r') as outfile:
        metadata = json.load(outfile)

    for key, value in metadata.items():
        if 'path' in value:
            model_path = value['path']
            starting_iteration = int(key)

    model = torch.load(model_path)
else:
    model = helper_functions.LinearNN
    metadata = {}


optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
criterion = nn.L1Loss()
model.to(device)

steps = 0
running_loss = 0
train_losses = []
test_losses = []
accuracies = []

for epoch in range(epochs_to_run):
    epoch = starting_iteration + epoch
    model.train()
    for results, inputs in train_loader:
        steps += 1
        inputs, results = inputs.to(device).float(), results.to(device)
        optimizer.zero_grad()
        logps = model(inputs)
        loss = criterion(logps.squeeze(1), results.type_as(logps))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss)
    print(f"Epoch {epoch + 1}/{epochs_to_run + starting_iteration}.. "
          f"Train loss: {running_loss:.3f}.. ")
    running_loss = 0

    save_path = f'models/{metadata_file}/music_creation_{model_name}_{epoch + 1}.pth'
    metadata[epoch + 1] = {
        'running_loss': running_loss / len(train_loader.dataset),
    }

    if epoch % save_every == save_every - 1:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        metadata[epoch + 1]['path'] = save_path
        torch.save(model, save_path)

        with open(config_file, 'w') as outfile:
            json.dump(metadata, outfile)
