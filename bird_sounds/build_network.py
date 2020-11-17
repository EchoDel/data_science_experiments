from pathlib import Path

from bird_sounds import helper_functions
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from matplotlib import pyplot as plt

metadata_path = Path('../bird_sounds/ff1010bird_metadata.csv')

# Quick test of the helper function
metadata = helper_functions.load_metadata(metadata_path)

data, rate = helper_functions.load_sound_file(metadata.iloc[0,2])

frequency_graph = helper_functions.spectrogram_creation(data, rate, None)

# plt.pcolormesh(frequency_graph[1], frequency_graph[0], frequency_graph[2], shading='gouraud')


# Start model definition

model = helper_functions.AlexNet(num_classes=2)
device = 'cuda'


train_loader = torch.utils.data.DataLoader(
    helper_functions.BirdCalls(Path('../bird_sounds/ff1010bird_metadata.csv'), False),
    batch_size=1)

test_loader = torch.utils.data.DataLoader(
    helper_functions.BirdCalls(Path('../bird_sounds/ff1010bird_metadata.csv'), True),
    batch_size=1)

optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.L1Loss()
model.to(device)

epochs = 10
steps = 0
running_loss = 0
print_every = 5
train_losses, test_losses = [], []

for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        logps = model.forward(test_inputs)
        batch_loss = criterion(logps, test_labels)
        test_loss += batch_loss.item()

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    train_losses.append(running_loss / len(train_loader))
    test_losses.append(test_loss / len(test_loader))
    print(f"Epoch {epoch + 1}/{epochs}.. "
          f"Train loss: {running_loss / print_every:.3f}.. "
          f"Test loss: {test_loss / len(test_loader):.3f}.. "
          f"Test accuracy: {accuracy / len(test_loader):.3f}")
    running_loss = 0
    model.train()

torch.save(model, 'aerialmodel.pth')


