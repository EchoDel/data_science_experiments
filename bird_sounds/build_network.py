import json
from pathlib import Path

from bird_sounds import helper_functions
import torch
from torch import nn
from torch import optim
from torchvision import models, transforms

from matplotlib import pyplot as plt

model_name = 'Modified_AlexNet'
metadata_file = 'ff1010bird_metadata.csv'

# Start model definition
if model_name == 'AlexNet':
    model = models.AlexNet(num_classes=1)
elif model_name == 'Modified_AlexNet':
    model = helper_functions.AlexNet(num_classes=1)
elif model_name == 'resnet101':
    model = models.resnet101(num_classes=1)
elif model_name == 'resnet18':
    model = models.resnet18(num_classes=1)

device = 'cuda'

transformations = transforms.transforms.Compose([
    transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
])

train_loader = torch.utils.data.DataLoader(
    helper_functions.BirdCalls(Path('../bird_sounds/' + metadata_file), False,
                               x_size=224, y_size=224,
                               transformations=transformations),
    batch_size=16)

test_loader = torch.utils.data.DataLoader(
    helper_functions.BirdCalls(Path('../bird_sounds/' + metadata_file), True,
                               x_size=224, y_size=224,
                               transformations=transformations),
    batch_size=50)

optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()
model.to(device)

epochs = 40
steps = 0
running_loss = 0
save_every = 5
train_losses = []
test_losses = []
accuracies = []
metadata = {}

for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model(inputs)
        loss = criterion(logps.squeeze(1), labels.type_as(logps))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_logps = model.forward(test_inputs)
            batch_loss = criterion(test_logps.squeeze(1), test_labels.type_as(test_logps))
            test_loss += batch_loss.item()

            top_class = test_logps.gt(0.5)
            equals = top_class == test_labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item() * len(test_labels)

    train_losses.append(running_loss / len(train_loader.dataset))
    test_losses.append(test_loss / len(test_loader.dataset))
    accuracies.append(accuracy / len(test_loader.dataset))
    print(f"Epoch {epoch + 1}/{epochs}.. "
          f"Train loss: {running_loss / len(train_loader.dataset):.3f}.. "
          f"Test loss: {test_loss / len(test_loader.dataset):.3f}.. "
          f"Test accuracy: {accuracy / len(test_loader.dataset):.3f}")
    running_loss = 0
    train_loader.dataset.shuffle()
    model.train()

    save_path = f'models/{metadata_file}/birdcalls_{model_name}_{epoch + 1}.pth'
    metadata[epoch + 1] = {
        'running_loss': running_loss / len(train_loader.dataset),
        'test_loss': test_loss / len(test_loader.dataset),
        'accuracy': accuracy / len(test_loader.dataset)
    }

    if epoch == 0:
        metadata[epoch + 1]['path'] = save_path
        torch.save(model, save_path)
    elif epoch % save_every == 1:
        metadata[epoch + 1]['path'] = save_path
        torch.save(model, save_path)
    elif accuracy / len(test_loader.dataset) > max(accuracies):
        metadata[epoch + 1]['path'] = save_path
        torch.save(model, save_path)


with open(f'models/{metadata_file}/metadata{model_name}.json', 'w') as outfile:
    json.dump(metadata, outfile)

plt.plot(range(epochs), train_losses, label='Train Losses')
plt.plot(range(epochs), test_losses,  label='Test Losses')
plt.plot(range(epochs), accuracies,  label='Test Accuracy')
plt.legend()
