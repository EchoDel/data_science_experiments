from pathlib import Path
import torch
from torch import optim
import json
from torch import nn

import bookingdotcom.helper_functions as helper_functions

cache_location = Path('bookingdotcom/cache/')
epochs = 1000
save_every = 5
device = 'cuda'
model_location = Path('models/bookingdotcom/')
config_file = model_location / 'metadata.json'

connected_node_features = torch.load(cache_location / 'connected_node_features.pkl')
trips = torch.load(cache_location / 'trip_properties.pkl')


train_loader = torch.utils.data.DataLoader(
    helper_functions.BookingLoader(trips=trips.copy(),
                                   connected_node_features=connected_node_features.copy(),
                                   training=True,
                                   number_of_classes=67566,
                                   training_percentage=0.2),
    batch_size=16)

test_loader = torch.utils.data.DataLoader(
    helper_functions.BookingLoader(trips=trips.copy(),
                                   connected_node_features=connected_node_features.copy(),
                                   training=False,
                                   number_of_classes=67566,
                                   training_percentage=0.2),
    batch_size=16)


if config_file.exists():
    with open(config_file, 'r') as outfile:
        metadata = json.load(outfile)

    for key, value in metadata.items():
        if 'path' in value:
            model_path = value['path']
            starting_iteration = int(key)

    model = torch.load(model_path)

else:
    model = helper_functions.LinearNN(city_numbers=67566)
    metadata = {}
    starting_iteration = 0


optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()
model.to(device)

steps = 0
running_loss = 0
train_losses = []
test_losses = []
accuracies = []
metadata = {}

for epoch in range(epochs):
    for final_city, trip_cities, previous_cities, node_features in train_loader:
        steps += 1

        trip_cities = trip_cities.to_dense().to(device)
        previous_cities = previous_cities.to_dense().to(device)
        closeness = node_features.to_dense()[:, 0:1, ].to(device)
        betweenness = node_features.to_dense()[:, 1:2, ].to(device)
        triangles = node_features.to_dense()[:, 2:, ].to(device)
        final_city = final_city.to(device)

        optimizer.zero_grad()
        logps = model(closeness, betweenness, triangles, trip_cities, previous_cities)
        # times by the cities which previously have been visited by the latest city
        valid_cities = (closeness > 0).float()
        logps = logps * valid_cities

        loss = criterion(logps.squeeze(1), final_city.type_as(logps))
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

    save_path = model_location / f'booking_model_{epoch + 1}.pth'
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


with open(config_file, 'w') as outfile:
    json.dump(metadata, outfile)
