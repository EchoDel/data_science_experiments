from pathlib import Path
import torch
from torch import optim
import json
from torch import nn

import bookingdotcom.helper_functions as helper_functions

cache_location = Path('bookingdotcom/cache/')
epochs = 1000
save_every = 10
device = 'cuda'
device = 'cpu'
model_location = Path('models/bookingdotcom/')
config_file = model_location / 'metadata.json'

connected_node_features = torch.load(cache_location / 'connected_node_features.pkl')
trips = torch.load(cache_location / 'trip_properties.pkl')


train_loader = torch.utils.data.DataLoader(
    helper_functions.BookingLoader(trips=trips.copy(),
                                   connected_node_features=connected_node_features.copy(),
                                   training=True,
                                   number_of_classes=67566,
                                   training_percentage=0.8),
    batch_size=256)

test_loader = torch.utils.data.DataLoader(
    helper_functions.BookingLoader(trips=trips.copy(),
                                   connected_node_features=connected_node_features.copy(),
                                   training=False,
                                   number_of_classes=67566,
                                   training_percentage=0.8),
    batch_size=256)


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


optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()
model.to(device)

steps = 0
running_loss = 0
train_losses = []
test_losses = []
accuracies = []
metadata = {}

for epoch in range(epochs):
    for final_city, trip_cities, previous_cities, node_features, trip_id in train_loader:
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
        for test_final_city, test_trip_cities, test_previous_cities, test_node_features, trip_id in test_loader:
            test_trip_cities = test_trip_cities.to_dense().to(device)
            test_previous_cities = test_previous_cities.to_dense().to(device)
            closeness = test_node_features.to_dense()[:, 0:1, ].to(device)
            betweenness = test_node_features.to_dense()[:, 1:2, ].to(device)
            triangles = test_node_features.to_dense()[:, 2:, ].to(device)
            test_final_city = test_final_city.to(device)

            test_logps = model(closeness, betweenness, triangles, test_trip_cities, test_previous_cities)

            valid_cities = (closeness > 0).float()
            test_logps = test_logps * valid_cities

            predictions = []
            for final_city_, maximum_cities in zip(test_final_city.topk(1)[1], test_logps.topk(4)[1]):
                predictions.append(final_city_ in maximum_cities)

            accuracy += sum(predictions) / len(test_final_city)

    train_losses.append(running_loss / len(train_loader.dataset))
    test_losses.append(test_loss / len(test_loader.dataset))
    accuracies.append(accuracy)
    print(f"Epoch {epoch + 1}/{epochs}.. "
          f"Train loss: {running_loss / len(train_loader.dataset):.3f}.. "
          f"Test accuracy: {accuracy:.3f}")
    running_loss = 0
    model.train()

    save_path = model_location / f'booking_model_{epoch + 1}.pth'
    metadata[epoch + 1] = {
        'running_loss': running_loss / len(train_loader.dataset),
        'accuracy': accuracy / len(test_loader.dataset)
    }

    if epoch == 0:
        model_location.mkdir(parents=True, exist_ok=True)
        metadata[epoch + 1]['path'] = str(save_path)
        torch.save(model, save_path)
    elif epoch % save_every == 1:
        metadata[epoch + 1]['path'] = str(save_path)
        torch.save(model, save_path)
    elif accuracy / len(test_loader.dataset) > max(accuracies):
        metadata[epoch + 1]['path'] = str(save_path)
        torch.save(model, save_path)


with open(config_file, 'w') as outfile:
    json.dump(metadata, outfile)
