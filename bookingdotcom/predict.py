from pathlib import Path
import torch
import pandas as pd

import bookingdotcom.helper_functions as helper_functions

cache_location = Path('bookingdotcom/cache/')
epochs = 1000
save_every = 10
device = 'cuda'
device = 'cpu'
model_location = Path('models/bookingdotcom/')
config_file = model_location / 'metadata.json'
model_path = model_location / 'booking_model_11.pth'

connected_node_features = torch.load(cache_location / 'connected_node_features.pkl')
trips = torch.load(cache_location / 'test_trip_properties.pkl')

test_loader = torch.utils.data.DataLoader(
    helper_functions.BookingLoader(trips=trips.copy(),
                                   connected_node_features=connected_node_features.copy(),
                                   training=True,
                                   number_of_classes=67566,
                                   training_percentage=1),
    batch_size=8000)


model = torch.load(model_path)
model.eval()
output = []

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

        selected_city = test_logps.topk(4)

        selected_cities = selected_city[1].squeeze(1).detach().numpy()
        indexes = trip_id

        selected_cities = pd.DataFrame(selected_cities)
        selected_cities.index = trip_id
        output.append(selected_cities)
        print('done')

final_output = pd.concat(output)
final_output.to_csv(model_location / 'output.csv')
