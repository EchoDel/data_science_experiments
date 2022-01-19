import pandas as pd

room_features = ['8Ball', 'Beer', 'Board', 'Books', 'Bookshelf', 'Boombox', 'Cap', 'Fish', 'Coffee', 'Flower', 'Glasses',
                 'Guitar', 'Lamp', 'Light', 'Martini', 'Moneyplant', 'Pizza', 'Cheeseplant', 'Rubberplant', 'Skull',
                 'Trailingplant', 'Umbrella',  'Windowday', 'Windownight', 'Wine']

rooms = []

for x in room_features:
    for y in room_features:
        if y == x:
            continue
        for z in room_features:
            if (z == x) or (z == x):
                continue
            rooms.append([x, y, z])


rooms = pd.DataFrame(rooms)

rooms.to_csv('dead_fellas_rooms/rooms.csv')
rooms.sample(500).to_csv('dead_fellas_rooms/sampled_rooms.csv')
