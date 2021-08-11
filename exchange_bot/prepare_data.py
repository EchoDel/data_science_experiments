from pathlib import Path
import pickle as pkl

import pandas as pd

temp_file = Path('./cache/exchange_bot_data.pkl')
temp_file.parent.mkdir(exist_ok=True, parents=True)

try:
    final_data = pkl.load(open(temp_file, 'rb'))

except Exception as E:
    print(E)
    source_data_location = '../data/exchange_bot/20YD.csv'

    source_data = pd.read_csv(source_data_location)

    source_data['timestamp'] = pd.to_datetime(source_data['Date'] +
                                              source_data['Time'],
                                              format='%Y.%m.%d%H:%M')

    source_data = source_data.drop('Time', axis='columns')
    source_data = source_data.drop('Date', axis='columns')

    source_data = source_data.reset_index(drop=True)

    source_data = source_data.drop('High', axis='columns')
    source_data = source_data.drop('Low', axis='columns')
    source_data = source_data.drop('Open', axis='columns')
    source_data = source_data.drop('Volume', axis='columns')

    for observation in range(1, 41):
        observation_period = observation * 60
        source_data['EMA' + str(observation_period)] = \
            source_data.Close.ewm(observation_period).mean()

    pkl.dump(source_data, open(temp_file, 'wb'))

    final_data = source_data.copy()
