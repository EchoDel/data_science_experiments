from pathlib import Path
from bird_sounds import helper_functions
import numpy as np

from matplotlib import pyplot as plt

metadata_path = Path('../bird_sounds/ff1010bird_metadata.csv')


metadata = helper_functions.load_metadata(metadata_path)

data, rate = helper_functions.load_sound_file(metadata.iloc[0,2])

frequency_graph = helper_functions.spectogram_creation(data, rate, None)

plt.pcolormesh(frequency_graph[1], frequency_graph[0], frequency_graph[2], shading='gouraud')