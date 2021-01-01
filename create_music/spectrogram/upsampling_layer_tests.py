import torch
import torch.nn as nn

input_x = 17
input_y = 17
input_z = 512

input_tensor = torch.randn(1, input_z, input_y, input_x)

previous_x = 0
for z in range(input_z):
    for y in range(input_y):
        for x in range(input_x):
            input_tensor[0, z, y, x] = previous_x
            previous_x += 1


# Test Pixel Shuffle

pixel_shuffle_output = nn.PixelShuffle(16)(input_tensor)
pixel_shuffle_output.shape

# Test Flatten

flatten_output = nn.Flatten(2)(input_tensor)
flatten_output.shape

# Test .view

view_output = input_tensor.view(1, 512, 289)
view_output.shape
