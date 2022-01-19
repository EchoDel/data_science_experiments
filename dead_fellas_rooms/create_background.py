import itertools
import json
import random

from PIL import Image, ImageDraw, ImageFilter, ImageOps

# Read in all the config file
with open('dead_fellas_rooms/base_images/png/samples.json') as json_file:
    feature_data = json.load(json_file)

silhouette = Image.open('dead_fellas_rooms/base_images/png/silhouette.png')
possible_location = ['top_left', 'bottom_left', 'top_right', 'bottom_right']


def setup_base_image():
    # Setup a blank image
    mode = 'RGB'
    size = (1000, 1000)
    color = (0, 0, 0)

    image = Image.new(mode, size, color)

    # add the silhouette
    draw = ImageDraw.Draw(image)
    draw.ellipse((1, 1, 998, 998), fill=(255, 255, 255), outline=(255, 255, 255))
    image = image.filter(ImageFilter.GaussianBlur(2))
    mask = image.filter(ImageFilter.GaussianBlur(2)).convert('L')

    return image, mask


def draw_silhouette_layer(image, mask):
    image.paste(silhouette, (0, 0), silhouette)
    return image

def draw_circle_layer(image, mask):
    image.paste(mask, (0, 0), ImageOps.invert(mask))
    return image

number_of_items = 3


def place_item(base_image: Image, item: str, location):
    item_image = Image.open(item['image_location'])
    # need to move based on the size of the image
    if location == 'top_left':
        start_location = (300, 100)
    elif location == 'top_right':
        start_location = (800, 100)
    elif location == 'bottom_left':
        start_location = (50, 700)
    elif location == 'bottom_right':
        start_location = (800, 700)

    paste_location = tuple(x + y - int(z / 2) + random.randint(-30, 30)
                           for x, y, z in zip(start_location,
                                              feature_data[items[2]]['offset'],
                                              item_image.size))
    base_image.paste(item_image, paste_location, item_image)




# Create images from all the combinations
for n, items in enumerate(itertools.combinations(feature_data.keys(), number_of_items)):
    # skip over if there are too many of the same type of item
    # currently skips over if there is more than one from the same category
    if len(set([feature_data[x]['location'] for x in items])) != 3:
        continue
    # skips if two or more items have to be on the left
    left_items = [x for x in items if feature_data[x]['side'] == 'left']
    right_items = [x for x in items if feature_data[x]['side'] == 'right']
    other_items = [x for x in items if not feature_data[x]['side'] in ['right', 'left']]
    if len(left_items) > 1:
        continue
    # skips if two or more items are on the right
    if len(right_items) > 1:
        continue
    # create a dict of items and location based on the rules to be defined. Test case provided

    locations = {}

    if left_items:
        if feature_data[left_items[0]]['location'] == 'floor':
            locations['bottom_left'] = feature_data[left_items[0]]
        elif feature_data[left_items[0]]['location'] == 'roof':
            locations['top_left'] = feature_data[left_items[0]]
        else:
            locations[random.sample(['bottom_left', 'top_left'], 1)[0]] = feature_data[left_items[0]]

    if right_items:
        if feature_data[right_items[0]]['location'] == 'floor':
            locations['bottom_right'] = feature_data[right_items[0]]
        elif feature_data[right_items[0]]['location'] == 'roof':
            locations['top_right'] = feature_data[right_items[0]]
        else:
            locations[random.sample(['bottom_right', 'top_right'], 1)[0]] = feature_data[right_items[0]]

    for item in other_items:
        remaining_location = [x for x in possible_location if x not in locations.keys()]
        if feature_data[item]['location'] == 'floor':
            location = random.sample(
                [x for x in remaining_location if 'bottom' in x], 1)[0]
        elif feature_data[item]['location'] == 'roof':
            location = random.sample(
                [x for x in remaining_location if 'top' in x], 1)[0]
        else:
            location = random.sample(remaining_location, 1)[0]

        locations[location] = feature_data[item]

    # produce the image
    im, mask = setup_base_image()

    for location, item_path in locations.items():
        place_item(im, item_path, location)

    im = draw_circle_layer(im, mask)
    im.save(f'dead_fellas_rooms/outputs/test{n}_{"-".join(items)}.png')
    im = draw_silhouette_layer(im, mask)
    im.save(f'dead_fellas_rooms/outputs/test_silhouette_{n}_{"-".join(items)}.png')
    # break




