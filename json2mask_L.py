import json
from PIL import Image, ImageDraw

file_path = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\VIA_coord_json.json'

with open(file_path, 'r') as file:
    data = json.load(file)

shape_attributes = data['GP3_orto_A3.png1084532140']['regions']

all_regions = []

for attribute in range(0, len(shape_attributes)):
    x_coord = shape_attributes[attribute]['shape_attributes']['all_points_x']
    y_coord = shape_attributes[attribute]['shape_attributes']['all_points_y']

    classes = shape_attributes[attribute]['region_attributes']
    for terrain_class in classes:
        if classes[f'{terrain_class}'] == '1':
            mask_class = terrain_class

    coord_tuples = [(x, y) for x, y in zip(x_coord, y_coord)]

    all_regions.append([coord_tuples, mask_class])

image = Image.new('L', (16453, 16453))
draw = ImageDraw.Draw(image)

for region in all_regions:
    mask_class = region[-1]
    coord_tuples = region[-2]
    if mask_class == 'tree canopy':
        fill_color = 1
    elif mask_class == 'water':
        fill_color = 2
    elif mask_class == 'rock':
        fill_color = 3
    elif mask_class == 'dirt':
        fill_color = 4
    elif mask_class == 'sand':
        fill_color = 5
    elif mask_class == 'waterlilly':
        fill_color = 6
    elif mask_class == 'none':
        fill_color = 7
    elif mask_class == 'swamp':
        fill_color = 8
    elif mask_class == 'grass':
        fill_color = 9

    draw.polygon(coord_tuples, fill_color)

image.save('C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\mask_foundation_8bit.png')
image.show()