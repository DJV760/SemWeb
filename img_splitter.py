from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None

large_img = Image.open('C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\GP3_orto_A3_2.png')
width, height = 512, 512

output_directory = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\512_images'

for i in range(0, large_img.width, width):
    for j in range(0, large_img.height, height):
        box = (i, j, i+width, j+height)
        region = large_img.crop(box)
        filename = f'Image_{i}_{j}.png'
        output_path = os.path.join(output_directory, filename)
        region.save(output_path)